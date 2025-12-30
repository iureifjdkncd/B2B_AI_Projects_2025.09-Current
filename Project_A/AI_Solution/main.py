import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.det import apply_enhanced_det
from src.utils import *
from src.visualization import *
from sklearn.metrics import recall_score, confusion_matrix

plt.rcParams["font.family"] = "DejaVu Sans"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = None

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    file_path = os.path.join(args.save_train_result_path, f"checkpoint.ckpt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "accuracy_list": accuracy_list,
        },
        file_path,
    )

def freeze_model_parameters(model):
    """
    TranAD 모델의 Fine-tuning을 위한 파라미터 Freezing 함수.
    
    Args:
        model: 학습할 모델 객체
    """

    print(f"{color.YELLOW}Start Freezing Model Parameters (Target: Encoder)...{color.ENDC}")
    
    frozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        
        # 1. Positional Encoding 및 Transformer Encoder 고정
        if 'transformer_encoder' in name or 'pos_encoder' in name:
            param.requires_grad = False
            frozen_count += 1
            
        # 2. Decoder 및 FCN(Fully Connected Network)은 학습 가능하도록 유지
        # 'transformer_decoder1', 'transformer_decoder2', 'fcn' 은 True 유지
        else:
            param.requires_grad = True
            
    print(f"{color.OKBLUE}Freezing Completed. {frozen_count}/{total_count} layers are frozen.{color.ENDC}")
    print(f"{color.OKBLUE}Trainable components: Decoders & FCN{color.ENDC}")

def load_model(modelname, dims, window_size, batch_size, args):
    import src.models  # type: ignore[import]

    model_class = getattr(src.models, modelname)
    model = model_class(dims, window_size, batch_size).double()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1
    accuracy_list = []

    # 모델 파일 경로가 지정되었거나, Resume/Finetune 모드인 경우 로드 시도
    if args.model_path or args.resume or args.finetune:
        
        # 경로 설정 (지정 안 했으면 기본 경로 사용)
        load_path = args.model_path if args.model_path else os.path.join(args.save_train_result_path, "checkpoint.ckpt")
        
        if not os.path.exists(load_path):
             # Test 모드에서는 파일 없으면 에러, Train(New)이면 그냥 진행
             if args.phase == 'test':
                 raise FileNotFoundError(f"모델 파일이 없습니다: {load_path}")
             else:
                 print("체크포인트가 없어 새로 학습을 시작합니다.")
                 return model, optimizer, scheduler, epoch, accuracy_list

        print(f"Loading checkpoint from: {load_path}")
        checkpoint = torch.load(load_path)

        # -----------------------------------------------------------
        # CASE 1: Resume (이어서 학습)
        # -----------------------------------------------------------
        if args.phase == 'train' and args.resume:
            print(">> Mode: RESUME Training")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # Optimizer 상태 복구
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"]
            accuracy_list = checkpoint["accuracy_list"]

        # -----------------------------------------------------------
        # CASE 2: Fine-tune (미세 조정)
        # -----------------------------------------------------------
        elif args.phase == 'train' and args.finetune:
            print(">> Mode: FINE-TUNE Training")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            # 1. Freeze 적용
            freeze_model_parameters(model)

            # 2. Optimizer 재설정 및 LR 감소 적용
            finetune_lr = args.learning_rate * 0.1  # 예: 기존 LR의 1/10
            print(f">> Re-initializing Optimizer with reduced LR: {finetune_lr}")
            
            # Freeze된 파라미터는 업데이트 대상에서 제외해야 함
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=finetune_lr, 
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        # -----------------------------------------------------------
        # CASE 3: Test (추론)
        # -----------------------------------------------------------
        elif args.phase == 'test':
            print(">> Mode: INFERENCE")
            model.load_state_dict(checkpoint["model_state_dict"])
            # 추론 시에는 Optimizer, Epoch 등은 필요 없음

        # -----------------------------------------------------------
        # CASE 4: New Training (파일은 있지만 무시하고 새로 시작)
        # -----------------------------------------------------------
        else:
            print(">> Mode: NEW Training (Ignoring checkpoint)")
            # 아무것도 로드하지 않음

    else:
        print(">> Mode: NEW Training (No checkpoint provided)")

    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction="mean" if training else "none")
    feats = dataO.shape[1]
    if "TranAD" in model.name:
        l = nn.MSELoss(reduction="none")
        bs = model.batch
        dataloader = DataLoader(data, batch_size=bs)
        n = epoch + 1
        l1s = []
        if training:
            for d in dataloader:
                d = d.to(device)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = (
                    l(z, elem)
                    if not isinstance(z, tuple)
                    else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                )
                if isinstance(z, tuple):
                    z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f"Epoch {epoch},\tL1 = {np.mean(l1s)}")
            return np.mean(l1s), optimizer.param_groups[0]["lr"]
        else:
            all_loss = []
            all_preds = []

            total_batches = len(dataloader)
            desc = "Testing" if epoch == 0 else f"Inference (Epoch {epoch})"

            with tqdm(
                total=total_batches,
                desc=desc,
                unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            ) as pbar:
                for batch_idx, d in enumerate(dataloader):
                    d = d.to(device)
                    window = d.permute(1, 0, 2)
                    local_bs = d.shape[0]
                    elem = window[-1, :, :].view(1, local_bs, feats)
                    with torch.no_grad():
                        z = model(window, elem)
                        if isinstance(z, tuple):
                            z = z[1]
                    batch_loss = l(z, elem)[0]
                    all_loss.append(batch_loss.detach().cpu().numpy())
                    all_preds.append(z.detach().cpu().numpy()[0])
                    pbar.update(1)
                    pbar.set_postfix(
                        {"Batch": f"{batch_idx+1}/{total_batches}", "Shape": f"{local_bs}x{feats}"}
                    )
            loss_full = np.concatenate(all_loss, axis=0)
            pred_full = np.concatenate(all_preds, axis=0)
            return loss_full, pred_full


def _resolve_dataset_path(path_hint: str, description: str) -> str:
    if not path_hint:
        raise ValueError(f"{description} 경로가 지정되지 않았습니다.")
    if not os.path.exists(path_hint):
        raise FileNotFoundError(f"{description} 경로를 찾을 수 없습니다: {path_hint}")
    return path_hint



def load_text(file_path: str) -> list:
    """텍스트 파일에서 한 줄씩 읽어 리스트로 반환"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def _load_columns(dataset_dir: str, columns_file: str):
    columns_path = os.path.join(dataset_dir, columns_file)
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"컬럼 목록 파일을 찾을 수 없습니다: {columns_path}")
    return load_text(columns_path)


def _scale_split(data: np.ndarray, scaler_path: str, fit_required: bool):
    # scaler파일이 저장되는 경로가 부재하면 생성
    scaler_dir = os.path.dirname(scaler_path)
    if scaler_dir and not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir, exist_ok=True)
    if fit_required or not os.path.exists(scaler_path):
        scaled, scaler = scaling(data, None, scaler_path)
        return scaled, scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"스케일러 파일을 찾을 수 없습니다: {scaler_path}. 먼저 학습을 실행해주세요."
        )
    scaler = joblib.load(scaler_path)
    scaled = scaling(data, scaler)
    return scaled, scaler

# 학습 데이터 로드
def _load_training_split(args, fit_required: bool):
    dataset_path = Path(_resolve_dataset_path(args.train_dataset, "학습 데이터"))
    columns = _load_columns(dataset_path.parent, args.columns)
    train_path = dataset_path
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv 파일을 찾을 수 없습니다: {train_path}")

    train_df = pd.read_csv(train_path)
    file_number = train_df["file_number"].values

    data = train_df[columns].values
    scaler_path = os.path.join(args.save_train_result_path, "TranAD_scaler.pkl")
    scaled_data, scaler = _scale_split(data, scaler_path, fit_required=fit_required)

    train_loader = DataLoader(
        scaled_data, batch_size=scaled_data.shape[0], pin_memory=True, num_workers=4
    )

    return {
        "train_loader": train_loader,
        "train_file_number": file_number,
        "columns": columns,
        "feature_dim": len(columns),
        "train_scaler": scaler,
    }

# Evaluation 데이터 로드
def _load_eval_split(args, scaler, columns):
    if not args.eval_dataset:
        return None
    
    dataset_path = Path(_resolve_dataset_path(args.eval_dataset, "Eval 데이터"))
    eval_path = dataset_path
    
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"eval.csv 파일을 찾을 수 없습니다: {eval_path}")

    eval_df = pd.read_csv(eval_path)
    eval_df['attack'] = 1
    file_number = eval_df["file_number"].values
    
    scaled_data = scaling(eval_df[columns].values, scaler)
    
    eval_loader = DataLoader(
        scaled_data, batch_size=scaled_data.shape[0], pin_memory=True, num_workers=4
    )
    
    return {
        "eval_loader": eval_loader,
        "eval_file_number": file_number
    }

def _load_calibration_split(args, scaler, columns):
    # Note: run_inference 단계에서도 calibration_normal_dataset을 로드하기 위해 사용됨
    if scaler is None:
        raise RuntimeError("스케일러가 준비되지 않았습니다. 먼저 학습을 실행했는지 확인하세요.")

    dataset_calibration_normal_path = args.calibration_normal_dataset
    if dataset_calibration_normal_path and os.path.exists(dataset_calibration_normal_path):
        dataset_calibration_normal_path = Path(dataset_calibration_normal_path)
    else:
        # Inference 단계에서 선택적으로 로드할 때 파일이 없으면 그냥 빈 dict 리턴
        return {}

    columns = _load_columns(dataset_calibration_normal_path.parent, args.columns)

    calibration_data = {
        "columns": columns,
        "feature_dim": len(columns),
    }

    if os.path.exists(dataset_calibration_normal_path):
        calibration_normal_df = pd.read_csv(dataset_calibration_normal_path)
        calibration_normal_file_number = (
            calibration_normal_df["file_number"].values if "file_number" in calibration_normal_df.columns else None
        )
        calibration_normal_scaled = scaling(calibration_normal_df[columns].values, scaler)
        calibration_normal_loader = DataLoader(
            calibration_normal_scaled,
            batch_size=calibration_normal_scaled.shape[0],
            pin_memory=True,
            num_workers=4,
        )
        calibration_data.update({
            "calibration_normal_loader": calibration_normal_loader,
            "calibration_normal_file_number": calibration_normal_file_number,
        })

    return calibration_data

def _load_inference_split(args, scaler, columns):
    if scaler is None:
        raise RuntimeError("스케일러가 준비되지 않았습니다. 먼저 학습을 실행했는지 확인하세요.")

    dataset_test_path = Path(_resolve_dataset_path(args.test_dataset, "추론 데이터"))
    columns = _load_columns(dataset_test_path.parent, args.columns)
    test_path = dataset_test_path
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv 파일을 찾을 수 없습니다: {test_path}")

    test_df = pd.read_csv(test_path)
    test_df['attack'] = 1
    test_file_number = test_df["file_number"].values
    test_scaled = scaling(test_df[columns].values, scaler)
    test_loader = DataLoader(
        test_scaled, batch_size=test_scaled.shape[0], pin_memory=True, num_workers=4
    )
    inference_data = {
        "test_loader": test_loader,
        "test_file_number": test_file_number,
        "columns": columns,
        "feature_dim": len(columns),
    }
    return inference_data


def prepare_dataloaders(args):
    loaders = {}
    train_phase = args.phase == "train"
    infer_phase = args.phase == "test"

    train_scaler = None
    columns = None

    if train_phase:
        fit_required = not (args.resume or args.finetune)

        # [수정] Fine-tuning 시 Scaler 로드 경로 처리
        if args.finetune and args.model_path:
            # Base 모델 폴더에서 Scaler 로드
            base_dir = os.path.dirname(args.model_path)
            base_scaler_path = os.path.join(base_dir, "TranAD_scaler.pkl")
            
            # 현재(새로운) 결과 폴더로 복사해둘 경로
            current_scaler_path = os.path.join(args.save_train_result_path, "TranAD_scaler.pkl")
            
            if os.path.exists(base_scaler_path):
                print(f"Copying scaler from base model: {base_scaler_path}")
                scaler = joblib.load(base_scaler_path)
                joblib.dump(scaler, current_scaler_path) # 새 폴더에 저장 (나중에 쓰기 위해)
            else:
                raise FileNotFoundError(f"Base scaler not found at {base_scaler_path}")

        train_data = _load_training_split(args, fit_required=fit_required)
        loaders.update(train_data)
        train_scaler = train_data["train_scaler"]
        columns = train_data["columns"]
        
        # Eval Dataset Load
        if args.eval_dataset:
            eval_data = _load_eval_split(args, train_scaler, columns)
            if eval_data:
                loaders.update(eval_data)

    if infer_phase:
        train_scaler = os.path.join(args.save_train_result_path, "TranAD_scaler.pkl")
        train_scaler = joblib.load(train_scaler)
        
        # Test 데이터 로드
        columns_test = _load_columns(Path(args.test_dataset).parent, args.columns)
        inference_data = _load_inference_split(args, train_scaler, columns_test)
        loaders.update(inference_data)

        # Calibration Normal 데이터 로드 (Inference 시 임계치 보정용)
        if args.calibration_normal_dataset:
             # 컬럼 정의가 다를 수 있으므로 경로 기반으로 다시 로드 시도
             try:
                columns_calib = _load_columns(Path(args.calibration_normal_dataset).parent, args.columns)
                calibration_data = _load_calibration_split(args, train_scaler, columns_calib)
                loaders.update(calibration_data)
             except Exception as e:
                 print(f"{color.WARNING}Calibration data loading failed: {e}{color.ENDC}")

    return loaders

def build_phase_context(args, model, loaders):
    context = {}
    
    # Train Dataset Context
    if "train_loader" in loaders:
        train_tensor = next(iter(loaders["train_loader"])).to(device)
        train_dataset = SlidingWindowDataset(
            train_tensor,
            loaders["train_file_number"],
            model,
            args.window_size,
            device=device,
        )
        train_window_info = collect_window_outputs(
            train_dataset,
            loaders["train_file_number"],
            args.window_size
        )
        context.update({
            "train_dataset": train_dataset,
            "train_windows": train_window_info["windows"],
            "train_file_number_windowed": train_window_info.get("file_numbers_windowed"),
            "columns": loaders["columns"],
        })

    # Eval Dataset Context
    if "eval_loader" in loaders:
        eval_tensor = next(iter(loaders["eval_loader"])).to(device)
        eval_dataset = SlidingWindowDataset(
            eval_tensor,
            loaders["eval_file_number"],
            model,
            args.window_size,
            device=device,
        )
        eval_window_info = collect_window_outputs(
            eval_dataset,
            loaders["eval_file_number"],
            args.window_size
        )
        context.update({
            "eval_dataset": eval_dataset,
            "eval_windows": eval_window_info["windows"],
            "eval_file_number_windowed": eval_window_info.get("file_numbers_windowed"),
        })

    # Test Context
    if "test_loader" in loaders:
        test_tensor = next(iter(loaders["test_loader"])).to(device)
        test_dataset = SlidingWindowDataset(test_tensor, loaders["test_file_number"], model, args.window_size, device=device)
        test_window_info = collect_window_outputs(test_dataset, loaders["test_file_number"], args.window_size)
        context.update({"test_dataset": test_dataset, "test_windows": test_window_info["windows"], "test_file_number_windowed": test_window_info.get("file_numbers_windowed"), "columns": loaders["columns"]})
    
    # Calibration Normal Context (for Inference)
    if "calibration_normal_loader" in loaders:
        tensor = next(iter(loaders["calibration_normal_loader"])).to(device)
        dataset = SlidingWindowDataset(tensor, loaders["calibration_normal_file_number"], model, args.window_size, device=device)
        window_info = collect_window_outputs(dataset, loaders["calibration_normal_file_number"], args.window_size)
        context.update({"calibration_normal_dataset": dataset, "calibration_normal_windows": window_info["windows"], "calibration_normal_file_number_windowed": window_info.get("file_numbers_windowed")})

    return context

# train 단계
def run_training(args, model, optimizer, scheduler, context):
    
    train_dataset = context["train_dataset"]
    train_windows = context["train_windows"]
    
    # -------------------------------------------------------
    # 1. Training Loop
    # -------------------------------------------------------
    start_epoch = 0
    accuracy_list = []
    train_display = args.train_dataset or "unknown-dataset"
    print(f"{color.HEADER}Training {args.model} on {train_display}{color.ENDC}")
    
    num_epochs = args.epoch
    last_epoch = start_epoch
    start_time = time()
    
    for e in tqdm(list(range(start_epoch + 1, start_epoch + num_epochs + 1))):
        lossT, lr = backprop(e, model, train_dataset, train_windows, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
        last_epoch = e
        
    print(f"{color.BOLD}Training time: {time() - start_time:10.4f} s{color.ENDC}")
    save_model(model, optimizer, scheduler, last_epoch, accuracy_list)
    plot_accuracies(accuracy_list, args)

    # -------------------------------------------------------
    # 2. Score Calculation
    # -------------------------------------------------------
    # Train Data Score
    lossT, _ = backprop(0, model, context["train_dataset"], context["train_windows"], optimizer, scheduler, training=False)
    lossTfinal = np.mean(lossT, axis=1)

    # Eval Data Score (존재 시)
    lossEvalFinal = None
    if "eval_dataset" in context:
        lossEval, _ = backprop(0, model, context["eval_dataset"], context["eval_windows"], optimizer, scheduler, training=False)
        lossEvalFinal = np.mean(lossEval, axis=1)

    # -------------------------------------------------------
    # 3. Threshold Setting (핵심 수정 부분)
    # -------------------------------------------------------
    threshold = None

    # [Case A] Fine-tuning 모드: 기존 모델의 임계치 로드 (업데이트 X)
    if args.finetune and args.model_path:
        print(f"{color.HEADER}Fine-tuning Mode: Loading existing threshold from base model...{color.ENDC}")
        base_dir = os.path.dirname(args.model_path)
        base_result_path = os.path.join(base_dir, "results_train.pkl")
        
        if os.path.exists(base_result_path):
            prev_results = load_pickle(base_result_path)
            threshold = prev_results["threshold"]
            print(f"{color.OKBLUE}Loaded Threshold: {threshold:.6f} (No update){color.ENDC}")
        else:
            raise FileNotFoundError(f"Base model results not found at {base_result_path}")

    # [Case B] 일반 학습 모드: 임계치 새로 계산
    else:
        if lossEvalFinal is not None and args.calib_by_eval == "True":
            # Eval 데이터 기반 보정
            _, threshold = apply_enhanced_det(
                train_errors=lossTfinal,
                test_errors=lossEvalFinal,
                min_anomaly_duration=3,
                return_thresholds=True
            )
            print(f"{color.OKBLUE}Calibrated Threshold: {threshold:.6f}{color.ENDC}")
        else:
            # Train 데이터 기반 계산 (DET/POT)
            threshold = apply_enhanced_det(train_errors=lossTfinal)
            print(f"{color.OKBLUE}Threshold: {threshold:.6f}{color.ENDC}")

    # -------------------------------------------------------
    # 4. Metrics & Plotting (공통 실행)
    # -------------------------------------------------------
    # Eval 데이터가 있을 경우에만 메트릭 계산 및 플롯
    if "eval_dataset" in context and lossEvalFinal is not None:
        
        # DataFrame 생성
        train_file_nums = context["train_file_number_windowed"]
        df_train = pd.DataFrame({
            "score": lossTfinal,
            "label": 0,
            "file_number": train_file_nums
        })
        
        eval_file_nums = context["eval_file_number_windowed"]
        df_eval = pd.DataFrame({
            "score": lossEvalFinal,
            "label": 1,
            "file_number": eval_file_nums
        })
        
        # Merge
        df_merged = pd.concat([df_train, df_eval], ignore_index=True)
        df_merged = df_merged.sort_values(by=["file_number", "label"]).reset_index(drop=True)
        
        # Metrics
        combined_scores = df_merged["score"].values
        combined_labels = df_merged["label"].values
        combined_preds = (combined_scores > threshold).astype(int)
        
        from src.det import remove_short_anomalies
        combined_preds = remove_short_anomalies(combined_preds, min_duration=3)
        
        recall = recall_score(combined_labels, combined_preds)
        cm = confusion_matrix(combined_labels, combined_preds)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0
            
        print(f"{color.OKBLUE}Evaluation Recall: {recall:.4f}, FPR: {fpr:.4f}{color.ENDC}")
        
        # Plotting
        plot_result_eval_combined(
            df_merged=df_merged,
            combined_preds=combined_preds,
            threshold=threshold,
            fpr=fpr,
            recall=recall,
            args=args
        )
        
        results = {
            "model": "TranAD",
            "train_anomaly_score": lossTfinal,
            "eval_anomaly_score": lossEvalFinal,
            "combined_metrics": {"recall": recall, "fpr": fpr},
            "threshold": threshold,
        }
    else:
        # Eval 데이터가 없는 경우
        results = {
            "model": "TranAD",
            "train_anomaly_score": lossTfinal,
            "threshold": threshold,
        }

    save_pickle(os.path.join(args.save_train_result_path, "results_train.pkl"), results)

# inference 단계
def run_inference(args, model, optimizer, scheduler, context):
    torch.zero_grad = True
    model.eval()
    test_display = args.test_dataset or "unknown-dataset"
    print(f"{color.HEADER}Testing {args.model} on {test_display}{color.ENDC}")

    # 1. Calibration using Calibration Normal Dataset (If available)
    if "calibration_normal_dataset" in context and context["calibration_normal_dataset"] is not None:
        print(f"{color.HEADER}Calibrating threshold using Calibration Normal Dataset...{color.ENDC}")
        loss_calib, _ = backprop(
            0,
            model,
            context["calibration_normal_dataset"],
            context["calibration_normal_windows"],
            optimizer,
            scheduler,
            training=False,
        )
        lossCalibFinal = np.mean(loss_calib, axis=1)
        base_dir = os.path.dirname(args.model_path)
        base_result_path = os.path.join(base_dir, "results_train.pkl")
        
        if os.path.exists(base_result_path):
            prev_results = load_pickle(base_result_path)
            lossTfinal = prev_results["train_anomaly_score"]
            lossEvalFinal = prev_results["eval_anomaly_score"]
        else:
            raise FileNotFoundError(f"Base model results not found at {base_result_path}")
        
        # 정상 데이터(Calibration Normal)만 사용하여 POT 알고리즘 등으로 임계치 산출
        _, threshold = apply_enhanced_det(
            train_errors=lossTfinal,
            test_errors=lossEvalFinal,
            train_errors_B=lossCalibFinal,
            min_anomaly_duration=3,
            return_thresholds=True
        )
        print(f"{color.OKBLUE}Recalibrated Threshold: {threshold:.6f}{color.ENDC}")
        
    else:
        # Fallback: 기존 저장된 임계치 사용
        print(f"{color.WARNING}Calibration dataset not found. Using pre-trained threshold.{color.ENDC}")
        if os.path.exists(os.path.join(args.save_train_result_path, "results_train.pkl")):
            threshold = load_pickle(os.path.join(args.save_train_result_path, "results_train.pkl"))["threshold"]
        else:
            raise FileNotFoundError("Pre-trained threshold not found.")

    # 2. Run Inference on Test Dataset
    test_df = pd.read_csv(args.test_dataset)
    original_index_mapping = test_df[['original_index', 'segment_original_start', 'segment_original_end']].values
   
    loss, y_pred = backprop(
        0,
        model,
        context["test_dataset"],
        context["test_windows"],
        optimizer,
        scheduler,
        training=False,
    )
    lossFinal = np.mean(loss, axis=1)

    # 새로운 경로 생성 (test 모드)
    kst = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(kst).strftime("%y%m%d_%H%M%S")
    setting = f"test_window_{args.window_size}_{current_time}"
    args.save_test_result_path = os.path.join(
        args.save_result_dir, args.task_name, setting
    )
    path = args.save_test_result_path
    if not os.path.exists(args.save_test_result_path):
        os.makedirs(path)
    print(f"Creating new directory: {args.save_test_result_path}")

    y_pred = (lossFinal > threshold).astype(int)
    from src.det import remove_short_anomalies
    y_pred = remove_short_anomalies(y_pred, min_duration=3)

    # 기존 인덱스로 매핑된 예측 결과 생성
    prediction_with_original_index = map_to_original_index(
        y_pred,
        lossFinal,
        original_index_mapping, 
        args.window_size
    )

    n_anomaly_window = max(5, len(context["test_windows"]))
    top_10_indices = np.argsort(lossFinal)[::-1][:n_anomaly_window].copy()
    top_10_windows = context["test_windows"][top_10_indices]


    results = {
        "model": "TranAD",
        "threshold": threshold,
        "prediction": prediction_with_original_index["prediction"],
        "anomaly_score": prediction_with_original_index["anomaly_score"],
        "anomaly_window": top_10_windows,
    }

    plot_results(
        anomaly_score=prediction_with_original_index["anomaly_score"],
        pred=prediction_with_original_index["prediction"],
        threshold=threshold,
        task=args.task_name,
        columns=context["columns"],
        save_plot_dir=path,
    )
    save_pickle(os.path.join(path, "results_test.pkl"), results)

def parse_arguments():

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    kst = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(kst).strftime("%y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Time-Series Anomaly Detection")
    parser.add_argument("--train_dataset", type=str, default="", help="학습 데이터 디렉토리")
    parser.add_argument("--eval_dataset", type=str, default="", help="평가(Eval) 데이터 디렉토리 (Optional)")
    parser.add_argument("--test_dataset", type=str, default="", help="추론 데이터 디렉토리")
    parser.add_argument("--calibration_normal_dataset", type=str, default="", help="calib 정상 데이터 (Inference 보정용)")
    parser.add_argument("--calib_by_eval", type=str, default="True", help="calib 이상 데이터 사용 여부 (True, False)")
    parser.add_argument("--model", type=str, default="TranAD", help="model name")
    parser.add_argument("--save_result_dir", type=str, default="result/")
    parser.add_argument("--model_path", type=str, default=None, help="로드할 모델 파일 경로 (.ckpt)")
    parser.add_argument("--is_calibration", type=str2bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="중단된 학습을 재개 (Optimizer, Epoch 포함 로드)")
    parser.add_argument("--finetune", action="store_true", help="사전 학습 모델을 기반으로 미세 조정 (가중치만 로드)")
    parser.add_argument("--columns", type=str, default="column_list.txt")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--task_name", type=str, default="A2-DT1-DI4-T1")
    parser.add_argument("--scaler_path", type=str, default="")
    parser.add_argument("--det_threshold", type=bool, default=True)
    parser.add_argument("--phase", type=str, choices=["train", "test"], default="train") # calibration 삭제
    parser.add_argument("--use_checkpoint_dir", type=str, default="")
    parsed_args = parser.parse_args()

    # 1. 사용자가 명시적으로 체크포인트 폴더를 지정한 경우
    if parsed_args.use_checkpoint_dir:
        parsed_args.save_train_result_path = parsed_args.use_checkpoint_dir
        print(f"Using existing checkpoint directory: {parsed_args.save_train_result_path}")
    # 2. 모델 파일 경로(--model_path)가 있고, [수정] Fine-tuning이 아닌 경우 (Resume, Test)
    #    -> 기존 폴더를 그대로 사용 (이어서 학습하거나 추론)
    elif parsed_args.model_path and not parsed_args.finetune:
        parsed_args.save_train_result_path = os.path.dirname(parsed_args.model_path)
        print(f"Setting result path to model's directory: {parsed_args.save_train_result_path}")
    # 3. 그 외 (New Training 혹은 Fine-tuning) -> 새로운 폴더 생성
    else:
        setting = f"train_window_{parsed_args.window_size}_{current_time}"
        if parsed_args.finetune:
            setting += "_finetune" # 구분하기 쉽게 이름 추가
            
        parsed_args.setting = setting
        parsed_args.save_train_result_path = os.path.join(
            parsed_args.save_result_dir, parsed_args.task_name, setting
        )
        # 새 폴더 생성
        if not os.path.exists(parsed_args.save_train_result_path):
            os.makedirs(parsed_args.save_train_result_path)
            print(f"Creating new directory: {parsed_args.save_train_result_path}")

    parsed_args.index = 0

    return parsed_args


def main(cli_args):
    global args
    args = cli_args
    loaders = prepare_dataloaders(args)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        args.model, loaders["feature_dim"], args.window_size, args.batch_size, args
    )
    model.to(device)

    context = build_phase_context(args, model, loaders)

    if args.phase == "train":
        run_training(args, model, optimizer, scheduler, context)

    if args.phase == "test":
        run_inference(args, model, optimizer, scheduler, context)

if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)