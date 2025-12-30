import sys
import json
import streamlit as st
import streamlit.components.v1 as components 
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import shap
import io
import time
import subprocess
import threading
import queue
import signal
import os
import gc
import importlib
from glob import glob
import pickle
import joblib
import webbrowser
import hashlib
import shlex
import time
import shutil
import re
from io import StringIO
import threading
import urllib.parse
import psutil
import difflib
from pathlib import Path

from reportlab.platypus import * 
from reportlab.lib.styles import * 
from reportlab.lib.pagesizes import * 
from reportlab.lib.units import * 
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))


########## Function 함수 작성 ############
def clean_memory():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        print(f"After empty_cache: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"GPU 이름: {torch.cuda.get_device_name(device)}")
            allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB 단위
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            total = torch.cuda.get_device_properties(device).total_memory / 1024**2
            print(f"현재 사용 중 (allocated): {allocated:.2f} MB")
            print(f"예약됨 (reserved): {reserved:.2f} MB")
            print(f"총 메모리: {total:.2f} MB")
        else:
            print("GPU를 사용할 수 없습니다.")
        for i in range(10):
            torch.cuda.empty_cache()
            gc.collect()
        print(gc.collect())


def convert_to_df(data, column_name):
    if isinstance(data, pd.Series):
        df = data.reset_index()
        df.columns = ["original_index", column_name]
        return df
    elif isinstance(data, str):
        cleaned = re.sub(r"^original_index\s*", "", data.strip())
        cleaned = re.sub(r"Name:.*$", "", cleaned, flags=re.MULTILINE).strip()
        df = pd.read_csv(StringIO(cleaned),sep=r"\s+",header=None,names=["original_index", column_name])
        df["original_index"] = df["original_index"].astype(int)
        return df
    else:
        raise ValueError(f"Unsupported data type for {column_name}: {type(data)}")


def load_file(file):
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == "csv":
        df = pd.read_csv(file)
    elif file_extension in ["xlsx", "xls"]:
        df = pd.read_excel(file)
        # Excel -> CSV 다운로드 버튼 제공
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"⬇️ {file.name} CSV로 다운로드",
            data=csv_buffer.getvalue(),
            file_name=file.name.rsplit('.', 1)[0] + ".csv",
            mime="text/csv"
        )
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        df = None
    return df


def preprocess_and_save(uploaded_files, module, mode, task_name):
    st.session_state.preprocess_messages = [
        msg for msg in st.session_state.preprocess_messages if not msg.startswith(f"✅ {mode}")]
    feature_type = st.session_state.common_feature_type 
    drive_pattern = st.session_state.drive_pattern
    gvw = st.session_state.common_gvw
    min_length = st.session_state.common_min_length
    engine_threshold = st.session_state.engine_threshold
    engine_threshold_high = st.session_state.engine_threshold_high
    fuel_threshold = st.session_state.fuel_threshold

    base_dir = os.path.join(BASE_DIR,"data_preprocessed") # os.abspath 추가 
    if mode == "train":
        save_dir = os.path.join(base_dir, f"normal_gvw{gvw}", feature_type, drive_pattern)
    elif mode == "eval":
        save_dir = os.path.join(base_dir, f"Eval_{task_name}", feature_type, drive_pattern)
    elif mode =='fine_tune':
        save_dir = os.path.join(base_dir, f"normal_gvw{gvw}_fine_tune", feature_type, drive_pattern)
    elif mode == "test":
        save_dir = os.path.join(base_dir, task_name, feature_type, drive_pattern)
    elif mode == "normal_calibration":
        save_dir = os.path.join(base_dir, "normal_engine_B", feature_type, drive_pattern)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data.csv")
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name.lower())
    df_list = []
    total_files = len(uploaded_files)
    progress = st.progress(0)
    with st.spinner(f"{mode} 데이터 전처리 중..."):
        status_text = st.empty()
        for i, f in enumerate(uploaded_files):
            try:
                f.seek(0)
            except Exception:
                pass
            try:
                if hasattr(module, "load_file"):
                    df = module.load_file(f)
                else:
                    df = pd.read_csv(f)
            except Exception as e:
                st.error(f"⚠ 파일을 읽을 수 없습니다: {f.name}\n{e}")
                continue

            if df is None or df.empty:
                st.warning(f"⚠ 파일이 비어있습니다: {f.name}")
                continue
            # NaN 처리
            df = module.NaN_to_value(df)
            df["file_number"] = i
            df_list.append(df)
            percent = int((i + 1) / total_files * 40)
            progress.progress(percent)
            status_text.text(f"파일 처리 중: {i + 1}/{total_files} ({f.name})")
        if len(df_list) == 0:
            st.error("⚠ 전처리 가능한 파일이 없습니다. (EmptyDataError 예방됨)")
            return None, None
        all_df = pd.concat(df_list, ignore_index=True)
        status_text.text("전처리 파이프라인 적용 중...")
        df_processed = module.preprocess_pipeline(all_df,drive_pattern,min_length,engine_threshold,engine_threshold_high,fuel_threshold)
        progress.progress(70)
        if df_processed is None or df_processed.empty:
            st.warning("⚠ 전처리 중 오류가 발생했습니다. 현재 데이터를 다시 확인하세요.")
            st.info("전처리 및 주행패턴 설정 결과, 전처리 이후 데이터 개수가 0개로 확인됐습니다.")
            return None, None
        if feature_type == "main_feature":
            MAIN_FEATURE = [
                'Air_temp_Value','Air_pres_Value','Exhaust_temp_Value','Exhaust_pres_Value',
                'Turbo_speed','J1939CM_VNT_Position','EGR_Position_demand',
                'J1939CM_Egr_position','EGR_Gas_temp_Value','TSE_Turbine_in_temp']
            keep_cols = MAIN_FEATURE + ['file_number','original_index','segment_original_start','segment_original_end']
            df_processed = df_processed[keep_cols]

        try:
            status_text.text("CSV 저장 중...")
            df_processed.to_csv(save_path, index=False)
            progress.progress(90)
            exclude_cols = ['file_number','attack','original_index','segment_original_start','segment_original_end']
            column_list_path = os.path.join(save_dir, 'column_list.txt')
            with open(column_list_path, "w", encoding="utf-8") as f:
                for col in df_processed.columns:
                    if col not in exclude_cols:
                        f.write(f"{col}\n")
            status_text.text("연속 인덱스 그룹 분석 중...")
            module.analyze_consecutive_index_groups_by_condition(
                df_processed, task_name, drive_pattern, save_dir)
            progress.progress(100)
        except PermissionError:
            st.error(f"⚠ 파일 저장 권한 오류: {save_path}")
            return None, None
        status_text.text("✅ 데이터 전처리 완료")

        # # # --- 세션에 Threshold 및 min_length 값 저장 ---
        st.session_state["engine_threshold"] = engine_threshold
        st.session_state["engine_threshold_high"] = engine_threshold_high
        st.session_state["fuel_threshold"] = fuel_threshold
        st.session_state["feature_type"] = feature_type
        st.session_state["gvw"] = gvw
        st.session_state["drive_pattern"] = drive_pattern

        if mode == "train":
            st.session_state["df_train_preview"] = df_processed
            st.session_state["task_name_train"] = task_name
        elif mode =="eval":
            st.session_state["df_eval_preview"] = df_processed
            st.session_state["task_name_eval"] = task_name
        elif mode == "test":
            st.session_state["df_test_preview"] = df_processed
            st.session_state["task_name_test"] = task_name
            save_path_txt = "test_session_info.txt"
            with open(save_path_txt, "w", encoding="utf-8") as f:
                f.write(f"{feature_type}\n")
                f.write(f"{st.session_state.common_level}\n")
                f.write(f"{gvw}\n")
                f.write(f"{task_name}\n")
        elif mode == "normal_calibration":
            st.session_state["df_calibration_preview"] = df_processed
            st.session_state["task_name_calibration"] = task_name
    return df_processed, save_dir


def data_upload_ui(col, label, key_prefix, mode, task_name, module):
    if mode =='test':
        col.markdown(f"###### 📂 {label} 데이터 업로드 (엔진이상탐지 유형: {task_name})")
    elif mode=='eval':
        col.markdown(f"###### 📂 Validation 데이터 업로드 (엔진이상탐지 유형: {task_name})")
    elif mode=='eval':
        col.markdown(f"###### 📂 Validation 데이터 업로드 (엔진이상탐지 유형: {task_name})")
    elif mode =='normal_calibration':
        col.markdown(f"###### 📂 Normal Calibration 데이터 업로드")
    else:
        col.markdown(f"###### 📂 {label} 데이터 업로드")

    # --- uploader 키 관리 ---
    if f"{key_prefix}_files_key" not in st.session_state:
        st.session_state[f"{key_prefix}_files_key"] = 0
    uploader_key = f"{key_prefix}_files_{st.session_state[f'{key_prefix}_files_key']}"

    # --- 파일 업로더 ---
    uploaded_files = col.file_uploader(
        f"{label} 파일 업로드",
        type=["csv"],
        accept_multiple_files=True,
        key=uploader_key)
    if uploaded_files:
        st.session_state[f"{key_prefix}_uploaded"] = uploaded_files

    files_to_process = st.session_state.get(f"{key_prefix}_uploaded")
    if files_to_process:
        if col.button(f"🗑 데이터 초기화", key=f"{key_prefix}_reset"):
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith(key_prefix)]
            for k in keys_to_delete:
                del st.session_state[k]
            if mode =='train':
                extra_keys = ["df_train_preview","train_save_dir","train_csv_path","column_list","saved_params","preprocess_done","params_synced"]
            elif mode == 'eval':
                extra_keys = ["df_eval_preview","eval_save_dir","eval_csv_path","eval_column_list","saved_params","preprocess_done","params_synced"]
            elif mode =='fine_tune':
                extra_keys = ["df_fine_tune_preview","fine_tune_save_dir","fine_tune_csv_path","fine_tune_column_list","saved_params","preprocess_done","params_synced"]
            else:
                extra_keys = []
            for k in extra_keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state[f"{key_prefix}_files_key"] = st.session_state.get(f"{key_prefix}_files_key", 0) + 1
            st.experimental_rerun()
            return None, None

    if files_to_process:
        if module is None:
            col.warning("⚠ 전처리 모듈을 먼저 업로드하세요.")
            return None, None
        if col.button(f"🚀 전처리 실행", key=f"{key_prefix}_process"):
            df, save_dir = preprocess_and_save(files_to_process, module, mode, task_name)
            if save_dir:
                st.session_state[f"{key_prefix}_save_dir"] = save_dir
            st.session_state[f"df_{key_prefix}_preview"] = df
            return df, save_dir
    return None, None


def run_subprocess_and_stream(cmd, env=None):
    st.info(f"실행 명령어: {cmd}")
    log_box = st.empty()
    text_container = ""
    if isinstance(cmd, str):
        cmd_parsed = shlex.split(cmd)
    else:
        cmd_parsed = cmd
    p = subprocess.Popen(cmd_parsed,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,env=env, universal_newlines=True)
    try:
        for line in iter(p.stdout.readline, ""):
            if line == "" and p.poll() is not None:
                break
            text_container += line
            log_box.text_area("실시간 로그", value=text_container[-50000:], height=400)
        p.wait()
    except Exception as e:
        p.kill()
        text_container += f"\n[ERROR] {e}\n"
    return p.returncode, text_container


def find_latest_train_checkpoint(result_dir, task_name):
    pattern = os.path.join(result_dir, task_name, "train_window_*", "checkpoint.ckpt")
    matches = glob(pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]

def find_latest_test_result(result_dir, task_name):
    pattern = os.path.join(result_dir, task_name, "test_window_*", "results_test.pkl")
    matches = glob(pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def activate_model_parameter_sidebar():
    with st.sidebar.expander("📌 TranAD 파라미터 설정", expanded=True):
        # ---------------- Epoch ----------------
        if "epoch" not in st.session_state:
            st.session_state.epoch = 1
        def sync_epoch_slider():
            st.session_state.epoch = st.session_state.epoch_slider
            st.session_state.epoch_input = st.session_state.epoch
        def sync_epoch_input():
            st.session_state.epoch = st.session_state.epoch_input
            st.session_state.epoch_slider = st.session_state.epoch

        col1, col2 = st.columns([2, 1])
        with col1:
            st.slider("Epoch 수", 1, 100, value=st.session_state.epoch,step=1, key="epoch_slider", on_change=sync_epoch_slider)
        with col2:
            st.number_input(" ", 1, 100, value=st.session_state.epoch,step=1, key="epoch_input", on_change=sync_epoch_input)

        # ---------------- Window Size ----------------
        if "window_size" not in st.session_state:
            st.session_state.window_size = 60
        def sync_window_slider():
            st.session_state.window_size = st.session_state.window_slider
            st.session_state.window_input = st.session_state.window_size
        def sync_window_input():
            st.session_state.window_size = st.session_state.window_input
            st.session_state.window_slider = st.session_state.window_size
        col1, col2 = st.columns([2, 1])
        with col1:
            st.slider("Window Size", 10, 200, value=st.session_state.window_size,step=1, key="window_slider", on_change=sync_window_slider)
        with col2:
            st.number_input(" ", 10, 200, value=st.session_state.window_size,step=1, key="window_input", on_change=sync_window_input)

        # ---------------- Batch Size ----------------
        if "batch_size" not in st.session_state:
            st.session_state.batch_size = 64
        def sync_batch_slider():
            st.session_state.batch_size = st.session_state.batch_slider
            st.session_state.batch_input = st.session_state.batch_size
        def sync_batch_input():
            st.session_state.batch_size = st.session_state.batch_input
            st.session_state.batch_slider = st.session_state.batch_size
        col1, col2 = st.columns([2, 1])
        with col1:
            st.slider("Batch Size", 16, 1028, value=st.session_state.batch_size,step=1, key="batch_slider", on_change=sync_batch_slider)
        with col2:
            st.number_input(" ", 16, 1028, value=st.session_state.batch_size,step=1, key="batch_input", on_change=sync_batch_input)

        # ---------------- Learning Rate ----------------
        if "learning_rate" not in st.session_state:
            st.session_state.learning_rate = 1e-3
        def sync_lr_slider():
            st.session_state.learning_rate = st.session_state.lr_slider
            st.session_state.lr_input = st.session_state.learning_rate
        def sync_lr_input():
            st.session_state.learning_rate = st.session_state.lr_input
            st.session_state.lr_slider = st.session_state.learning_rate
        col1, col2 = st.columns([2, 1])
        with col1:
            st.slider("Learning Rate", 1e-6, 1e-1, value=st.session_state.learning_rate,step=1e-5, format="%.6f", key="lr_slider", on_change=sync_lr_slider)
        with col2:
            st.number_input(" ", 1e-6, 1e-1, value=float(st.session_state.learning_rate),step=1e-5, format="%.6f", key="lr_input", on_change=sync_lr_input)

        # ---------------- Session 저장 ----------------
        epoch = st.session_state.epoch 
        window_size= st.session_state.window_size
        batch_size= st.session_state.batch_size
        learning_rate = st.session_state.learning_rate
        st.session_state["epoch"] = epoch
        st.session_state["window_size"] = window_size
        st.session_state["batch_size"] = batch_size
        st.session_state["learning_rate"] = learning_rate
    return epoch, window_size, batch_size, learning_rate



def sync_model_params_from_ui():
    st.session_state.epoch = st.session_state.get("epoch_input", st.session_state.epoch)
    st.session_state.window_size = st.session_state.get("window_input", st.session_state.window_size)
    st.session_state.batch_size = st.session_state.get("batch_input", st.session_state.batch_size)
    st.session_state.learning_rate = st.session_state.get("lr_input", st.session_state.learning_rate)


def train_full_run():
    col1, col2 = st.columns([15, 1])
    with col1:
        df_train_normal, train_save_dir_normal = data_upload_ui(st,"Train","train_normal",mode="train",task_name="Normal",module=module)
    with col2:
        st.button("?", help="현재 정상 엔진 데이터에 대한 전처리를 Sidebar 메뉴의 ⚙️공통 설정/🔧주행패턴 세부사항 설정값을 토대로 진행합니다.")
    if df_train_normal is not None:
        st.session_state["df_train_preview"] = df_train_normal
    if train_save_dir_normal is not None:
        st.session_state["train_save_dir"] = train_save_dir_normal
    try:
        task_name_train = st.session_state.get("task_name_train")
        task_name_test = st.session_state.get("test_task_fault")
        feature_type = st.session_state.get("feature_type")
        gvw = st.session_state.get("gvw")
        drive_pattern = st.session_state.get("drive_pattern")
        train_path_dir = st.session_state.get("train_save_dir")
        if train_path_dir:
            train_csv = os.path.join(train_path_dir, "data.csv")
            column_list_path = os.path.join(train_path_dir, "column_list.txt")
            st.session_state["train_csv_path"] = train_csv
            st.session_state["column_list"] = column_list_path
        else:
            train_csv = st.session_state.get("train_csv_path")
            column_list_path = st.session_state.get("column_list")

        df_train_preview = st.session_state.get("df_train_preview")
        if df_train_preview is not None and train_csv:
            with open(column_list_path, "w", encoding="utf-8") as f:
                for col in df_train_preview.columns:
                    if col not in ['file_number','attack','original_index','segment_original_start','segment_original_end']:
                        f.write(f"{col}\n")
            st.session_state["df_train_preview"] = df_train_preview
            st.session_state["train_csv_path"] = train_csv
            st.session_state["column_list"] = column_list_path
            st.success(f"✅ Train 데이터 저장 완료: {train_csv}")
            st.session_state.preprocess_done = True
            try:
                data_params = {
                    "Data Shape": df_train_preview.shape,
                    "Train Task": task_name_train,
                    "Test Task": task_name_test,
                    "Feature Type": feature_type,
                    "GVW": gvw,
                    "Drive Pattern": drive_pattern,
                    "Min Length": common_min_length,
                    "Engine Threshold": engine_threshold,
                    "Engine Threshold High": engine_threshold_high,
                    "Fuel Threshold": fuel_threshold,
                }
            except Exception as e:
                st.error(f"⚠ data_params 생성 중 오류: {e}")
                data_params = {}

            st.session_state.data_params = data_params
            norm_saved = {k: "" if v is None else str(v) for k, v in data_params.items()}
            st.session_state.saved_params = norm_saved
            df_params = pd.DataFrame([(k, v) for k, v in norm_saved.items()], columns=["Parameter", "Value"])
            st.session_state.df_params = df_params
            st.session_state.params_synced = True
        else:
            if not st.session_state.preprocess_done:
                st.warning("⚠ Train 데이터가 존재하지 않거나 전처리를 수행하지 않았습니다.")

        df_params = st.session_state.get("df_params")
        if st.session_state.params_synced:
            toggle_key = "df_params_toggle"
            default_toggle = st.session_state.get(toggle_key, True)
            col1, col2 = st.columns([15, 1])
            with col1:
                show_params = st.checkbox("📊 Project Management / 전처리 공통 설정 및 주행패턴 세부사항", value=default_toggle, key=toggle_key)
            with col2:
                st.button("?", help="⚙️ 공통 설정 / 🔧 주행패턴 세부사항 전처리 설정에 대한 저장된 학습데이터가 존재하다는 것을 의미합니다.")
            if show_params:
                if df_params is not None and not df_params.empty:
                    rows, cols = df_train_preview.shape
                    st.markdown(
                        f"""
                        <div style="
                            padding:12px;
                            border-radius:10px;
                            font-size:13px;
                            color:inherit;">
                        
                        <b>📝 Project Management</b><br>
                        • 엔진 이상탐지 유형&nbsp;&nbsp;: {task_name_test}</code><br><br>

                        <hr style='border:2px solid #999;'>

                        <b>⚙️ 공통 설정</b><br>
                        • Feature Type : {feature_type}</code><br>
                        • GVW : {gvw}</code><br>
                        • Drive Pattern : {drive_pattern}</code><br><br>
                        
                        <b>🔧 주행패턴 세부사항 설정</b><br>
                        • Engine Threshold:
                        {engine_threshold}</code> /
                        {engine_threshold_high}</code><br>
                        • Fuel Threshold: {fuel_threshold}</code><br>
                        • Min Length : {common_min_length}</code><br><br>

                        <b>📊 학습 데이터 크기</b><br>
                        • Data Shape&nbsp;&nbsp;: ({rows}, {cols})</code><br><br>
                        </div>
                        """,
                        unsafe_allow_html=True)
                    csv_data = df_train_preview.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(label="⬇️ 학습 데이터 CSV 다운로드",data=csv_data,file_name="train_data.csv",mime="text/csv")
                    st.success("✅ Task 및 주행패턴 등 정보 변경 시 전처리 재실행 필수")


        if df_train_preview is not None and not df_train_preview.empty:
            toggle_df_key = "df_train_preview_toggle"
            default_toggle_df = st.session_state.get(toggle_df_key, False)
            show_summary = st.checkbox("📊 학습 데이터 요약 통계량", value=default_toggle_df, key=toggle_df_key)
            if show_summary:
                df_summary = df_train_preview.drop(['file_number','original_index','segment_original_start','segment_original_end'],axis=1).describe()
                st.dataframe(df_summary, width=1200)  
                st.success("✅ 요약 통계량 확인 완료")

            
        if df_train_preview is not None and not df_train_preview.empty:
            toggle_plot_key = "df_train_plot_toggle"
            default_toggle_plot = st.session_state.get(toggle_plot_key, False)
            show_plot = st.checkbox("📈 학습 데이터 시각화", value=default_toggle_plot, key=toggle_plot_key)
            if show_plot:
                numeric_cols = df_train_preview.drop(['file_number','original_index','segment_original_start','segment_original_end'],axis=1).select_dtypes(include="number").columns.tolist()
                max_len = len(df_train_preview)
                if "range_idx" not in st.session_state:
                    st.session_state.range_idx = (0, max_len)
                    
                start_idx, end_idx = st.slider("Index Range",min_value=0,max_value=max_len,step=1,key="range_idx") #value=st.session_state.range_idx
                if start_idx > end_idx:
                    st.session_state.range_idx = (end_idx, end_idx)
                    start_idx, end_idx = end_idx, end_idx

                if "selected_multi_cols" not in st.session_state:
                    st.session_state.selected_multi_cols = []
                multi_options = ["🔄 전체 변수 선택"] + numeric_cols
                selected_cols = st.multiselect("📈 시각화할 변수들 선택",multi_options,key="selected_multi_cols") #default=st.session_state.selected_multi_cols

                if "🔄 전체 변수 선택" in selected_cols:
                    selected_cols = numeric_cols
                if selected_cols:
                    with st.expander("📊 Plot 결과 보기", expanded=True):
                        for col in selected_cols:
                            sns.set(font_scale=1.0)
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(df_train_preview[col].iloc[start_idx:end_idx],color='blue')
                            ax.set_title(f"{col} Plot (Index {start_idx} ~ {end_idx})")
                            ax.set_xlabel("Index")
                            ax.set_ylabel(col)
                            st.pyplot(fig)
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            st.download_button(label=f"📥 {col} Plot Download",data=buf,file_name=f"{col}_Train_plot.png",mime="image/png")

    except Exception as e:
        if not st.session_state.preprocess_done:
            st.warning("⚠ Train 데이터가 존재하지 않거나 전처리를 수행하지 않았습니다.")

    if st.session_state.saved_params is not None:
        current_params = {
            "Data Shape": "" if st.session_state.get("df_train_preview") is None else str(st.session_state.get("df_train_preview").shape),
            "Train Task": "" if st.session_state.get("task_name_train") is None else str(st.session_state.get("task_name_train")),
            "Test Task": "" if st.session_state.get("test_task_fault") is None else str(st.session_state.get("test_task_fault")),
            "Feature Type": "" if st.session_state.get("feature_type") is None else str(st.session_state.get("feature_type")),
            "GVW": "" if st.session_state.get("gvw") is None else str(st.session_state.get("gvw")),
            "Drive Pattern": "" if st.session_state.get("drive_pattern") is None else str(st.session_state.get("drive_pattern")),
            "Min Length": "" if st.session_state.get("common_min_length") is None else str(st.session_state.get("common_min_length")),
            "Engine Threshold": "" if st.session_state.get("engine_threshold") is None else str(st.session_state.get("engine_threshold")),
            "Engine Threshold High": "" if st.session_state.get("engine_threshold_high") is None else str(st.session_state.get("engine_threshold_high")),
            "Fuel Threshold": "" if st.session_state.get("fuel_threshold") is None else str(st.session_state.get("fuel_threshold")),}
        diffs = []
        for k, saved_v in st.session_state.saved_params.items():
            curr_v = current_params.get(k, "")
            if str(saved_v) != str(curr_v):
                diffs.append(f"{k}: {saved_v} → {curr_v}")

    df_params = None
    data_params = None
    if "df_params" in st.session_state and "data_params" in st.session_state:
        df_params = st.session_state.df_params
        data_params = st.session_state.data_params

    if "train_init" not in st.session_state:
        st.session_state.train_init = False
    if "train_ready" not in st.session_state:
        st.session_state.train_ready = False
    if "train_logs" not in st.session_state:
        st.session_state.train_logs = ""
    if "train_run_done" not in st.session_state:
        st.session_state.train_run_done = False
    if "latest_train_ckpt_dir" not in st.session_state:
        st.session_state.latest_train_ckpt_dir = None
    
    st.divider()
    col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
    with col1:
        st.markdown("#### ▶️ Deep Learning기반 이상탐지 학습모델 구축 (TranAD)")
    with col2:
        st.button("?",help="TranAD는 정상 엔진 시계열 데이터를 기준으로 학습하고, 동일 엔진의 불량 데이터를 통해 이상 탐지 성능을 검증·향상시키는 Transformer 기반 이상탐지 모델입니다.")
    if not st.session_state.train_init:
        if st.button("▶ 시작"):
            st.success("데이터 전처리 확인 완료. 학습을 시작할 수 있습니다.")
            try:
                train_task_value = df_params.loc[df_params['Parameter'] == 'Train Task', 'Value'].iloc[0]
                if train_task_value is not None and len(diffs) == 0:
                    st.session_state.train_init = True
                    st.session_state.train_ready = False
                    st.success("✅ 학습모델 단계 이동 성공 (설정값 변경 시 전처리 재실행 필요)")
                    st.experimental_rerun() 
                else:
                    st.warning("⚠ 학습모델 단계로 이동 불가 (전처리 변경됨)")
            except Exception as e:
                st.error(f"⚠ Train Task 확인 오류: {e}")
        st.stop()   
    

    # ================================================================
    # 1️⃣ STEP 1: TranAD 파라미터 설정 단계
    # ================================================================
    epoch, window_size, batch_size, learning_rate = activate_model_parameter_sidebar()
    param_toggle = "model_params_area_open"

    col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
    with col1:
        if st.button("📌 모델 파라미터 설정"):
            st.session_state[param_toggle] = not st.session_state.get(param_toggle, False)
            st.session_state.train_ready = True
    with col2:
        st.button("?", help="학습모델의 파라미터는 Sidebar 메뉴의 📌 TranAD 파라미터 설정을 통해서 조정할 수 있습니다.")
    if st.session_state.get(param_toggle, False):
        st.write("✅ 학습 파라미터 설정 완료")
        if all(v is not None for v in [epoch, window_size, batch_size, learning_rate]):
            model_params = {"Epoch": epoch,"Window Size": window_size,"Batch Size": batch_size,"Learning Rate": learning_rate}
            df_model_params = pd.DataFrame(model_params.items(), columns=["Parameter", "Value"])
            st.table(df_model_params)
            st.success("✅ 파라미터 변경 시 모델 재학습 필수")

    # ================================================================
    # 2️⃣ STEP 2: Eval 업로드 및 학습 실행 단계
    # ================================================================
    col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
    with col1:
        df_eval_fault, eval_save_dir = data_upload_ui(st,"Eval","eval_fault",mode="eval",task_name=st.session_state.get("test_task_fault"),module=module)
    with col2:
        st.button("?", help="엔진 이상탐지 검증 대상인 데이터에 대한 전처리를 수행합니다. \n\n 해당 검증용 이상데이터는 학습모델의 이상탐지 능력을 향상시킵니다.")

    if df_eval_fault is not None:
        st.session_state["df_eval_preview"] = df_eval_fault
    if eval_save_dir is not None:
        st.session_state["eval_save_dir"] = eval_save_dir
    try:
        eval_path_dir = st.session_state.get("eval_save_dir")
        if eval_path_dir:
            eval_csv = os.path.join(eval_path_dir, "data.csv")
            eval_column_list_path = os.path.join(eval_path_dir, "column_list.txt")
            st.session_state["eval_csv_path"] = eval_csv
            st.session_state["eval_column_list"] = eval_column_list_path
        else:
            eval_csv = st.session_state.get("eval_csv_path")
            eval_column_list_path = st.session_state.get("eval_column_list")
        df_eval_preview = st.session_state.get("df_eval_preview")
        if df_eval_preview is not None and eval_csv:
            with open(eval_column_list_path, "w", encoding="utf-8") as f:
                for col in df_eval_preview.columns:
                    if col not in ['file_number','attack','original_index','segment_original_start','segment_original_end']:
                        f.write(f"{col}\n")
            st.success(f"✅ Eval 데이터 저장 완료: {eval_csv}")
    except Exception as e:
        st.error(f"⚠ Eval 데이터 처리 중 오류: {e}")

    # 버튼 표시 조건
    eval_ready = (
        st.session_state.get("df_eval_preview") is not None and
        st.session_state.get("eval_csv_path") is not None and
        st.session_state.get("eval_column_list") is not None)
    if eval_ready:
        col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
        with col1:
            if st.button("🚀 TranAD 학습"):
                train_file = pd.read_csv(os.path.join(train_path_dir, 'data.csv'), encoding='utf-8')
                eval_file = pd.read_csv(os.path.join(eval_path_dir, 'data.csv'), encoding='utf-8')
                if "TB" in task_name_test or "turbo" in task_name_test:
                    main_task = "Turbo_problem"
                elif "egr_sw" in task_name_test:
                    main_task = "EGR_SW"
                elif "egr_hw" in task_name_test:
                    main_task = "EGR_HW"
                else:
                    main_task = task_name_test
                result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
                os.makedirs(result_path, exist_ok=True)
                log_path = os.path.join(BASE_DIR, "logs")
                os.makedirs(log_path, exist_ok=True)
                log_file = os.path.join(log_path,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}.txt")

                st.session_state['train_path_dir'] = train_path_dir
                st.session_state['eval_path_dir'] = eval_path_dir
                cmd = (
                    f"python main.py --phase train "
                    f"--model TranAD "
                    f"--task_name \"{task_name_test}\" "
                    f"--train_dataset \"{os.path.join(train_path_dir, 'data.csv')}\" "
                    f"--eval_dataset \"{os.path.join(eval_path_dir, 'data.csv')}\" "
                    f"--columns \"{column_list_path}\" "
                    f"--epoch {st.session_state.epoch} "
                    f"--window_size {st.session_state.window_size} "
                    f"--batch_size {st.session_state.batch_size} "
                    f"--learning_rate {st.session_state.learning_rate} "
                    f"--save_result_dir \"{result_path}\" ")
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0"
                with st.spinner("학습 진행 중..."):
                    rc, logs = run_subprocess_and_stream(cmd, env=env)
                st.session_state.train_logs = logs
                st.session_state.train_run_done = (rc == 0)
                try:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(logs)
                    st.success(f"학습 로그 저장됨: {log_file}")
                except:
                    st.error("로그 파일 저장 실패")
                if rc == 0:
                    ckpt = find_latest_train_checkpoint(result_path, task_name_test)
                    if ckpt:
                        ckpt_dir = os.path.dirname(ckpt)
                        st.session_state.latest_train_ckpt_dir = ckpt
                        st.success(f"학습 체크포인트 저장 완료: {ckpt}")
                    else:
                        st.error("체크포인트를 찾을 수 없습니다. main.py 로그를 확인하세요.")
                else:
                    st.error("학습 실패. 로그를 확인하세요.")

        with col2:
            st.button("?",help="현재 Project Management 및 전처리 설정에 대한 학습모델을 구축한 이후, 학습 완료 페이지로 전환됩니다.")


    # ---------------- Training Graph & Anomaly Score Plot 다운로드 ----------------
    if st.session_state.train_run_done and st.session_state.get("latest_train_ckpt_dir"):
        st.text_area("학습 로그", value=st.session_state.train_logs[-5000:], height=300)
        col1, col2 = st.columns(2)  
        pdf_path = os.path.join(os.path.dirname(st.session_state.latest_train_ckpt_dir),"train_loss_plot.pdf")
        with col1:
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(label="📥 Train Result Graph (New)",data=f.read(),file_name="train_loss_plot.pdf",mime="application/pdf",use_container_width=True)
        html_plot_path = os.path.join(os.path.dirname(st.session_state.latest_train_ckpt_dir),"anomaly_score.html")
        with col2:
            if os.path.exists(html_plot_path):
                with open(html_plot_path, "rb") as f:
                    st.download_button(label="📥 Train Anomaly Score Plot (New)",data=f.read(),file_name="train_result_plot.html",mime="text/html",use_container_width=True)
                    
        ##### 데이터 전처리 & 모델 기초 파라미터 정보 Json 저장 
        active_list = [data_params]
        try:
            save_path = os.path.join(BASE_DIR,result_path,task_name_test, "data_preprocessing_parameters.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(active_list, f, indent=4)
            train_params = {"epoch": epoch,"window_size": window_size,"batch_size": batch_size,"learning_rate": learning_rate}
            result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
            pattern = f"train_window_{st.session_state.window_size}_*"
            result_window_dirs = [
                d for d in glob(os.path.join(result_base_dir , pattern))
                if os.path.isdir(d)]
            latest_result_dir = max(result_window_dirs, key=os.path.getctime)
            dest_path = os.path.join(latest_result_dir,"data_preprocessing_parameters.json")
            shutil.move(save_path, dest_path)
           
            save_path = os.path.join(BASE_DIR, result_path, task_name_test, "model_training_parameters.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(train_params, f, indent=4)
            result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
            pattern = f"train_window_{st.session_state.window_size}_*"
            result_window_dirs = [
                d for d in glob(os.path.join(result_base_dir , pattern))
                if os.path.isdir(d)]
            latest_result_dir = max(result_window_dirs, key=os.path.getctime)
            dest_path = os.path.join(latest_result_dir,"model_training_parameters.json")
            shutil.move(save_path, dest_path)
            st.success("학습을 성공적으로 마쳤습니다. 학습 결과는 Train Model Archive로 이동합니다.")

        except FileNotFoundError:
            st.warning("📁 결과 디렉토리를 찾을 수 없습니다. 재학습을 실행하세요.")
        except ValueError:

            st.warning("⚠️ 학습 결과 디렉토리가 아직 생성되지 않았습니다. 재학습을 실행하세요.")
        except Exception as e:
            print("ee")
            #st.warning(f"전처리 공통 설정 및 주행패턴 세부사항이 변경됐습니다. 재학습을 실행하세요.")


def delete_other_folders(recall_list):
    st.markdown("📁 학습결과 폴더 선택")
    folder_names = [item["folder"] for item in recall_list]
    selected = st.selectbox("학습 폴더를 선택하세요 (선택 안할 시 가장 최근에 학습한 모델을 사용합니다):", folder_names)
    if st.button("🚀 선택한 폴더 제외하고 나머지 삭제"):
        selected_item = next(item for item in recall_list if item["folder"] == selected)
        st.success(f"✅ 선택된 폴더는 유지됩니다: {selected}")
        for item in recall_list:
            if item["folder"] != selected:
                folder_path = item["path"]
                try:
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                        st.warning(f"🧹 삭제됨: {item['folder']} ({folder_path})")
                    else:
                        st.error(f"❌ 경로 없음: {folder_path}")
                except Exception as e:
                    st.error(f"⚠ 삭제 실패 ({item['folder']}): {e}")
        st.success("🎉 작업 완료! 선택된 폴더만 남았습니다.")


def delete_incomplete_train_results(base_result_dir):
    train_window_dirs = [
        d for d in glob(os.path.join(base_result_dir, "train_window_*"))
        if os.path.isdir(d)]
    for d in train_window_dirs:
        result_pkl = os.path.join(d, "results_train.pkl")
        if not os.path.exists(result_pkl):
            try:
                shutil.rmtree(d)
            except Exception as e:
                st.error(f"⚠ 삭제 실패 ({os.path.basename(d)}): {e}")
    st.success("✅ 미완료 학습 결과 폴더 삭제 완료")


def update_model_data_params(ckpt):
    train_window_dir = os.path.basename(os.path.dirname(ckpt))
    parts = train_window_dir.split("_")
    window_size = int(parts[2])
    result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
    pattern = f"train_window_{window_size}_*"
    train_window_dirs = [
        d for d in glob(os.path.join(result_base_dir, pattern))
        if os.path.isdir(d)]
    is_finetune = "finetune" in train_window_dir.lower()
    if is_finetune:
        train_window_dirs = [
            d for d in train_window_dirs
            if "finetune" in os.path.basename(d).lower()]
    else:
        train_window_dirs = [
            d for d in train_window_dirs
            if "finetune" not in os.path.basename(d).lower()]
    latest_result_dir = max(train_window_dirs, key=os.path.getctime)
    load_preprocessing_params = os.path.join(latest_result_dir,"data_preprocessing_parameters.json")
    load_model_params = os.path.join(latest_result_dir,"model_training_parameters.json")
    return latest_result_dir, load_preprocessing_params,load_model_params


def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("HYSMyeongJo-Medium", 8)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, f"- {doc.page} -")
    canvas.restoreState()

def load_summary_txt(page, base_dir):
    file_map = {
        1: "page_1_summary.txt",
        2: "page_2_summary.txt",
        3: "page_3_summary.txt",}
    file_name = file_map.get(page)
    if file_name is None:
        return ""
    file_path = os.path.join(base_dir, "summaries", file_name)
    if not os.path.exists(file_path):
        return f"[ERROR] Summary file not found: {file_name}"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

    
def generate_summary_pdf(summary_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer,pagesize=A4,rightMargin=20 * mm,leftMargin=20 * mm,topMargin=20 * mm,bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    # 최상단 대제목 스타일
    title_style = ParagraphStyle(name="KoreanTitle",parent=styles["Normal"],fontName="HYSMyeongJo-Medium",fontSize=16,leading=22,spaceAfter=16,spaceBefore=4,)
    # 본문 스타일
    body_style = ParagraphStyle(name="KoreanBody",parent=styles["Normal"],fontName="HYSMyeongJo-Medium",fontSize=10,leading=14,spaceAfter=6)
    # 1~6번 제목 스타일
    section_style = ParagraphStyle(name="KoreanSection",parent=styles["Normal"],fontName="HYSMyeongJo-Medium",fontSize=13,leading=18,spaceBefore=12,spaceAfter=8,)
    story = []
    for line in summary_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        # ✅ 최상단 대제목
        if line.startswith("[") and line.endswith("]"):
            story.append(Paragraph(line, title_style))
            story.append(Spacer(1, 6))
            continue
        # 1. 2. 3. ... 섹션 제목 판별
        if (
            line.startswith("1.") or
            line.startswith("2.") or
            line.startswith("3.") or
            line.startswith("4.") or
            line.startswith("5.") or
            line.startswith("6.") or 
            line.startswith("7.")):
            story.append(Paragraph(line, section_style))
        else:
            story.append(Paragraph(line, body_style))
    doc.build(story,onFirstPage=draw_footer,onLaterPages=draw_footer)
    buffer.seek(0)
    return buffer.getvalue()




##################################################################################################################################################################################################################
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def has_korean(path: str) -> bool:
    return bool(re.search(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", path))
if has_korean(BASE_DIR):
    st.warning(
        "⚠️ 실행 경로에 한글이 포함되어 있습니다.\n\n"
        "현재 경로:\n"
        f"`{BASE_DIR}`\n\n"
        "모델 학습 및 torch.save 과정에서 오류가 발생할 수 있으므로\n"
        "**전체 영문 경로로 이동 혹은 변경 후 다시 실행하세요.**")
    st.stop()
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)


current_task = st.session_state.get("task_select_ui", "TB_fault")
st.sidebar.markdown(f"# 🗂️ Project Management : {current_task}") #📝
with st.sidebar.expander("⚙️ 엔진 이상탐지 문제 정의", expanded=False):
    TASK_FILE = os.path.join(BASE_DIR, "task_list.txt")
    DEFAULT_TASKS = ["turbo_sw_per2","turbo_sw_per4","TB_fault","TB_gasket_problem","TB_fouling","egr_sw","egr_hw_per50","egr_hw_per90"]
    NEW_TASK_LABEL = "새로운 Task..."
    # ---------------- Default Reset ----------------
    if st.button("🔄 Task Reset"):
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            for task in DEFAULT_TASKS:
                f.write(task + "\n")
        st.session_state.task_options = DEFAULT_TASKS + [NEW_TASK_LABEL]
        st.session_state.task_select_ui = "TB_fault"
        st.session_state.test_task_fault = "TB_fault"
        st.session_state.test_task_calib = "TB_fault"
        st.session_state.new_task = ""

    if not os.path.exists(TASK_FILE):
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            for task in DEFAULT_TASKS:
                f.write(task + "\n")
    if "task_options" not in st.session_state:
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            tasks = [line.strip() for line in f if line.strip()]
        if NEW_TASK_LABEL not in tasks:
            tasks.append(NEW_TASK_LABEL)
        st.session_state.task_options = tasks
        
    if "test_task_fault" not in st.session_state:
        st.session_state.test_task_fault = "TB_fault"
    if "task_select_ui" not in st.session_state:
        st.session_state.task_select_ui = st.session_state.test_task_fault

    st.selectbox("📌 Task 선택 (엔진 이상탐지 유형)",options=st.session_state.task_options,key="task_select_ui",index=st.session_state.task_options.index(st.session_state.task_select_ui))
    if st.session_state.task_select_ui == NEW_TASK_LABEL:
        new_task = st.text_input("새로운 Task 이름을 입력하세요",value=st.session_state.get("new_task", ""),key="new_task_input")
        if new_task and new_task not in st.session_state.task_options:
            st.session_state.task_options.insert(-1, new_task)
            with open(TASK_FILE, "w", encoding="utf-8") as f:
                for task in st.session_state.task_options:
                    if task != NEW_TASK_LABEL:
                        f.write(task + "\n")
            st.session_state.test_task_fault = new_task
            st.session_state.test_task_calib = new_task
            st.session_state.new_task = new_task      
    else:
        st.session_state.test_task_fault = st.session_state.task_select_ui
        st.session_state.test_task_calib = st.session_state.task_select_ui
        st.session_state.new_task = ""


st.sidebar.markdown("### ⚙️ App 제어")
with st.sidebar.expander("Reset/Memory Clean ", expanded=False):
    if st.button("🧹 Reset ( 전체 설정 초기화)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ 설정 초기화 완료. 페이지를 새로고침합니다...")
        st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
    if st.button("💾 메모리 정리"):
        st.success("✅ 메모리 정리 완료")
        clean_memory()



st.sidebar.markdown("### ⚙️ Device 설정")
with st.sidebar.expander("CPU/GPU Environment", expanded=False):
    device_status_placeholder = st.empty()
    device_option = st.radio("💻 Device 선택",["GPU", "CPU"],index=0,key="device_option")
    if st.button("⚡ Device 적용"):
        if device_option == "GPU":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                device_status_placeholder.success("⚡ GPU 활성화")
            else:
                device_status_placeholder.warning("❌ GPU 사용 불가 → CPU로 전환")
        else:
            device = torch.device("cpu")
            gc.collect()
            device_status_placeholder.success("⚡ CPU 활성화")
        st.session_state.device = device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.sidebar.markdown("### ⚙️ 핵심 기능 선택")
page = st.sidebar.radio("## Page", ["Data Upload/Preprocess & Model Training","Anomaly Detection/Causal Analysis", "Trained Model Fine-Tuning"])
if "preprocessing_module" not in st.session_state:
    st.session_state.preprocessing_module = None
if "process" not in st.session_state:
    st.session_state.process = None
if "log_text" not in st.session_state:
    st.session_state.log_text = ""
if "progress" not in st.session_state:
    st.session_state.progress = 0.0


if page == "Data Upload/Preprocess & Model Training":
    top_left, top_right = st.columns([4, 1])
    with top_right:
        #pdf_bytes = generate_summary_pdf(get_page_summary(1))
        summary_text = load_summary_txt(page=1, base_dir=BASE_DIR)
        pdf_bytes = generate_summary_pdf(summary_text)
        st.download_button(label="📄기능 설명",data=pdf_bytes,file_name="기능설명_Page1.pdf",mime="application/pdf",use_container_width=True)

    st.markdown(f"#### 📊 학습 데이터 (정상엔진) 업로드 & 전처리")
    with st.sidebar.expander("⚙️ 공통 설정", expanded=False):
        if st.button("▶️ 전처리 패키지 업로드"):
            status_text = st.empty()
            steps = ["패키지 로딩 준비", "전처리 모듈 로딩", "기능 준비 완료"]
            with st.spinner("전처리 패키지 로딩 중..."):
                for step in steps:
                    status_text.text(f"⏳ {step} ...")
                    time.sleep(0.5)
                st.session_state.preprocessing_module = importlib.import_module("src.preprocessing")
                status_text.text("✅ 전처리 모듈 로딩 완료")

        if "common_feature_type" not in st.session_state:
            st.session_state.common_feature_type = "full_feature"
        if "common_level" not in st.session_state:
            st.session_state.common_level = "Desc"
        if "drive_pattern" not in st.session_state:
            st.session_state.drive_pattern = "desc"
        if "common_gvw" not in st.session_state:
            st.session_state.common_gvw = 100
        if "engine_threshold_by_level" not in st.session_state:
            st.session_state.engine_threshold_by_level = {"Low": 1100,"Mid": 1100,"High": 1300,"Desc": 700}
        if "engine_threshold_high_by_level" not in st.session_state:
            st.session_state.engine_threshold_high_by_level = {"Mid": 1300}
        if "fuel_threshold_by_level" not in st.session_state:
            st.session_state.fuel_threshold_by_level = {"Low": 15.0,"Mid": 15.0,"High": 15.0,"Desc": 15.0}
        if "common_min_length_by_level" not in st.session_state:
            st.session_state.common_min_length_by_level = {"Low":60,"Mid":60,"High":60,"Desc":60}
        if "prev_level" not in st.session_state:
            st.session_state.prev_level = st.session_state.common_level
       
       
        def sync_feature_type():
            st.session_state.common_feature_type = st.session_state.feature_type_selectbox
        def sync_gvw():
            st.session_state.common_gvw = st.session_state.gvw_selectbox
        def sync_drive_pattern():
            st.session_state.common_level = st.session_state.drive_pattern_selectbox
            st.session_state.drive_pattern = st.session_state.common_level.lower()
        levels = ["Low", "Mid", "High", "Desc"]
        common_feature_type = st.selectbox("📌 Feature Type",["full_feature", "main_feature"],key="feature_type_selectbox",index=["full_feature","main_feature"].index(st.session_state.common_feature_type),on_change=sync_feature_type)
        common_gvw = st.selectbox("📌 GVW Level",[0,50,100],key="gvw_selectbox",index=[0,50,100].index(st.session_state.common_gvw),on_change=sync_gvw)
        selected_level = st.selectbox("📌 Drive Pattern 선택",levels,key="drive_pattern_selectbox",index=levels.index(st.session_state.common_level),on_change=sync_drive_pattern)
        st.session_state.common_feature_type = common_feature_type 
        st.session_state.common_gvw = common_gvw
        st.session_state.common_level = selected_level
        st.session_state.drive_pattern = st.session_state.common_level.lower()

        with st.sidebar.expander("🔧 주행패턴 세부사항 설정", expanded=False):
            if st.session_state.prev_level != selected_level:
                prev = st.session_state.prev_level
                curr = selected_level
                st.session_state.engine_threshold_by_level[curr] = st.session_state.engine_threshold_by_level.get(prev, 0)
                st.session_state.fuel_threshold_by_level[curr] = st.session_state.fuel_threshold_by_level.get(prev, 15.0)
                st.session_state.common_min_length_by_level[curr] = st.session_state.common_min_length_by_level.get(prev, 60)
                if curr == "Mid":
                    st.session_state.engine_threshold_high_by_level["Mid"] = st.session_state.engine_threshold_high_by_level.get(prev, 1300)
                st.session_state.prev_level = curr

            engine_key = f"engine_{selected_level}"
            if engine_key not in st.session_state:
                st.session_state[engine_key] = st.session_state.engine_threshold_by_level[selected_level]
            engine_threshold = st.number_input(f"📌 {selected_level} Engine Speed Threshold",min_value=0,value=st.session_state[engine_key],step=1,key=engine_key,
                                                on_change=lambda: st.session_state.engine_threshold_by_level.update({selected_level: st.session_state[engine_key]}))

            if selected_level == "Mid":
                high_key = f"engine_high_{selected_level}"
                if high_key not in st.session_state:
                    st.session_state[high_key] = st.session_state.engine_threshold_high_by_level["Mid"]
                engine_threshold_high = st.number_input("📌 Mid Upper Engine Speed Threshold",min_value=0,value=st.session_state[high_key],step=1,
                    key=high_key,on_change=lambda: st.session_state.engine_threshold_high_by_level.update({"Mid": st.session_state[high_key]}))
            else:
                engine_threshold_high = None

            fuel_key = f"fuel_{selected_level}"
            if fuel_key not in st.session_state:
                st.session_state[fuel_key] = st.session_state.fuel_threshold_by_level[selected_level]
            fuel_threshold = st.number_input("📌 Fuel Threshold",min_value=0.0,value=st.session_state[fuel_key],step=0.1,key=fuel_key,
                                            on_change=lambda: st.session_state.fuel_threshold_by_level.update({selected_level: st.session_state[fuel_key]}))

            minlen_key = f"minlen_{selected_level}"
            if minlen_key not in st.session_state:
                st.session_state[minlen_key] = st.session_state.common_min_length_by_level[selected_level]
            common_min_length = st.number_input("📌 최소 연속 구간 길이",min_value=1,value=st.session_state[minlen_key],step=1,key=minlen_key,
                                    on_change=lambda: st.session_state.min_length_by_level.update({selected_level: st.session_state[minlen_key]}))

            st.session_state.engine_threshold = engine_threshold 
            st.session_state.engine_threshold_high = engine_threshold_high
            st.session_state.fuel_threshold = fuel_threshold
            st.session_state.common_min_length = common_min_length

            if st.button("🔄 현재 주행패턴 설정 초기화"):
                default_engine_threshold_map = {"Low": 1100,"Mid": 1100,"High": 1300,"Desc": 700}
                default_engine_threshold_high_map = {"Mid": 1300}
                default_engine_threshold = default_engine_threshold_map[selected_level]
                st.session_state.engine_threshold = default_engine_threshold
                st.session_state.engine_threshold_by_level[selected_level] = default_engine_threshold
                if selected_level == "Mid":
                    default_high = default_engine_threshold_high_map["Mid"]
                    st.session_state.engine_threshold_high = default_high
                    st.session_state.engine_threshold_high_by_level["Mid"] = default_high
                else:
                    st.session_state.engine_threshold_high = None
                st.session_state.fuel_threshold = 15.0
                st.session_state.common_min_length = 60
                st.success(f"✅ {selected_level} 주행패턴 설정이 기본값으로 초기화되었습니다.")

                engine_threshold = default_engine_threshold 
                fuel_threshold = 15.0
                common_min_length = 60

                st.session_state.engine_threshold = engine_threshold
                st.session_state.fuel_threshold= fuel_threshold
                st.session_state.common_min_length = common_min_length

                st.write("🔽 주행패턴 Reset 완료")
                if selected_level== "Mid":
                    params_dict = {"engine_threshold": engine_threshold,"engine_threshold_high": engine_threshold_high, "fuel_threshold": fuel_threshold,"common_min_length": common_min_length}
                else:
                    params_dict = {"engine_threshold": engine_threshold,"fuel_threshold": fuel_threshold,"common_min_length": common_min_length}
                st.write(params_dict)


    
    # ---------- 세션 초기화 (처음에 한 번만) ----------
    if "preprocess_messages" not in st.session_state:
        st.session_state.preprocess_messages = []
    if "train_uploaded" not in st.session_state:
        st.session_state.train_uploaded = False
    if "df_train_preview" not in st.session_state:
        st.session_state.df_train_preview = None
    if "df_eval_preview" not in st.session_state:
        st.session_state.df_eval_preview = None
    if "train_save_dir" not in st.session_state:
        st.session_state.train_save_dir = None
    if "eval_save_dir" not in st.session_state:
        st.session_state.eval_save_dir = None
    if "saved_params" not in st.session_state:
        st.session_state.saved_params = None
    if "preprocess_done" not in st.session_state:
        st.session_state.preprocess_done = False
    if "params_synced" not in st.session_state:
        st.session_state.params_synced = False
    if "train_csv_path" not in st.session_state:
        st.session_state.train_csv_path = None
    if "eval_csv_path" not in st.session_state:
        st.session_state.eval_csv_path = None
    if "column_list" not in st.session_state:
        st.session_state.column_list = None


    # ---------- (기존) 데이터 업로드 / 전처리 코드 실행 ----------
    module = st.session_state.get("preprocessing_module")
    if module is None:
        col1, col2 = st.columns([20, 1])
        with col1:
            st.info("⚙️ 전처리 모듈 활성화 후 데이터 업로드 및 전처리 가능")
        with col2:
            st.button("!", help="새로고침 및 App 재실행을 할 때 Sidebar 메뉴의 ⚙️ 공통 설정에서 ▶️ 전처리 패키지 업로드를 반드시 수행해야 합니다.")

    #### 기존 학습 / 검증 전처리 데이터 / 학습모델 저장 여부 확인
    st.session_state["task_name_train"] = "Normal"
    task_name_train = st.session_state.get("task_name_train")
    task_name_test = st.session_state.get("test_task_fault")
    feature_type = st.session_state.common_feature_type
    gvw = st.session_state.common_gvw
    min_length = st.session_state.common_min_length
    fuel_threshold = st.session_state.fuel_threshold
    engine_threshold = st.session_state.engine_threshold
    engine_threshold_high = st.session_state.engine_threshold_high
    drive_pattern = st.session_state.get("drive_pattern")
    common_min_length = st.session_state.common_min_length
    st.session_state.feature_type = feature_type
    st.session_state.gvw = gvw
    
    train_path_dir = os.path.join(BASE_DIR,"data_preprocessed", f"normal_gvw{gvw}", feature_type, drive_pattern)
    eval_path_dir = os.path.join(BASE_DIR,"data_preprocessed",f"Eval_{task_name_test}",feature_type,drive_pattern)                          
    if "TB" in task_name_test or "turbo" in task_name_test:
        main_task = "Turbo_problem"
    elif "egr_sw" in task_name_test:
        main_task = "EGR_SW"
    elif "egr_hw" in task_name_test:
        main_task = "EGR_HW"
    else:
        main_task = task_name_test
    result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
    column_list_path = os.path.join(train_path_dir, "column_list.txt")
    ckpt_load = find_latest_train_checkpoint(result_path, task_name_test)

    if not ckpt_load or not os.path.exists(ckpt_load):
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
            
    

   
    obliviate = False 
    paths_ok = (os.path.exists(train_path_dir) and os.path.exists(eval_path_dir)and os.path.exists(column_list_path) and os.path.exists(result_path) and os.path.exists(ckpt_load))
    if paths_ok:
        st.success("✅ 현재 ⚙️ 공통 설정 및 🔧 주행패턴 세부사항 설정에 대한 전처리 데이터 및 학습 결과가 존재합니다. (학습완료 페이지 이동 완료)")
        #### 1.) 데이터 정보 
        latest_result_dir, load_preprocessing_params,load_model_params = update_model_data_params(ckpt_load)    

        df_train_normal = pd.read_csv(os.path.join(train_path_dir,"data.csv"),encoding='cp949')
        if df_train_normal is not None:
            st.session_state["df_train_preview"] = df_train_normal
        if train_path_dir is not None:
            st.session_state["train_save_dir"] = train_path_dir

        train_path_dir = st.session_state.get("train_save_dir")
        train_csv = os.path.join(train_path_dir, "data.csv")
        column_list_path = os.path.join(train_path_dir, "column_list.txt")
        st.session_state["train_csv_path"] = train_csv
        st.session_state["column_list"] = column_list_path
        
        df_train_preview = st.session_state.get("df_train_preview")
        with open(column_list_path, "w", encoding="utf-8") as f:
            for col in df_train_preview.columns:
                if col not in ['file_number','attack','original_index','segment_original_start','segment_original_end']:
                    f.write(f"{col}\n")
        st.session_state["df_train_preview"] = df_train_preview
        st.session_state["train_csv_path"] = train_csv
        st.session_state["column_list"] = column_list_path
        st.session_state.preprocess_done = True
        
        data_params = {
            "Data Shape": df_train_preview.shape,
            "Train Task": task_name_train,
            "Test Task": task_name_test,
            "Feature Type": feature_type,
            "GVW": gvw,
            "Drive Pattern": drive_pattern,
            "Min Length": common_min_length,
            "Engine Threshold": engine_threshold,
            "Engine Threshold High": engine_threshold_high,
            "Fuel Threshold": fuel_threshold}
        st.session_state.data_params = data_params

        norm_saved = {k: "" if v is None else str(v) for k, v in data_params.items()}
        st.session_state.saved_params = norm_saved
        df_params = pd.DataFrame([(k, v) for k, v in norm_saved.items()], columns=["Parameter", "Value"])
        st.session_state.df_params = df_params
        st.session_state.params_synced = True

        df_params = st.session_state.get("df_params")
        if st.session_state.params_synced:
            toggle_key = "df_params_toggle"
            default_toggle = st.session_state.get(toggle_key, True)
            col1, col2 = st.columns([15, 1])
            with col1:
                show_params = st.checkbox("📊 Project Management / 전처리 공통 설정 및 주행패턴 세부사항", value=default_toggle, key=toggle_key)
            with col2:
                st.button("?", help="⚙️ 공통 설정 / 🔧 주행패턴 세부사항 전처리 설정에 대한 저장된 학습데이터가 존재하다는 것을 의미합니다.")
            if show_params:
                if df_params is not None and not df_params.empty:
                    rows, cols = df_train_preview.shape
                    st.markdown(
                        f"""
                        <div style="
                            padding:12px;
                            border-radius:10px;
                            font-size:13px;
                            color:inherit;">
                        
                        <b>📝 Project Management</b><br>
                        • 엔진 이상탐지 유형&nbsp;&nbsp;: {task_name_test}</code><br><br>

                        <hr style='border:2px solid #999;'>

                        <b>⚙️ 공통 설정</b><br>
                        • Feature Type : {feature_type}</code><br>
                        • GVW : {gvw}</code><br>
                        • Drive Pattern : {drive_pattern}</code><br><br>
                        
                        <b>🔧 주행패턴 세부사항 설정</b><br>
                        • Engine Threshold:
                        {engine_threshold}</code> /
                        {engine_threshold_high}</code><br>
                        • Fuel Threshold: {fuel_threshold}</code><br>
                        • Min Length : {common_min_length}</code><br><br>

                        <b>📊 학습 데이터 크기</b><br>
                        • Data Shape&nbsp;&nbsp;: ({rows}, {cols})</code><br><br>
                        </div>
                        """,
                        unsafe_allow_html=True)
                    csv_data = df_train_preview.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(label="⬇️ 학습 데이터 CSV 다운로드",data=csv_data,file_name="train_data.csv",mime="text/csv")
                    st.success("✅ 파라미터 변경 시 전처리 재실행 필수")
                    

        toggle_df_key = "df_train_preview_toggle"
        default_toggle_df = st.session_state.get(toggle_df_key, False)
        show_summary = st.checkbox("📊 학습 데이터 요약 통계량", value=default_toggle_df, key=toggle_df_key)
        if show_summary:
            df_summary = df_train_preview.drop(['file_number','original_index','segment_original_start','segment_original_end'],axis=1).describe()
            st.dataframe(df_summary, width=1200)  
            st.success("✅ 요약 통계량 확인 완료")
            
        toggle_plot_key_pretrain = "df_pretrain_plot_toggle"
        default_toggle_plot_pretrain = st.session_state.get(toggle_plot_key_pretrain, False)
        show_plot = st.checkbox("📈 학습 데이터 시각화", value=default_toggle_plot_pretrain, key=toggle_plot_key_pretrain)
        if show_plot:
            numeric_cols = df_train_preview.drop(['file_number','original_index','segment_original_start','segment_original_end'],axis=1).select_dtypes(include="number").columns.tolist()
            max_len = len(df_train_preview)
            if "range_idx" not in st.session_state:
                st.session_state.range_idx = (0, max_len)
                
            start_idx, end_idx = st.slider("Index Range",min_value=0,max_value=max_len,step=1,key="range_idx") 
            if start_idx > end_idx:
                st.session_state.range_idx = (end_idx, end_idx)
                start_idx, end_idx = end_idx, end_idx

            if "selected_multi_cols" not in st.session_state:
                st.session_state.selected_multi_cols = []

            multi_options = ["🔄 전체 변수 선택"] + numeric_cols
            selected_cols = st.multiselect("📈 시각화할 변수들 선택",multi_options,key="selected_multi_cols") 
            if "🔄 전체 변수 선택" in selected_cols:
                selected_cols = numeric_cols
            if selected_cols:
                with st.expander("📊 Plot 결과 보기", expanded=True):
                    for col in selected_cols:
                        sns.set(font_scale=1.0)
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(df_train_preview[col].iloc[start_idx:end_idx])
                        ax.set_title(f"{col} Plot (Index {start_idx} ~ {end_idx})")
                        ax.set_xlabel("Index")
                        ax.set_ylabel(col)
                        st.pyplot(fig)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.download_button(label=f"📥 {col} Plot Download",data=buf,file_name=f"{col}_Train_plot.png",mime="image/png")

        df_params = None
        data_params = None
        if "df_params" in st.session_state and "data_params" in st.session_state:
            df_params = st.session_state.df_params
            data_params = st.session_state.data_params

     
        if os.path.exists(load_preprocessing_params) ==False:
            st.warning("⚠️ 학습을 중단했습니다.⚙️ App 제어에서 🧹 Reset ( 전체 설정 초기화)를 눌러주세요.")
            undone_dir_path = os.path.dirname(load_preprocessing_params)
            shutil.rmtree(undone_dir_path)
            st.stop()

        if ckpt_load and load_preprocessing_params:
            with open(load_preprocessing_params, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            loaded_params = loaded_json[0]
            differences = []
            for key in data_params:
                val_current = data_params.get(key)
                val_json = loaded_params.get(key)
                if key == "Data Shape" and isinstance(val_current, tuple):
                    val_current = list(val_current)
                if val_current != val_json:
                    differences.append({
                        "Parameter": key,
                        "Current Value": val_current,
                        "Trained Value": val_json})
                    
            if not differences:
                st.info("✅ 기존 학습 및 현재 전처리 설정이 모두 일치합니다.")
                existing_ckpt_dir = os.path.dirname(os.path.dirname(ckpt_load))
                subfolders = [
                    os.path.join(existing_ckpt_dir, folder)
                    for folder in os.listdir(existing_ckpt_dir)
                    if os.path.isdir(os.path.join(existing_ckpt_dir, folder))]
                recall_list = []
                for folder in subfolders:
                    pkl_path = os.path.join(folder, "results_train.pkl")
                    if os.path.exists(pkl_path):
                        try:
                            with open(pkl_path, "rb") as f:
                                data = pickle.load(f)
                            recall_value = data.get("combined_metrics", {}).get("recall", None)
                            fpr_value = data.get("combined_metrics", {}).get("fpr", None)
                            thres_value = data.get("threshold",None)
                            recall_list.append({
                                "folder": os.path.basename(folder),
                                "path": folder,     # ★ 필수 ★
                                "recall": recall_value,
                                "FPR":fpr_value,
                                "Threshold":thres_value})
                        except Exception as e:
                            st.error(f"❗ {folder} → results_train.pkl 읽기 실패: {e}")
                
                recall_list = sorted(recall_list,key=lambda x: ( x["recall"] is None,-(x["recall"] or 0),x["FPR"] is None,(x["FPR"] or float("inf"), x["Threshold"] is None,(x["Threshold"] or float("inf")))))
                base_result_dir = os.path.join(result_path, task_name_test)
                history_toggle_key = "trained_history_open"

                st.markdown("---")
                st.markdown("#### 🗄️ 학습모델 이력")
                col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                with col1:
                    if st.button("📂 Train Model Archive"):
                        st.session_state[history_toggle_key] = not st.session_state.get(history_toggle_key, False)
                with col2:
                    st.button("?", help="현재 전처리/주행패턴 설정에 대한 학습 결과 이력입니다.")
                if st.session_state.get(history_toggle_key, False):
                    col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                    with col1:
                        display_list = [{"Folder": item["folder"],"Recall": item["recall"],"FPR": item["FPR"] ,"Threshold": item["Threshold"]} for item in recall_list]
                        st.table(display_list)
                    with col2:
                        st.button("?",help="Recall : 실제 이상 중 모델이 올바르게 탐지한 비율\n\n""FPR (False Positive Rate) : 정상 중 이상으로 잘못 판단한 비율\n\n""Threshold : 이상 여부를 판단하는 기준값 \n\n 원하는 기준을 통해서 이상탐지를 수행할 학습모델을 선택할 수 있습니다.")

                    delete_other_folders(recall_list)
                    delete_incomplete_train_results(base_result_dir)

                    ckpt_update = find_latest_train_checkpoint(result_path, task_name_test)
                    st.session_state.latest_train_ckpt_dir = ckpt_update
                    latest_result_dir, load_preprocessing_params,load_model_params = update_model_data_params(ckpt_update)    
                    if st.session_state.get("latest_train_ckpt_dir"):      
                        st.success('✅ Trained History Checkpoint 존재.\n\n 현재 학습된 모델로 이상탐지 수행 혹은 추가/재학습을 진행할 수 있습니다')
                        col1, col2 = st.columns(2)  
                        pdf_path = os.path.join(os.path.dirname(st.session_state.latest_train_ckpt_dir),"train_loss_plot.pdf")
                        with col1:
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as f:
                                    st.download_button(label="📥 Train Result Graph (History)",data=f.read(),file_name="train_loss_plot.pdf",mime="application/pdf",use_container_width=True)
                        html_plot_path = os.path.join(os.path.dirname(st.session_state.latest_train_ckpt_dir),"anomaly_score.html")
                        with col2:
                            if os.path.exists(html_plot_path):
                                with open(html_plot_path, "rb") as f:
                                    st.download_button(label="📥 Train Anomaly Score Plot (History)",data=f.read(),file_name="train_result_plot.html",mime="text/html",use_container_width=True)
            else:
                st.error("❌ 주행패턴 세부사항이 변경됐습니다.")
                obliviate = True
                paths_ok = False

        
        if obliviate == False:
            toggle_key_retrain = "retrain_params_toggle"
            default_toggle_retrain = st.session_state.get(toggle_key_retrain, False)
            st.markdown("---")
            st.markdown("#### 🛠️ 재학습 옵션")
            col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
            with col1:
                retrain_params = st.checkbox("🔄 모델 파라미터 재학습",value=default_toggle_retrain,key=toggle_key_retrain)
            with col2:
                st.button("?", help="Sidebar 메뉴의 📌 TranAD 파라미터 설정을 통해서 업데이트를 수행할 수 있습니다.")
            if not retrain_params:
                with open(load_model_params, "r", encoding="utf-8") as f:
                    model_params = json.load(f)
                    epoch = model_params['epoch'] 
                    window_size= model_params['window_size']
                    batch_size= model_params['batch_size']
                    learning_rate = model_params['learning_rate']
                    st.session_state["epoch"] = epoch
                    st.session_state["window_size"] = window_size
                    st.session_state["batch_size"] = batch_size
                    st.session_state["learning_rate"] = learning_rate
                    
                toggle_key_open_params = "open_params_toggle"
                default_toggle_open_params = st.session_state.get(toggle_key_open_params, False)
                col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                with col1:
                    open_params = st.checkbox("📊 현재 학습 파라미터 보기",value=default_toggle_open_params,key=toggle_key_open_params)
                with col2:
                    st.button("?", help="Sidebar 메뉴의 📌 TranAD 파라미터 설정을 통해서 지정한 값들입니다.")
                if open_params:
                    model_params = {"Epoch": epoch,"Window Size": window_size,"Batch Size": batch_size,"Learning Rate": learning_rate}
                    df_model_params = pd.DataFrame(model_params.items(), columns=["Parameter", "Value"])
                    st.table(df_model_params)
                    st.success("✅ 파라미터 변경 시 모델 재학습 필수")

            else:
                epoch, window_size, batch_size, learning_rate = activate_model_parameter_sidebar()
                st.session_state.epoch = epoch
                st.session_state.window_size = window_size
                st.session_state.batch_size = batch_size
                st.session_state.learning_rate = learning_rate
                if "train_trigger" not in st.session_state:
                    st.session_state.train_trigger = False
                if "train_running" not in st.session_state:
                    st.session_state.train_running = False
                if "train_run_done" not in st.session_state:
                    st.session_state.train_run_done = False
                if "train_logs" not in st.session_state:
                    st.session_state.train_logs = ""
                if "latest_train_ckpt_dir" not in st.session_state:
                    st.session_state.latest_train_ckpt_dir = None

                toggle_key_open_params_update = "open_params_toggle_update"
                col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                with col1:
                    open_params_update = st.checkbox("📊 학습 파라미터 업데이트",value=st.session_state.get(toggle_key_open_params_update, False),key=toggle_key_open_params_update)
                with col2:
                    st.button("?", help="Sidebar 메뉴의 📌 TranAD 파라미터 설정을 통해서 파라미터를 업데이트합니다. ")

                if open_params_update:
                    model_params = {"Epoch": epoch,"Window Size": window_size,"Batch Size": batch_size,"Learning Rate": learning_rate}
                    st.table(pd.DataFrame(model_params.items(), columns=["Parameter", "Value"]))

                toggle_key_retrain_params = "model_params_toggle_update"

                col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                with col1:
                    model_params_update = st.checkbox("🚀 모델 재학습 실행",value=st.session_state.get(toggle_key_retrain_params, False),key=toggle_key_retrain_params)
                with col2:
                    st.button("?", help="업데이트한 파라미터를 토대로 모델을 재학습합니다. 현재 표시되는 Log는 최근에 학습한 결과입니다.")
                if model_params_update:
                    st.info("파라미터 재학습 설정이 완료되었습니다. 재학습을 실행하면 아래의 결과가 업데이트됩니다.")
                    if st.button("▶️ 재학습 시작"):
                        st.session_state.train_trigger = True
                        st.session_state.train_run_done = False
                        st.session_state.train_logs = ""

                if st.session_state.train_trigger and not st.session_state.train_running:
                    st.session_state.train_running = True
                    st.session_state.train_trigger = False

                if st.session_state.train_running:
                    if "TB" in task_name_test or "turbo" in task_name_test:
                        main_task = "Turbo_problem"
                    elif "egr_sw" in task_name_test:
                        main_task = "EGR_SW"
                    elif "egr_hw" in task_name_test:
                        main_task = "EGR_HW"
                    else:
                        main_task = task_name_test
                    result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
                    os.makedirs(result_path, exist_ok=True)
                    log_path = os.path.join(BASE_DIR, "logs")
                    os.makedirs(log_path, exist_ok=True)
                    log_file = os.path.join(log_path,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}.txt")
                    cmd = (
                        f"python main.py --phase train "
                        f"--model TranAD "
                        f"--task_name \"{task_name_test}\" "
                        f"--train_dataset \"{os.path.join(train_path_dir, 'data.csv')}\" "
                        f"--eval_dataset \"{os.path.join(eval_path_dir, 'data.csv')}\" "
                        f"--columns \"{column_list_path}\" "
                        f"--epoch {epoch} "
                        f"--window_size {window_size} "
                        f"--batch_size {batch_size} "
                        f"--learning_rate {learning_rate} "
                        f"--save_result_dir \"{result_path}\" ")

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = "0"
                    with st.spinner("⏳ 모델 학습 진행 중..."):
                        rc, logs = run_subprocess_and_stream(cmd, env=env)
                    st.session_state.train_logs = logs
                    st.session_state.train_run_done = (rc == 0)
                    st.session_state.train_running = False
                    try:
                        with open(log_file, "w", encoding="utf-8") as f:
                            f.write(logs)
                    except:
                        pass
                    if rc == 0:
                        ckpt = find_latest_train_checkpoint(result_path, task_name_test)
                        if ckpt:
                            st.session_state.latest_train_ckpt_dir = ckpt

                if st.session_state.train_run_done:
                    is_finetune = (
                        st.session_state.latest_train_ckpt_dir is not None
                        and isinstance(st.session_state.latest_train_ckpt_dir, str)
                        and "finetune" in st.session_state.latest_train_ckpt_dir.lower()
                        and os.path.isfile(st.session_state.latest_train_ckpt_dir))
                    
                    logs_dir_finetuned = os.path.join(BASE_DIR, "logs")
                    finetuned_txt_files = [
                        f for f in os.listdir(logs_dir_finetuned)
                        if "finetuned" in f.lower()
                        and f.lower().endswith(".txt")
                        and os.path.isfile(os.path.join(logs_dir_finetuned, f))]
                    latest_finetuned_txt = None
                    latest_finetuned_content = ""

                    if finetuned_txt_files and is_finetune==True: 
                        latest_finetuned_txt = max(finetuned_txt_files,key=lambda f: os.path.getmtime(os.path.join(logs_dir_finetuned, f)))
                        txt_path = os.path.join(logs_dir_finetuned, latest_finetuned_txt)
                        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                            latest_finetuned_content = f.read()
                        st.text_area("학습 로그 (Latest finetuned)",value=latest_finetuned_content[-5000:],height=300)
                    else:
                        st.text_area("학습 로그",value=st.session_state.train_logs[-5000:],height=300)

                    if st.session_state.latest_train_ckpt_dir:
                        base_dir = os.path.dirname(st.session_state.latest_train_ckpt_dir) # 항상 가장 최신 파일 잡음 (finetune 포함)
                        col1, col2 = st.columns(2)
                        pdf_path = os.path.join(base_dir, "train_loss_plot.pdf")
                        with col1:
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📥 Train Result Graph (Updated)",f.read(),file_name="train_loss_plot.pdf",mime="application/pdf",use_container_width=True)
                        html_plot_path = os.path.join(base_dir, "anomaly_score.html")
                        with col2:
                            if os.path.exists(html_plot_path):
                                with open(html_plot_path, "rb") as f:
                                    st.download_button("📥 Train Anomaly Score Plot (Updated)",f.read(),file_name="train_result_plot.html",mime="text/html",use_container_width=True)

                        active_list = [data_params]
                        try:
                            save_path = os.path.join(BASE_DIR,result_path,task_name_test, "data_preprocessing_parameters.json")
                            with open(save_path, "w", encoding="utf-8") as f:
                                json.dump(active_list, f, indent=4)
                            train_params = {"epoch": epoch,"window_size": window_size,"batch_size": batch_size,"learning_rate": learning_rate}
                            result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
                            pattern = f"train_window_{st.session_state.window_size}_*"
                            result_window_dirs = [
                                d for d in glob(os.path.join(result_base_dir , pattern))
                                if os.path.isdir(d)]
                            latest_result_dir = max(result_window_dirs, key=os.path.getctime)
                            dest_path = os.path.join(latest_result_dir,"data_preprocessing_parameters.json")
                            shutil.move(save_path, dest_path)
                         
                            save_path = os.path.join(BASE_DIR, result_path, task_name_test, "model_training_parameters.json")
                            with open(save_path, "w", encoding="utf-8") as f:
                                json.dump(train_params, f, indent=4)
                            result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
                            pattern = f"train_window_{st.session_state.window_size}_*"
                            result_window_dirs = [
                                d for d in glob(os.path.join(result_base_dir , pattern))
                                if os.path.isdir(d)]
                            latest_result_dir = max(result_window_dirs, key=os.path.getctime)
                            dest_path = os.path.join(latest_result_dir,"model_training_parameters.json")
                            shutil.move(save_path, dest_path)

                        except FileNotFoundError:
                            st.warning("📁 결과 디렉토리를 찾을 수 없습니다. 재학습을 실행하세요.")
                        except ValueError:
                            st.warning("⚠️ 학습 결과 디렉토리가 아직 생성되지 않았습니다. 재학습을 실행하세요.")
                        except Exception as e:
                            st.error(f"❌ 파라미터 저장 중 오류 발생: {e}")

            #---------------- 전체 재학습 트리거 ----------------
            full_key_retrain = "retrain_full_toggle"
            col1, col2 = st.columns([15, 1])
            with col1:
                retrain_full = st.checkbox( "🔄 전체 재학습",value=st.session_state.get(full_key_retrain, False),key=full_key_retrain)
            with col2:
                st.button("?",help=("모델 파라미터 재학습 창이 열려 있을 시 끄고 다시 🔄 전체 재학습을 수행하세요.\n\n""필요 시 진행을 위해 ▶️ TranAD 모델 학습 실행 하단의 ▶ 시작을 여러 번 눌러주세요."))
            if retrain_full:
                st.warning("⚠️ 학습/검증 데이터 전처리 및 모델 학습 전 과정을 처음부터 다시 진행합니다.")
                st.session_state.force_full_retrain = True
                sync_model_params_from_ui()
                try:
                    train_full_run()
                except:
                     st.warning("파라미터 재학습 창이 열려 있을 시 끄고 다시 🔄 전체 재학습을 수행하세요.")

        else:
            pass
        
    if paths_ok==False or obliviate==True:
        st.warning(
    "❌ 현재 설정에 대한 학습 결과가 없습니다.\n\n"
    "--------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
    "1.) 최초 작업 실행 시:\n\n"
    "• 전 과정 실시 : 필수\n\n"
    
    "--------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
    "2.) Project Management 설정만 변경 시:\n\n"
    "• 학습 데이터 재학습: 선택\n\n"
    "• 검증 데이터 재업로드 및 전처리 후 재학습: 필수 \n\n"
    "--------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
    "3.) 그 외 설정 함께 변경 시:\n\n"
    "• 학습 데이터 재학습: 필수\n\n"
    "• 검증 데이터 재업로드 및 전처리 후 재학습: 필수")
        train_full_run()
        
    
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################


if page == "Anomaly Detection/Causal Analysis":
    top_left, top_right = st.columns([4, 1])
    with top_right:
        #pdf_bytes = generate_summary_pdf(get_page_summary(2))
        summary_text = load_summary_txt(page=2, base_dir=BASE_DIR)
        pdf_bytes = generate_summary_pdf(summary_text)
        st.download_button(label="📄기능 설명",data=pdf_bytes,file_name="기능설명_Page2.pdf",mime="application/pdf",use_container_width=True)

    st.markdown(f"#### 📉 Anomaly Detection")
    if "df_test_preview" not in st.session_state:
        st.session_state.df_test_preview = None
    if "test_csv_path" not in st.session_state:
        st.session_state.test_csv_path = None
    if "test_save_dir" not in st.session_state:
        st.session_state.test_save_dir = None
    if "use_calibration" not in st.session_state:
        st.session_state.use_calibration = True
    if "df_calibration_preview" not in st.session_state:
        st.session_state.df_calibration_preview = None
    if "calibration_csv_path" not in st.session_state:
        st.session_state.calibration_csv_path = None
    if "calibration_save_dir" not in st.session_state:
        st.session_state.calibration_save_dir = None
    
    task_name_train = st.session_state.get("task_name_train")
    task_name_test = st.session_state.get("test_task_fault")
    feature_type = st.session_state.get("feature_type") or st.session_state.common_feature_type
    gvw = st.session_state.get("gvw") or st.session_state.common_gvw
    drive_pattern = st.session_state.get("drive_pattern")
    common_min_length = st.session_state.get("common_min_length") or st.session_state.common_min_length
    engine_threshold = st.session_state.get("engine_threshold") or st.session_state.engine_threshold
    engine_threshold_high = st.session_state.get("engine_threshold_high") or st.session_state.engine_threshold_high
    fuel_threshold = st.session_state.get("fuel_threshold")
    module = st.session_state.get("preprocessing_module")
    log_path = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_path, exist_ok=True)

    if "TB" in task_name_test or "turbo" in task_name_test:
        main_task = "Turbo_problem"
    elif "egr_sw" in task_name_test:
        main_task = "EGR_SW"
    elif "egr_hw" in task_name_test:
        main_task = "EGR_HW"
    else:
        main_task = task_name_test
    result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
    ckpt_recent = find_latest_train_checkpoint(result_path, task_name_test)
    if ckpt_recent ==None:
        st.warning("⚠️ 학습 및 추론을 수행할 준비가 완료되지 않았습니다 .\n\n"
                   "먼저 **Data Upload and Preprocess** 페이지의 작업을 완성하세요.")
        st.stop()  

    required_keys = ['task_name_train','test_task_fault','feature_type','gvw','drive_pattern','common_min_length','engine_threshold','fuel_threshold']#['feature_type'] 
    missing = [k for k in required_keys if k not in st.session_state or st.session_state[k] in [None, ""]]
    if missing:
        st.warning("⚠️ 학습 및 추론을 수행할 준비가 완료되지 않았습니다 .\n\n"
                   "먼저 **Data Upload and Preprocess** 페이지의 작업을 완성하세요.")
        st.stop()   


    with st.expander("##### 📊 추론 / 보정 데이터 준비", expanded=True): #
        col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
        with col1:
             df_test_fault, test_save_dir = data_upload_ui(st,"Test","test_fault",mode="test",task_name=st.session_state.get("test_task_fault"),module=module)
        with col2:
            st.button("?", help="엔진 이상탐지 예측 대상인 데이터에 대한 전처리를 수행합니다.\n\n전처리는 1페이지의 ⚙️공통 설정/🔧주행패턴 세부사항 설정값을 토대로 진행됩니다.")
        if df_test_fault is not None:
            st.session_state["df_test_preview"] = df_test_fault
        if test_save_dir is not None:
            st.session_state["test_save_dir"] = test_save_dir
        try:
            test_path_dir = st.session_state.get("test_save_dir")
            if test_path_dir:
                test_csv = os.path.join(test_path_dir, "data.csv")
                test_column_list_path = os.path.join(test_path_dir, "column_list.txt")
                st.session_state["test_csv_path"] = test_csv
                st.session_state["test_column_list"] = test_column_list_path
            else:
                test_csv = st.session_state.get("test_csv_path")
                test_column_list_path = st.session_state.get("test_column_list")
            df_test_preview = st.session_state.get("df_test_preview")
            if df_test_preview is not None and test_csv:
                with open(test_column_list_path, "w", encoding="utf-8") as f:
                    for col in df_test_preview.columns: # state rerun 누적 정보 방지
                        if col not in ['file_number','attack','original_index','segment_original_start','segment_original_end','prediction','anomaly_score','threshold']:
                            f.write(f"{col}\n")
                st.success(f"✅ Test 데이터 저장 완료: {test_csv}")
        except Exception as e:
            st.error(f"⚠ Test 데이터 처리 중 오류: {e}")

        # 버튼 표시 조건
        test_ready = (
            st.session_state.get("df_test_preview") is not None and
            st.session_state.get("test_csv_path") is not None and
            st.session_state.get("test_column_list") is not None)
        if test_ready:
            use_checkbox = st.checkbox("📌 Normal Calibration 수행 여부", value=True)
            st.session_state["use_calibration"] = use_checkbox
            # --- Calibration 수행 조건 ---
            if st.session_state["use_calibration"]:

                # 1) 기존 Calibration 파일이 존재 여부 확인
                calibration_csv_path = os.path.join(BASE_DIR, "data_preprocessed","normal_engine_B",feature_type,drive_pattern,"data.csv")
                calibration_save_dir = os.path.join(BASE_DIR, "data_preprocessed","normal_engine_B",feature_type,drive_pattern)
                calibration_exists = (calibration_csv_path is not None and os.path.exists(calibration_csv_path) and os.path.exists(calibration_save_dir))
                if calibration_exists:
                    reupload = st.checkbox("📤 Normal Calibration Data 새로 업로드하기")
                    if not reupload:
                        st.success(f"📁 기존 Normal Calibration 데이터를 재사용합니다.\n: {calibration_csv_path}")
                        try:
                            calibration_csv = pd.read_csv(calibration_csv_path)
                            st.session_state["df_calibration_preview"] = calibration_csv
                            st.session_state.calibration_csv_path = calibration_csv_path
                            st.session_state.calibration_save_dir = calibration_save_dir 
                        except Exception as e:
                            st.error(f"⚠ 기존 Calibration 데이터를 불러오는 중 오류: {e}")
                    else:
                        col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                        with col1:
                            df_normal_calibration, calib_save_dir = data_upload_ui(st,"Calibration","test_calibration",mode="normal_calibration",task_name=st.session_state.get("test_task_fault"),module=module)
                        with col2:
                            st.button("?", help="다른 엔진의 정상데이터에 대한 전처리를 수행합니다.\n\n 전처리는 1페이지의 ⚙️공통 설정/🔧주행패턴 세부사항 설정값을 토대로 진행됩니다. \n\n해당 데이터는 학습모델의 오탐지 위험 중 과검률을 완화하는 데에 사용됩니다.")
                        if df_normal_calibration is not None:
                            st.session_state["df_calibration_preview"] = df_normal_calibration
                        if calib_save_dir is not None:
                            st.session_state["calibration_save_dir"] = calib_save_dir
                        try:
                            calibration_path_dir = st.session_state.get("calibration_save_dir")
                            if calibration_path_dir:
                                calibration_csv = os.path.join(calibration_path_dir, "data.csv")
                                calibration_column_list_path = os.path.join(calibration_path_dir, "column_list.txt")
                                st.session_state["calibration_csv_path"] = calibration_csv
                                st.session_state["calibration_column_list"] = calibration_column_list_path
                            else:
                                calibration_csv = st.session_state.get("calibration_csv_path")
                                calibration_column_list_path = st.session_state.get("calibration_column_list")
                            df_calibration_preview = st.session_state.get("df_calibration_preview")
                            if df_calibration_preview is not None and calibration_csv:
                                with open(calibration_column_list_path, "w", encoding="utf-8") as f:
                                    for col in df_calibration_preview.columns:
                                        if col not in ['file_number','attack','original_index','segment_original_start','segment_original_end']:
                                            f.write(f"{col}\n")
                                st.success(f"✅ New Calibration 데이터 저장 완료: {calibration_csv}")
                        except Exception as e:
                            st.error(f"⚠ Calibration 데이터 처리 중 오류: {e}")
                else:
                    col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                    with col1:
                        df_normal_calibration, calib_save_dir = data_upload_ui(st,"Calibration","test_calibration",mode="normal_calibration",task_name=st.session_state.get("test_task_fault"),module=module)
                    with col2:
                        st.button("?", help="다른 엔진의 정상데이터에 대한 전처리를 수행합니다.\n\n해당 데이터는 학습모델의 오탐지 위험 중 과검률을 완화하는 데에 사용됩니다.")
                    if df_normal_calibration is not None:
                        st.session_state["df_calibration_preview"] = df_normal_calibration
                    if calib_save_dir is not None:
                        st.session_state["calibration_save_dir"] = calib_save_dir
                    try:
                        calibration_path_dir = st.session_state.get("calibration_save_dir")
                        if calibration_path_dir:
                            calibration_csv = os.path.join(calibration_path_dir, "data.csv")
                            calibration_column_list_path = os.path.join(calibration_path_dir, "column_list.txt")
                            st.session_state["calibration_csv_path"] = calibration_csv
                            st.session_state["calibration_column_list"] = calibration_column_list_path
                        else:
                            calibration_csv = st.session_state.get("calibration_csv_path")
                            calibration_column_list_path = st.session_state.get("calibration_column_list")
                        df_calibration_preview = st.session_state.get("df_calibration_preview")
                        if df_calibration_preview is not None and calibration_csv:
                            with open(calibration_column_list_path, "w", encoding="utf-8") as f:
                                for col in df_calibration_preview.columns:
                                    if col not in ['file_number','attack','original_index','segment_original_start','segment_original_end']:
                                        f.write(f"{col}\n")
                            st.success(f"✅ Calibration 데이터 저장 완료: {calibration_csv}")
                    except Exception as e:
                        st.error(f"⚠ Calibration 데이터 처리 중 오류: {e}")

                result_path_test = os.path.join(BASE_DIR, "result", main_task,f"Target_Inference_{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}")
                log_test_file = os.path.join(log_path,f"Target_Inference_{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}.txt")

            else:
                st.info("⚙️ Calibration 옵션이 꺼져 있습니다. Calibration 과정을 생략합니다.")
                st.session_state.df_calibration_preview = None
                st.session_state.calibration_csv_path = None
                st.session_state.calibration_save_dir = None
                result_path_test = os.path.join(BASE_DIR, "result", main_task,f"Target_Inference_No_Calibration_{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}")
                log_test_file = os.path.join(log_path,f"Target_Inference_No_Calibration_{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}.txt")
            os.makedirs(result_path_test, exist_ok=True)
            st.session_state["result_path_test"] = result_path_test
            st.session_state["log_test_file"] = log_test_file

        result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
        ckpt_load = find_latest_train_checkpoint(result_path, task_name_test)
        st.session_state["latest_train_ckpt_dir"] = ckpt_load

        model_params_path = os.path.join(BASE_DIR, result_path, task_name_test, "model_training_parameters.json")
        if os.path.exists(model_params_path ):
            with open(model_params_path , "r", encoding="utf-8") as f:
                loaded_model_params = json.load(f)  
    
    with st.expander("##### ▶️ 추론 실행", expanded=True):
        col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
        with col1:
            if st.button("🚀 TranAD 추론 실행"):
                test_csv = st.session_state.get("test_csv_path")
                test_column_list = st.session_state.get("column_list") 
                use_calibration = st.session_state.get("use_calibration")
                calibration_csv = st.session_state.get("calibration_csv_path")
                result_path_test = st.session_state.get("result_path_test")
                log_test_file = st.session_state.get("log_test_file")
                ckpt_dir = st.session_state.get("latest_train_ckpt_dir")
                window_size = st.session_state.get("window_size") or loaded_model_params.get("window_size")
                batch_size = st.session_state.get("batch_size") or loaded_model_params.get("batch_size")
                epoch = st.session_state.get("epoch") or loaded_model_params.get("epoch")
                st.session_state["window_size"] = window_size
                st.session_state["batch_size"] = batch_size
                st.session_state["epoch"] = epoch

                if not test_csv or not test_column_list:
                    st.error("⚠️ Test 데이터 준비가 완료되지 않았습니다.")
                    st.stop()
                if not ckpt_dir:
                    st.error("⚠️ 학습 체크포인트가 존재하지 않습니다. 먼저 학습을 완료하세요.")
                    st.stop()
                cmd = (
                    f"python main.py --phase test "
                    f"--model TranAD "
                    f"--task_name \"{task_name_test}\" "
                    f"--test_dataset \"{test_csv}\" "
                    f"--columns \"{test_column_list}\" "
                    f"--window_size {window_size} "
                    f"--batch_size {batch_size} "
                    f"--save_result_dir \"{result_path_test}\" "
                    f"--model_path \"{ckpt_dir}\" ")
                if use_calibration and calibration_csv:
                    cmd += f" --calibration_normal_dataset \"{calibration_csv}\" "
                else:
                    st.write("Calibration 과정을 생략합니다")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0"

                with st.spinner("추론 진행 중..."):
                    rc, logs = run_subprocess_and_stream(cmd, env=env)
                st.session_state.test_logs = logs
                st.session_state.test_run_done = (rc == 0)
                try:
                    with open(log_test_file, "w", encoding="utf-8") as f:
                        f.write(logs)
                    st.success(f"Inference 로그 저장됨: {log_test_file}")
                except:
                    st.error("Inference 로그 저장 실패")

                if rc == 0:
                    latest_result = find_latest_test_result(result_path_test, task_name_test)
                    if latest_result:
                        st.session_state.latest_test_result = latest_result
                        st.success(f"추론 결과 저장 완료: {latest_result}")
                    else:
                        st.error("추론 결과 파일을 찾을 수 없습니다.")
                else:
                    st.error("추론 실패. 로그를 확인하세요.")
        with col2:
            st.button("!", help="추론 이후 결과 plot과 csv파일을 다운받는 것을 권장합니다.")

        if st.session_state.get("test_run_done") and st.session_state.get("latest_test_result"):
            st.text_area("추론 로그", value=st.session_state.test_logs[-5000:], height=300)
            html_plot_path = os.path.join(os.path.dirname(st.session_state.latest_test_result), f"{task_name_test}_anomaly_score_plotly.html")
            if os.path.exists(html_plot_path):
                with open(html_plot_path, "rb") as f:
                    st.download_button(
                        label="📥 Test Result Plot (New)",
                        data=f.read(),
                        file_name=f"{task_name_test}_test_result_plot.html",
                        mime="text/html")
                    
            results_pkl = st.session_state.latest_test_result
            if results_pkl and os.path.exists(results_pkl):
                with open(results_pkl, "rb") as f:
                    result_data = pickle.load(f)
                prediction_raw = result_data["prediction"]
                anomaly_raw = result_data["anomaly_score"]
                threshold_raw = result_data['threshold']

                df_pred = convert_to_df(prediction_raw, "prediction")
                df_anom = convert_to_df(anomaly_raw, "anomaly_score")
                df_result = df_pred.merge(df_anom, on="original_index").reset_index(drop=True)
                df_result['threshold'] = threshold_raw
                st.write("###### 📊 Prediction / Anomaly Score Table for Test Data")
                filtered_df = df_test_preview[df_test_preview["original_index"].isin(df_result["original_index"])].reset_index(drop=True).copy()
                for col in ["prediction", "anomaly_score", "threshold"]:
                    if col in filtered_df.columns:
                        filtered_df.drop(columns=[col], inplace=True)
                df_test_predicted = filtered_df.merge(df_result, on="original_index", how="left").reset_index(drop=True)

                normal_count = df_test_predicted['prediction'].value_counts().get(1, 0)
                fault_count = df_test_predicted['prediction'].value_counts().get(0, 0)
                total_count = normal_count + fault_count
                fault_ratio = fault_count / total_count if total_count > 0 else 0
                st.write("Normal Predict Count:", normal_count)
                st.write("Fault Predict Count:", fault_count)
                st.write("Predicted Fault Ratio (%):", round(fault_ratio * 100, 2), "%")
                st.write(df_test_predicted)
                st.session_state.df_test_preview = df_test_predicted.copy()

                csv = df_test_predicted.to_csv(index=False).encode("utf-8")
                st.download_button(label="📥 Download Predicted Test Data",data=csv,file_name=f"{task_name_test}_predicted.csv",mime="text/csv")
                
    st.info("추론 완료 이후 인과분석을 수행할 수 있습니다.")
    if st.session_state.df_test_preview is not None and st.session_state.get("test_run_done") and st.session_state.get("latest_test_result"):
        df_test_predicted_cut = st.session_state.df_test_preview
        with st.expander("###### 🔎 Anomaly Detection Data 구간 선택 ( 추론 결과 상세 관찰)", expanded=True):
            min_idx = int(df_test_predicted_cut["original_index"].min())
            max_idx = int(df_test_predicted_cut["original_index"].max())
            data_key = f"{min_idx}_{max_idx}"
            if "start_idx" not in st.session_state:
                st.session_state.start_idx = min_idx
            if "end_idx" not in st.session_state:
                st.session_state.end_idx = max_idx
            if "idx_range_key" not in st.session_state or st.session_state.idx_range_key != data_key:
                st.session_state.idx_range_key = data_key
                st.session_state.start_idx = min_idx
                st.session_state.end_idx = max_idx

            def sync_idx_range():
                st.session_state.start_idx = st.session_state.slider_range[0]
                st.session_state.end_idx = st.session_state.slider_range[1]
            st.slider("Original Index Range",min_value=min_idx,max_value=max_idx,value=(st.session_state.start_idx, st.session_state.end_idx),step=1,key="slider_range",on_change=sync_idx_range)
            start_idx = st.session_state.start_idx
            end_idx = st.session_state.end_idx
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Start Original Index",min_value=min_idx,max_value=max_idx,step=1,key="start_idx")
            with col2:
                st.number_input("End Original Index",min_value=min_idx,max_value=max_idx,step=1,key="end_idx")

            # ------------------- Validation -------------------
            if st.session_state.start_idx > st.session_state.end_idx:
                st.error("Start index must be less than or equal to End index.")
                st.session_state.start_idx = st.session_state.end_idx
            else:
                selected_df_new = df_test_predicted_cut[
                    (df_test_predicted_cut["original_index"] >= start_idx) &
                    (df_test_predicted_cut["original_index"] <= end_idx)].set_index("original_index") 
                selected_df_new ['original_index'] = selected_df_new.index         
                if "selected_df" not in st.session_state or st.session_state.selected_df is None or not selected_df_new.equals(st.session_state.selected_df):
                    st.session_state.selected_df = selected_df_new
                st.write(f"📄 Rows Selected: {len(st.session_state.selected_df)}")
                st.dataframe(st.session_state.selected_df)
                csv = st.session_state.selected_df.to_csv(index=False).encode("utf-8")
                st.download_button(label="📥 Download Selected Data",data=csv,
                    file_name=f"{task_name_test}_test_selected_{start_idx}_{end_idx}.csv",mime="text/csv")
                
        selected_df = st.session_state.get("selected_df", None)
        toggle_test_selected_key = "df_test_selected_preview_toggle"
        default_toggle_selected_df = st.session_state.get(toggle_test_selected_key, False)
        show_summary_selected_df = st.checkbox("📊 Anomaly Detection Data 요약 통계량 보기", value=default_toggle_selected_df, key=toggle_test_selected_key)
        if show_summary_selected_df:
            selected_df_summary = selected_df.drop(['file_number','original_index','segment_original_start','segment_original_end','prediction','anomaly_score','threshold'],axis=1).describe()
            if selected_df_summary.iloc[0:1].T.sum().values[0] !=0.0:
                st.dataframe(selected_df_summary, width=1200)  
                st.success("✅ 요약 통계량 확인 완료")
            else:
                st.warning("⚠ 현재 구간에 해당하는 데이터가 존재하지 않습니다.")


        toggle_plot_key_selected = "df_test_selected_plot_toggle"
        default_toggle_plot_selected = st.session_state.get(toggle_plot_key_selected, False)
        show_plot_selected = st.checkbox("📈 Anomaly Detection Data 시각화 보기", value=default_toggle_plot_selected, key=toggle_plot_key_selected)
        if show_plot_selected:
            numeric_cols = selected_df.drop(['file_number','original_index','segment_original_start','segment_original_end','prediction','anomaly_score','threshold'],axis=1).select_dtypes(include="number").columns.tolist()
            try:
                min_idx = int(selected_df.index.min())
                max_idx = int(selected_df.index.max())
                multi_options = ["🔄 전체 변수 선택"] + numeric_cols
                selected_cols = st.multiselect("📈 시각화할 변수들 선택",multi_options,key="selected_multi_cols_selected") 
                if "🔄 전체 변수 선택" in selected_cols:
                    selected_cols = numeric_cols
                if selected_cols:
                    with st.expander("📊 Plot 결과 보기", expanded=True):
                        for col in selected_cols:
                            sns.set(font_scale=1.0)
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(selected_df[col].loc[start_idx:end_idx],color='blue')
                            ax.set_title(f"{col} Plot (Original Index {start_idx} ~ {end_idx})")
                            ax.set_xlabel("Index")
                            ax.set_ylabel(col)
                            st.pyplot(fig)
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            st.download_button(label=f"📥 {col} Plot Download",data=buf,file_name=f"{col}_Anomaly_Detection_plot.png",mime="image/png")
            except:
                st.warning("⚠ 현재 구간에 해당하는 데이터가 존재하지 않습니다.")

    if st.session_state.df_test_preview is not None and st.session_state.get("test_run_done") and st.session_state.get("latest_test_result"):
        st.markdown("---")
        st.markdown(f"#### 🔗 Causal Analysis")
        with st.expander("##### ▶️ Phase 1", expanded=True):
            if "pcmci_analysis_logs" not in st.session_state:
                st.session_state.pcmci_analysis_logs = ""
            if "pcmci_analysis_run_done" not in st.session_state:
                st.session_state.pcmci_analysis_run_done = False

            # ---------------- 필수 정보 업로드 재확인 & 세션 업데이트 ----------------
            #### 기본 설정 정보 
            task_name_test = st.session_state.get("test_task_fault")
            feature_type = st.session_state.get("feature_type")
            gvw = st.session_state.get("gvw")
            drive_pattern = st.session_state.get("drive_pattern")

            #### 모델 설정 정보
            window_size = st.session_state["window_size"]
            batch_size = st.session_state["batch_size"]
            epoch = st.session_state["epoch"]

            #### Train / Test 결과 경로 설정 정보
            if "TB" in task_name_test or "turbo" in task_name_test:
                main_task = "Turbo_problem"
            elif "egr_sw" in task_name_test:
                main_task = "EGR_SW"
            elif "egr_hw" in task_name_test:
                main_task = "EGR_HW"
            else:
                main_task = task_name_test
            result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
            ckpt_load = find_latest_train_checkpoint(result_path, task_name_test)
            st.session_state["latest_train_ckpt_dir"] = ckpt_load
            ckpt_dir = st.session_state["latest_train_ckpt_dir"]
            result_train_path = os.path.dirname(ckpt_dir)
            use_calibration = st.session_state.use_calibration
            if use_calibration == True:
                result_path_test = os.path.join(BASE_DIR, "result", main_task,f"Target_Inference_{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}")
            else:
                result_path_test = os.path.join(BASE_DIR, "result", main_task,f"Target_Inference_No_Calibration_{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}")
            latest_result = find_latest_test_result(result_path_test, task_name_test)
            st.session_state["latest_test_result"] = latest_result
            result_dir = st.session_state["latest_test_result"] 
            result_test_path = os.path.dirname(result_dir)

            #### Train / Test 데이터 경로 설정 정보 
            normal_dataset = st.session_state.get("train_csv_path")
            abnormal_dataset = st.session_state.get("test_csv_path")
            column_list_path = st.session_state.get("column_list")

            
            #### Log & Save Directory 생성
            save_dir = os.path.join(BASE_DIR, "causal_trace","results",f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}")
            os.makedirs(save_dir, exist_ok=True)
            log_dir = os.path.join(BASE_DIR,"causal_trace", "logs",f"{task_name_test}")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}.txt")

            # ------------------- CFG_DICT & Params 구성 -------------------
            cfg_dict = {
                "log_dir": log_dir + "/",    # 위에서 작성한 log_dir 활용
                "save_dir_compare_result": save_dir + "/",
                "task": f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}",
                "vars_highlight": "None",

                "abnormal": {
                    "dataset_path": abnormal_dataset,
                    "save_dir": os.path.join(save_dir, "abnormal"),
                    "problem_type": main_task,
                    "gvw": gvw,
                    "drive_pattern": drive_pattern,
                    "feature_type": feature_type},

                "normal": {
                    "dataset_path": normal_dataset,
                    "save_dir": os.path.join(save_dir, "normal"),
                    "problem_type": main_task,
                    "gvw": gvw,
                    "drive_pattern": drive_pattern,
                    "feature_type": feature_type},

                "result_train_path": result_train_path,
                "result_test_path": result_test_path}
            
            base_params_pcmci = {
                "downsample_rate": 1,
                "cond_ind_test": "PARCORR",
                "tau_max": 5,
                "tau_min": 0,
                "min_length": 200,
                "max_length": 1000,
                "alpha_level": 0.05,
                "combine_segments": True,
                "remove_bidirectional": True,
                "use_parallel": True}

            cfg_json_path = os.path.join(log_dir, "cfg_dict_input.json")
            with open(cfg_json_path, "w") as f:
                json.dump(cfg_dict,f)
            pcmci_json_path = os.path.join(log_dir, "base_params_pcmci_input.json")
            with open(pcmci_json_path, "w") as f:
                json.dump(base_params_pcmci,f)

            defaults = {
                "pcmci_log_queue": queue.Queue(),
                "pcmci_logs_buffer": "",
                "pcmci_analysis_logs": "",
                "pcmci_analysis_run_done": False,
                "pcmci_running": False,
                "pcmci_process": None,
                "pcmci_progress": 0,
                "pcmci_status": "대기 중"}
            for k, v in defaults.items():
                if k not in st.session_state:
                    st.session_state[k] = v

            # ------------------- 백그라운드 실행 함수 -------------------
            def run_pcmci_bg(cmd, env, log_file, log_queue):
                try:
                    log_queue.put(("STATUS", "PCMCI 프로세스 시작"))
                    log_queue.put(("PROGRESS", 10))
                    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
                    log_queue.put(("PROCESS", process))
                    log_queue.put(("STATUS", "PCMCI 실행 중"))
                    log_queue.put(("PROGRESS", 30))
                    for line in iter(process.stdout.readline, ""):
                        log_queue.put(("LOG", line))
                        if "PCMCI finished" in line:
                            log_queue.put(("PROGRESS", 90))
                    process.stdout.close()
                    rc = process.wait()
                    log_queue.put(("PROGRESS", 100))
                    log_queue.put(("DONE", rc == 0))
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("".join([]))  
                except Exception as e:
                    log_queue.put(("ERROR", str(e)))

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 PCMCI 분석", disabled=st.session_state.pcmci_running):
                    st.session_state.pcmci_running = True
                    st.session_state.pcmci_analysis_run_done = False
                    st.session_state.pcmci_logs_buffer = ""
                    st.session_state.pcmci_progress = 0
                    st.session_state.pcmci_status = "초기화"
                    cmd = [
                        sys.executable,
                        os.path.join(BASE_DIR, "causal_trace", "causal_trace.py"),
                        "--cfg_dict", cfg_json_path,
                        "--base_params_pcmci", pcmci_json_path,
                        "--dir_log", log_dir,
                        "--seed", "42"]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = "0"
                    threading.Thread(target=run_pcmci_bg,args=(cmd, env, log_file, st.session_state.pcmci_log_queue),daemon=True,).start()
            with col2:
                if st.button("⛔ 강제 중단", disabled=not st.session_state.pcmci_running):
                    proc = st.session_state.pcmci_process
                    if proc:
                        proc.kill()
                        st.session_state.pcmci_status = "사용자에 의해 중단됨"
                        st.session_state.pcmci_running = False
                        st.session_state.pcmci_analysis_run_done = True
                        st.warning("PCMCI 프로세스가 강제 종료되었습니다.")

            try:
                while True:
                    msg = st.session_state.pcmci_log_queue.get_nowait()
                    msg_type, payload = msg
                    if msg_type == "LOG":
                        st.session_state.pcmci_logs_buffer += payload
                    elif msg_type == "STATUS":
                        st.session_state.pcmci_status = payload
                    elif msg_type == "PROGRESS":
                        st.session_state.pcmci_progress = payload
                    elif msg_type == "PROCESS":
                        st.session_state.pcmci_process = payload
                    elif msg_type == "DONE":
                        st.session_state.pcmci_analysis_run_done = payload
                        st.session_state.pcmci_running = False
                        st.session_state.pcmci_analysis_logs = st.session_state.pcmci_logs_buffer
                    elif msg_type == "ERROR":
                        st.session_state.pcmci_status = "오류 발생"
                        st.error(payload)
            except queue.Empty:
                pass
            
            col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
            with col1:
                st.markdown(f"**상태:** {st.session_state.pcmci_status} (Background Run)")
            with col2:
                st.button("?", help="다른 기능에 방해받지 않고 수행됩니다.\n\nLog에 즉각적으로 분석과정이 나타나지 않을 때는 다른 페이지로 잠깐 이동 혹은 기능 수행(Ex.📈 Anomaly Detection Data 시각화 보기)을 권장합니다.")
            st.progress(st.session_state.pcmci_progress)
            st.text_area("PCMCI 로그",value=st.session_state.pcmci_logs_buffer[-5000:],height=300,key="pcmci_log_area",)
            if st.session_state.pcmci_analysis_run_done and st.session_state.pcmci_analysis_logs:
                st.download_button("📄 PCMCI 로그 다운로드",data=st.session_state.pcmci_analysis_logs,file_name="pcmci_analysis.log",mime="text/plain",)
            if st.session_state.pcmci_running:
                time.sleep(0.5)
                #st.experimental_rerun()
            elif st.session_state.pcmci_analysis_run_done:
                st.success("✅ PCMCI 분석 종료")


            # ------------------- Phase 2 활성화 -------------------
            pcmci_normal_path = os.path.join(save_dir, "normal", "pcmci_results.pkl")
            pcmci_abnormal_path = os.path.join(save_dir, "abnormal", "pcmci_results.pkl")
            if st.session_state.pcmci_analysis_run_done:
                if os.path.exists(pcmci_normal_path) and os.path.exists(pcmci_abnormal_path):
                    st.success("✅ Normal / Abnormal PCMCI 분석 결과 완료. Phase 2를 수행할 수 있습니다.")
                    with open(column_list_path, "r", encoding="utf-8") as f:
                        columns = [line.strip() for line in f.readlines() if line.strip()]
                else:
                    st.warning("⚠️ 현재 PCMCI 분석에 부적합한 Normal or Abnormal 데이터셋이 존재합니다. 분석 로그를 확인하세요.")

           
            # =================== Phase 2 State ===================
            defaults_phase2 = {
                "causal_log_queue": queue.Queue(),
                "causal_logs_buffer": "",
                "causal_analysis_logs": "",
                "causal_running": False,
                "causal_run_done": False,
                "causal_process": None,
                "causal_progress": 0,
                "causal_status": "대기 중",
                "feature_importance_type": "ttest",
                "target_variable": None}
            for k, v in defaults_phase2.items():
                if k not in st.session_state:
                    st.session_state[k] = v

            def run_causal_bg(cmd, env, log_queue):
                try:
                    log_queue.put(("STATUS", "Causal Analysis 시작"))
                    log_queue.put(("PROGRESS", 10))
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env)
                    log_queue.put(("PROCESS", process))
                    log_queue.put(("STATUS", "Causal Analysis 실행 중"))
                    log_queue.put(("PROGRESS", 30))
                    for line in iter(process.stdout.readline, ""):
                        log_queue.put(("LOG", line))
                        if any(k in line.lower() for k in ["ttest", "cohen", "shap", "finished"]):
                            log_queue.put(("PROGRESS", 90))
                    process.stdout.close()
                    rc = process.wait()
                    log_queue.put(("PROGRESS", 100))
                    log_queue.put(("DONE", rc == 0))
                except Exception as e:
                    log_queue.put(("ERROR", str(e)))


        with st.expander("##### ▶️ Phase 2", expanded=True):
            # Default State
            if "threshold" not in st.session_state:
                st.session_state.threshold = 0.2
            if "max_depth" not in st.session_state:
                st.session_state.max_depth = 3
            if "top_k" not in st.session_state:
                st.session_state.top_k = 3
            with st.sidebar.expander("📌 인과분석 파라미터 설정", expanded=False):
                threshold = st.number_input("Threshold",min_value=0.0,max_value=1.0,value=st.session_state.threshold,step=0.01)
                max_depth = st.number_input("Max Depth",min_value=1,max_value=10,value=st.session_state.max_depth,step=1)
                top_k = st.number_input("Top K",min_value=1,max_value=20,value=st.session_state.top_k,step=1)

            base_params_causal_trace = {
                "window_size": window_size,
                "batch_size": batch_size,
                "threshold": threshold,
                "max_depth": max_depth,
                "top_k": top_k}
            
            col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
            with col1:
                st.write("Causal Analysis Parameters 조정:", base_params_causal_trace)
            with col2:
                st.button("?", help="Phase2를 수행하기 위한 인과분석 파라미터 집합입니다. \n\n Sidebar 메뉴의 📌 인과분석 파라미터 설정을 통해서 조정할 수 있습니다.")
            causal_json_path = os.path.join(log_dir, "base_params_causal_trace.json")
            with open(causal_json_path, "w") as f:
                json.dump(base_params_causal_trace,f)

            col_run, col_stop = st.columns(2)
            # ▶ 실행
            with col_run:
                if st.button("🚀 Causal Trace & Feature Importance 분석",disabled=st.session_state.causal_running):
                    if not st.session_state.pcmci_analysis_run_done:
                        st.warning("⚠️ PCMCI Phase 1을 먼저 완료하세요.")
                    else:
                        st.session_state.causal_running = True
                        st.session_state.causal_run_done = False
                        st.session_state.causal_logs_buffer = ""
                        st.session_state.causal_progress = 0
                        st.session_state.causal_status = "초기화"
                        cmd = [
                            sys.executable,
                            os.path.join(BASE_DIR, "causal_trace", "causal_trace.py"),
                            "--cfg_dict", cfg_json_path,
                            "--base_params_pcmci", pcmci_json_path,
                            "--base_params_causal_trace", causal_json_path,
                            "--dir_log", log_dir,
                            "--seed", "42"]
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = "0"
                        threading.Thread(target=run_causal_bg,args=(cmd, env, st.session_state.causal_log_queue),daemon=True,).start()

            # ⛔ 강제 중단
            with col_stop:
                if st.button("⛔ 강제 중단.", disabled=not st.session_state.causal_running):
                    proc = st.session_state.causal_process
                    if proc:
                        proc.kill()
                        st.session_state.causal_status = "사용자에 의해 중단됨"
                        st.session_state.causal_running = False
                        st.session_state.causal_run_done = True
                        st.warning("Causal Analysis 프로세스가 강제 종료되었습니다.")
            try:
                while True:
                    msg_type, payload = st.session_state.causal_log_queue.get_nowait()
                    if msg_type == "LOG":
                        st.session_state.causal_logs_buffer += payload
                    elif msg_type == "STATUS":
                        st.session_state.causal_status = payload
                    elif msg_type == "PROGRESS":
                        st.session_state.causal_progress = payload
                    elif msg_type == "PROCESS":
                        st.session_state.causal_process = payload
                    elif msg_type == "DONE":
                        st.session_state.causal_run_done = payload
                        st.session_state.causal_running = False
                        st.session_state.causal_analysis_logs = st.session_state.causal_logs_buffer
                    elif msg_type == "ERROR":
                        st.session_state.causal_status = "오류 발생"
                        st.error(payload)
            except queue.Empty:
                pass


            col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
            with col1:
                st.markdown(f"**상태:** {st.session_state.causal_status} (Background Run)")
            with col2:
                st.button("?", help="Phase1과 동일하게 다른 기능에 방해받지 않고 수행됩니다. \n\nLog에 즉각적으로 분석과정이 나타나지 않을 때는 다른 페이지로 잠깐 이동 혹은 기능 수행\n\n(Ex. Sidebar 메뉴의 🎯 인과분석 대상 변수 선택)을 권장합니다.")
            st.progress(st.session_state.causal_progress)
            st.text_area("Causal Trace & Feature Importance 분석 로그",value=st.session_state.causal_logs_buffer[-5000:],height=300,key="causal_log_area")
            completed_targets = []
            incomplete_targets = []
            with open(column_list_path, "r", encoding="utf-8") as f:
                columns = [line.strip() for line in f.readlines() if line.strip()]
            for col in columns:
                tree_png_all = os.path.join(save_dir, f"causal_tree_{col}.png")
                critical_csv_all = os.path.join(save_dir, f"critical_paths_result_{col}.csv")
                variable_contribution_all = os.path.join(save_dir, f"variable_contribution_{col}.csv")
                if os.path.exists(tree_png_all) and os.path.exists(critical_csv_all) and os.path.exists(variable_contribution_all):
                    completed_targets.append(col)
                else:
                    incomplete_targets.append(col)

            if completed_targets and not incomplete_targets:
                # ✅ ALL DONE
                st.success("✅ 모든 Target에 대한 Causal Trace 분석이 완료되었습니다.\n\n"
                    "Feature Importance 계산 완료 이전까지 인과분석 결과부터 확인할 수 있습니다.")
            elif completed_targets and incomplete_targets:
                st.warning("⚠️ 현재 일부 변수에 대한 Causal Trace 분석이 불가합니다.\n\n"
                    f"완료: {len(completed_targets)} / 전체: {len(columns)}\n\n")
            else:
                st.info("ℹ️ 아직 Causal Trace 결과가 생성되지 않았습니다.\n\n""분석이 진행 중이거나 대기 상태일 수 있습니다.")


            if completed_targets:
                columns = completed_targets.copy()
            else:
                columns = columns
            if "target_variable" not in st.session_state:
                st.session_state.target_variable = columns[0]
            with st.sidebar.expander("🎯 인과분석 대상 변수 선택", expanded=False):
                target_variable = st.selectbox("Target Variable",columns,index=columns.index(st.session_state.target_variable)if st.session_state.target_variable in columns else 0)
                

            tree_png = os.path.join(save_dir, f"causal_tree_{target_variable}.png")
            critical_csv = os.path.join(save_dir, f"critical_paths_result_{target_variable}.csv")
            variable_contribution_csv = os.path.join(save_dir, f"variable_contribution_{target_variable}.csv")

            
            # ---------- Feature Importance Type별 결과 확인 ----------
            importance_types = ["ttest", "cohend", "shap"]
            existing_importance_types = [t for t in importance_types if os.path.exists(os.path.join(save_dir, f"feature_importance_{t}.csv"))]
            if existing_importance_types:
                if ("feature_importance_type" not in st.session_state or st.session_state.feature_importance_type not in existing_importance_types):
                    st.session_state.feature_importance_type = existing_importance_types[0]
                with st.sidebar.expander("⚙️ Feature Importance 설정", expanded=False):
                    feature_importance_type = st.selectbox("📌 Feature Importance Type",existing_importance_types,index=existing_importance_types.index(st.session_state.feature_importance_type))
                
                importance_csv = os.path.join(save_dir,f"feature_importance_{feature_importance_type}.csv")
                top_features = pd.read_csv(importance_csv)['Feature'].head(5).tolist()
                top_features_str = ", ".join(top_features) if top_features else "N/A"
                feature_type_map = {"ttest": "T-Test Analysis","cohend": "Cohen's D-Test Analysis","shap": "TimeSHAP Analysis"}
                st.table(pd.DataFrame({
                    "Select Download Types": ["Feature Importance Type","Top 5 Features"],
                    "Value": [feature_type_map.get(feature_importance_type, "Unknown"),top_features_str]}))

                col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
                with col1:
                    st.info(f"🎯 Causal Analysis Target: {target_variable}")
                with col2:
                    st.button("?", help="이상탐지에 대한 원인변수 중요도 및 특정 변수에 대한 인과분석 결과를 관찰할 수 있습니다. \n\n Sidebar 메뉴의 ⚙️ Feature Importance 설정을 통해서 상위5개의 주요변수들을 우선 선별할 수 있습니다.")
            

                col1, col2, col3 , col4 = st.columns(4)
                with col1:
                    if os.path.exists(tree_png):
                        with open(tree_png, "rb") as f:
                            st.download_button(f"📥 Causal Tree",f,os.path.basename(tree_png),"image/png",use_container_width=True)
                    else:
                        st.info("Causal Tree 결과 없음")
                with col2:
                    if os.path.exists(critical_csv):
                        with open(critical_csv, "rb") as f:
                            st.download_button("📥 Critical Paths",f,os.path.basename(critical_csv),"text/csv",use_container_width=True)
                    else:
                        st.info("Critical Paths 결과 없음")
                with col3:
                    if os.path.exists(variable_contribution_csv):
                        with open(variable_contribution_csv, "rb") as f:
                            st.download_button("📥 Variable Contributions",f,os.path.basename(variable_contribution_csv),"text/csv",use_container_width=True)
                    else:
                        st.info("Variable Contributions 결과 없음")

                with col4:
                    if os.path.exists(importance_csv):
                        with open(importance_csv, "rb") as f:
                            st.download_button(f"📥 Feature Importance",f,os.path.basename(importance_csv),"text/csv",use_container_width=True)
                    else:
                        st.info("Feature Importance 결과 없음")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if os.path.exists(tree_png):
                        with open(tree_png, "rb") as f:
                            st.download_button("📥 Causal Tree",f,os.path.basename(tree_png),"image/png",use_container_width=True)
                    else:
                        st.info("Causal Tree 결과 없음")
                with col2:
                    if os.path.exists(critical_csv):
                        with open(critical_csv, "rb") as f:
                            st.download_button("📥 Critical Paths",f,os.path.basename(critical_csv),"text/csv",use_container_width=True)
                    else:
                        st.info("Critical Paths 결과 없음")


            # ---------- 7. 실행 상태 UX ----------
            if st.session_state.causal_running:
                st.warning("⏳ 분석이 실행 중입니다. 생성된 결과부터 확인할 수 있습니다.")
            elif st.session_state.causal_run_done:
                st.success("✅ Causal Analysis 완료")


    
######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################


if page== "Trained Model Fine-Tuning":
    top_left, top_right = st.columns([4, 1])
    with top_right:
        #pdf_bytes = generate_summary_pdf(get_page_summary(3))
        summary_text = load_summary_txt(page=3, base_dir=BASE_DIR)
        pdf_bytes = generate_summary_pdf(summary_text)
        st.download_button(label="📄기능 설명",data=pdf_bytes,file_name="기능설명_Page3.pdf",mime="application/pdf",use_container_width=True)
    st.markdown(f"#### 🛠️ Trained Model Fine-Tuning")

    df_train_preview = st.session_state.get("df_train_preview")
    task_name_train = st.session_state.get("task_name_train")
    task_name_test = st.session_state.get("test_task_fault")
    feature_type = st.session_state.get("feature_type") or st.session_state.common_feature_type
    gvw = st.session_state.get("gvw") or st.session_state.common_gvw
    drive_pattern = st.session_state.get("drive_pattern")
    common_min_length = st.session_state.get("common_min_length") or st.session_state.common_min_length
    engine_threshold = st.session_state.get("engine_threshold") or st.session_state.engine_threshold
    engine_threshold_high = st.session_state.get("engine_threshold_high") or st.session_state.engine_threshold_high
    fuel_threshold = st.session_state.get("fuel_threshold")
    module = st.session_state.get("preprocessing_module")
    log_path = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_path, exist_ok=True)

    if "TB" in task_name_test or "turbo" in task_name_test:
        main_task = "Turbo_problem"
    elif "egr_sw" in task_name_test:
        main_task = "EGR_SW"
    elif "egr_hw" in task_name_test:
        main_task = "EGR_HW"
    else:
        main_task = task_name_test
    result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
    ckpt_recent = find_latest_train_checkpoint(result_path, task_name_test)
    if ckpt_recent ==None:
        st.warning("⚠️ 학습 및 추론을 수행할 준비가 완료되지 않았습니다 .\n\n"
                   "먼저 **Data Upload and Preprocess** 페이지의 작업을 완성하세요.")
        st.stop()  

    required_keys = ['test_task_fault','window_size','feature_type','gvw','drive_pattern']#['feature_type'] 
    missing = [k for k in required_keys if k not in st.session_state or st.session_state[k] in [None, ""]]
    if missing:
        st.warning("⚠️ 학습 및 추론을 수행할 준비가 완료되지 않았습니다 .\n\n"
                   "먼저 **Data Upload and Preprocess** 페이지의 작업을 완성하세요.")
        st.warning("⚠️ 기존 학습이력이 존재할 경우 📂 Train 데이터 업로드 & 📌 모델 파라미터 설정 버튼을 누르세요.")
        st.stop()  


    with st.expander("##### 📌 TranAD Fine-Tuning", expanded=True):

        if "fine_tune_save_dir" not in st.session_state:
            st.session_state.fine_tune_save_dir = None
        col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
        with col1:
            df_fine_tune_normal, fine_tune_save_dir_normal = data_upload_ui(st,"Fine_Tune","fine_tuning_normal",mode="fine_tune",task_name="Normal_Fine_Tuning",module=module)
        with col2:
            st.button("?", help="현재 엔진에 대한 추가 정상데이터에 대한 전처리를 수행합니다. \n\n전처리는 1페이지의 ⚙️공통 설정/🔧주행패턴 세부사항 설정값을 토대로 진행됩니다.")

        if df_fine_tune_normal is not None:
            st.session_state["df_fine_tune_preview"] = df_fine_tune_normal
        if fine_tune_save_dir_normal is not None:
            st.session_state["fine_tune_save_dir"] = fine_tune_save_dir_normal
        df_fine_tune_preview = st.session_state.get("df_fine_tune_preview")
        fine_tune_path_dir = st.session_state["fine_tune_save_dir"]

        if "train_fine_tuning_logs" not in st.session_state:
            st.session_state.train_fine_tuning_logs = ""
        if "train_fine_tuning_run_done" not in st.session_state:
            st.session_state.train_fine_tuning_run_done = False
        result_path = os.path.join(BASE_DIR, "result", main_task,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_(Train)")
        ckpt_load = find_latest_train_checkpoint(result_path, task_name_test)
        st.session_state["latest_train_ckpt_dir"] = ckpt_load
        log_path = os.path.join(BASE_DIR, "logs")
        log_file = os.path.join(log_path,f"{main_task}_{feature_type}_gvw_{gvw}_{drive_pattern}_{task_name_test}_finetuned.txt")

        window_size = st.session_state["window_size"]
        batch_size = st.session_state["batch_size"]
        epoch = st.session_state["epoch"]
        learning_rate = st.session_state["learning_rate"]
        try:
            train_path_dir = st.session_state["train_path_dir"]
            eval_path_dir = st.session_state["eval_path_dir"]
            test_path_dir = st.session_state["test_save_dir"]
        except:
            train_path_dir = os.path.join(BASE_DIR,"data_preprocessed", f"normal_gvw{gvw}", feature_type, drive_pattern)
            eval_path_dir = os.path.join(BASE_DIR,"data_preprocessed",f"Eval_{task_name_test}",feature_type,drive_pattern)
            test_path_dir =  os.path.join(BASE_DIR,"data_preprocessed",task_name_test,feature_type,drive_pattern)
        ckpt_dir = st.session_state["latest_train_ckpt_dir"]
        column_list_path = st.session_state.get("column_list")

        if st.button("▶️ 정상데이터 추가학습 실행"): 
            col1, col2 = st.columns([15, 1])  # 비율로 너비 조정 가능
            with col1:
                st.info('현재 Task에 대한 Train / Eval 데이터를 활용한 모델 가중치 업데이트를 진행합니다.')
            with col2:
                st.button("?", help="1페이지에서 선택한 학습모델의 가중치를 업데이트합니다. 해당 과정은 추후 불량 오탐지에 대한 발생가능성을 완화합니다.")

            cmd = (
                f"python main.py "
                f"--model TranAD "
                f"--task_name \"{task_name_test}\" "
                f"--train_dataset \"{os.path.join(fine_tune_path_dir, 'data.csv')}\" "
                f"--eval_dataset \"{os.path.join(eval_path_dir, 'data.csv')}\" "
                f"--columns \"{column_list_path}\" "
                f"--epoch {epoch} "
                f"--window_size {window_size} "
                f"--batch_size {batch_size} "
                f"--save_result_dir \"{result_path}\" "
                f"--phase train "
                f"--finetune "
                f"--model_path \"{ckpt_dir}\"")
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            with st.spinner("학습 진행 중..."):
                rc, logs = run_subprocess_and_stream(cmd, env=env)
            st.session_state.train_fine_tuning_logs = logs
            st.session_state.train_fine_tuning_run_done = (rc == 0)
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(logs)
                st.success(f"학습 로그 저장됨: {log_file}")
            except:
                st.error("로그 파일 저장 실패")
            if rc == 0:
                ckpt = find_latest_train_checkpoint(result_path, task_name_test)
                if ckpt:
                    ckpt_dir = os.path.dirname(ckpt)
                    st.session_state.latest_train_ckpt_dir = ckpt
                    st.success(f"학습 체크포인트 저장 완료: {ckpt}")
                else:
                    st.error("체크포인트를 찾을 수 없습니다. main.py 로그를 확인하세요.")
            else:
                st.error("학습 실패. 로그를 확인하세요.")

        # ---------------- Training Fine-Tuning Graph & Anomaly Score Plot 다운로드 ----------------
        if st.session_state.train_fine_tuning_run_done and st.session_state.get("latest_train_ckpt_dir"):
            st.text_area("Fine-Tuning 로그", value=st.session_state.train_fine_tuning_logs[-5000:], height=300)
            col1, col2 = st.columns(2)  
            pdf_path = os.path.join(os.path.dirname(st.session_state.latest_train_ckpt_dir),"train_loss_plot.pdf")
            with col1:
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(label="📥 Train Loss Plot (Fine-Tuned)",data=f.read(),
                            file_name="train_loss_plot.pdf",mime="application/pdf",use_container_width=True)

            html_plot_path = os.path.join(os.path.dirname(st.session_state.latest_train_ckpt_dir),"anomaly_score.html")
            with col2:
                if os.path.exists(html_plot_path):
                    with open(html_plot_path, "rb") as f:
                        st.download_button(label="📥 Train Result Plot (Fine-Tuned)",data=f.read(),
                            file_name="train_result_plot.html",mime="text/html",use_container_width=True)
                        
        try:
            data_params = {
                "Data Shape": df_train_preview.shape,
                "Train Task": task_name_train,
                "Test Task": task_name_test,
                "Feature Type": feature_type,
                "GVW": gvw,
                "Drive Pattern": drive_pattern,
                "Min Length": common_min_length,
                "Engine Threshold": engine_threshold,
                "Engine Threshold High": engine_threshold_high,
                "Fuel Threshold": fuel_threshold}
        except Exception as e:
            st.error(f"⚠ data_params 생성 중 오류: {e}")
            data_params = {}
                        
        active_list = [data_params]
        try:
            save_path = os.path.join(BASE_DIR,result_path,task_name_test, "data_preprocessing_parameters.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(active_list, f, indent=4)
            train_params = {"epoch": epoch,"window_size": window_size,"batch_size": batch_size,"learning_rate": learning_rate}
            result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
            pattern = f"train_window_{st.session_state.window_size}_*"
            result_window_dirs = [
                d for d in glob(os.path.join(result_base_dir , pattern))
                if os.path.isdir(d)]
            latest_result_dir = max(result_window_dirs, key=os.path.getctime)
            dest_path = os.path.join(latest_result_dir,"data_preprocessing_parameters.json")
            shutil.move(save_path, dest_path)
            
            save_path = os.path.join(BASE_DIR, result_path, task_name_test, "model_training_parameters.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(train_params, f, indent=4)
            result_base_dir = os.path.join(BASE_DIR,result_path,task_name_test)
            pattern = f"train_window_{st.session_state.window_size}_*"
            result_window_dirs = [
                d for d in glob(os.path.join(result_base_dir , pattern))
                if os.path.isdir(d)]
            latest_result_dir = max(result_window_dirs, key=os.path.getctime)
            dest_path = os.path.join(latest_result_dir,"model_training_parameters.json")
            shutil.move(save_path, dest_path)

        except FileNotFoundError:
            st.warning("📁 결과 디렉토리를 찾을 수 없습니다. 재학습을 실행하세요.")

        except ValueError:
            st.warning("⚠️ 학습 결과 디렉토리가 아직 생성되지 않았습니다. 재학습을 실행하세요.")

        except Exception as e:
            st.error(f"❌ 파라미터 저장 중 오류 발생: {e}")
                        
#######################################################################################################################################################


