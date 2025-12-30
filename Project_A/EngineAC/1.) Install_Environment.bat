@echo off
REM --- Anaconda 경로 자동 감지 ---
SET "CONDA_PATH="
FOR %%D IN (
    "%USERPROFILE%\anaconda3",
    "%USERPROFILE%\Miniconda3",
    "C:\ProgramData\Anaconda3",
    "C:\ProgramData\Miniconda3"
) DO (
    IF EXIST "%%D\Scripts\conda.exe" (
        SET "CONDA_PATH=%%D"
        GOTO :FOUND_CONDA
    )
)

:FOUND_CONDA
IF "%CONDA_PATH%"=="" (
    echo ERROR: Anaconda/Miniconda not found. Please install it first.
    pause
    exit /b 1
)

SET "PATH=%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin;%PATH%"
echo Using Anaconda at: %CONDA_PATH%

REM --- 현재 배치 파일 경로 기준 ---
SET "BASE_DIR=%~dp0"

REM --- 기존 Test2 환경 삭제 (있으면) ---
call conda env remove -y -n Test2

REM --- 환경 생성 ---
call conda create -y -n Test2 python=3.7.10

REM === requirements.txt가 있는 폴더 자동 탐지 ===
SET "REQ_DIR="
FOR /D %%F IN ("%BASE_DIR%*") DO (
    IF EXIST "%%F\requirements.txt" SET "REQ_DIR=%%F"
)

IF "%REQ_DIR%"=="" (
    echo ERROR: requirements.txt not found in any subfolder of %BASE_DIR%
    pause
    exit /b 1
)

REM --- 환경 활성화 후 설치 ---
call conda run -n Test2 pip install -r "%REQ_DIR%\requirements.txt"
call conda run -n Test2 conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
call conda run -n Test2 pip install jupyter notebook

REM --- 설치 확인 ---
call conda run -n Test2 pip list
call conda run -n Test2 python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"


REM === NVIDIA GPU Driver Check ===
echo Checking NVIDIA GPU driver...
conda run -n Test2 nvidia-smi >nul 2>&1

IF ERRORLEVEL 1 (
    echo.
    echo ======================================================
    echo   [WARNING] NVIDIA GPU 드라이버를 감지할 수 없습니다.
    echo ======================================================
    echo   이 솔루션은 CUDA GPU 연산을 위해
    echo   "NVIDIA GeForce RTX 3050" 드라이버가 필요합니다.
    echo.
    echo   아래 공식 페이지에서 드라이버를 설치하세요:
    echo   https://www.nvidia.com/ko-kr/drivers/
    echo.
    echo   설치 후 PC를 재부팅한 다음, 다시 실행해주세요.
    echo ======================================================
    echo.
)


echo All done!
pause
