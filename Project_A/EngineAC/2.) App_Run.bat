@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM === 배치 파일 위치 기준 ===
SET "BASE_DIR=%~dp0"

REM === Streamlit.py가 있는 폴더 자동 탐색 ===
SET "STREAMLIT_DIR="
FOR /D %%F IN ("%BASE_DIR%*") DO (
    IF EXIST "%%F\Streamlit.py" SET "STREAMLIT_DIR=%%F"
)

IF "%STREAMLIT_DIR%"=="" (
    echo ERROR: Streamlit.py not found in any subfolder of %BASE_DIR%
    pause
    exit /b 1
)

REM === Anaconda/Miniconda 경로 자동 탐색 ===
SET "CONDA_PATH="
FOR %%D IN (
    %USERPROFILE%\anaconda3
    %USERPROFILE%\Miniconda3
    C:\ProgramData\Anaconda3
    C:\ProgramData\Miniconda3
) DO (
    IF EXIST "%%D\Scripts\activate.bat" (
        SET CONDA_PATH=%%D
        GOTO :FOUND_CONDA
    )
)

:FOUND_CONDA
IF "%CONDA_PATH%"=="" (
    echo ERROR: Could not find Anaconda or Miniconda installation.
    pause
    exit /b 1
)

echo Using Anaconda at: %CONDA_PATH%

REM === Base 환경 activate (중요!) ===
CALL "%CONDA_PATH%\Scripts\activate.bat"

REM === Test2 환경 활성화 ===
CALL conda activate Test2

REM === Streamlit.py folder로 이동 ===
CD /D "%STREAMLIT_DIR%"

REM === Streamlit App 실행 ===
start "" streamlit run "Streamlit.py" --server.port 8501 --server.headless true

REM === 현재 PC 로컬 IP 자동 탐지 ===
FOR /F "tokens=2 delims=:" %%A IN ('ipconfig ^| findstr /R "IPv4"') DO (
    SET IP=%%A
    SET IP=!IP: =!
    GOTO :break
)
:break

REM === 브라우저 자동 실행 ===
start "" http://!IP!:8501
