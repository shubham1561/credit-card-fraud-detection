@echo off
rem Double-click to start the fraud-detection demo and open it in your browser.
rem First run sets everything up (a few minutes); later runs start in about 15 seconds.
title Card Fraud Detector
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Creating the Python environment...
    python -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
)

rem Install packages if any are missing (also repairs a half-installed .venv)
"%PY%" -c "import uvicorn, fastapi, httpx, sklearn, xgboost, scipy, yaml, joblib, pandas, matplotlib" 2>nul
if errorlevel 1 (
    echo Installing packages, this takes a few minutes the first time...
    "%PY%" -m pip install --upgrade pip -q
    "%PY%" -m pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
)

if not exist "data\raw\creditcard.csv" (
    echo Downloading the dataset...
    "%PY%" -m src.data --download || (pause & exit /b 1)
)

rem Train the model if it is missing or was saved by an incompatible library version
"%PY%" -c "import joblib; joblib.load('models/model_bundle.joblib')" 2>nul
if errorlevel 1 (
    echo Training the model, about 8 minutes, only needed once...
    "%PY%" -m src.train --protocol time || (pause & exit /b 1)
)

echo.
echo Starting the demo. Your browser will open at http://127.0.0.1:8000
echo Close this window to stop the server.
echo.
start "" /b "%PY%" scripts\open_when_ready.py http://127.0.0.1:8000
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
