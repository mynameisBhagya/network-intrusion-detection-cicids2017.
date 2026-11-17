@echo off
title Explainable NIDS - Streamlit Dashboard
echo =====================================================================
echo Starting Explainable NIDS Dashboard (CIC-IDS2017 Real Dataset)
echo =====================================================================

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_EXE=python
) else (
    set PY_EXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

echo Using Python: %PY_EXE%
echo Opening Streamlit Dashboard in your browser...
%PY_EXE% -m streamlit run app/dashboard.py
pause
