@echo off
title Explainable NIDS - Benchmark Experiments
echo =====================================================================
echo Running CIC-IDS2017 Benchmark Experiments (E1 - E5)
echo =====================================================================

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_EXE=python
) else (
    set PY_EXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

echo Using Python: %PY_EXE%
echo Executing E1-E5, statistical testing, model training, and figure generation...
%PY_EXE% -m src.experiments.run_all
pause
