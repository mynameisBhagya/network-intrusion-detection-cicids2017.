@echo off
title Explainable NIDS - Unit Tests
echo =====================================================================
echo Running Pytest Unit Verification Tests
echo =====================================================================

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_EXE=python
) else (
    set PY_EXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

echo Using Python: %PY_EXE%
%PY_EXE% -m pytest -v
pause
