@echo off
rem Double-click to start ClearMark (or drag a video onto this file).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up a private Python environment. This takes a minute...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python 3.10 or newer is needed. Get it from https://www.python.org/downloads/
        echo and tick "Add python.exe to PATH" while installing.
        pause
        exit /b 1
    )
)

rem (Re)install packages whenever requirements.txt changes.
fc /b requirements.txt ".venv\requirements.txt" >nul 2>&1
if errorlevel 1 (
    echo Installing packages...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installing packages failed - check your internet connection and run this again.
        pause
        exit /b 1
    )
    copy /y requirements.txt ".venv\requirements.txt" >nul
)

start "" ".venv\Scripts\pythonw.exe" app.py %*
