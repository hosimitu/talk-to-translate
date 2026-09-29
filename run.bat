@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe goto :NO_VENV

echo Starting Talk-to-Translate...
.venv\Scripts\python.exe main.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Application exited with error code: %ERRORLEVEL%
    pause
)
goto :EOF

:NO_VENV
echo [ERROR] Virtual environment (.venv) not found.
echo Please create .venv first.
pause
