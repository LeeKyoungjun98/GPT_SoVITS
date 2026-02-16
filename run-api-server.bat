@echo off
cd /d "%~dp0"

echo ============================================
echo   GPT-SoVITS TTS API Server
echo ============================================
echo.

if not "%~1"=="" goto :run_with_args

echo   Select mode:
echo.
echo     [1] WS only (HTTP) - default
echo     [2] WSS only (HTTPS)
echo     [3] WS + WSS both
echo.
set /p MODE_CHOICE="  Choice (1-3, default=1): "

set MODE=ws
if "%MODE_CHOICE%"=="2" set MODE=wss
if "%MODE_CHOICE%"=="3" set MODE=both

echo.
echo ============================================
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting API server (mode: %MODE%)...
echo.

python tts_api_server.py --mode %MODE%
goto :end

:run_with_args
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
python tts_api_server.py %*

:end
pause
