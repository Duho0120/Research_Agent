@echo off
setlocal
cd /d "%~dp0\.."
if "%PORT%"=="" set PORT=8080
echo Research Agent web UI
echo URL: http://127.0.0.1:%PORT%
echo.
echo Keep this window open while using the web UI.
python -B -m research_agent.web_app
endlocal
