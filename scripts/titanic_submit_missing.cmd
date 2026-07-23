@echo off
setlocal
cd /d "%~dp0\.."
echo [deprecated] Titanic-only prototype repair helper. Current agent runs submit through generic_workspace_auto_loop.py.
python scripts\titanic_submit_missing.py %*
endlocal
