@echo off
setlocal
cd /d "%~dp0\.."
echo [deprecated] Titanic-only prototype runner. Use scripts\agent.cmd for current agent runs.
python scripts\titanic_run_5_trials.py
endlocal
