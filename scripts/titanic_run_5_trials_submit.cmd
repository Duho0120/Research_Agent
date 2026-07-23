@echo off
setlocal
cd /d "%~dp0\.."
echo [deprecated] Titanic-only prototype runner. Use scripts\agent.cmd for current agent runs.
python scripts\titanic_run_5_trials.py --submit --wait-for-lb --poll-seconds 10
endlocal
