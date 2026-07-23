@echo off
setlocal
cd /d "%~dp0\.."
echo [deprecated] Titanic-only prototype loop. Use scripts\agent.cmd or generic_workspace_auto_loop.py for current agent runs.
python scripts\titanic_auto_submit_loop.py %*
endlocal
