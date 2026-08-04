@echo off
setlocal
cd /d "%~dp0.."
python -B -m research_agent.cli_app %*
endlocal
