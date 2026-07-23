@echo off
setlocal
cd /d "%~dp0.."
python -B -m kaggle_research_agent.cli_app %*
endlocal
