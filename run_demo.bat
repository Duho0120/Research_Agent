@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Preparing portable demo profile...
python scripts\prepare_demo_profile.py
if errorlevel 1 (
  echo Demo profile preparation failed.
  exit /b 1
)

echo.
echo [2/2] Running one-cycle demo...
python -m research_agent.cli demo-one-cycle --competition titanic --trial trial_001 --mock-plan-file experiments/titanic/trial_001/mock_plan_response.json --mock-response-file experiments/titanic/trial_001/mock_code_response.json --run-now --show-progress
exit /b %ERRORLEVEL%
