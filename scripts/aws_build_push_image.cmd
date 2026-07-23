@echo off
setlocal
if "%AWS_ACCOUNT_ID%"=="" (
  echo Set AWS_ACCOUNT_ID first.
  exit /b 1
)
if "%AWS_REGION%"=="" (
  echo Set AWS_REGION first.
  exit /b 1
)
if "%IMAGE_TAG%"=="" set IMAGE_TAG=latest

set REPOSITORY=research-agent
set IMAGE_URI=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%REPOSITORY%:%IMAGE_TAG%

aws ecr describe-repositories --repository-names %REPOSITORY% --region %AWS_REGION% >nul 2>nul
if errorlevel 1 (
  aws ecr create-repository --repository-name %REPOSITORY% --region %AWS_REGION%
  if errorlevel 1 exit /b 1
)

aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com
if errorlevel 1 exit /b 1

docker build -t %REPOSITORY%:%IMAGE_TAG% .
if errorlevel 1 exit /b 1

docker tag %REPOSITORY%:%IMAGE_TAG% %IMAGE_URI%
if errorlevel 1 exit /b 1

docker push %IMAGE_URI%
if errorlevel 1 exit /b 1

echo Pushed %IMAGE_URI%
endlocal
