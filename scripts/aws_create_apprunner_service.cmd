@echo off
setlocal
if "%AWS_REGION%"=="" set AWS_REGION=ap-northeast-1
set ROLE_NAME=research-agent-apprunner-ecr-role
set SERVICE_FILE=deploy\aws\apprunner-service.json

aws iam create-role --role-name %ROLE_NAME% --assume-role-policy-document file://deploy/aws/apprunner-ecr-trust-policy.json >nul 2>nul
aws iam attach-role-policy --role-name %ROLE_NAME% --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
if errorlevel 1 exit /b 1

aws apprunner create-service --cli-input-json file://%SERVICE_FILE% --region %AWS_REGION%
if errorlevel 1 exit /b 1

echo App Runner service creation started.
echo Check status with:
echo aws apprunner list-services --region %AWS_REGION%
endlocal
