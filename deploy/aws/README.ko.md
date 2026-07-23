# AWS 배포 가이드

이 프로젝트는 AWS에서 **작업 실행형 에이전트**로 배포하는 것을 우선합니다. 웹 서버처럼
항상 떠 있는 앱보다, 사용자별 실험 실행 요청이 들어오면 ECS/Fargate 또는 AWS Batch task를
하나 띄우는 구조가 더 안전합니다.

## 권장 구조

```text
사용자
→ API/운영 서버
→ 사용자별 Secret 확인
→ ECS/Fargate 또는 Batch task 실행
→ EFS에 runtime/state/artifact 저장
→ CloudWatch Logs로 진행 로그 확인
```

핵심 원칙:

- OpenAI/Kaggle 키는 서비스 소유자 키를 공유하지 않습니다.
- 사용자별 키를 Secrets Manager에 저장하고, 해당 사용자 task에만 주입합니다.
- `RESEARCH_AGENT_RUNTIME_DIR`는 EFS 경로로 지정합니다.
- `RESEARCH_AGENT_STORAGE_DIR`는 사용자/competition별 EFS 루트로 지정합니다.
- 같은 사용자/같은 competition은 같은 runtime 경로를 써야 lock이 작동합니다.
- ECS/Batch에서는 대화형 CLI가 아니라 루프 스크립트를 컨테이너의 main command로 실행합니다.

## 이미지 빌드/푸시

CMD에서:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
set AWS_ACCOUNT_ID=<account-id>
set AWS_REGION=ap-northeast-2
set IMAGE_TAG=latest
scripts\aws_build_push_image.cmd
```

## 제출용 URL 만들기: App Runner

과제 제출용 URL이 필요하면 App Runner가 가장 빠릅니다. 단, App Runner는 서울
`ap-northeast-2` 리전을 지원하지 않으므로 Tokyo `ap-northeast-1`을 사용합니다.
ECR의 `research-agent:latest`
이미지를 웹앱으로 실행하고, `https://...awsapprunner.com` 주소를 생성합니다.

먼저 이미지를 최신 코드로 다시 push합니다.

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
set AWS_ACCOUNT_ID=961341521085
set AWS_REGION=ap-northeast-1
set IMAGE_TAG=latest
scripts\aws_build_push_image.cmd
```

그 다음 App Runner 서비스를 생성합니다.

```bat
scripts\aws_create_apprunner_service.cmd
```

상태 확인:

```bat
aws apprunner list-services --region ap-northeast-1
```

`Status`가 `RUNNING`이 되면 `ServiceUrl` 값이 과제 제출용 URL입니다.

비용을 멈추려면 서비스 ARN을 확인한 뒤 삭제합니다.

```bat
aws apprunner delete-service --service-arn <SERVICE_ARN> --region ap-northeast-1
```

## Task definition

예시 파일:

```text
deploy/aws/task-definition.research-agent.example.json
```

바꿔야 하는 값:

```text
<account-id>
<region>
<user-id>
<competition>
fs-xxxxxxxxxxxxxxxxx
fsap-xxxxxxxxxxxxxxxxx
role 이름
```

## 일반 실험 실행 command

컨테이너 override 예시:

```text
deploy/aws/container-overrides.generic.example.json
```

실행 command:

```bash
python -B scripts/generic_workspace_auto_loop.py \
  --competition <competition-slug> \
  --start-trial trial_001 \
  --max-trials 5 \
  --submit \
  --kaggle-slug <competition-slug> \
  --code-writer \
  --allow-api
```

## Titanic 실행 command

컨테이너 override 예시:

```text
deploy/aws/container-overrides.titanic.example.json
```

```bash
python -B scripts/titanic_auto_submit_loop.py --start trial_004 --end trial_005
```

## 사용자별 Secret

최소 secret:

```text
research-agent/<user-id>/openai
research-agent/<user-id>/kaggle-username
research-agent/<user-id>/kaggle-key
```

운영 서버는 task 실행 시 현재 사용자에게 해당하는 secret ARN만 container secret으로 넘깁니다.
task role도 가능한 한 해당 사용자 secret prefix만 읽도록 제한하는 것이 좋습니다.

## EFS 디렉터리 예시

```text
/mnt/research-agent/users/<user-id>/<competition>/demo_workspaces
/mnt/research-agent/users/<user-id>/<competition>/experiments
/mnt/research-agent/users/<user-id>/<competition>/memory
/mnt/research-agent/users/<user-id>/<competition>/submissions
/mnt/research-agent/users/<user-id>/<competition>/runtime
```

컨테이너 시작 시 `scripts/docker_entrypoint.sh`가 `/app/demo_workspaces`, `/app/experiments`,
`/app/memory`, `/app/submissions` 등을 `RESEARCH_AGENT_STORAGE_DIR` 아래로 연결합니다. 그래서
컨테이너가 교체되어도 실험 상태와 산출물이 EFS에 남습니다.

`RESEARCH_AGENT_RUNTIME_DIR`는 lock/state/log 위치로 사용합니다. 같은 사용자/competition의
동시 실행을 막으려면 이 값이 항상 같은 경로여야 합니다.

## 운영 체크리스트

1. ECR 이미지 push 성공
2. ECS task가 CloudWatch Logs에 Python 시작 로그 출력
3. task role이 사용자 secret만 읽을 수 있음
4. EFS access point에 쓰기 가능
5. `RESEARCH_AGENT_RUNTIME_DIR`에 `auto_loop_state.json` 생성
6. 같은 runtime으로 task 2개 실행 시 lock 충돌로 하나가 거절됨
7. Kaggle 제출이 사용자 계정으로 기록됨
8. OpenAI quota 부족 시 trial이 blocked/failed로 기록되고 secret 값은 로그에 노출되지 않음
