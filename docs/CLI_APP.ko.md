# Research Agent 사용자용 CLI

## 일반 사용자는 이것만 실행하세요

Windows CMD:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
scripts\agent.cmd
```

이 명령 하나로 실험 선택, 새 실험 등록, 자동 실험 시작/중단 요청, 상태 확인,
에이전트 질문, 인사이트 기록, 피드백 요청 확인을 처리합니다.

빠른 상태 확인만 하고 싶을 때:

```bat
scripts\agent.cmd --no-sync --status
```

## 메뉴

대화형 CLI는 매 선택 후 화면을 지우고 현재 상태와 메뉴를 다시 그립니다.
이전 출력이 계속 쌓이지 않도록 하고, 직전 작업 결과는 `선택 >` 바로 위의 `최근 메시지` 영역에 남깁니다.
실험 실행 상태와 `auto_loop.log`의 마지막 기록은 `피드백 요청` 아래의 `진행 상태`와
`최근 로그` 영역에 표시됩니다. 5번 `상태 다시 보기`를 선택하면 이 영역도 함께 갱신됩니다.

1. `실험 바꾸기`: 이후 모든 메뉴의 기준 실험을 바꿉니다.
2. `새 실험 등록`: Kaggle slug/url 또는 로컬 실험 ID를 등록하고 워크스페이스를 만듭니다.
3. `자동 실험 시작`: 선택 실험을 백그라운드 자동 루프로 시작합니다. 시작 전 진행 횟수를 묻습니다. 숫자를 입력하면 해당 횟수만큼 진행하고, `c`를 입력하면 중단 요청 전까지 계속 진행하며, `q`를 입력하면 시작하지 않고 메뉴로 돌아갑니다.
4. `현재 실험 중단 요청`: 현재 trial의 로컬 실행, 제출, 점수 기록이 끝난 뒤 멈추도록 요청합니다.
5. `상태 다시 보기`: 상태 DB와 산출물 기록을 다시 읽습니다.
6. `에이전트에게 질문`: 선택 실험의 문서와 기록에 근거해 답변합니다. 답변은 질문 입력 바로 아래에 표시되고, `q`를 입력할 때까지 질문을 이어갈 수 있습니다.
7. `다음 실험에 반영할 인사이트 남기기`: 현재 실행은 건드리지 않고 다음 trial 계획에 전달합니다. 이미 같은 trial 기준 인사이트가 있으면 기존 내용을 먼저 보여주며, 새 내용을 입력하면 기존 인사이트를 덮어씁니다. `q`를 입력하면 기존 내용을 유지하고 메뉴로 돌아갑니다. 저장 후 제공한 인사이트 원문, 반영 예정 trial, 적용 개선안 요약을 최근 메시지에 표시합니다.
8. `피드백 요청 확인 (개수)`: 에이전트가 만든 대기 요청을 확인하고 답변합니다. 개수가 0이면 현재 답변할 요청이 없다는 뜻입니다.
9. `Trial 비교표 보기`: trial별 base, local/submit score, delta, 개선축, decision, best 여부를 비교합니다.
10. `폴더/DB 위치 열기`: 사용자용 산출물, 실행 워크스페이스, 실험 기록, 제출 파일, SQLite DB를 확인합니다.
11. `종료`: CLI만 종료합니다. 백그라운드 루프는 별도로 계속됩니다.

## 자동 실험 시작 시 안내되는 내용

`3. 자동 실험 시작`을 선택하면 다음 정보를 보여줍니다.

```text
실험을 시작하겠습니다. 몇 회 진행할까요?

- 숫자 입력: 입력한 횟수만큼 진행
- c 입력: 중단 요청 전까지 계속 진행
- q 입력: 시작하지 않고 메뉴로 돌아가기

진행 횟수/c/q> 2

자동 실험을 시작했습니다.

- 선택된 실험: titanic
- 시작 trial: trial_004
- 실행 범위: trial_004 -> trial_005 (2회)
- 진행 방식: 로컬 실행 -> Kaggle 제출 -> 제출 점수 기록 -> 다음 trial 계획
- 중단 요청 시: 현재 trial 제출/기록까지 마친 뒤 멈춤
- PID: <process id>
- 로그: demo_workspaces/_runtime/auto_loop.log
```

이미 실행 중이면 새 루프를 중복으로 띄우지 않고 현재 trial과 중단 방법을 안내합니다.

## 새 실험 등록

`2. 새 실험 등록`에서 예를 들어 다음처럼 입력할 수 있습니다.

```text
실험 ID 또는 Kaggle slug/url> https://www.kaggle.com/competitions/playground-series-s4e1
```

그러면 아래 경로가 만들어집니다.

```text
competitions/<experiment-id>/
demo_workspaces/<experiment-id>/
```

새 워크스페이스를 생성하면 기본 `test_step.py`, `train_step.py`, `predict_step.py`,
`workspace_config.json`, `execution_profile.yaml`이 준비됩니다. 데이터 파일은 사용자가
`demo_workspaces/<experiment-id>/data/`에 직접 넣어야 합니다.

## 실행 범위

Titanic은 기존 전용 루프를 사용합니다.

```text
scripts/titanic_auto_submit_loop.py
```

Titanic 자동 루프는 제출 제한이 없다는 전제에서 각 trial을 로컬 실행한 뒤 로컬 점수가
이전 trial보다 낮아도 Kaggle에 제출합니다. 다음 trial 판단은 로컬 점수만이 아니라 제출
점수까지 기록한 뒤 진행합니다.

새로 등록한 일반 실험은 실행 프로필 기반 공통 루프를 사용합니다.

```text
scripts/generic_workspace_auto_loop.py
```

공통 루프는 현재 다음 순서까지 지원합니다.

1. `execution_profile.yaml` 검증
2. workspace `test/train/predict` 실행
3. metrics 수집
4. 사용자용 산출물 정리
5. Kaggle 실험이면 `submission.csv` 제출
6. 제출 결과와 점수 기록
7. 다음 trial 계획 문맥 생성
8. 다음 trial부터 code writer가 개선안을 코드에 반영
9. 수정된 코드로 다시 로컬 실행/제출/기록

일반 실험은 `trial_001`에서 등록된 기본 baseline을 먼저 실행합니다. `trial_002`부터는 직전
trial의 결과, 제출 점수, 사용자 인사이트, continuation context를 바탕으로 code writer가
허용된 write scope 안에서 코드를 수정한 뒤 실행합니다.

OpenAI API quota 또는 네트워크가 막히면 해당 trial은 code writer 단계에서 blocked/failed로
남고, `auto_loop.log`와 해당 trial 산출물에 이유가 기록됩니다.

## 상태 파일

기본 상태 경로:

```text
demo_workspaces/_runtime/
```

주요 파일:

```text
cli_state.json
auto_loop_state.json
auto_loop.lock
pause.request
auto_loop.log
```

`RESEARCH_AGENT_RUNTIME_DIR`를 설정하면 상태와 잠금 파일을 다른 경로에 저장할 수 있습니다.
AWS에서는 이 경로를 EFS 같은 지속 공유 볼륨으로 지정하는 것을 권장합니다.

## 고급/개발자용 명령

일반 사용자는 보통 아래 명령을 직접 입력하지 않습니다. 자동 루프나 UI가 내부적으로
사용하는 세부 명령입니다.

```bat
python -B -m research_agent.cli status --competition titanic --sync
python -B -m research_agent.cli run-next-trial --competition titanic --trial trial_004 --run-now
python -B -m research_agent.cli record-submission --competition titanic --trial trial_004 ...
python -B -m research_agent.cli prepare-workspace --competition <id> ...
```

문제가 생겼을 때 원인을 좁히거나, 자동 루프 없이 특정 단계만 검증할 때 사용합니다.

## 후속 개발 예정

현재 CLI는 Windows CMD에서도 안정적으로 동작하도록 표준 입력/출력 기반 메뉴를 사용합니다.
후속 버전에서는 `rich` 또는 `prompt_toolkit` 기반의 Claude CLI 스타일 화면을 검토합니다.

예상 개선 방향:

- 선택 항목 하이라이트
- 상태 박스/패널 표시
- 로그 영역과 질문 입력 영역 분리
- 긴 답변 스크롤 또는 접기
- 웹 UI와 동일한 상태/액션 용어 유지
