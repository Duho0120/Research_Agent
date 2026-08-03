# 로컬 데스크톱 앱 실행 가이드

이 앱은 웹 서버를 띄우지 않고, 현재 PC에서 바로 실행되는 Tkinter 기반 로컬 데스크톱 앱이다.

현재 UI는 다크 테마와 카드형 패널을 사용한다. 실험 목록, 사용자 요청, trial 목록, 산출물 미리보기를 한 창에서 확인하는 관찰용 앱 v1이다.

## 실행

프로젝트 폴더에서 실행한다.

```cmd
cd /d C:\Users\ASUS\Desktop\Research_Agent
python -m research_agent.app
```

파일시스템과 SQLite 동기화를 생략하고 빠르게 열고 싶다면:

```cmd
python -m research_agent.app --no-sync
```

## 현재 제공 기능

- 전체 실험 목록 확인
- 실험별 trial 목록 확인
- trial별 사용자용 산출물 목록 확인
- trial 핵심 요약을 먼저 확인
- 실험 계획서, 파이프라인 구조도, 결과 파일을 앱에서 읽기 좋은 형태로 미리보기
- 다음 trial 실행 가능 상태 미리보기
- 대기 중인 사용자 요청 확인
- 사용자 요청에 답변 기록
- 특정 trial에 사용자 인사이트 직접 추가

## 산출물 읽기 방식

Trial을 선택하면 오른쪽 화면에 먼저 `0. 핵심 요약`이 표시된다.

추천 확인 순서:

1. 핵심 요약
2. 실험 계획서
3. 파이프라인 구조도
4. 실행 결과
5. 판단 카드

Markdown 원문은 앱 안에서 제목, 표, 불릿을 단순한 읽기 화면으로 변환해서 보여준다. 원문 파일을 그대로 확인하고 싶으면 `파일 열기` 버튼을 사용한다.

## 설계 의도

로컬 앱은 CLI 출력 문자열을 파싱하지 않는다. `kaggle_research_agent.interface_contract`의 공통 operation을 호출해서 상태를 가져온다.

따라서 나중에 웹 UI나 Slack/카카오톡형 메신저를 추가하더라도 동일한 operation을 재사용할 수 있다.

## 아직 포함하지 않은 것

- 앱 내부에서 trial을 직접 실행하는 버튼
- 실행 중인 trial의 실시간 로그 스트리밍
- 제출 승인 게이트 화면
- Slack/카카오톡 연동

이 기능들은 다음 단계에서 `run_next_trial`, approval gate, message adapter operation을 추가한 뒤 연결하는 것이 안정적이다.

## API 없는 발표용 채팅

OpenAI API 키 없이도 SQLite와 실험 문서 검색 결과로 채팅을 시연할 수 있습니다.

```bat
scripts\web_demo.cmd
```

이 모드에서는 `RESEARCH_AGENT_CHAT_DEMO_MODE=1`이 설정되며 외부 LLM API를 호출하지 않습니다.
채팅창에는 `DEMO · API 없이 로컬 근거로 답변`이라고 표시됩니다. 점수·베스트 질문은
SQLite에서 직접 답하고, 계획·파이프라인 질문은 로컬 문서 검색 근거를 보여줍니다.
일반 채팅과 마찬가지로 실험 계획이나 코드를 변경하지 않는 읽기 전용 기능입니다.

## 대화 기록

웹과 CLI의 에이전트 질문은 `memory/research_agent.sqlite3`의 `chat_sessions`,
`chat_messages` 테이블에 실험별로 저장됩니다.

- 실험을 바꾸면 해당 실험의 대화 기록으로 전환됩니다.
- 웹 채팅의 대화 선택 목록에서 이전 대화를 다시 열 수 있습니다.
- `+` 버튼으로 새 대화를 시작해도 기존 대화는 삭제되지 않습니다.
- 각 메시지는 질문 당시 참조한 trial과 답변 모드, 근거 경로를 함께 기록합니다.
- LLM 입력에는 현재 대화의 최근 메시지 6개만 포함되어 대화 누적에 따른 토큰 증가를 제한합니다.
- 채팅은 읽기 전용이며 실험 계획·코드·점수·연구 판단을 변경하지 않습니다.

로컬 SQLite 파일은 앱 재시작 후에도 유지됩니다. 컨테이너를 재배포한 뒤에도 기록을
유지하려면 외부 DB 또는 영구 볼륨에 `memory` 경로를 연결해야 합니다.

## 저비용 모델 설정

저비용 모델의 기본값은 `configs/policies/model_policy.yaml`에서 관리합니다. 배포 환경에서는
코드를 수정하지 않고 다음 환경변수로 재정의할 수 있습니다.

- `RESEARCH_AGENT_LOW_COST_MODEL`: 모든 저비용 호출
- `RESEARCH_AGENT_CHAT_MODEL`: 채팅만 재정의
- `RESEARCH_AGENT_SUMMARY_MODEL`: 사용자 요약 카드만 재정의
- `RESEARCH_AGENT_INSIGHT_MODEL`: 인사이트 해석만 재정의

기능별 환경변수가 공통 저비용 모델 환경변수보다 우선합니다.

## LangGraph 실험 런타임

CLI와 웹에서 시작하는 자동 실험은 기본적으로 LangGraph 운영 그래프를 사용합니다.
기존 계획, 코드 작성, 로컬 실행, 제출, 산출물 함수는 그대로 사용하고 실행 순서와
중단·재개 상태만 LangGraph가 관리합니다.

기본 trial 흐름:

```text
현재 trial 계획 확인·인사이트 반영
→ 코드 작성 및 기존 재시도 정책
→ 로컬 실행·점수 수집
→ Kaggle 제출 및 제출 점수 기록
→ 산출물·SQLite 동기화
→ 다음 trial 계획 생성
→ 중단 요청 확인
```

체크포인트는 기본적으로 아래 별도 SQLite 파일에 저장됩니다.

```text
demo_workspaces/_runtime/langgraph_checkpoints.sqlite3
```

`RESEARCH_AGENT_GRAPH_CHECKPOINT_DB`로 위치를 변경할 수 있습니다. 프로세스가 강제로
종료되어 실행 상태가 `running`으로 남은 경우에는 같은 graph thread의 미완료 노드부터
재개합니다. 정상 실패나 정책 차단은 기존과 동일하게 해당 trial부터 다시 시작합니다.

비상 호환 확인에만 기존 Python 루프를 사용할 수 있습니다.

```bat
python scripts\generic_workspace_auto_loop.py ... --legacy-runtime
```
