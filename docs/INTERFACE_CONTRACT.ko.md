# UI/CLI/메신저 공통 운영 계약

이 문서는 자율 ML 연구 에이전트를 CLI, UI, Slack/카카오톡형 메신저에서 같은 방식으로 다루기 위한 공통 계약을 정의한다.

핵심 원칙은 인터페이스가 내부 구현에 직접 의존하지 않는 것이다. CLI와 UI는 LangGraph 노드, 파일 구조, SQLite 테이블을 직접 조합하지 않고 `research_agent.interface_contract`의 operation 함수를 호출한다.

## 계층 구조

```text
Core Logic
- LangGraph node
- planner / coder / executor / recorder
- decision policy

Operation Service
- operations.py
- state_query.py
- state_db_sync.py

Interface Contract
- interface_contract.py

Interface
- CLI
- UI dashboard
- Slack/Kakao style messenger adapter
```

## 공통 응답 Envelope

모든 operation은 아래 구조를 반환한다.

```json
{
  "schema_version": "1.0",
  "ok": true,
  "action": "get_experiment",
  "status": "completed",
  "message": "Experiment status loaded.",
  "data": {},
  "next_actions": [],
  "warnings": [],
  "errors": [],
  "source": {
    "db_path": "memory/research_agent.sqlite3",
    "synced": true
  }
}
```

UI는 최소한 `ok`, `action`, `status`, `message`, `data`, `next_actions`만 보고 화면을 구성할 수 있어야 한다.

## v1 Operation

### `sync_state`

파일시스템 기반 실험 산출물을 SQLite 상태 DB에 동기화한다.

CLI 대응:

```bash
python -m research_agent.cli sync-state-db
```

### `list_experiments`

전체 실험 목록을 조회한다.

주요 사용처:

- 실험 목록 화면
- 각 실험의 상태, best trial, 현재 개선축, 다음 trial 표시

### `get_experiment`

특정 실험의 상세 상태와 trial 목록을 조회한다.

주요 사용처:

- 실험 상세 화면
- trial 목록 테이블
- 현재 개선축 상태 카드
- 다음 조치 카드

### `get_trial`

특정 trial의 상세 정보와 사용자 확인용 산출물 목록을 조회한다.

주요 사용처:

- trial 상세 화면
- 실험 계획서, 파이프라인 구조도, 코드, 결과, 판단, 로그/토큰 확인

### `preview_next_trial`

다음 trial 실행 전 현재 정책 상태를 확인한다.

주요 사용처:

- 다음 trial 미리보기 버튼
- 실행 가능 여부 표시
- 현재 개선축과 시도 횟수 표시

## v2 Operation

v2는 사람이 개입해야 하는 지점과 사용자의 자발적 인사이트를 UI/메신저에서 공통으로 다루기 위한 계약이다.

### `list_pending_requests`

사용자 응답을 기다리는 요청 목록을 조회한다.

입력:

```python
list_pending_requests(competition="titanic", status="pending")
```

반환 데이터 핵심:

```json
{
  "competition": "titanic",
  "status_filter": "pending",
  "request_count": 1,
  "requests": [
    {
      "request_id": "review_001",
      "competition": "titanic",
      "trial_id": "trial_004",
      "type": "human_review",
      "status": "pending",
      "priority": 5,
      "title": "Feature review",
      "message": "Please review whether to keep this feature-engineering axis.",
      "topic": "feature_review",
      "question": "Keep the engineered family feature?",
      "context_files": [],
      "questions": []
    }
  ]
}
```

### `get_pending_request`

특정 요청 1건을 상세 조회한다.

주요 사용처:

- UI 모달
- 메신저 답변 카드
- 사용자가 봐야 하는 파일과 질문 표시

### `respond_to_request`

사용자의 응답을 기록하고 pending request를 resolved 상태로 전환한다.

입력 예시:

```python
respond_to_request(
    "review_001",
    answers={"continue_axis": "continue"},
    free_text="같은 개선축에서 더 작은 변형을 먼저 시도해보자.",
)
```

처리 내용:

- `memory/<competition>/user_feedback.jsonl`에 응답 저장
- `experiments/<competition>/<trial>/user_review_response.md` 생성
- decision log에 사람 입력 사용 기록 추가
- SQLite `pending_actions` 상태를 `resolved`로 변경

### `submit_human_insight`

pending request가 없어도 사용자가 자발적으로 제공한 연구 인사이트를 저장한다.

입력 예시:

```python
submit_human_insight(
    "titanic",
    "trial_004",
    insight="모델 변경보다 결측치 처리 기준을 먼저 고정하는 것이 좋아 보인다.",
)
```

주요 사용처:

- UI의 “의견 추가” 버튼
- 메신저에서 사용자가 먼저 보낸 의견
- 다음 trial planning의 참고 memory

## Pending Request Schema

```json
{
  "request_id": "review_20260720_001",
  "competition": "titanic",
  "trial_id": "trial_004",
  "type": "human_review",
  "status": "pending",
  "priority": 5,
  "title": "사용자 판단 필요",
  "message": "현재 개선축을 계속 시도할지 확인이 필요합니다.",
  "topic": "axis_review",
  "question": "같은 개선축을 계속 시도할까요?",
  "context_files": [
    {
      "label": "파이프라인 구조도",
      "path": "runs/titanic/trial_004/02_pipeline_structure.ko.md"
    }
  ],
  "questions": [
    {
      "id": "continue_axis",
      "label": "같은 개선축을 계속 시도할까요?",
      "answer_type": "choice",
      "choices": [
        {"value": "continue", "label": "계속 시도"},
        {"value": "switch_axis", "label": "축 전환"},
        {"value": "pause", "label": "일시중단"}
      ],
      "required": true
    }
  ]
}
```

## UI 화면 매핑

```text
전체 실험 목록
  -> list_experiments

실험 상세 및 회차 목록
  -> get_experiment

회차 상세
  -> get_trial

다음 회차 미리보기
  -> preview_next_trial

대기 중인 사용자 요청 목록
  -> list_pending_requests

사용자 요청 상세
  -> get_pending_request

사용자 요청 응답
  -> respond_to_request

사용자 인사이트 직접 추가
  -> submit_human_insight

상태 새로고침
  -> sync_state + list/get 재조회
```

## 이후 후보

- `pause_experiment`
- `resume_experiment`
- `run_next_trial` envelope 통합
- 제출 승인 게이트 전용 operation
- 메신저 알림 전송 adapter
