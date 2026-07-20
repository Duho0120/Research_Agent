# DB / UI / Message Integration Architecture

## 목적

이 문서는 LangGraph/RAG 기반 자율 ML 연구 루프 위에 DB, UI, 메시지 프로그램 연동을 추가하기 위한 설계 기준을 정리합니다.

핵심 목표는 기존 파일 기반 연구 산출물을 유지하면서, 에이전트와 향후 UI/메신저가 안정적으로 상태를 조회할 수 있는 내부 저장 계층을 추가하는 것입니다.

## 설계 원칙

- 사용자가 보는 연구 산출물은 계속 파일로 남긴다.
- DB는 원본 산출물을 대체하지 않고, 조회와 상태 관리를 위한 색인 계층으로 둔다.
- SQLite는 정형 상태와 메타데이터를 관리한다.
- Chroma는 RAG 검색 고도화를 위한 후속 확장으로 보류한다.
- UI/메신저는 파일을 직접 훑지 않고 SQLite를 먼저 조회한다.
- 상세 문서와 코드는 SQLite에 전문을 저장하지 않고 파일 경로로 연결한다.
- 외부 API, 제출, 고비용 학습, 사용자 피드백 요청은 명시적인 상태와 pending action으로 남긴다.

## 현재 구조와 유지할 것

현재 프로젝트는 다음 파일 기반 구조를 중심으로 동작합니다.

```text
competitions/<competition>/
  execution_profile.yaml
  competition_data_card.json

demo_workspaces/<competition>/
  data/
  src/
  outputs/
  train_step.py
  predict_step.py
  test_step.py

experiments/<competition>/<trial>/
  metrics.json
  graph_state.json
  node_events.jsonl
  internal/
    code_snapshot/
    pipeline_structure.json
    decision_card.json
    trial_memory_card.json
  user_view/

runs/<competition>/<trial>/
  README.ko.md
  01_plan.ko.md
  02_pipeline_structure.ko.md
  03_code_pipeline.ko.md
  04_result.ko.md
  code/

memory/<competition>/
  decision_cards.jsonl
  token_usage.jsonl
  document_index.jsonl
  demo_trial_index.jsonl
```

이 구조는 그대로 유지합니다. 특히 `runs/`와 `experiments/<competition>/<trial>/internal/code_snapshot/`은 재현성과 사람이 직접 확인하는 사용성을 위해 계속 필요합니다.

## 저장 계층 역할 분리

```text
파일 시스템
  원본 연구 산출물, 코드 스냅샷, 실행 로그, 사용자용 Markdown, metrics.json

SQLite
  실험/Trial 상태, 점수, 개선축, best trial, 제출 이력, pending action, 파일 경로 색인

Chroma
  후속 확장: 계획서, 파이프라인 구조도, decision card, trial memory card, research notes의 의미 검색
```

### 파일 시스템

파일 시스템은 연구의 원본 기록입니다.

- 사용자가 직접 읽을 수 있어야 합니다.
- Git/SFTP/압축 공유에 유리해야 합니다.
- 특정 trial을 재현할 수 있는 코드와 문서가 남아야 합니다.
- DB가 깨져도 파일만으로 최소 복구가 가능해야 합니다.

### SQLite

SQLite는 에이전트와 UI가 빠르게 조회하는 정형 상태 저장소입니다.

주요 용도:

- 등록된 실험 목록 조회
- 현재 진행 중인 trial 상태 조회
- best trial 조회
- trial별 점수와 개선축 비교
- 제출 상태와 LB 점수 기록
- 사용자 승인/피드백 요청 상태 관리
- 산출물 파일 경로 조회
- 토큰 사용량 요약

SQLite에는 큰 로그 전문이나 코드 전문을 넣지 않습니다. 대신 파일 경로, 요약, 상태, 점수, 축 정보만 저장합니다.

### Chroma

Chroma는 RAG 검색을 위한 의미 검색 DB입니다. 다만 현재 에이전트는 파일 기반 RAG와 context pack을 이미 사용하고 있으므로, 초기 DB 구현 범위에서는 Chroma를 도입하지 않습니다.

주요 용도:

- 이전 실험 계획서 검색
- 파이프라인 구조도 검색
- decision card 검색
- trial memory card 검색
- research notes/rules 검색
- "이전에 비슷한 개선을 했는가?" 질문에 대한 근거 검색

Chroma는 정형 상태 판단의 source of truth가 아닙니다. 정형 판단은 SQLite와 파일 JSON을 기준으로 하고, Chroma는 나중에 파일 기반 RAG만으로 부족할 때 LLM에게 필요한 근거 문서를 더 잘 찾는 용도로 추가합니다.

## 권장 데이터 흐름

```text
LangGraph node 실행
  -> 파일 산출물 생성
  -> SQLite 동기화
  -> 파일 기반 RAG 색인 갱신
  -> UI/메신저 상태 조회 가능
```

Trial 완료 시점의 흐름:

```text
collect_metrics
  -> metrics.json 저장
  -> record_result/process_workspace_result
  -> decision_card.json, trial_memory_card.json 저장
  -> organize_trial_artifacts
  -> runs/<competition>/<trial>/ 사용자 산출물 생성
  -> SQLite upsert
  -> 파일 기반 RAG index update
```

## SQLite 최소 스키마 후보

1차 구현에서는 아래 테이블만 도입합니다.

```text
competitions
trials
trial_scores
trial_artifacts
trial_decisions
token_usage
submissions
pending_actions
```

### competitions

실험/대회 단위 상태를 저장합니다.

필드 후보:

- `competition_id`
- `platform`
- `topic`
- `metric`
- `objective`
- `status`
- `workspace_path`
- `created_at`
- `updated_at`

### trials

trial 단위 상태를 저장합니다.

필드 후보:

- `competition_id`
- `trial_id`
- `status`
- `source_trial_id`
- `recommended_base_trial`
- `plan_type`
- `primary_change_axis`
- `created_at`
- `updated_at`

### trial_scores

점수와 비교 정보를 저장합니다.

필드 후보:

- `competition_id`
- `trial_id`
- `metric`
- `objective`
- `local_score`
- `lb_score`
- `local_status`
- `lb_status`
- `is_best_local`
- `is_best_lb`

### trial_artifacts

사용자와 UI가 열 파일 경로를 저장합니다.

필드 후보:

- `competition_id`
- `trial_id`
- `artifact_type`
- `path`
- `is_user_facing`
- `created_at`

예시 `artifact_type`:

- `plan_ko`
- `pipeline_structure_ko`
- `code_pipeline_ko`
- `result_ko`
- `code_snapshot`
- `metrics_json`
- `graph_state`
- `node_events`

### trial_decisions

축 유지/전환과 다음 계획 제약을 저장합니다.

필드 후보:

- `competition_id`
- `trial_id`
- `decision`
- `change_axis`
- `active_axis`
- `axis_attempt_count`
- `axis_attempt_limit`
- `recommended_base_trial`
- `rejected_axes_json`
- `rejected_candidates_json`
- `planner_constraints_json`

### token_usage

LLM 호출 비용 요약을 저장합니다.

필드 후보:

- `competition_id`
- `trial_id`
- `provider`
- `model`
- `call_type`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `created_at`

### submissions

대회 제출과 LB 피드백을 저장합니다.

필드 후보:

- `competition_id`
- `trial_id`
- `platform`
- `submission_file`
- `status`
- `lb_score`
- `rank`
- `submitted_at`
- `requires_user_approval`

### pending_actions

UI/메신저가 사용자에게 보여줄 대기 작업을 저장합니다.

필드 후보:

- `action_id`
- `competition_id`
- `trial_id`
- `action_type`
- `status`
- `priority`
- `message`
- `payload_json`
- `created_at`
- `resolved_at`

예시 `action_type`:

- `api_key_required`
- `data_required`
- `approval_required`
- `human_review_required`
- `submission_ready`
- `run_blocked`
- `cost_approval_required`

## Chroma 색인 대상 - 후속 확장

Chroma는 현재 구현 범위에서 제외합니다. 나중에 의미 검색 품질이 필요해지면 아래 문서를 제한적으로 색인합니다.

```text
runs/<competition>/<trial>/01_plan.ko.md
runs/<competition>/<trial>/02_pipeline_structure.ko.md
runs/<competition>/<trial>/03_code_pipeline.ko.md
runs/<competition>/<trial>/04_result.ko.md
experiments/<competition>/<trial>/internal/decision_card.json
experiments/<competition>/<trial>/internal/trial_memory_card.json
memory/<competition>/research_notes.md
memory/<competition>/rules.md
```

색인 메타데이터 후보:

- `competition_id`
- `trial_id`
- `document_type`
- `path`
- `metric`
- `local_score`
- `change_axis`
- `decision`
- `created_at`

## UI/메신저에서 필요한 조회 단위

UI와 메시지 프로그램은 처음부터 복잡한 기능을 갖지 않습니다. SQLite 기반으로 아래 조회를 제공하는 것을 1차 목표로 둡니다.

```text
실험 목록
현재 실행 중인 trial
최근 trial 요약
best trial
pending action 목록
사용자가 볼 산출물 경로
토큰 사용량 요약
제출 준비/제출 결과 상태
```

메신저 연동은 먼저 알림 중심으로 설계합니다.

```text
trial started
stage changed
trial completed
trial blocked
approval required
human review requested
submission completed
```

사용자 응답을 받는 기능은 나중 단계로 둡니다. 먼저 메시지 전송과 상태 조회가 안정화된 뒤, 승인/피드백 입력을 붙이는 것이 안전합니다.

## 구현 단계

### 1단계: 설계 문서와 스키마 확정

- 이 문서를 기준으로 SQLite 최소 스키마를 확정합니다.
- 기존 파일 구조를 source of truth로 유지할지, 특정 상태를 SQLite source of truth로 승격할지 결정합니다.

권장 결정:

- 연구 산출물 source of truth: 파일
- trial 상태 조회 source of truth: SQLite
- RAG 검색 source of truth: 현재는 파일 기반 index/context pack, 후속 확장 시 Chroma 색인 추가

### 2단계: SQLite 저장소 추가

- `kaggle_research_agent/state_db.py` 추가
- DB 기본 경로: `memory/research_agent.sqlite3`
- schema migration 함수 추가
- upsert 함수 추가
- 조회 함수 추가

### 3단계: 파일 기반 기록을 SQLite에 동기화

- 기존 trial 기록을 훑어 SQLite에 반영하는 명령 추가
- 새 trial 완료 시 자동 upsert 연결
- 기존 파일 기반 fallback 유지

구현 상태:

- `kaggle_research_agent/state_db_sync.py` 추가
- `kaggle_research_agent/state_db_auto.py` 추가
- `sync-state-db` CLI 명령 추가
- `demo_one_cycle._finish()`와 `workspace_result_cycle._finish()`에서 best-effort 자동 sync 수행
- 동기화 대상:
  - `competitions/<competition>/execution_profile.yaml`
  - `experiments/<competition>/<trial>/metrics.json`
  - `experiments/<competition>/<trial>/internal/*.json`
  - `runs/<competition>/<trial>/*.ko.md`
  - `memory/<competition>/token_usage.jsonl`
  - `submissions/<competition>/submission_log.jsonl`
- 같은 파일 기반 기록을 여러 번 동기화해도 SQLite row가 중복 생성되지 않도록 upsert 방식으로 처리
- SQLite sync 실패는 실험 실패로 전파하지 않고 `state_db_sync.status=failed`로 결과에 남김

사용 예:

```text
python -m research_agent.cli sync-state-db --competition titanic
python -m research_agent.cli sync-state-db
```

### 4단계: CLI 조회 명령 추가

구현 상태:

- `kaggle_research_agent/state_query.py` 추가
- SQLite를 읽어 UI/메신저가 재사용할 수 있는 summary dict 생성
- CMD 확인용 renderer 추가
- `--json` 옵션으로 구조화 출력 지원
- `--sync` 옵션으로 조회 직전 파일 기반 기록을 SQLite에 재동기화 가능

지원 명령:

```text
python -m research_agent.cli list-experiments --sync
python -m research_agent.cli show-experiment --competition <name> --sync
python -m research_agent.cli show-trial --competition <name> --trial <trial_id> --sync
python -m research_agent.cli show-experiment --competition <name> --json
```

이 명령들은 최종 사용자 UI가 아니라, UI/메신저가 붙기 전 상태 조회 계층을 검증하는 개발자용 인터페이스입니다.

### 5단계: Chroma 색인 추가 - 보류

- 현재 단계에서는 구현하지 않습니다.
- 기존 `document_index.jsonl` 기반 RAG를 유지합니다.
- 후속 확장 시 Chroma writer를 추가하고, Chroma가 없거나 설치되지 않은 환경에서는 파일 기반 RAG로 fallback합니다.
- 색인 대상 문서를 작게 제한해 embedding 비용을 관리합니다.

### 6단계: UI/메신저 연동 준비

- UI/메신저가 SQLite를 조회하도록 status service를 정의
- pending action을 메시지 알림으로 변환
- 사용자 응답 처리는 별도 단계로 분리

## 이번 브랜치의 범위

`codex/db-ui-message-design` 브랜치에서는 다음을 목표로 합니다.

- DB/UI/메신저 설계 문서 작성
- SQLite 최소 스키마와 저장소 구현
- 기존 파일 기록을 SQLite에 동기화하는 안전한 인덱서 구현
- Chroma는 후속 확장으로 보류하고 SQLite 중심 상태 계층을 먼저 구현
- UI/메신저가 조회할 상태 API 또는 CLI 계약 정의

실제 웹 UI, Slack, 카카오톡 연동은 이 브랜치의 후반 또는 다음 브랜치에서 진행합니다.

## 의도적으로 미루는 것

- 기존 파일 기반 산출물 제거
- 모든 문서 전문의 SQLite 저장
- Chroma 단독 source of truth화
- 복잡한 웹 대시보드
- 메시지 프로그램에서 직접 trial 실행/중단/수정까지 처리하는 기능
- Docker 기반 실행 격리 구현

## 요약

DB 도입은 사용자 경험을 크게 바꾸기 위한 작업이 아니라, 내부 상태 조회와 장기 운영 안정성을 높이는 작업입니다.

사용자는 계속 `runs/<competition>/<trial>/`의 계획서, 파이프라인 구조도, 결과 파일, 코드 스냅샷을 확인할 수 있습니다. SQLite는 에이전트와 UI가 현재 상태를 빠르게 조회하기 위한 구조화 계층입니다. Chroma는 당장 필수 구성요소가 아니라, 파일 기반 RAG가 부족해질 때 추가할 수 있는 후속 확장입니다.
