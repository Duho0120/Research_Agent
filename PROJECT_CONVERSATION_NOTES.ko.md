# 프로젝트 대화 요약 노트

이 문서는 구현용 `README.md`가 아니라, 프로젝트 진행 중 사용자가 짚어준 핵심 관점과 의사결정을 정리하는 기록입니다.  
목적은 에이전트가 기능 구현만 따라가는 것이 아니라, 프로젝트 오너의 판단 기준과 연구 철학을 잃지 않도록 하는 것입니다.

## 1. 프로젝트의 본질

이 프로젝트는 단순히 모델을 학습시키고 성능을 올리는 프로젝트가 아니다.

기존에는 딥러닝 모델 학습 중심으로 작업했지만, 이번 프로젝트의 목표는 다음에 가깝다.

- 실험 계획을 세우는 프로그램
- 실험 결과를 해석하는 프로그램
- 이전 실험을 기억하는 프로그램
- 필요하면 학습을 실행하고 결과를 다시 반영하는 프로그램
- 장기적으로는 자율 연구자처럼 행동하는 에이전트 시스템

즉, 핵심은 “좋은 모델 하나 만들기”가 아니라 **연구 과정을 자동화하고 구조화하는 시스템을 만드는 것**이다.

## 2. 사용자는 프로젝트 결정권자다

사용자는 “이 프로젝트 만들어줘”라고 요청하면 에이전트가 구현은 할 수 있지만, 그 경우 사용자가 맥락을 모른 채 기능 구현과 목표 달성만 따라가게 될 위험이 있다고 지적했다.

따라서 프로젝트는 다음 순서를 지나야 한다.

- 전체 틀 이해
- 중요한 개념과 도구 이해
- 자율성 범위 결정
- 시스템 구조 기획
- 단계별 구현
- 검토
- 이후 실제 목표를 잡고 목표 달성까지 진행

에이전트는 사용자의 의견을 단순 명령으로만 받아들이지 말고, 사용자가 올바른 결정을 내릴 수 있도록 구조와 맥락을 계속 설명해야 한다.

## 3. 단계적 구현 전략

한 번에 완전 자율 연구자를 만들지 않고, 자율성 레벨을 나눠 구현한다.

현재 구현 및 검토한 범위:

- Level 0: 프로젝트 골격, state, memory, trial 기록 구조
- Level 1: Planner, Evaluator, Memory Agent 기반 연구 보조 루프
- Level 2: Colab job queue skeleton
- Level 4: config 기반 실험 구조

아직 전면 구현하지 않은 범위:

- LangGraph 기반 main graph
- 완전 자동 코드 수정 Agent
- 실제 Kaggle 제출 자동화
- Colab 자동 실행 제어
- 카카오톡 기반 승인/피드백 채널
- 완전 자율 반복 연구 루프

현재는 구현 후 검토 가능한 1차 골격을 만든 상태이며, 다음부터는 실제 목표를 잡고 점진적으로 확장한다.

## 4. Local 우선, Colab은 필요할 때만

초기에는 metrics가 없으면 Colab job을 만드는 방향으로 구현됐지만, 사용자는 로컬에서 가능한 작업은 로컬에서 먼저 진행하고 싶다고 밝혔다.

이에 따라 실행 정책은 다음과 같이 정리됐다.

- 기본 실행 backend는 `local`
- Colab은 사용자가 명시적으로 선택하거나, 나중에 Main Agent가 필요하다고 판단할 때만 사용
- GPU가 꼭 필요하거나 실행 시간이 길거나 로컬 리소스가 부족한 경우에 Colab을 고려

현재 구현 기준:

- `create-job` 기본 backend는 local
- `--backend colab`을 명시해야 Colab job 생성
- `run-local` 또는 `cycle --run-now`로 로컬 실행 가능

향후 LangGraph에서는 Main Agent가 다음 조건을 보고 local/colab/ask_user를 결정하도록 설계한다.

- GPU 필요 여부
- 예상 실행 시간
- 로컬 장치 사용 가능 여부
- 이전 로컬 실행 실패 원인
- 사용자의 실행 정책

## 5. LangGraph 도입 방향

현재 코드는 LangGraph가 아니라 Python CLI 기반이다.

하지만 함수 구조는 LangGraph로 옮기기 쉽게 분리되어 있다.

- `propose_plan()` → plan node
- `validate_config()` → validation node
- `run_local_job()` → local execution node
- `create_job(..., backend="colab")` → colab job node
- `evaluate_trial()` → evaluation node
- `remember_trial()` → memory node

향후 LangGraph 구조는 다음과 같은 형태가 적절하다.

```text
START
  ↓
load_state
  ↓
plan_trial
  ↓
validate_config
  ↓
decide_execution_backend
  ├─ run_local
  ├─ create_colab_job
  ├─ ask_user
  └─ wait_for_metrics
  ↓
collect_results
  ↓
evaluate
  ↓
diagnose_errors
  ↓
remember
  ↓
decide_next_action
```

중요한 점:

- 아직 LangGraph 코드는 반영하지 않는다.
- 먼저 설계와 판단 기준을 명확히 한다.
- 초기 LangGraph 버전은 LLM 판단보다 rule-based routing으로 시작하는 것이 좋다.

## 6. Claude/Codex 연동에 대한 관점

Claude와 Codex를 반드시 연동할 필요는 없다.

초기에는 하나의 구조와 명확한 state/memory 설계가 더 중요하다.

다만 장기적으로는 역할을 나눌 수 있다.

- Codex: 코드 수정, repo 작업, 실행 오류 디버깅
- Claude: 긴 연구 노트 요약, 전략 비평, 실패 원인 분석
- Main Agent: 어떤 모델에게 어떤 일을 맡길지 결정

초기 구현에서는 멀티모델보다 안정적인 workflow가 우선이다.

## 7. 카카오톡 개입 채널 아이디어

사용자는 작업 중간에 아이디어나 주의사항을 카카오톡으로 제시하고 싶다고 말했다.

현재 판단:

- 카카오톡은 알림 채널로 먼저 쓰는 것이 현실적
- 사용자의 입력은 카카오톡 메시지 직접 수신보다 링크 기반 inbox가 적절
- 에이전트는 중요한 판단 지점에서 사용자에게 알림을 보내고, 사용자는 승인/주의사항/아이디어를 남기는 구조가 좋다

향후 구조:

```text
Main Agent
  ↓
needs_user_input
  ↓
Kakao notification
  ↓
User inbox
  ↓
human_notes.jsonl
  ↓
Main Agent가 다음 cycle 전에 반영
```

## 8. Human-in-the-loop는 핵심 기능이다

사용자는 중요한 관점을 제시했다.

자율 연구자라고 해서 모든 판단을 에이전트가 해서는 안 된다.  
특히 이미지, 영상, ROI, 라벨링, 시각적 의미 판단이 필요한 경우에는 사람의 육안 판단이 필요하다.

환자 행동 인식 프로젝트의 실제 사례:

- 원본 데이터는 이미지/영상 기반이었다.
- 사용자가 직접 GUI 라벨링 도구를 만들어 라벨링했다.
- ROI 영역 feature를 추가할 때도 GUI 도구를 만들고, 사용자가 직접 침대 라인을 잡았다.
- 특정 시나리오의 오탐 결과를 눈으로 확인해보니, 의자 때문에 ROI 영역이 좁아지고 환자가 앉아 있어도 골반 좌표가 ROI 밖으로 넘어가는 문제가 있었다.
- 이 경우 모델은 실제 행동 때문이 아니라 ROI 정의와 시각적 맥락 때문에 혼동할 수 있다.

따라서 Human Review는 예외 처리나 수동 디버깅이 아니라, 그래프의 정식 노드로 들어가야 한다.

추천 노드:

```text
diagnose_errors
  ↓
need_human_review?
  ├─ no  → plan_next_trial
  └─ yes → prepare_review_pack
              ↓
            request_human_review
              ↓
            ingest_human_feedback
              ↓
            update_memory
              ↓
            revise_strategy
```

Human Review가 필요한 트리거:

- 특정 scenario/view에 오류가 과도하게 몰림
- Bed Exit/Wandering처럼 라벨 경계가 행동적으로 애매함
- ROI 기반 feature가 성능에 큰 영향을 줌
- 좌표만으로 오류 원인을 설명하기 어려움
- Fall 같은 안전 중요 클래스의 false negative 발생
- 새 feature가 실제 영상 의미와 맞는지 확인 필요

에이전트의 역할:

- 사람이 봐야 할 샘플을 자동으로 골라준다.
- 시각화 pack을 만든다.
- 사용자의 관찰을 구조화해서 memory에 저장한다.
- 그 판단을 다음 trial 전략으로 바꾼다.

사람의 역할:

- 실제 시각적 의미 판단
- 라벨 기준 판단
- ROI 정의 판단
- 행동 경계 판단

핵심 원칙:

> 자율성은 사람을 배제하는 것이 아니라, 사람이 봐야 할 순간을 정확히 찾아내고 그 판단을 다음 실험으로 연결하는 능력이다.

## 9. 환자 행동 인식 모델 분석에서 얻은 현재 baseline 요약

사용자가 제공한 노트북:

`C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\Notebooks\V07_COM_Lean_Method_4-Torch_XPU.ipynb`

프로젝트:

- 병상 환자 행동 인식
- 30프레임 skeleton 시계열 모델
- 4-class 분류: Normal, Bed Exit, Wandering, Fall
- Transformer 기반
- ROI dynamic feature 사용
- LDAM loss, DRW, Bed Exit/Wandering auxiliary head 사용

baseline 성능:

- selection score: 0.7926
- accuracy: 0.8952
- macro F1: 0.8326
- Bed Exit F1: 0.7792
- Fall F1: 0.95
- Fall recall: 0.96

관찰된 병목:

- Bed Exit/Wandering 혼동
- C1 view에서 가장 높은 error rate: 0.1686
- C3는 상대적으로 안정적: 0.0772
- `00620_H_D_SY` 시나리오에서 23/60 오류
- C1에서 Bed Exit/Wandering 혼동 window 13개

현재 생성된 trial:

- `trial_001_v07_baseline`: baseline 기록
- `trial_002_c1_bed_wandering_focus`: C1과 Bed Exit/Wandering 혼동 개선 계획

## 10. 문서 언어 정책

사용자는 `.md` 파일을 한글로 읽고 싶다고 말했다.

정리된 정책:

- 사람이 읽는 기본 문서: 한글
- 에이전트/LLM 참고용 문서: 영어본도 보존
- `config.yaml`, `metrics.json`, 코드, schema key는 영어 유지

파일 구성 예:

```text
plan.md      # 한글 기본본
plan.ko.md   # 한글 명시본
plan.en.md   # 영어본
```

현재 `trial_002_c1_bed_wandering_focus`에는 다음 파일이 있다.

- `plan.md`
- `plan.ko.md`
- `plan.en.md`
- `improvement_candidates.md`
- `improvement_candidates.ko.md`
- `improvement_candidates.en.md`

## 11. 다음 설계에서 잊지 말아야 할 것

앞으로 구현할 때 다음을 계속 유지해야 한다.

- 기능보다 연구 흐름이 중요하다.
- Kaggle 대회별로 state, experiments, memory, jobs, configs를 분리해 기록이 섞이지 않게 한다.
- Local 실행을 기본으로 한다.
- Colab은 필요할 때만 선택한다.
- LangGraph는 Main Agent의 판단 흐름을 명시하기 위해 도입한다.
- Human-in-the-loop는 정식 graph branch로 설계한다.
- 사용자의 시각적 판단과 도메인 판단을 memory에 남긴다.
- 한글 문서는 사용자의 판단을 돕는 기본 인터페이스다.
- 영어 문서는 에이전트와 외부 도구가 안정적으로 참고하도록 보존한다.

## 12. 비용 효율적 자율 연구 정책

사용자는 자율 연구자가 LLM을 계속 호출하는 시스템이 아니라, 일반 코드와 rule-based gate가 반복 작업을 처리하고 LLM은 중요한 판단 지점에서만 호출되는 구조여야 한다고 정리했다.

정책화한 핵심 원칙:

- Local 우선 실행
- LLM 호출 최소화
- 명확한 판단은 rule-based로 처리
- 비싼 판단만 Main Agent/LLM에게 맡김
- 사람의 검토가 필요한 순간은 Human Review 노드로 분기
- 모든 trial 결과와 판단 근거는 memory에 구조화해서 저장

1차 설계 문서:

```text
docs/policies/execution_decision_policy.ko.md
docs/policies/human_review_policy.ko.md
docs/policies/review_pack_schema.ko.md
```

다음 구현 방향:

- `token_policy.yaml`, `execution_policy.yaml`, `human_review_policy.yaml` 추가
- 정책 파일을 읽는 rule-based gate 구현
- LLM 호출 전후를 `decision_log.jsonl`에 기록
- `review_pack/` 생성 기능 추가
- local 실패 원인 분류 추가

현재 반영 상태:

- `configs/policies/token_policy.yaml`
- `configs/policies/execution_policy.yaml`
- `configs/policies/human_review_policy.yaml`
- `research_agent/policies.py`
- `research_agent/agents/policy_gate.py`
- `research_agent/agents/review_pack.py`

구현된 판단:

- 실행 전 `decide_execution`으로 local, wait, ask_user, create job 계열 판단
- `local_run.log` 기반 local failure type 분류
- diagnosis 기반 Human Review policy 판단
- token budget과 호출 사유 기반 `should_call_llm` 판단
- Human Review 필요 시 표준 `review_pack/` 생성

추가 반영:

- `log_llm_decision`으로 LLM 호출/비호출 판단을 `decision_log.jsonl`에 남긴다.
- CLI `decide-llm`으로 token policy 판단을 직접 확인할 수 있다.
- CLI `request-review`는 Human Review policy를 평가하고, 필요하면 표준 `review_pack/`을 생성한다.

Human Review feedback loop 반영:

- `record-feedback`는 `review_pack/human_feedback.md`와 `review_pack/human_feedback.json`을 쓴다.
- `review_pack/manifest.json`의 상태를 `feedback_recorded`로 바꾼다.
- `decision_log.jsonl`에 `decision_type: human_feedback`, `user_input_used: true`를 기록한다.
- `plan-next`는 source trial의 최신 사용자 피드백을 읽어 `change_validation`, `change_model_family`, `prepare_sota_research` 같은 결정을 다음 전략에 반영한다.
