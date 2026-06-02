# 자율 연구 루프와 사용자 리뷰 설계

## 목적

이 문서는 현재 `Autonomous Kaggle Research Agent`를 완전 자율 연구 에이전트로 확장하기 위한 중심 설계를 정의한다.

최종 목표는 Kaggle 같은 대회에서 에이전트가 스스로 실험을 계획하고, 실행하고, 제출하고, 리더보드 결과를 해석하고, 문제 해결 방안을 세우고, 코드 수정과 재실험을 반복하는 구조를 만드는 것이다.

이번 문서의 초점은 실제 Kaggle API, LangGraph, 코드 수정 Agent를 바로 구현하는 것이 아니다. 먼저 Main Agent가 어떤 상태를 읽고, 어떤 판단을 하며, 언제 사용자 의견을 받고, 제출 이력과 최고 실험본을 어떻게 보존할지 정의한다.

## 현재 프로젝트 위치

현재 프로젝트에는 다음 범위가 구현되어 있다.

- Level 0: competition state, memory, trial 기록 구조
- Level 1: planner, evaluator, memory updater, 단일 trial cycle
- Level 2: local/Colab job queue skeleton
- Level 4: config 기반 실험 구조와 validation

아직 구현하지 않은 범위는 다음과 같다.

- Level 3: Kaggle 제출과 leaderboard 결과 수집
- Level 5: 결과를 바탕으로 코드를 수정하는 Engineer Agent
- Level 6: 제출, 진단, 코드 수정, 재제출을 반복하는 완전 자율 루프
- LangGraph 기반 graph runtime
- Kakao 또는 외부 알림 연동

이 설계는 현재 CLI 기반 함수들을 나중에 LangGraph node로 옮기기 쉽게 경계를 잡는다.

## 핵심 원칙

### 사용자 리뷰는 범용 루프다

`User Review`는 이미지, 영상, ROI, 라벨 검토 전용 기능이 아니다.

모든 대회에서 에이전트가 불확실성, 문제점, 개선 후보, 큰 변경 방향을 사용자에게 설명하고, 사용자의 도메인 판단과 아이디어를 다음 실험과 코드 수정에 반영하는 범용 의사결정 루프다.

환자 행동 인식 대회의 ROI/라벨 검토는 이 범용 루프의 한 예시일 뿐이다.

### 제출은 결과만 남기는 작업이 아니다

Kaggle 제출은 `submission.csv`를 올리는 동작만 의미하지 않는다. 제출 전 현재 점수와 순위, 제출 후 새 점수와 순위, 제출에 사용한 파일, version name, best 여부를 모두 기록해야 한다.

제출에 사용한 실험 파일은 보존한다. 최고 점수 실험본은 사람이 바로 알아볼 수 있게 표시한다.

### 개선 정체 시 전략을 확장한다

기존 방법을 계속 튜닝해도 더 이상 개선되지 않으면, Main Agent는 모델 변경, 모델 구조 변경, feature pipeline 변경, ensemble, SOTA 후보 탐색으로 전략을 확장해야 한다.

이 전환은 큰 비용과 위험이 있으므로 사용자에게 근거와 후보를 설명한 뒤 진행한다.

## Main Agent 루프

Main Agent는 연구 루프의 판단자다. 각 노드는 파일 기반 state, memory, trial artifact를 입력으로 받고 다음 행동을 결정한다.

```text
load_context
  -> plan_trial
  -> validate_trial
  -> decide_execution
  -> run_or_wait
  -> evaluate_result
  -> diagnose_trial
  -> decide_user_review
  -> collect_user_feedback
  -> revise_strategy
  -> remember_result
  -> decide_submission
  -> submit_to_kaggle
  -> record_submission_result
  -> decide_strategy_escalation
  -> decide_next_action
```

초기 구현에서는 모든 노드를 한 번에 자동 실행하지 않는다. 현재 함수와 가까운 부분부터 확장한다.

## Node 설계

### load_context

읽는 정보:

- `competitions/<competition>/state.yaml`
- `memory/<competition>/research_notes.md`
- `memory/<competition>/rules.md`
- `memory/<competition>/trial_index.jsonl`
- `memory/<competition>/user_feedback.jsonl`
- 최근 trial artifact
- submission history
- best trial summary

출력:

- competition profile
- current state
- recent trials
- known rules
- user feedback summary
- current best trial
- submission budget/status

### plan_trial

현재 `propose_plan()`을 확장한다.

역할:

- 다음 trial의 primary hypothesis 작성
- 바꿀 요소를 하나 또는 소수로 제한
- 이전 실패와 사용자 피드백 반영
- 큰 전략 전환이 필요한 경우 후보를 명시

출력:

- `experiments/<competition>/<trial>/plan.md`
- `experiments/<competition>/<trial>/config.yaml`

### validate_trial

현재 `validate_config()`를 사용한다.

역할:

- config가 allowed search space 안에 있는지 확인
- validation strategy와 model family가 동시에 바뀌는지 확인
- 큰 변경이면 user review 또는 strategy escalation이 필요한지 표시

### decide_execution

역할:

- local 실행
- Colab job 생성
- metrics 대기
- 사용자 판단 요청

기본 정책:

- local을 기본 backend로 둔다.
- Colab은 명시적 설정이나 Main Agent의 정책 판단이 있을 때만 사용한다.
- 예상 실행 비용이 크거나 GPU가 필요하면 사용자 확인을 요청할 수 있다.

### evaluate_result

현재 `evaluate_trial()`을 확장한다.

역할:

- CV score와 metric별 결과를 읽는다.
- LB score가 있으면 CV/LB 관계를 기록한다.
- diversity, leakage warning, validation mismatch를 점검한다.
- 결과 요약을 `evaluation.md`로 저장한다.

### diagnose_trial

`diagnose_trial`은 특정 데이터 유형에 묶이지 않는 범용 진단 노드다.

진단 항목:

- CV 개선 여부
- LB/CV 괴리
- metric별 약점
- segment/group/fold/feature/pattern별 문제 집중
- leakage 또는 validation 문제 의심
- 실패 반복 여부
- 제출 후 rank/score 변화
- 현재 방법의 개선 여지
- 다음 개선 후보
- 사용자에게 물어볼 쟁점

출력:

- `experiments/<competition>/<trial>/diagnosis.md`
- structured diagnosis object

예시:

- 이미지/영상 대회: ROI, 라벨 기준, 시각적 의미 판단
- tabular 대회: feature leakage, validation split, domain feature 해석
- time-series 대회: 시간 누수, windowing, seasonality, anomaly 처리
- NLP 대회: prompt/template, tokenization, label ambiguity, domain shift

### decide_user_review

`decide_user_review`는 사용자 의견이 필요한지 판단한다.

트리거:

- 문제 원인이 metrics만으로 설명되지 않는다.
- 도메인 판단이 필요하다.
- 큰 코드 수정이나 모델 구조 변경을 하려 한다.
- 제출 전략 또는 validation strategy를 바꾸려 한다.
- leakage, label ambiguity, data split 문제가 의심된다.
- 최근 trial이 반복 실패했고 전략 전환이 필요하다.
- SOTA 후보 적용 비용과 위험을 사용자가 알아야 한다.

출력:

- review가 필요하면 `user_review_request.md`
- review가 필요 없으면 이유와 다음 action

### collect_user_feedback

사용자 의견을 구조화해 저장한다.

저장 위치:

```text
memory/<competition>/user_feedback.jsonl
experiments/<competition>/<trial>/user_review_response.md
```

기록 필드:

```text
time
competition
trial_id
topic
question
user_feedback
decision
follow_up_action
```

### revise_strategy

사용자 피드백과 diagnosis를 바탕으로 다음 전략을 수정한다.

역할:

- current_focus 갱신
- promising_directions 추가
- forbidden_directions 추가
- validation review 여부 표시
- 코드 수정 요청 초안 생성

### decide_submission

제출 여부를 판단한다.

정책:

- 제출 전 현재 best LB score와 rank를 확인한다.
- submission limit과 오늘 제출 횟수를 확인한다.
- CV 개선이 seed variance보다 작으면 제출하지 않는다.
- leakage 의심이 있으면 제출하지 않고 review를 요청한다.
- 큰 전략 전환 trial은 제출 전 사용자 확인을 요청할 수 있다.

출력:

- submit
- skip_submit
- ask_user
- wait_for_artifact

### submit_to_kaggle

향후 Kaggle API 연동 노드다.

역할:

- 제출 전 leaderboard 상태 확인
- version name 생성 또는 검증
- 지정된 `submission.csv` 제출
- 제출 결과 score/rank 수집

이번 설계 문서에서는 실제 Kaggle API 호출을 구현하지 않는다.

### record_submission_result

제출 결과를 보존한다.

저장 위치:

```text
submissions/<competition>/submission_log.jsonl
experiments/<competition>/<trial>/submission_result.md
experiments/<competition>/<trial>/VERSION.md
experiments/<competition>/<trial>/artifacts/
experiments/<competition>/BEST_TRIAL.md
memory/<competition>/best_trial.json
```

submission log 필드:

```text
submission_id
competition
trial_id
version_name
submitted_at
submission_file
cv_score
previous_lb_score
previous_rank
submitted_lb_score
submitted_rank
score_delta
rank_delta
is_best
notes
```

최고 실험본 표시:

- `experiments/<competition>/BEST_TRIAL.md`에는 사람이 읽는 요약을 둔다.
- `memory/<competition>/best_trial.json`에는 구조화된 best trial 정보를 둔다.
- best trial directory에는 `BEST_MARKER.md`를 둘 수 있다.

### decide_strategy_escalation

개선 정체를 감지하고 전략 전환을 결정한다.

트리거:

- 최근 N개 trial이 개선되지 않았다.
- 같은 모델/feature 계열에서 score 변화가 seed noise 안에 머문다.
- CV는 개선되지만 LB가 반복 악화된다.
- current best 대비 rank가 의미 있게 오르지 않는다.
- 현재 방법의 개선 후보가 소진됐다.

전략 단계:

```text
Level A: current method refinement
  - hyperparameter
  - feature
  - validation 안정화
  - threshold
  - seed
  - ensemble

Level B: model family change
  - LightGBM -> CatBoost/XGBoost/NN
  - Transformer -> TCN/RNN/Graph/TabTransformer
  - baseline model -> stronger known family

Level C: architecture redesign
  - feature pipeline 변경
  - multi-stage model
  - auxiliary task
  - ensembling/blending 구조
  - data augmentation strategy

Level D: SOTA exploration
  - 최근 대회/논문/공개 solution 후보 조사
  - 적용 가능성 평가
  - 구현 비용과 예상 이득 비교
  - 사용자 승인 후 큰 코드 변경
```

큰 전환은 `decide_user_review`와 연결한다.

## 저장 구조

추가할 구조:

```text
memory/<competition>/user_feedback.jsonl
memory/<competition>/decision_log.jsonl
memory/<competition>/best_trial.json

submissions/<competition>/submission_log.jsonl

experiments/<competition>/BEST_TRIAL.md

experiments/<competition>/<trial>/diagnosis.md
experiments/<competition>/<trial>/user_review_request.md
experiments/<competition>/<trial>/user_review_response.md
experiments/<competition>/<trial>/submission_result.md
experiments/<competition>/<trial>/VERSION.md
experiments/<competition>/<trial>/BEST_MARKER.md
experiments/<competition>/<trial>/artifacts/
```

기존 구조와의 관계:

- `memory/<competition>/trial_index.jsonl`은 실험 단위 요약을 유지한다.
- `submission_log.jsonl`은 제출 단위 기록을 유지한다.
- 하나의 trial이 여러 번 제출될 수 있으므로 trial log와 submission log를 분리한다.
- `BEST_TRIAL.md`는 사람이 읽는 현재 최고본 안내서다.
- `best_trial.json`은 에이전트가 읽는 구조화된 최고본 상태다.

## Version Name 규칙

version name은 사람이 보아도 실험 의도를 알 수 있어야 한다.

기본 형식:

```text
<competition>_<trial_id>_<short_focus>_vNN
```

예시:

```text
patient_action_skeleton_trial_002_c1_boundary_v01
demo_local_demo_001_baseline_v01
```

규칙:

- 같은 trial에서 제출이 반복되면 suffix를 올린다.
- 큰 모델 구조 변경은 focus에 반영한다.
- best가 된 version은 `BEST_TRIAL.md`와 `best_trial.json`에 기록한다.

## Decision Log

Main Agent는 중요한 판단을 `decision_log.jsonl`에 남긴다.

기록 대상:

- 제출 여부 판단
- 사용자 리뷰 요청 여부
- validation strategy 변경
- model family 변경
- SOTA 후보 적용 결정
- best trial 갱신
- 중단 또는 대기 결정

필드:

```text
time
competition
trial_id
decision_type
decision
reason
evidence
user_input_used
next_action
```

## 구현 순서

이 설계 이후 구현 계획은 다음 순서를 따른다.

1. `diagnose_trial` 기본 구현
2. `user_review_request.md` 생성과 `user_feedback.jsonl` 저장
3. `decision_log.jsonl` 기록
4. Main Agent cycle에 diagnosis/user review decision 연결
5. submission artifact와 version metadata 구조 추가
6. Kaggle 제출 전후 score/rank 기록 설계 구현
7. best trial 표시와 best metadata 갱신
8. strategy escalation 판단 구현
9. Code Editing Agent 설계와 연결
10. LangGraph runtime 도입

## 이번 단계에서 제외하는 것

이번 설계 문서는 다음을 실제 구현 범위로 삼지 않는다.

- LangGraph 라이브러리 실제 도입
- Kaggle API 제출 구현
- 코드 자동 수정 Agent 구현
- Kakao 실제 연동
- Colab 완전 자동 제어
- 특정 대회 전용 진단 로직 구현
- 외부 SOTA 검색 자동화

단, 위 항목들이 들어갈 노드와 저장 구조는 문서에 자리로 남긴다.

## 성공 기준

이 설계가 성공하려면 다음이 가능해야 한다.

- 어떤 대회든 결과를 `diagnose_trial`로 해석할 수 있다.
- 특정 도메인 예시에 갇히지 않고 사용자 의견을 받을 수 있다.
- 제출 전후 score/rank와 사용한 artifact가 보존된다.
- 최고 점수 실험본을 사람이 바로 찾을 수 있다.
- 개선 정체 시 작은 튜닝을 반복하지 않고 모델/구조/SOTA 방향으로 전환할 수 있다.
- 이후 구현 계획이 명확한 작은 단계로 나뉜다.

