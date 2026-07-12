# 프로젝트 변경 이력

이 문서는 포트폴리오 작성과 프로젝트 회고를 위해 주요 변경사항을 날짜/시간 순서로 정리한 기록입니다.  
시간 기준은 `Asia/Seoul`입니다.

## 2026-05-30 02:00-03:20 KST

### 1차 프로젝트 골격 구현

요약:

- Autonomous Kaggle Research Agent의 Level 0/1/2/4 범위를 우선 구현했다.
- 완전 자율 연구자 이전 단계로, 실험 계획/평가/기억/Job 생성이 가능한 기본 구조를 만들었다.

주요 변경:

- Python package `kaggle_research_agent` 생성
- CLI 진입점 `kaggle_research_agent.cli` 구현
- 기본 폴더 구조 생성
  - `competitions/`
  - `experiments/`
  - `memory/`
  - `jobs/`
  - `configs/`
  - `colab/`
- `state.yaml`, `trial_index.jsonl`, `research_notes.md`, `rules.md` 기반 memory 구조 도입
- Planner/Evaluator/Memory Agent 초안 구현
- config validation 구조 구현
- Colab worker skeleton 추가

주요 파일:

- `README.md`
- `kaggle_research_agent/cli.py`
- `kaggle_research_agent/planner_agent.py`
- `kaggle_research_agent/evaluator_agent.py`
- `kaggle_research_agent/memory_agent.py`
- `kaggle_research_agent/config_validator.py`
- `colab/worker.py`
- `colab/worker_notebook.ipynb`

검증:

```powershell
python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli create-job --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli evaluate --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli remember --competition demo --trial trial_001
```

결과:

- demo competition 기준 config 검증, job 생성, 평가, memory 업데이트가 정상 동작했다.

## 2026-05-30 03:20-03:35 KST

### Main cycle 추가 및 구조 검토

요약:

- 단일 명령으로 trial 계획, config 검증, job 생성 또는 평가/기억 업데이트를 수행하는 `cycle` 흐름을 추가했다.

주요 변경:

- `main_agent.py` 추가
- `cycle` CLI 명령 추가
- YAML parser의 child node 추정 문제 수정
- README에 cycle 사용법 추가

주요 파일:

- `kaggle_research_agent/main_agent.py`
- `kaggle_research_agent/simple_yaml.py`
- `kaggle_research_agent/cli.py`
- `README.md`

검증:

```powershell
python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_002 --no-job
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_003
```

결과:

- `trial_002`는 metrics가 없어 `waiting_for_metrics`까지 진행
- `trial_003`은 job 생성까지 진행

## 2026-05-30 03:40-03:55 KST

### 환자 행동 인식 노트북 분석 및 baseline trial 등록

요약:

- 사용자가 제공한 환자 행동 인식 노트북을 분석하고, 현재 성능을 baseline trial로 등록했다.
- 새로운 개선 방향으로 C1 view와 Bed Exit/Wandering 혼동에 집중하는 trial을 설계했다.

분석 대상:

```text
C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\Notebooks\V07_COM_Lean_Method_4-Torch_XPU.ipynb
```

노트북 요약:

- 30-frame skeleton time-series classification
- 4-class 분류: Normal, Bed Exit, Wandering, Fall
- PyTorch Transformer 기반
- ROI dynamic feature 사용
- LDAM loss, DRW, Bed Exit/Wandering auxiliary head 사용
- scenario-group split 사용

baseline 성능:

- selection score: `0.7926`
- accuracy: `0.8952`
- macro F1: `0.8326`
- Bed Exit F1: `0.7792`
- Fall F1: `0.95`
- Fall recall: `0.96`

관찰된 병목:

- Bed Exit/Wandering 혼동
- C1 view error rate: `0.1686`
- C3 view error rate: `0.0772`
- `00620_H_D_SY` 시나리오에서 23/60 오류
- C1에서 Bed Exit/Wandering 혼동 13건

생성한 trial:

- `trial_001_v07_baseline`
- `trial_002_c1_bed_wandering_focus`

주요 파일:

- `competitions/patient_action_skeleton/overview.md`
- `competitions/patient_action_skeleton/data_notes.md`
- `experiments/patient_action_skeleton/trial_001_v07_baseline/plan.md`
- `experiments/patient_action_skeleton/trial_001_v07_baseline/metrics.json`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/plan.md`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/improvement_candidates.md`

검증:

```powershell
python -B -m kaggle_research_agent.cli validate-config --competition patient_action_skeleton --trial trial_001_v07_baseline
python -B -m kaggle_research_agent.cli validate-config --competition patient_action_skeleton --trial trial_002_c1_bed_wandering_focus
python -B -m kaggle_research_agent.cli evaluate --competition patient_action_skeleton --trial trial_001_v07_baseline
python -B -m kaggle_research_agent.cli remember --competition patient_action_skeleton --trial trial_001_v07_baseline
```

결과:

- baseline trial이 `accept_as_candidate`로 평가됨
- `patient_action_skeleton/state.yaml`에 baseline이 best trial로 기록됨

## 2026-05-30 03:55-04:05 KST

### Local 우선 실행 정책 반영

요약:

- 사용자가 로컬에서 가능한 작업은 로컬에서 먼저 진행하고 싶다고 요청했다.
- 이에 따라 기본 실행 backend를 local로 변경하고, Colab은 명시적으로 선택할 때만 사용하도록 구조를 바꿨다.

주요 변경:

- `backend: local | colab` 개념 추가
- `create-job` 기본값을 local로 설정
- `run-local` CLI 명령 추가
- `cycle --run-now`로 로컬 실행 후 metrics 평가/기억까지 이어지는 흐름 추가
- demo local trainer 추가

주요 파일:

- `kaggle_research_agent/job_manager.py`
- `kaggle_research_agent/main_agent.py`
- `kaggle_research_agent/cli.py`
- `scripts/demo_train.py`
- `README.md`

검증:

```powershell
python -B -m kaggle_research_agent.cli cycle --competition demo --trial local_demo_001 --run-now --run-command "python scripts/demo_train.py --config experiments/demo/local_demo_001/config.yaml --output experiments/demo/local_demo_001 --score 0.83"
python -B -m kaggle_research_agent.cli create-job --competition demo --trial local_demo_002
python -B -m kaggle_research_agent.cli create-job --competition demo --trial colab_demo_001 --backend colab
```

결과:

- `cycle --run-now`가 local run, metrics 생성, evaluation, memory update까지 정상 수행
- local job은 `backend: local`
- colab job은 `backend: colab`

## 2026-05-30 04:05-04:20 KST

### 실험 계획 문서 한글/영어 분리

요약:

- 사용자가 `.md` 문서를 한글로 읽고 싶다고 요청했다.
- 사람이 읽는 기본 문서는 한글로 두고, 에이전트/LLM 참고용 영어 문서는 `.en.md`로 보존하는 정책을 적용했다.

정책:

- `plan.md`: 한글 기본본
- `plan.ko.md`: 한글 명시본
- `plan.en.md`: 영어본
- `config.yaml`, `metrics.json`, 코드, schema key는 영어 유지

주요 파일:

- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/plan.md`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/plan.ko.md`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/plan.en.md`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/improvement_candidates.md`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/improvement_candidates.ko.md`
- `experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/improvement_candidates.en.md`

결과:

- 사용자는 한글 문서를 기본으로 읽을 수 있고, 에이전트는 영어본도 참고할 수 있게 됐다.

## 2026-05-30 04:20-04:35 KST

### 프로젝트 대화 요약 노트 생성

요약:

- 구현용 README와 별도로, 사용자의 핵심 의견과 프로젝트 철학을 정리하는 문서를 만들었다.

주요 내용:

- 프로젝트의 본질
- 사용자가 프로젝트 결정권자라는 점
- 단계적 구현 전략
- Local 우선 / Colab 선택 정책
- LangGraph 도입 방향
- Claude/Codex 연동 관점
- 카카오톡 개입 채널 아이디어
- Human-in-the-loop 필요성
- 환자 행동 인식 baseline 분석 요약
- 한글/영어 문서 정책

주요 파일:

- `PROJECT_CONVERSATION_NOTES.ko.md`

결과:

- 이후 프로젝트 진행 중 사용자의 판단 기준과 설계 철학을 잃지 않도록 별도 기록이 생겼다.

## 2026-05-30 04:35 KST

### 포트폴리오용 변경 이력 문서 생성

요약:

- 포트폴리오 작성과 회고를 위해 날짜/시간 기반 변경 이력 문서를 만들었다.

주요 파일:

- `PROJECT_CHANGELOG.ko.md`

활용 목적:

- 프로젝트 개발 과정 설명
- 포트폴리오 타임라인 작성
- 어떤 문제를 어떤 설계로 해결했는지 회고
- 나중에 GitHub README, 블로그, 발표 자료로 확장

## 2026-05-30 이후

### 대회별 workspace 분리 강화

요약:

- Kaggle 대회마다 별도 작업 단위를 유지할 수 있도록 공용 성격이 강했던 `memory`, `jobs`, `configs`를 대회별 하위 폴더 기준으로 분리했다.
- 기존 `competitions/<competition>`와 `experiments/<competition>` 구조에 더해, 실행 큐와 연구 기억도 대회별로 격리되도록 정리했다.

주요 변경:

- `memory/<competition>/research_notes.md`
- `memory/<competition>/rules.md`
- `memory/<competition>/trial_index.jsonl`
- `jobs/<competition>/*.yaml`
- `configs/<competition>/allowed_space.yaml`

설계 의미:

- 여러 Kaggle 대회를 동시에 다뤄도 trial 기록, 연구 노트, job queue, search space가 서로 섞일 위험을 줄인다.
- 기존 전역 파일은 fallback/legacy 성격으로 남기고, 새로 쓰는 데이터는 대회별 경로를 우선 사용한다.

## 2026-06-01 KST

### 비용 효율적 자율 연구 정책 1차 설계

요약:

- LLM을 계속 호출하는 구조가 아니라, rule/tool/Python이 반복 작업을 처리하고 LLM은 중요한 판단 지점에서만 호출하는 방향을 정책 문서로 정리했다.
- 코드 구현에 앞서 `local / colab / ask_user / wait_for_metrics`, Human Review 트리거, Review Pack 구조를 먼저 명확히 했다.

생성한 정책 문서:

- `docs/policies/execution_decision_policy.ko.md`
- `docs/policies/human_review_policy.ko.md`
- `docs/policies/review_pack_schema.ko.md`

핵심 결정:

- 실행 backend는 local-first를 기본으로 한다.
- Colab은 명시 선택, resource failure, GPU 필요, 긴 실행 시간 같은 근거가 있을 때만 후보가 되며 기본적으로 사용자 확인을 거친다.
- Human Review는 이미지/영상 전용 예외 기능이 아니라, validation, label ambiguity, feature leakage, strategy shift, submission approval까지 포괄하는 범용 의사결정 branch다.
- Review Pack은 `experiments/<competition>/<trial_id>/review_pack/` 아래에 `manifest.json`, `summary.ko.md`, `questions.ko.md`, `cases.jsonl`, `metrics_snapshot.json` 등을 저장하는 구조로 정의했다.

### 비용 효율 정책 2차 구현

요약:

- 1차 정책 문서를 기계가 읽을 수 있는 yaml 설정과 rule-based gate로 옮겼다.
- Orchestrator cycle에 execution decision 기록과 human review policy 판단을 연결했다.
- Human Review가 필요한 diagnosis에는 표준 `review_pack/`을 생성할 수 있게 했다.

추가한 정책 파일:

- `configs/policies/token_policy.yaml`
- `configs/policies/execution_policy.yaml`
- `configs/policies/human_review_policy.yaml`

주요 구현:

- `kaggle_research_agent/policies.py`
  - 정책 yaml 로딩
  - 기본 정책 fallback
- `kaggle_research_agent/agents/policy_gate.py`
  - `decide_execution`
  - `classify_local_failure`
  - `decide_human_review`
  - `should_call_llm`
- `kaggle_research_agent/agents/review_pack.py`
  - `review_pack/manifest.json`
  - `summary.ko.md`
  - `questions.ko.md`
  - `cases.jsonl`
  - `metrics_snapshot.json`

검증:

```powershell
python -B -m unittest discover -s tests -v
```

결과:

- `66 tests`
- `OK`

### 비용 효율 정책 3차 구현

요약:

- LLM 호출/비호출 판단을 `decision_log.jsonl`에 남기는 흐름을 추가했다.
- `request-review` CLI가 diagnosis만 작성하는 것이 아니라 Human Review policy를 평가하고, 필요하면 표준 `review_pack/`까지 생성하도록 연결했다.

주요 변경:

- `kaggle_research_agent/agents/policy_gate.py`
  - `log_llm_decision` 추가
  - `should_call_llm` 결과를 `decision_type: llm_call`로 기록
- `kaggle_research_agent/cli.py`
  - `decide-llm` 명령 추가
  - `request-review` 명령에서 `decide_human_review`와 `prepare_review_pack` 연결
- `tests/test_policy_gate.py`
  - LLM decision logging 테스트 추가
- `tests/test_cli_loop_core.py`
  - `request-review`가 `review_pack/`을 생성하는지 확인
  - `decide-llm`이 decision log를 남기는지 확인

검증:

```powershell
python -B -m unittest discover -s tests -v
```

결과:

- `69 tests`
- `OK`

### Human Review Feedback Loop 4차 구현

요약:

- Human Review가 요청과 review pack 생성에서 끝나지 않고, 사용자 피드백이 review pack, memory, decision log, 다음 실험 계획에 연결되도록 보강했다.

주요 변경:

- `record_user_feedback`
  - `memory/<competition>/user_feedback.jsonl` 유지
  - `experiments/<competition>/<trial_id>/user_review_response.md` 유지
  - `review_pack/human_feedback.md` 생성
  - `review_pack/human_feedback.json` 생성
  - `review_pack/manifest.json`의 `status`를 `feedback_recorded`로 갱신
  - `decision_log.jsonl`에 `decision_type: human_feedback`, `user_input_used: true` 기록
- `propose_next_experiment`
  - source trial의 최신 `user_feedback.jsonl`을 읽음
  - `change_validation` 피드백을 `validation_review` 전략으로 반영
  - `change_model_family`, `prepare_sota_research` 같은 decision도 전략 선택에 반영할 수 있게 함

검증:

```powershell
python -B -m unittest discover -s tests -v
```

결과:

- `72 tests`
- `OK`

### README 최신 workflow 정리

요약:

- README의 오래된 Colab 중심 `Main workflow`를 local-first, policy gate, Human Review, submission approval 흐름에 맞게 갱신했다.
- Current Status에 policy files, execution/LLM decision logging, closed Human Review feedback loop를 반영했다.
- 최신 테스트 기준을 `72 tests OK`로 갱신했다.

### Local failure artifact 5차 구현

요약:

- local 실행 실패 시 `experiments/<competition>/<trial_id>/local_failure.json`과 `local_failure.md`를 저장하도록 했다.
- 실패 artifact에는 command, exit code, failure type, matched pattern, log tail, suggested next action을 남긴다.
- execution policy gate가 `local_failure.json`을 우선 읽고, 없을 때만 `local_run.log` 패턴 매칭으로 fallback하도록 했다.
- resource failure는 Colab 승인 후보로, missing dependency는 dependency fix 후보로 분기하도록 next action을 구분했다.

주요 변경:

- `kaggle_research_agent/agents/experiment_runner.py`
  - `write_local_failure_artifact`
  - `render_local_failure`
- `kaggle_research_agent/agents/policy_gate.py`
  - `classify_local_failure(..., use_artifact=True)`
  - execution decision evidence에 `local_failure_artifact_path` 추가
- `tests/test_experiment_runner.py`
  - local 실패 artifact 생성 테스트 추가
- `tests/test_policy_gate.py`
  - policy gate가 artifact를 우선 사용하는지 테스트 추가

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `74 tests`
- `OK`

### Pipeline Improvement Planner 6차 구현

요약:

- 다음 trial에서 어떤 파이프라인 개선 축을 바꿀지 선택하는 `Pipeline Improvement Planner`를 추가했다.
- 모델 변경은 여러 개선 축 중 하나로만 다루고, validation, preprocessing, feature engineering, augmentation, sampling, loss/metric alignment, hyperparameter, training recipe, post-processing, human review 같은 축을 함께 고려하도록 했다.
- CV/LB 불일치나 leakage 의심이 있으면 validation을 우선하고, model family/architecture/pretraining 변경을 보호 축으로 둔다.
- segment/group/fold/view 등 오류 집중이 있으면 error analysis와 human review를 우선한다.
- 반복 정체나 best prediction과 높은 상관이 있으면 model family와 pretraining strategy를 검토한다.

주요 변경:

- `configs/policies/pipeline_improvement_policy.yaml`
- `docs/policies/pipeline_improvement_policy.ko.md`
- `kaggle_research_agent/agents/pipeline_planner.py`
  - `plan_pipeline_improvement`
  - `pipeline_improvement_plan.json/md` 생성
- `kaggle_research_agent/agents/research_planner.py`
  - `plan-next`가 source trial의 pipeline improvement plan을 evidence로 사용
- `kaggle_research_agent/cli.py`
  - `plan-improvement` 명령 추가
- `tests/test_pipeline_planner.py`
  - validation 우선, error analysis/human review 우선, model family 우선 테스트 추가
- `tests/test_cli_loop_core.py`
  - `plan-improvement` CLI 테스트 추가

검증:

```powershell
python -B -m unittest tests.test_pipeline_planner -v
python -B -m unittest tests.test_cli_loop_core.CliLoopCoreTest.test_plan_improvement_command_creates_pipeline_improvement_plan -v
python -B -m unittest tests.test_research_planner_next_experiment -v
```

### Pipeline Patch Planner 역할 분리

요약:

- `research_planner.py` 안에 있던 `prepare_patch_plan` 책임을 별도 `pipeline_patch_planner.py`로 분리했다.
- Research Planner는 다음 실험의 연구 가설과 `next_experiment.md` 생성에 집중한다.
- Pipeline Patch Planner는 `next_experiment.md`와 source config를 읽어 `config.yaml`, `code_patch_plan.md/json`을 생성한다.
- CLI와 Orchestrator는 새 모듈의 `prepare_patch_plan`을 사용하도록 import 경로를 변경했다.

주요 변경:

- `kaggle_research_agent/agents/pipeline_patch_planner.py` 추가
- `kaggle_research_agent/agents/research_planner.py`에서 patch planning 코드 제거
- `kaggle_research_agent/cli.py` import 변경
- `kaggle_research_agent/agents/orchestrator.py` import 변경
- `tests/test_research_planner_patch_plan.py`가 새 모듈 경로를 검증하도록 변경
- `README.md`의 agent architecture와 workflow를 역할 분리에 맞게 갱신

검증:

```powershell
python -B -m unittest tests.test_research_planner_patch_plan -v
python -B -m unittest tests.test_cli_loop_core.CliLoopCoreTest.test_prepare_patch_command_creates_code_patch_plan tests.test_orchestrator_diagnosis.OrchestratorDiagnosisTest.test_cycle_can_apply_next_patch_and_run_next_trial -v
python -B -m unittest discover -s tests -v
```

결과:

- `78 tests`
- `OK`

### Pipeline Patch Planner 7차 고도화

요약:

- `pipeline_patch_planner.py`가 source trial의 `pipeline_improvement_plan.json`을 읽어 `primary_axis`를 code/config patch plan에 반영하도록 했다.
- 이제 `sampling`, `loss_metric_alignment`, `pretraining_strategy`, `augmentation`, `post_processing` 같은 개선 축이 단순 controlled refinement로 뭉개지지 않고, 각각의 config 변경과 target file, implementation step으로 번역된다.
- `pretraining_strategy`, model-level axis, human review 필요 축은 `requires_user_approval`을 켜도록 했다.

주요 동작:

- `sampling`
  - `training.sampler: balanced`
  - `training.sampling_weight_source: train_labels`
  - dataset target file 추가
- `loss_metric_alignment`
  - `training.loss: metric_aligned`
  - `training.class_weights: auto`
  - `post_processing.threshold_sweep: true`
  - losses/post-processing target file 추가
- `pretraining_strategy`
  - `model.pretraining.mode: partial_finetune`
  - pretrained backbone 승인 확인 step 추가
  - freeze/unfreeze schedule step 추가
- `augmentation`
  - `augmentation.enabled: true`
  - domain-safe light augmentation branch target 추가
- `post_processing`
  - threshold/smoothing post-processing branch target 추가

검증:

```powershell
python -B -m unittest tests.test_research_planner_patch_plan -v
python -B -m unittest tests.test_cli_loop_core.CliLoopCoreTest.test_prepare_patch_command_creates_code_patch_plan tests.test_orchestrator_diagnosis.OrchestratorDiagnosisTest.test_cycle_can_apply_next_patch_and_run_next_trial -v
```

## 2026-05-31 KST

### 자율 연구 루프 core 설계 및 1차 구현

요약:

- `diagnose_trial`, `User Review`, decision log, submission tracking, best trial 표시를 첫 구현 단위로 확정하고 구현했다.
- 실제 Kaggle API와 Code Editing Agent는 다음 구현 단계로 분리했다.
- `cycle`은 metrics가 있는 trial을 평가한 뒤, memory 갱신 전에 diagnosis와 decision log를 먼저 남긴다.

주요 변경:

- `diagnosis_agent.py` 추가
- `decision_logger.py` 추가
- `user_review_agent.py` 추가
- `submission_tracker.py` 추가
- `diagnose`, `request-review`, `record-feedback`, `record-submission` CLI 명령 추가
- `experiments/<competition>/BEST_TRIAL.md`
- `memory/<competition>/best_trial.json`
- `memory/<competition>/decision_log.jsonl`
- `memory/<competition>/user_feedback.jsonl`
- `submissions/<competition>/submission_log.jsonl`

주요 계획/설계 파일:

- `docs/superpowers/specs/2026-05-30-autonomous-research-loop-with-user-review-design.md`
- `docs/superpowers/plans/2026-05-31-autonomous-research-loop-core.md`

검증:

```powershell
python -B -m unittest discover -s tests -v
python -B -m kaggle_research_agent.cli diagnose --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job
python -B -m kaggle_research_agent.cli record-submission --competition demo --trial trial_001 --version-name demo_trial_001_baseline_v01 --submission-file experiments/demo/trial_001/submission.csv --cv-score 0.83 --previous-lb-score 0.80 --previous-rank 120 --submitted-lb-score 0.84 --submitted-rank 90 --objective maximize --notes "Manual leaderboard entry"
```

### Next experiment planner 추가

요약:

- 진단 결과와 제출 결과를 읽어 다음 trial의 연구 방향을 `next_experiment.md`로 남기는 `next_experiment_agent.py`를 추가했다.
- 개선 여지가 남아 있으면 `controlled_refinement`, LB/순위 악화가 있으면 `validation_review`, 실패 누적 또는 전략 상승 신호가 있으면 `model_family_change` 또는 `sota_architecture_attempt`로 전환한다.
- `cycle --next-trial` 옵션으로 평가, 진단, 기억 업데이트 뒤 다음 실험 제안까지 이어갈 수 있게 했다.

주요 변경:

- `kaggle_research_agent/next_experiment_agent.py`
- `plan-next` CLI 명령
- `cycle --next-trial` CLI 옵션
- `tests/test_next_experiment_agent.py`
- `kaggle_research_agent/patch_planner_agent.py`
- `prepare-patch` CLI 명령
- `apply-patch` CLI 명령
- `cycle --prepare-next-patch` CLI 옵션
- `cycle --apply-next-patch`, `--next-run-command` CLI 옵션
- `scripts/demo_train.py`가 config의 `model.type`과 feature 설정을 읽어 metrics에 반영하도록 개선

검증:

```powershell
python -B -m unittest discover -s tests -v
python -B -m kaggle_research_agent.cli plan-next --competition demo --source-trial trial_001 --next-trial trial_002
python -B -m kaggle_research_agent.cli prepare-patch --competition demo --source-trial trial_001 --next-trial trial_002
python -B -m kaggle_research_agent.cli apply-patch --competition demo --trial trial_002 --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job --next-trial trial_002
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job --next-trial trial_002 --prepare-next-patch
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job --next-trial trial_002 --prepare-next-patch --apply-next-patch --next-run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
python -B -m kaggle_research_agent.cli run-local --competition demo --trial trial_002 --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_002 --no-job --next-trial trial_003 --prepare-next-patch
```

### 최소 Code Editing Agent 실행 루프 추가

요약:

- `code_patch_plan.json`을 읽어 target file 존재 여부와 config validation을 확인하는 `code_edit_agent.py`를 추가했다.
- `apply-patch --run-command`로 prepared patch plan을 실제 local run, evaluation, diagnosis까지 이어갈 수 있게 했다.
- `cycle --apply-next-patch --next-run-command ...`로 현재 trial 평가 후 다음 trial 계획, patch plan, 적용, 실행까지 한 번에 진행하는 최소 루프를 검증했다.

검증:

```powershell
python -B -m kaggle_research_agent.cli apply-patch --competition demo --trial trial_003 --run-command "python scripts/demo_train.py --config experiments/demo/trial_003/config.yaml --output experiments/demo/trial_003"
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_003 --no-job --next-trial trial_004 --prepare-next-patch --apply-next-patch --next-run-command "python scripts/demo_train.py --config experiments/demo/trial_004/config.yaml --output experiments/demo/trial_004"
python -B -m unittest discover -s tests -v
```

### 6개 상위 에이전트 물리 통합 리팩터링

요약:

- 세부 `*_agent.py` 모듈을 제거하고 `kaggle_research_agent/agents/` 아래 6개 상위 에이전트 모듈로 물리 통합했다.
- 공개 Python import 경로는 `kaggle_research_agent.agents.*` 기준으로 변경했다.
- CLI 명령 이름과 사용자-facing 동작은 유지했다.
- README는 6개 상위 에이전트와 infrastructure/tool 구분이 보이도록 정리했다.

상위 에이전트:

- `agents/orchestrator.py`
- `agents/research_planner.py`
- `agents/experiment_runner.py`
- `agents/result_analyst.py`
- `agents/memory.py`
- `agents/submission.py`

검증:

```powershell
python -B -m unittest discover -s tests -v
rg --files | rg "_agent\\.py$"
```

### Submission Agent CLI 연결

요약:

- 기존 `Submission Agent`의 `submit_trial` 흐름을 `submit-trial` CLI 명령으로 노출했다.
- 제출 전/후 점수와 순위를 함께 기록하고, `submission_run.md/json`, `submission_result.md`, `VERSION.md`, `BEST_MARKER.md` 생성을 CLI에서 확인할 수 있게 했다.
- `--before-command`, `--submit-command`, `--after-command` 훅은 열어두었지만, 실제 Kaggle API/CLI 제출과 리더보드 polling은 다음 단계로 남겼다.

검증:

```powershell
python -B -m unittest tests.test_cli_loop_core.CliLoopCoreTest.test_submit_trial_command_records_submission_run -v
```

### Submission 안전 게이트 구조 보강

요약:

- 실제 Kaggle 연동 전에 `prepare-submission` CLI를 추가해 `submit_manifest.md/json`을 먼저 생성하도록 했다.
- 이 준비 단계는 제출 로그, `VERSION.md`, `BEST_MARKER.md`를 변경하지 않으므로 leaderboard 제출 전 검토 지점으로 사용할 수 있다.
- 외부 Kaggle CLI 호출을 `kaggle_research_agent/integrations/kaggle_cli.py` adapter로 분리해, 다음 단계의 실제 Kaggle submit/polling 구현이 Submission Agent 기록 규칙을 흔들지 않도록 했다.

검증:

```powershell
python -B -m unittest tests.test_submission_agent.SubmissionAgentTest.test_prepare_submission_writes_manifest_without_marking_best tests.test_submission_agent.SubmissionAgentTest.test_prepare_submission_blocks_when_metrics_are_missing tests.test_cli_loop_core.CliLoopCoreTest.test_prepare_submission_command_creates_manifest_only -v
```

### Kaggle CLI adapter 1차 구체화

요약:

- `kaggle_research_agent/integrations/kaggle_cli.py`에 shell 문자열 대신 인자 리스트 기반 helper를 추가했다.
- Kaggle CLI 설치 확인, 인증 설정 확인, competition submit, leaderboard 조회 명령을 모두 구조화된 결과로 반환하도록 했다.
- 실제 Kaggle 네트워크 호출 없이 fake runner를 주입해 adapter 동작을 검증할 수 있게 했다.

검증:

```powershell
python -B -m unittest tests.test_kaggle_cli_integration -v
```

### Leaderboard 파싱 및 polling adapter 추가

요약:

- Kaggle leaderboard 출력이 CSV 또는 단순 table 형태일 때 target team의 score/rank를 파싱하는 `parse_leaderboard`를 추가했다.
- 제출 직후 leaderboard 반영 지연을 고려해 `poll_leaderboard`를 추가했다.
- timeout 시 score/rank를 `None`으로 유지해, Submission Agent의 best marker가 잘못 갱신되지 않도록 다음 연결 단계의 기반을 마련했다.

검증:

```powershell
python -B -m unittest tests.test_kaggle_cli_integration tests.test_submission_agent -v
```

### `submit-trial` Kaggle adapter 직접 연결

요약:

- `submit-trial`에 Kaggle CLI 직접 제출 옵션을 추가했다.
- 제출 전 `kaggle --version`, `kaggle config view`를 통해 CLI/인증 상태를 확인한다.
- Kaggle submit 후 leaderboard polling이 성공해 score/rank를 찾은 경우에만 `submission_log.jsonl`, `VERSION.md`, `BEST_MARKER.md`를 갱신한다.
- 인증 실패 또는 leaderboard timeout이면 `submission_run.md/json`만 남기고 최고 실험본 표시는 바꾸지 않는다.
- `kaggle.json`은 표준 위치 `C:\Users\ASUS\.kaggle\kaggle.json`으로 이동했다. 키 내용은 출력하지 않았다.

검증:

```powershell
python -B -m unittest tests.test_submission_agent tests.test_cli_loop_core tests.test_kaggle_cli_integration -v
```

### Kaggle competition inspect 기능 추가

요약:

- 대회 링크 또는 slug를 받아 `competition_slug`로 정규화하는 기능을 추가했다.
- `inspect-competition` CLI를 추가해 Kaggle CLI/auth 확인 후 competition file listing을 수집한다.
- 결과는 `competitions/<slug>/competition_inspection.md/json`에 저장된다.
- 이 기능은 6개 에이전트 구조를 늘리지 않고, 외부 Kaggle 정보 수집용 infrastructure/tool로 유지했다.

검증:

```powershell
python -B -m unittest tests.test_kaggle_cli_integration tests.test_competition_inspector tests.test_cli_loop_core.CliLoopCoreTest.test_inspect_competition_command_creates_inspection_files -v
```

### `start-competition` 온보딩 기능 추가

요약:

- 대회 링크 또는 slug를 받아 inspection, workspace 초기화, 기본 문서 작성, 첫 `trial_001` 계획 생성을 한 번에 수행하는 `start-competition` CLI를 추가했다.
- 생성/갱신되는 주요 산출물:
  - `competitions/<slug>/competition_inspection.md/json`
  - `competitions/<slug>/overview.md`
  - `competitions/<slug>/data_notes.md`
  - `memory/<slug>/research_notes.md`
  - `experiments/<slug>/trial_001/plan.md`
  - `experiments/<slug>/trial_001/config.yaml`
- inspection이 인증 실패 등으로 막히면 trial 계획을 만들지 않고 inspection 결과만 남긴다.

검증:

```powershell
python -B -m unittest tests.test_competition_onboarding tests.test_competition_inspector tests.test_cli_loop_core tests.test_kaggle_cli_integration -v
```

### `run-auto-loop` 안전 자동 루프 추가

요약:

- Orchestrator Agent에 `run_auto_research_loop`를 추가했다.
- 여러 trial을 순차적으로 돌리면서 `run_cycle`, next experiment, patch plan 생성을 재사용한다.
- 기본 제출 정책은 `never`이며, 이번 단계에서는 제출을 자동 실행하지 않는다.
- `stop_no_improvement`로 개선 없는 trial 반복 시 중단할 수 있게 했다.
- CLI `run-auto-loop`를 추가했다.

검증:

```powershell
python -B -m unittest tests.test_orchestrator_diagnosis tests.test_cli_loop_core -v
```

## 2026-06-01 KST

### Patch Validator 8차 구현

요약:

- 준비된 `code_patch_plan.json`을 실제 적용하기 전에 검사하는 `Patch Validator Agent`를 추가했다.
- 이제 patch 적용은 단순히 계획 파일을 읽는 단계에서 끝나지 않고, target file 존재 여부, 생성된 config 유효성, validation command 존재 여부, user approval 필요 여부, protected axis 침범 여부를 먼저 확인한다.
- `pretraining_strategy`처럼 비용이 크거나 사용자의 명시 승인이 필요한 변경은 `--user-approved` 없이는 blocked 처리된다.
- submission artifact를 patch target으로 삼는 계획은 금지해서 연구 코드 수정과 leaderboard 제출 기록이 섞이지 않도록 했다.
- `apply-patch` 실행 전에도 같은 검증을 거치도록 연결해, CLI 단독 검증과 실제 patch 적용 경로가 같은 안전 게이트를 공유한다.

주요 변경:

- `kaggle_research_agent/agents/patch_validator.py` 추가
  - `validate_patch_plan`
  - `patch_validation.json`
  - `patch_validation.md`
- `kaggle_research_agent/cli.py`
  - `validate-patch` 명령 추가
  - `--user-approved` 옵션 추가
- `kaggle_research_agent/agents/experiment_runner.py`
  - `apply_patch_plan()` 시작 전에 patch validation 실행
  - blocked 사유를 `code_edit_result.md/json`에 함께 기록
- `tests/test_patch_validator.py` 추가
- 기존 CLI / experiment runner 테스트 fixture에 `validation_commands` 반영

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `85 tests`
- `OK`

### Patch Validation 8.5차 정리

요약:

- 8차에서 추가한 Patch Validator를 단일 안전 게이트로 정리했다.
- `apply_patch_plan()`이 target/config 검증을 직접 반복하지 않고, `validate_patch_plan()` 결과의 `missing_targets`, `config_errors`, `issues`를 그대로 사용하도록 변경했다.
- patch validation 결과를 `memory/<competition>/decision_log.jsonl`에도 `decision_type: patch_validation`으로 기록하도록 했다.
- 이제 patch plan이 `ready`였는지 `blocked`였는지, 어떤 issue 때문에 막혔는지, 어떤 pipeline axis와 target file을 검사했는지 memory에서 추적할 수 있다.

주요 변경:

- `kaggle_research_agent/agents/patch_validator.py`
  - validation 결과 decision log 기록 추가
- `kaggle_research_agent/agents/experiment_runner.py`
  - 중복 target/config validation 제거
  - Patch Validator 결과를 단일 gate로 사용
- `tests/test_patch_validator.py`
  - patch validation decision log 테스트 추가
- `tests/test_experiment_runner.py`
  - `apply_patch_plan()`이 Patch Validator 결과를 단일 gate로 사용하는지 테스트 추가

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `87 tests`
- `OK`

### Coding Handoff Agent 9차 구현

요약:

- 검증된 `code_patch_plan.json`을 실제 코딩 작업자에게 넘기기 위한 표준 handoff 단계를 추가했다.
- 이번 단계는 Codex/API를 직접 호출하는 구현이 아니라, 안전하게 넘길 수 있는 `coding_agent_request.md`와 `coding_handoff.json`을 생성하는 단계다.
- patch validation이 `ready`인 경우에만 코딩 요청서를 생성한다.
- patch validation이 `blocked`이면 `coding_handoff.json`만 남기고, 실제 코딩 요청서는 생성하지 않는다.
- handoff 결과도 `memory/<competition>/decision_log.jsonl`에 `decision_type: coding_handoff`로 기록해 추적 가능하게 했다.

주요 산출물:

- `experiments/<competition>/<trial_id>/coding_handoff.json`
  - handoff status
  - strategy / pipeline_axis
  - target files
  - config changes
  - implementation steps
  - validation commands
  - blocking issues
- `experiments/<competition>/<trial_id>/coding_agent_request.md`
  - 코딩 에이전트에게 넘길 사람이 읽기 쉬운 작업 요청서
  - objective, target files, required changes, implementation steps, guardrails, validation commands 포함

주요 변경:

- `kaggle_research_agent/agents/coding_handoff.py` 추가
  - `prepare_coding_handoff`
  - `render_coding_agent_request`
- `kaggle_research_agent/cli.py`
  - `prepare-handoff` 명령 추가
  - `--user-approved` 옵션 지원
- `tests/test_coding_handoff.py` 추가
- `tests/test_cli_loop_core.py`
  - `prepare-handoff` CLI 테스트 추가
- `README.md`
  - Coding Handoff Agent와 최신 workflow 반영

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `90 tests`
- `OK`

## 2026-06-02 KST

### Competition Data Onboarding 10차 구현

요약:

- Kaggle 대회를 시작한 뒤 실제 데이터 구조를 이해하기 위한 `Data Onboarding Agent`를 추가했다.
- 로컬 데이터는 `data/<competition>/` 아래에 둔다.
- 로컬 데이터가 있으면 CSV schema를 읽어 train/test/sample_submission 역할, target 후보, id 후보, task type 후보를 추정한다.
- 로컬 데이터가 아직 없으면 `competition_inspection.json`의 파일 목록을 바탕으로 제한적인 profile을 만들고, data download가 필요하다고 표시한다.
- 사람이 확인해야 할 애매한 부분은 `human_review_questions`로 남긴다.

주요 산출물:

- `competitions/<competition>/data_profile.json`
  - status
  - local_data_dir
  - task_type
  - target_candidates
  - id_candidates
  - file role/schema profile
  - human_review_questions
  - next_steps
- `competitions/<competition>/data_profile.md`
  - 사람이 읽기 쉬운 데이터 프로필 요약

주요 변경:

- `kaggle_research_agent/data_onboarding.py` 추가
  - `profile_competition_data`
  - `render_data_profile`
- `kaggle_research_agent/paths.py`
  - `data_dir`
  - `competition_data_dir`
- `kaggle_research_agent/cli.py`
  - `profile-data` 명령 추가
- `kaggle_research_agent/competition_onboarding.py`
  - `start-competition` 이후 `data_profile.md/json` 생성 연결
- `tests/test_data_onboarding.py` 추가
- `tests/test_cli_loop_core.py`
  - `profile-data` CLI 테스트 추가
- `tests/test_competition_onboarding.py`
  - `start-competition` data profile 생성 테스트 추가
- `README.md`
  - Data Onboarding Agent, `profile-data`, `data/<competition>/` 반영

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `93 tests`
- `OK`

### Baseline Pipeline Generator 11차 구현

요약:

- `data_profile.md/json`을 바탕으로 첫 로컬 baseline 학습/제출 파이프라인을 생성하는 `Baseline Generator Agent`를 추가했다.
- 현재 11차 범위는 tabular CSV 대회에 대한 최소 baseline이다.
- 생성된 baseline은 복잡한 모델이 아니라 majority-class baseline이며, 목적은 성능 최적화가 아니라 로컬 실행, metrics 생성, submission 형식 검증이다.
- 데이터가 아직 준비되지 않았거나 target column을 찾지 못하면 `blocked` 상태로 기록하고 script를 만들지 않는다.

주요 산출물:

- `experiments/<competition>/<trial_id>/baseline_pipeline.json`
  - status
  - task_type
  - target_column
  - id_columns
  - run_command
  - expected outputs
  - blocking issues
- `experiments/<competition>/<trial_id>/baseline_plan.md`
  - 사람이 읽기 쉬운 baseline 실행 계획
- `experiments/<competition>/<trial_id>/baseline_train.py`
  - stdlib 기반 tabular majority-class baseline
  - 실행 시 `metrics.json`과 `submission.csv` 생성

주요 변경:

- `kaggle_research_agent/baseline_generator.py` 추가
  - `generate_baseline_pipeline`
  - `render_baseline_plan`
- `kaggle_research_agent/cli.py`
  - `generate-baseline` 명령 추가
- `tests/test_baseline_generator.py` 추가
  - baseline pipeline 생성 테스트
  - 생성된 baseline script 실행 테스트
  - data profile 미준비 상태 blocked 테스트
- `tests/test_cli_loop_core.py`
  - `generate-baseline` CLI 테스트 추가
- `README.md`
  - Baseline Generator Agent와 `generate-baseline` 흐름 반영

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `97 tests`
- `OK`

### 중복 설계 정리

요약:

- 12차 Model Candidate Advisor로 넘어가기 전에 기존 설계 중 중복 책임이 생길 수 있는 부분을 정리했다.
- 원칙은 “한 판단은 한 모듈이 책임지고, 다음 단계는 그 산출물을 읽는다”로 정했다.

정리한 경계:

- `Pipeline Patch Planner`
  - config/code patch plan 생성만 담당한다.
  - 실행 전 최종 config validation 책임은 갖지 않는다.
  - `validation_errors` 필드를 제거해 Patch Validator와 책임이 겹치지 않게 했다.
- `Patch Validator`
  - patch 적용 전 최종 안전 gate를 담당한다.
  - target file, config validity, approval, protected axis, submission artifact 검사를 여기로 집중한다.
- `Coding Handoff Agent`
  - 이미 존재하는 `patch_validation.json`이 있으면 재검증하지 않고 재사용한다.
  - 불필요한 `patch_validation` decision log 중복 기록을 줄였다.
- `Data Onboarding Agent`
  - data schema와 target/task 후보 판단을 담당한다.
- `Baseline Generator Agent`
  - 기존 `data_profile.json` snapshot을 우선 사용한다.
  - data schema를 매번 다시 판단하지 않도록 하여 Data Onboarding과 책임을 분리했다.

주요 변경:

- `kaggle_research_agent/agents/coding_handoff.py`
  - 기존 `patch_validation.json` 재사용 로직 추가
- `kaggle_research_agent/baseline_generator.py`
  - 기존 `data_profile.json` snapshot 우선 사용
- `kaggle_research_agent/agents/pipeline_patch_planner.py`
  - `validation_errors` 생성 제거
- `tests/test_coding_handoff.py`
  - patch validation decision log 중복 방지 테스트 추가
- `tests/test_baseline_generator.py`
  - existing data profile snapshot 사용 테스트 추가
- `tests/test_research_planner_patch_plan.py`
  - patch plan이 validation 책임을 갖지 않는지 테스트 추가

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `99 tests`
- `OK`

### Model Candidate Advisor 12차 구현

요약:

- Planning Agent 내부 도구로 `Model Candidate Advisor`를 추가했다.
- 목적은 대회/task/data profile/최근 trial evidence를 보고 다음에 검토할 모델 계열과 학습 전략을 구조화하는 것이다.
- 이 도구는 모델 변경을 기본값으로 만들지 않는다.
- validation 문제가 의심되거나 `pipeline_improvement_plan.json`에서 모델 변경이 protected axis로 지정되면 모델 변경을 보류한다.
- DINOv2 같은 특정 사례에 과적합하지 않고, task type별 일반 후보군과 guardrail을 제안한다.

주요 산출물:

- `experiments/<competition>/<trial>/model_candidates.json`
- `experiments/<competition>/<trial>/model_candidates.md`

주요 변경:

- `kaggle_research_agent/agents/model_advisor.py` 추가
  - `advise_model_candidates()` 추가
  - tabular/image/text/unknown task type별 후보 모델군 제안
  - pretrained fine-tuning vs train-from-scratch 판단
  - validation 보호축일 때 `defer_model_change`로 분기
- `kaggle_research_agent/cli.py`
  - `advise-models` CLI 명령 추가
- `tests/test_model_advisor.py` 추가
  - tabular baseline-friendly 후보 테스트
  - image pretrained vision 후보 테스트
  - validation primary axis에서 model change 보호 테스트
- `tests/test_cli_loop_core.py`
  - `advise-models` CLI 산출물 생성 테스트 추가
- `README.md`
  - Planning Agent 내부 도구로 `Model Candidate Advisor` 반영
  - `advise-models` 사용법 추가

설계 원칙:

- 모델 선택은 pipeline 개선축 중 하나일 뿐이다.
- validation, data split, leakage, human review 문제가 우선이면 모델 변경을 미룬다.
- 최신 모델 웹 검색은 이번 범위에 포함하지 않고, 이후 별도 research/search 단계로 분리한다.
- LangGraph runtime은 아직 적용하지 않고, 나중에 node로 옮기기 쉬운 함수형 내부 도구로 유지한다.

### Coding Request Schema 13차 구현

요약:

- 코드 작성까지 포함한 자동 루프를 연결하기 전에 Codex/API 코딩 작업자에게 전달할 요청 계약을 표준화했다.
- 코딩 작업자의 책임은 코드 수정으로 제한하고, 학습 실행과 Kaggle 제출은 다음 단계로 분리했다.
- 새 파일을 생성해야 하는 patch plan이 기존 Patch Validator에서 차단되는 문제를 발견하고, 선언된 새 파일만 허용하는 `create_files` 계약을 추가했다.

주요 변경:

- `docs/policies/coding_request_schema.ko.md` 추가
  - 입력 context 파일
  - 허용 쓰기 파일
  - 선언된 새 파일
  - 금지 경로
  - 실행 제약
  - `coding_result.json/md` 결과 계약 정의
- `kaggle_research_agent/agents/coding_handoff.py`
  - `schema_version`, `request_id`, `objective` 추가
  - `context_files`, `allowed_write_files`, `create_files`, `forbidden_paths` 추가
  - `execution_constraints`, `required_output` 추가
  - Markdown 요청서에 입력/쓰기/금지/결과 계약 섹션 추가
- `kaggle_research_agent/agents/pipeline_patch_planner.py`
  - 실제로 존재하지 않는 신규 코드 모듈을 `create_files`에 선언
- `kaggle_research_agent/agents/patch_validator.py`
  - `create_files`에 선언된 누락 target만 허용
  - 선언되지 않은 누락 target은 기존처럼 차단
- 테스트
  - coding handoff schema 필드와 Markdown 섹션 검증
  - 선언된 새 파일 target 허용 검증
  - sampling patch plan의 신규 dataset 모듈 선언 검증

설계 원칙:

- 코딩 작업자는 `allowed_write_files` 밖을 수정하지 않는다.
- 새 파일은 `create_files`에 명시된 경우에만 생성한다.
- 코딩 단계에서는 학습과 제출을 실행하지 않는다.
- 코드 수정 결과는 다음 자동 검증 단계가 읽을 수 있는 구조로 남긴다.

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `104 tests`
- `OK`

### LangGraph One-Trial Cycle 14차 구현

요약:

- 기존 Python 기반 agent/tool 함수들을 제거하지 않고, LangGraph `StateGraph` orchestration layer를 추가했다.
- 이번 범위는 전체 auto-loop 교체가 아니라 보수적인 one-trial `cycle` 흐름을 LangGraph node/edge로 감싸는 것이다.
- 기존 `cycle` CLI는 유지하고, 새 `run-graph-cycle` CLI를 추가해 비교 가능한 전환 경로를 만들었다.

주요 변경:

- `kaggle_research_agent/graph/` 패키지 추가
  - `state.py`: graph state schema
  - `nodes.py`: 기존 Python 도구를 감싼 node 함수와 routing 함수
  - `research_graph.py`: `StateGraph`, `START`, `END`, conditional edge wiring
- `kaggle_research_agent/cli.py`
  - `run-graph-cycle` 명령 추가
- `requirements.txt`
  - `langgraph>=1.0.5` 추가
- 테스트
  - 완료된 trial에서 graph가 evaluate/diagnose/remember까지 실행하는지 검증
  - metrics가 없을 때 execution decision 후 job request를 생성하는지 검증
  - CLI `run-graph-cycle` 검증

설계 원칙:

- Python 함수는 실제 작업 단위로 유지한다.
- LangGraph는 순서, 분기, 상태 전달만 담당한다.
- 한 번에 전체 `run-auto-loop`를 교체하지 않고 one-trial cycle부터 적용한다.
- 향후 Human Review, Code Writing Agent, submission approval branch를 graph node로 확장한다.

### 5개 Top-level Agent 구조로 공식 아키텍처 재정렬

요약:

- 초기 이미지의 구조에 맞춰 공식 프로젝트 설명을 5개 top-level agent 구조로 정리했다.
- 코드 파일을 물리적으로 합치지는 않았다.
- 기존의 작은 모듈들은 각 top-level agent가 사용하는 internal tool/module로 재분류했다.
- 이렇게 해서 기능을 버리지 않으면서도 프로젝트가 10개 이상의 별도 에이전트처럼 보이는 문제를 줄였다.

공식 5개 agent:

- `Knowledge and Memory Agent`
  - competition inspection
  - data profiling
  - research notes / rules / trial memory
  - review packs and user feedback
- `Planning Agent`
  - trial planning
  - pipeline improvement planning
  - baseline generation planning
  - patch planning
  - coding handoff preparation
- `Training and Execution Agent`
  - local / Colab job creation
  - local run execution
  - patch application
- `Evaluation and Decision Agent`
  - metric evaluation
  - diagnosis
  - execution / LLM / human-review policy gates
  - patch validation
  - submission approval and result recording
- `Feedback and Orchestration Agent`
  - cycle
  - run-auto-loop
  - decision log coordination
  - memory update and next-trial feedback loop

주요 변경:

- `README.md`
  - `What is included`를 5개 top-level agent 기준으로 재작성
  - `Current Status`의 세부 agent 명칭을 internal tool 표현으로 변경
  - `Agent Architecture`를 5-agent 구조로 재작성
  - `kaggle_research_agent/agents/` 설명을 top-level agent module이 아니라 internal tools로 정정

검증:

```powershell
python -B -m compileall -q kaggle_research_agent
python -B -m unittest discover -s tests -v
```

결과:

- `99 tests`
- `OK`

### Coding Result Validator 15차 구현

요약:

- Codex/API 기반 코딩 작업자가 실제로 붙기 전에, 코딩 결과물을 그대로 신뢰하지 않고 `coding_handoff.json` 계약에 맞는지 검사하는 결과 검증 게이트를 추가했다.
- 이번 단계는 외부 API 호출이 아니라 `coding_result.json/md` 산출물을 검증하는 rule-based harness다.
- 실제 코딩 에이전트 연결 전에도 전체 루프의 모양을 테스트할 수 있도록 `write-code-dry-run` 명령을 추가했다.

주요 변경:

- `kaggle_research_agent/agents/coding_result_validator.py` 추가
  - `validate_coding_result`
  - `create_dry_run_coding_result`
  - `render_coding_result_validation`
- `validate-coding-result` CLI 추가
  - 필수 필드, status 값, 변경 파일 범위, forbidden path 접촉 여부 검증
  - 결과를 `coding_result_validation.json/md`에 저장
  - `decision_log.jsonl`에 `decision_type: coding_result_validation` 기록
- `write-code-dry-run` CLI 추가
  - 외부 API 호출 없이 `blocked` 상태의 `coding_result.json/md` placeholder 생성
  - 실제 코드 수정은 수행하지 않음
- `docs/policies/coding_request_schema.ko.md`
  - 결과 검증 단계와 저장 파일 명시
- `README.md`
  - Current Status, CLI flow, Agent Architecture, 명령 예시에 15차 반영
- 테스트 추가
  - 정상 completed 결과 승인
  - 허용 범위 밖 파일 수정 차단
  - 필수 필드/status 오류 차단
  - dry-run placeholder 생성
  - CLI `write-code-dry-run`, `validate-coding-result` 검증

설계 원칙:

- 코드 작성은 이후 Codex/API worker가 맡더라도, 결과 검증은 가벼운 Python rule gate로 처리한다.
- `allowed_write_files`와 `create_files` 밖의 변경은 차단한다.
- `data/`, `submissions/`, `submission.csv`, `metrics.json` 같은 학습/제출 산출물은 코딩 단계에서 수정하지 않는다.
- 이 단계는 기존 5개 top-level agent 구조를 늘리지 않고, Evaluation and Decision Agent의 내부 검증 도구로 둔다.

### Code Writer Adapter 16차 구현

요약:

- `coding_handoff.json`을 실제 코딩 모델/API 또는 mock client에 전달하는 Code Writer Adapter를 추가했다.
- 모델/API가 로컬 파일을 직접 수정하는 구조가 아니라, JSON `file_updates`를 반환하고 Python adapter가 허용된 파일만 쓰는 구조로 설계했다.
- 실제 OpenAI API 호출은 `--allow-api`가 있을 때만 가능하게 했고, 테스트와 로컬 검증은 `--mock-response-file`로 비용 없이 수행할 수 있게 했다.

주요 변경:

- `kaggle_research_agent/agents/code_writer_adapter.py` 추가
  - `run_code_writer`
  - `build_code_writer_payload`
  - `OpenAIResponsesClient`
  - `FileResponseClient`
- `run-code-writer` CLI 추가
  - `--mock-response-file`로 외부 API 없이 adapter 경로 검증
  - `--allow-api`가 있을 때만 실제 API client 사용
  - `--model` 기본값은 `gpt-5`
- `configs/policies/token_policy.yaml`, `kaggle_research_agent/policies.py`
  - `code_writing` LLM 호출 사유 추가
- `coding_result_validator.py`
  - `file_updates.path`도 `allowed_write_files` / `create_files` / `forbidden_paths` 기준으로 검증
- `README.md`, `docs/policies/coding_request_schema.ko.md`
  - Code Writer Adapter 사용법과 안전 원칙 반영

설계 원칙:

- API 호출은 token policy와 명시 플래그를 모두 통과해야 한다.
- 모델이 반환한 경로가 허용 범위를 벗어나면 파일을 쓰기 전에 차단한다.
- adapter가 파일을 쓴 뒤에는 즉시 `validate-coding-result`를 실행한다.
- 이번 구현은 5개 top-level agent 구조를 늘리지 않고, Training/Execution Agent의 내부 실행 도구와 Evaluation/Decision Agent의 검증 게이트를 연결한다.

공식 API 참고:

- OpenAI Responses API는 공식 문서 기준 `POST https://api.openai.com/v1/responses`를 사용한다.

### Validation Command Runner 17차 구현

요약:

- accepted 된 `coding_result_validation.json` 이후에 handoff의 `validation_commands`를 실행하는 검증 runner를 추가했다.
- Code Writer Adapter에 `--run-validation-commands` 옵션을 연결해, mock/API 코딩 결과가 accepted일 때 검증 명령까지 이어서 실행할 수 있게 했다.
- accepted가 아닌 결과에서는 명령을 실행하지 않고 `blocked`로 기록한다.

주요 변경:

- `kaggle_research_agent/agents/validation_command_runner.py` 추가
  - `run_validation_commands`
  - `render_validation_run`
- `run-validation-commands` CLI 추가
  - `coding_result_validation.json`이 accepted일 때만 실행
  - 결과를 `validation_run.json/md`에 저장
  - 각 명령 로그를 `validation_command_001.log` 형식으로 저장
  - `decision_log.jsonl`에 `decision_type: validation_commands` 기록
- `run-code-writer --run-validation-commands` 옵션 추가
  - code writer 결과가 accepted인 경우에만 validation command runner 실행
- `README.md`, `docs/policies/coding_request_schema.ko.md`
  - 17차 흐름과 명령 예시 반영

설계 원칙:

- 코드 작성 결과 검증과 학습 실행은 분리한다.
- validation command는 코드 수정이 안전하게 accepted 된 뒤에만 실행한다.
- 명령 stdout/stderr는 별도 로그 파일로 남겨 이후 diagnosis나 human review 입력으로 사용할 수 있게 한다.
- 이번 단계는 5개 top-level agent를 늘리지 않고, Training/Execution Agent의 내부 실행 도구로 둔다.

### Post-Validation Executor 18차 구현

요약:

- `validation_run.status == passed` 이후에만 실제 trial 실행 단계로 넘어가는 실행 게이트를 추가했다.
- 기존 `execution_policy`와 `decide_execution`을 재사용해 local-first job creation 또는 명시적 local run으로 연결한다.
- validation이 failed/blocked이면 job 생성과 local 실행을 모두 차단한다.

주요 변경:

- `kaggle_research_agent/agents/post_validation_executor.py` 추가
  - `run_after_validation`
  - `render_post_validation_execution`
- `run-after-validation` CLI 추가
  - `--run-command`
  - `--run-now`
  - `--backend local|colab`
- 결과 파일 추가
  - `post_validation_execution.json`
  - `post_validation_execution.md`
- `decision_log.jsonl`
  - `decision_type: post_validation_execution` 기록
- `README.md`, `docs/policies/coding_request_schema.ko.md`
  - post-validation execution gate 흐름과 명령 예시 반영

설계 원칙:

- 코드 수정 검증과 실제 trial 실행은 분리한다.
- `validation_run.status`가 `passed`일 때만 실행 정책을 평가한다.
- 기본은 local-first job creation이며, 즉시 실행은 `--run-now`가 있을 때만 수행한다.
- 새 top-level agent를 만들지 않고 Training/Execution Agent의 내부 게이트로 둔다.

### Safe Execution Chain 19차 구현

요약:

- `run-code-writer -> validate-coding-result -> run-validation-commands -> run-after-validation` 흐름을 한 번에 실행하는 guarded chain을 추가했다.
- 각 단계는 새 로직으로 우회하지 않고, 16~18차에서 만든 기존 gate를 그대로 사용한다.
- 중간 단계가 실패하면 즉시 중단하고 다음 실행 단계로 넘어가지 않는다.

주요 변경:

- `kaggle_research_agent/agents/safe_execution_chain.py` 추가
  - `run_safe_execution_chain`
  - `render_safe_execution_chain`
- `run-safe-execution-chain` CLI 추가
  - `--mock-response-file`
  - `--allow-api`
  - `--run-command`
  - `--run-now`
  - `--backend local|colab`
- 결과 파일 추가
  - `safe_execution_chain.json`
  - `safe_execution_chain.md`
- `decision_log.jsonl`
  - `decision_type: safe_execution_chain` 기록
- `README.md`, `docs/policies/coding_request_schema.ko.md`
  - safe execution chain 흐름과 명령 예시 반영

설계 원칙:

- 자동 체인은 편의 wrapper일 뿐이며, 개별 안전 gate를 생략하지 않는다.
- code writer가 accepted가 아니면 validation command를 실행하지 않는다.
- validation command가 passed가 아니면 local run/job creation으로 가지 않는다.
- 새 top-level agent를 추가하지 않고 Training/Execution Agent 내부 chain으로 둔다.

## 2026-06-05 KST

### Safe Execution Chain 20차 오케스트레이션 연결

요약:

- 19차에서 만든 `run-safe-execution-chain`을 기존 one-trial cycle에 선택적으로 연결했다.
- 기본 `cycle`과 `run-graph-cycle` 동작은 유지하고, `--run-safe-chain`을 명시했을 때만 안전 체인을 실행하도록 했다.
- 코드 작성, coding result validation, validation commands, post-validation execution gate를 모두 재사용하므로 중복 gate나 우회 경로를 만들지 않았다.

주요 변경:

- `run_cycle()`에 `run_safe_chain`, `safe_chain_mock_response_file`, `safe_chain_allow_api`, `safe_chain_model` 옵션 추가
- LangGraph `StateGraph`에 `safe_chain` 노드와 조건부 edge 추가
- `cycle` CLI에 `--run-safe-chain`, `--mock-response-file`, `--safe-chain-model`, `--safe-chain-allow-api` 추가
- `run-graph-cycle` CLI에도 동일 옵션 추가
- README와 coding request policy에 cycle 내부 안전 체인 실행 원칙 반영

검증:

```powershell
python -B -m unittest tests.test_orchestrator_diagnosis.OrchestratorDiagnosisTest.test_cycle_can_run_safe_execution_chain_when_requested -v
python -B -m unittest tests.test_research_graph.ResearchGraphTest.test_graph_cycle_can_run_safe_execution_chain_when_requested -v
python -B -m unittest tests.test_cli_loop_core.CliLoopCoreTest.test_cycle_command_can_run_safe_execution_chain -v
```

결과:

- 오케스트레이터 cycle에서 `safe_execution_chain.json`과 job artifact 생성 확인
- LangGraph cycle에서 동일한 안전 체인 분기 확인
- CLI `cycle --run-safe-chain` 경로 확인

## 2026-06-05 KST

### ETRI Human Understanding 21차 DACON 온보딩

요약:

- 사용자가 제공한 ETRI Human Understanding handoff 문서를 읽고, 이 대회가 Kaggle이 아니라 DACON 계열 외부 대회임을 반영했다.
- Kaggle CLI 제출/리더보드 polling을 사용하지 않는 `manual_external` 운영 방식으로 competition memory를 구성했다.
- V11, V15, V16의 핵심 trial 기록을 Research Agent가 읽을 수 있는 state, notes, rules, trial_index, metrics/config/plan artifact로 이식했다.

주요 변경:

- `competitions/etri_human_understanding/`
  - `state.yaml`
  - `overview.md`
  - `data_notes.md`
  - `metric.md`
  - `data_profile.json`
- `memory/etri_human_understanding/`
  - `research_notes.md`
  - `rules.md`
  - `trial_index.jsonl`
  - `best_trial.json`
  - `decision_log.jsonl`
- `experiments/etri_human_understanding/`
  - `trial_v11_public_baseline`
  - `trial_v15_subject_temporal_deviation`
  - `trial_v16_causal_rolling_baseline`
- `configs/etri_human_understanding/allowed_space.yaml`
- ETRI 온보딩 회귀 테스트 추가

핵심 기록:

- V11은 Public LB 약 `0.5984`의 신뢰 baseline으로 기록했다.
- V15는 local CV 개선에도 Public LB `0.5994936146`으로 악화되어 subject personalization 위험 사례로 기록했다.
- V16은 Subject-hole `0.584915`, Tail `0.596018`의 local best이지만 Public LB 미확인 상태로 기록했다.
- 규칙에 `random KFold 금지`, `Subject-hole/Tail 필수`, `Kaggle CLI 사용 금지`, `V11 대비 target별 변화량 추적`을 반영했다.

검증:

```powershell
python -B -m unittest tests.test_etri_onboarding.EtriOnboardingTest.test_etri_competition_onboarding_files_exist_and_preserve_key_context -v
```

다음 후보:

- V16 safe Public LB가 이미 제출되었는지 확인하고 `record-submission` 또는 submission metadata로 기록
- V17 Target Chain Stacking 계획 생성
- DACON/external submission adapter를 별도 기능으로 추가할지 검토

## 2026-06-05 KST

### Research Operating Protocol 22차 구현

요약:

- 특정 ETRI 대회에만 맞춘 기능이 아니라, 어떤 대회/데이터가 들어와도 에이전트가 같은 연구 방식으로 사고하도록 Research Operating Protocol을 추가했다.
- 핵심 흐름은 `Current State -> Evidence -> Risk -> Candidate Actions -> Recommended Next Trial -> Do Not Change -> Need User Check -> Execution Plan`이다.
- 다음 실험을 바로 코드 수정으로 보내지 않고, safe/main/aggressive 후보와 리스크를 먼저 정리하도록 했다.

주요 변경:

- `kaggle_research_agent/agents/research_protocol.py` 추가
  - `build_research_protocol`
  - `render_research_protocol`
- `research-protocol` CLI 추가
- `configs/policies/research_operating_policy.yaml` 추가
- `docs/policies/research_operating_protocol.ko.md` 추가
- `plan-next`가 source trial에 metrics가 있을 경우 research protocol을 생성하고, next experiment 문서에 protocol risk/user check/do-not-change 정보를 포함하도록 연결
- ETRI V16 trial에 실제 protocol artifact 생성
  - `research_protocol.json`
  - `research_protocol.md`

프로토콜 판단 예시:

- local CV가 좋아졌지만 LB가 나빠지면 `validation_review`를 우선한다.
- local best인데 Public/외부 leaderboard evidence가 없으면 `safe_submission_or_holdout_confirmation`을 우선한다.
- 작은 데이터/적은 subject/group, validation suspected, 외부 수동 제출 대회는 risk flag로 남긴다.
- local-only evidence만으로 public baseline을 대체하지 않는다.

검증:

```powershell
python -B -m unittest tests.test_research_protocol -v
python -B -m unittest tests.test_research_planner_next_experiment -v
python -B -m kaggle_research_agent.cli research-protocol --competition etri_human_understanding --trial trial_v16_causal_rolling_baseline --next-trial trial_v17_target_chain_stacking
```

ETRI V16 결과:

- risk: `medium`
- strategy: `safe_submission_or_holdout_confirmation`
- 이유: local best지만 Public LB가 없고, validation suspected/small data/manual external submission 리스크가 있음

## 2026-06-09 KST

### 2차 진행: LLM 호출 수 자동 집계 23차 구현

요약:

- 토큰 비용 폭주를 줄이기 위한 1차 수정으로, `decision_log.jsonl` 기반 LLM 호출 수 자동 집계를 추가했다.
- 기존 CLI 인자로 `--trial-llm-calls`, `--strategy-calls-today`를 직접 넘기는 방식은 유지하되, 값을 생략하면 memory의 decision log에서 자동 계산한다.
- code writer / safe execution chain도 같은 token policy gate를 사용하도록 연결했다.

주요 변경:

- `kaggle_research_agent/agents/policy_gate.py`
  - `count_llm_calls_from_decision_log` 추가
  - `should_call_llm`이 competition/trial_id를 받으면 자동 집계 사용
  - `log_llm_decision`이 자동 집계 결과를 evidence에 기록
- `kaggle_research_agent/agents/code_writer_adapter.py`
  - code writing LLM 호출 전 자동 집계 기반 token gate 사용
- `kaggle_research_agent/agents/safe_execution_chain.py`
  - safe chain의 LLM call counter 인자를 optional로 변경
- `kaggle_research_agent/cli.py`
  - LLM call counter CLI 기본값을 `None`으로 변경하여 생략 시 자동 집계 활성화
- `tests/test_policy_gate.py`
  - decision log 자동 집계 테스트 추가
  - 수동 카운터 override 테스트 추가
  - code writer token decision 집계 테스트 추가

검증:

```powershell
python -B -m pytest tests/test_policy_gate.py -q
python -B -m pytest -q
```

결과:

- `tests/test_policy_gate.py`: 11 passed
- 전체 테스트: 140 passed
- `.pytest_cache` 쓰기 권한 warning 2개가 있었지만 테스트 실패는 없음

효과:

- trial별 LLM 호출 수와 일일 strategy call 수를 사용자가 매번 직접 입력하지 않아도 된다.
- 반복 연구 루프에서 token policy가 실제 memory 기록을 기준으로 동작할 수 있게 되었다.
- 기존 수동 입력 방식은 유지되어, 필요한 경우 명시 카운터로 override 가능하다.

### 2차 진행: Trial별 Token Usage 기록 24차 구현

요약:

- 토큰 절감 기능을 무리하게 더 넣기 전에, 실제 API 응답의 token usage를 관찰할 수 있는 계량 장치를 추가했다.
- code writer API 응답에 `usage` 필드가 있으면 `memory/<competition>/token_usage.jsonl`에 기록한다.
- 응답에 usage가 없는 경우 기존 흐름은 그대로 유지된다.

주요 변경:

- `kaggle_research_agent/agents/memory.py`
  - `log_token_usage` 추가
  - `normalize_token_usage` 추가
  - OpenAI Responses 형식(`input_tokens`, `output_tokens`, `total_tokens`)과 Chat Completions식 형식(`prompt_tokens`, `completion_tokens`)을 모두 정규화
- `kaggle_research_agent/agents/code_writer_adapter.py`
  - code writer API 응답 저장 후 usage가 있으면 token usage log 기록
  - code writer decision evidence에 `token_usage` 추가
- `tests/test_code_writer_adapter.py`
  - mock API response usage가 `token_usage.jsonl`에 기록되는지 검증
- `tests/test_memory_review.py`
  - token usage normalization과 JSONL append 검증

검증:

```powershell
python -B -m pytest tests/test_code_writer_adapter.py tests/test_memory_review.py -q
python -B -m pytest -q
```

결과:

- 관련 테스트: 8 passed
- 전체 테스트: 142 passed
- `.pytest_cache` 쓰기 권한 warning 2개가 있었지만 테스트 실패는 없음

효과:

- token usage 기록은 직접 비용을 줄이지는 않지만, 어떤 trial/call_type/model에서 비용이 발생하는지 추적할 수 있게 한다.
- 향후 context 상한선, prompt summary, 모델 등급 분리 정책을 실제 사용량 근거로 결정할 수 있다.

## 2026-07-06 KST

### 전역 Research Protocol 단순화

요약:

- ETRI 연구 과정에서 사용한 CV/LB, public baseline, safe/main/aggressive 후보 구분이 전역 Research Protocol에 과하게 반영된 문제를 수정했다.
- 전역 프로토콜은 어떤 주제에도 적용 가능한 최소 공통 연구 흐름만 유지한다.
- leaderboard 비교 같은 심화 판단은 `configs/<competition>/research_policy.yaml`에서 명시적으로 켠 경우에만 동작한다.

전역 출력:

```text
Current State
Evidence
Issues
Candidate Actions
Recommended Action
Constraints
User Questions
Execution Plan
```

주요 변경:

- safe/main/aggressive 후보 강제 제거
- 전역 CV/LB conflict 판단 제거
- `diagnose_trial`의 leaderboard 비교를 대회별 선택 정책으로 변경
- 대회 특화 판단은 필요한 competition workspace에서만 별도 `research_policy.yaml`로 활성화 가능
- `plan-next`는 단순화된 protocol 정보를 참고 자료로 포함하되 기존 전략 선택을 덮어쓰지 않음

### 범용 Execution Profile 1차 구현

요약:

- 특정 대회나 기존 프로젝트를 구현 기준으로 사용하지 않고, 사용자가 나중에 지정하는 임의의 대회/연구 프로젝트 경로를 연결하는 범용 계약을 추가했다.
- 이번 단계에서는 외부 코드를 실행하거나 수정하지 않고 profile 형식과 안전성만 검증한다.

주요 변경:

- `kaggle_research_agent/execution_profile.py`
  - profile load
  - 필수 필드 검증
  - 절대 project/Python 경로 검증
  - test/train/predict 명령 형식 검증
  - metrics/submission artifact 경로 검증
  - allowed/forbidden write scope 충돌 차단
- `validate-execution-profile` CLI 추가
- `configs/execution_profile.example.yaml` 범용 예시 추가
- `docs/policies/execution_profile_schema.ko.md` 추가
- 특정 대회의 실행 profile은 추가하지 않음

검증 결과 파일:

```text
competitions/<workspace>/execution_profile_validation.json
competitions/<workspace>/execution_profile_validation.md
```

### 범용 Workspace 준비 2차 구현

요약:

- 사용자가 지정한 임의의 로컬 프로젝트 경로나 연구 주제를 대회별 독립 workspace로 준비하는 `prepare-workspace` 기능을 추가했다.
- 특정 대회의 코드, 모델, metric, validation 방식을 전역 구현에 하드코딩하지 않는다.
- 자동 탐지는 초안만 만들며 외부 코드를 실행하거나 수정하지 않는다.

주요 기능:

- source path와 topic 기록
- 제한 깊이·파일 수 기반 workspace inventory 생성
- 관례적 test/train/predict entrypoint 감지
- metrics/submission artifact 후보 감지
- 코드 수정 허용 후보와 데이터/artifact 보호 경로 초안 생성
- 불확실한 항목은 `needs_review` 질문으로 기록
- topic만 있으면 `needs_project_path` 상태로 workspace 생성

CLI:

```powershell
python -B -m kaggle_research_agent.cli prepare-workspace --competition <workspace> --source-path "<path>" --topic "<objective>"
```

산출물:

```text
workspace_source.json/md
workspace_inventory.json/md
workspace_preparation.json/md
execution_profile.yaml
execution_profile_validation.json/md
```

### 범용 Workspace Pipeline 실행 3차 구현

요약:

- 검증된 Execution Profile을 실제 로컬 실행에 연결하는 `workspace_runner`를 추가했다.
- 특정 대회, 데이터 형식, 모델 종류를 전역 실행 로직에 포함하지 않았다.
- `--run-now`가 없는 호출은 계획과 판단 근거만 기록하고 외부 명령을 실행하지 않는다.
- 승인된 호출은 `test -> train -> predict` 순서로 실행하며 첫 실패에서 중단한다.
- profile validation 실패, 명령 실패, 산출물 누락을 서로 다른 상태로 기록한다.

CLI:

```powershell
python -B -m kaggle_research_agent.cli run-workspace-pipeline --competition <workspace> --trial trial_001
python -B -m kaggle_research_agent.cli run-workspace-pipeline --competition <workspace> --trial trial_001 --run-now
```

주요 산출물:

```text
experiments/<workspace>/<trial>/workspace_run.json
experiments/<workspace>/<trial>/workspace_run.md
experiments/<workspace>/<trial>/workspace_logs/*.log
memory/<workspace>/decision_log.jsonl
```

상태:

- `planned`: 실행 승인 대기
- `blocked`: profile validation 실패
- `failed`: 명령 실행 실패
- `incomplete_artifacts`: 예상 산출물 누락
- `completed`: 명령과 산출물 검사 통과

다음 단계:

- metrics 산출물의 범용 schema 검증 및 trial 수집 연결
- 제출은 이번 단계에서 실행하지 않음

### 범용 Workspace Metrics 수집 4차 구현

요약:

- 완료된 workspace pipeline의 JSON metrics를 trial 표준 `metrics.json`으로 수집하는 collector를 추가했다.
- 외부 metrics 원본은 수정하지 않고 원본 필드를 trial 복사본에 보존한다.
- 외부 JSON의 `cv_score`를 우선 사용하고, 없으면 선택적 `metrics_contract.source_key` dot path를 사용한다.
- 숫자 필드를 임의 추측하지 않으며 mapping이 불명확하면 `needs_review`로 전환한다.
- Kaggle/DACON 로컬 metrics 수집은 공통 처리하고 leaderboard 점수 수집은 submission 계층에 남긴다.

CLI:

```powershell
python -B -m kaggle_research_agent.cli collect-workspace-metrics --competition <workspace> --trial trial_001
```

상태:

- `collected`: trial 표준 metrics 생성
- `needs_review`: 대표 점수 key 확인 필요
- `blocked`: 실행/profile/artifact/JSON 문제

산출물:

```text
experiments/<workspace>/<trial>/metrics.json
experiments/<workspace>/<trial>/metrics_collection.json
experiments/<workspace>/<trial>/metrics_collection.md
memory/<workspace>/decision_log.jsonl
```

다음 단계:

- 수집된 표준 metrics를 `evaluate -> diagnose -> remember` 흐름에 안전하게 연결
- leaderboard 수집과 submission 실행은 4차 범위에 포함하지 않음

## 2026-07-07 KST

### Workspace Result Cycle 5차 구현

요약:

- 수집된 표준 metrics를 `evaluate -> diagnose -> remember`에 연결했다.
- Human Review를 매 trial마다 요청하지 않고 `request_now / defer / no_review` timing으로 분리했다.
- 비긴급 review는 정상 완료 trial 2개 이상일 때 요청하고, 첫 baseline에서는 competition queue에 누적한다.
- validation/leakage, label 경계 모호성, 안전 중요 false negative, 필수 정보 누락은 성숙도와 무관하게 즉시 요청한다.
- review timing과 무관하게 객관적 trial 결과는 memory에 기록한다.
- 동일 trial의 memory 중복 기록을 차단한다.
- 기존 review pack이 사용자 피드백 대기 중이면 후속 비긴급 review 요청을 queue로 보낸다.

CLI:

```powershell
python -B -m kaggle_research_agent.cli process-workspace-result --competition <workspace> --trial trial_001
```

상태:

- `completed`
- `completed_review_deferred`
- `awaiting_human_review`
- `already_processed`
- `blocked`

산출물:

```text
experiments/<competition>/<trial>/workspace_result_cycle.json
experiments/<competition>/<trial>/workspace_result_cycle.md
memory/<competition>/deferred_review_queue.json
```

## 2026-07-09 KST

### Workspace Next Experiment Gate 6차 구현

요약:

- `process-workspace-result` 이후 다음 trial 계획으로 넘어갈지 판단하는 workspace 전용 gate를 추가했다.
- Human Review가 필요한 경우에도 항상 전체 루프를 멈추지 않고, review 성격에 따라 `can_continue / continue_with_caution / must_wait`로 분기한다.
- non-urgent review는 `user_review_request.md`를 생성한 뒤 다음 실험 계획을 계속 만들 수 있게 했다.
- urgent 또는 blocking review는 `user_review_request.md`를 생성하고 다음 trial 계획 생성을 차단한다.
- 다음 trial에는 `continuation_context.md/json`을 남겨 pending review 상태에서 생성된 계획임을 명시한다.

주요 파일:

- `kaggle_research_agent/workspace_next_gate.py`
- `tests/test_workspace_next_gate.py`
- `docs/workspace_next_experiment_gate.ko.md`
- `docs/superpowers/plans/2026-07-09-workspace-next-experiment-gate.md`

CLI:

```powershell
python -B -m kaggle_research_agent.cli plan-next-workspace-trial --competition <workspace> --source-trial trial_001 --next-trial trial_002
```

산출물:

```text
experiments/<workspace>/<source-trial>/workspace_next_gate.json
experiments/<workspace>/<source-trial>/workspace_next_gate.md
experiments/<workspace>/<source-trial>/user_review_request.md
experiments/<workspace>/<next-trial>/next_experiment.md
experiments/<workspace>/<next-trial>/continuation_context.json
experiments/<workspace>/<next-trial>/continuation_context.md
memory/<workspace>/decision_log.jsonl
```

상태:

- `planned`: review 없이 다음 trial 계획 생성
- `planned_with_deferred_review`: deferred review를 가진 채 다음 trial 계획 생성
- `planned_with_pending_review`: 사용자 피드백 요청을 남긴 채 caution mode로 다음 trial 계획 생성
- `blocked_human_review`: 사용자 판단 전에는 다음 trial 계획 생성을 차단
- `blocked_missing_result_cycle`: 선행 result-cycle 산출물 없음

다음 단계:

- 다음 trial 계획을 코드 수정 handoff로 넘기는 workspace용 연결 단계를 추가한다.

### Workspace Coding Handoff 7차 구현

요약:

- `plan-next-workspace-trial`이 만든 다음 trial 계획을 외부 프로젝트 코드 수정 요청서로 변환하는 `prepare-workspace-handoff` CLI를 추가했다.
- 이 단계는 외부 프로젝트 파일을 수정하지 않고, Execution Profile의 `write_scope`를 기준으로 코딩 에이전트 요청 계약만 생성한다.
- `continuation_mode == must_wait`이면 사용자 피드백 전에는 handoff 생성을 차단한다.
- Execution Profile validation이 ready가 아니면 handoff를 차단한다.
- metrics/submission artifact는 `forbidden_paths`에 자동 포함해 코드 작성 에이전트가 결과/제출 파일을 덮어쓰지 않게 했다.

주요 파일:

- `kaggle_research_agent/workspace_coding_handoff.py`
- `tests/test_workspace_coding_handoff.py`
- `docs/workspace_coding_handoff.ko.md`
- `docs/superpowers/plans/2026-07-09-workspace-coding-handoff.md`

CLI:

```powershell
python -B -m kaggle_research_agent.cli prepare-workspace-handoff --competition <workspace> --trial trial_002
```

산출물:

```text
experiments/<workspace>/<trial>/workspace_coding_handoff.json
experiments/<workspace>/<trial>/workspace_coding_agent_request.md
memory/<workspace>/decision_log.jsonl
```

다음 단계:

- workspace coding handoff를 mock/API code writer로 연결하고, 결과 검증 단계에서 허용 scope를 다시 확인한다.

### Workspace Code Writer 8차 구현

요약:

- `workspace_coding_handoff.json`을 받아 mock/API code writer 응답을 처리하는 `run-workspace-code-writer` CLI를 추가했다.
- code writer 응답의 `changed_files`와 `file_updates[].path`는 Execution Profile의 `project_root` 기준 상대 경로로 해석한다.
- 허용 경로는 `allowed_write_paths`, 금지 경로는 `forbidden_paths`를 기준으로 검증한다.
- 절대 경로, `..` 포함 경로, Windows drive absolute 경로, metric/submission artifact 수정은 차단한다.
- 허용된 `file_updates`만 외부 프로젝트에 적용하고, 결과를 `workspace_coding_result_validation`으로 다시 검증한다.
- mock response file을 통해 API 호출 없이도 code writer 경로를 재현할 수 있게 했다.

주요 파일:

- `kaggle_research_agent/workspace_code_writer.py`
- `tests/test_workspace_code_writer.py`
- `docs/workspace_code_writer.ko.md`
- `docs/superpowers/plans/2026-07-09-workspace-code-writer.md`

CLI:

```powershell
python -B -m kaggle_research_agent.cli run-workspace-code-writer --competition <workspace> --trial trial_002 --mock-response-file mock_response.json
python -B -m kaggle_research_agent.cli validate-workspace-coding-result --competition <workspace> --trial trial_002
```

산출물:

```text
experiments/<workspace>/<trial>/workspace_coding_api_request.json
experiments/<workspace>/<trial>/workspace_coding_api_response.json
experiments/<workspace>/<trial>/workspace_coding_result.json
experiments/<workspace>/<trial>/workspace_coding_result.md
experiments/<workspace>/<trial>/workspace_coding_result_validation.json
experiments/<workspace>/<trial>/workspace_coding_result_validation.md
memory/<workspace>/decision_log.jsonl
memory/<workspace>/token_usage.jsonl
```

다음 단계:

- accepted workspace coding result 이후 validation command 실행과 workspace pipeline 재실행으로 연결한다.

### Workspace After-Coding Cycle 9차 구현

요약:

- accepted workspace code result 이후 실행/평가 루프로 재진입하는 `run-workspace-after-coding` CLI를 추가했다.
- `workspace_coding_result_validation.status == accepted`가 아니면 외부 workspace command를 실행하지 않고 차단한다.
- `--run-now`가 없으면 dry-run으로 `workspace_run.status == planned`까지만 기록한다.
- `--run-now`가 있으면 기존 `run_workspace_pipeline -> collect_workspace_metrics -> process_workspace_result`를 순서대로 호출한다.
- 이 단계까지 오면 코드 수정 이후 metrics 수집, 평가/진단, memory 업데이트로 돌아오는 1차 실험 루프가 닫힌다.

주요 파일:

- `kaggle_research_agent/workspace_after_coding.py`
- `tests/test_workspace_after_coding.py`
- `docs/workspace_after_coding_cycle.ko.md`
- `docs/superpowers/plans/2026-07-09-workspace-after-coding-cycle.md`

CLI:

```powershell
python -B -m kaggle_research_agent.cli run-workspace-after-coding --competition <workspace> --trial trial_002
python -B -m kaggle_research_agent.cli run-workspace-after-coding --competition <workspace> --trial trial_002 --run-now
```

산출물:

```text
experiments/<workspace>/<trial>/workspace_after_coding_cycle.json
experiments/<workspace>/<trial>/workspace_after_coding_cycle.md
experiments/<workspace>/<trial>/workspace_run.json
experiments/<workspace>/<trial>/metrics_collection.json
experiments/<workspace>/<trial>/workspace_result_cycle.json
memory/<workspace>/decision_log.jsonl
```

다음 단계:

- 데모/운영 편의를 위해 1-cycle 명령을 얇게 묶고, 각 gate 결과를 한 화면 또는 요약 파일로 보여준다.

다음 단계:

- `plan-next-experiment`를 result-cycle 상태와 연결
- `awaiting_human_review`에서는 사용자 피드백 전까지 다음 계획을 차단
