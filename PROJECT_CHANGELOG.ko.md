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
