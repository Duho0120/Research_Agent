# Execution Decision Policy

## 목적

이 문서는 Main Agent가 trial 실행 시 `local`, `colab`, `ask_user`, `wait_for_metrics` 중 어떤 경로를 선택할지 판단하는 기준을 정의한다.

기본 철학은 **Local 우선, Colab 선택, 위험하거나 비용이 큰 판단은 사용자 확인**이다. LLM은 이 판단을 매번 대신하지 않는다. 명확한 조건은 rule-based gate가 처리하고, 애매하거나 비용이 큰 경우만 Main Agent/LLM 또는 사용자에게 넘긴다.

## 입력

Execution decision은 다음 정보를 입력으로 사용한다.

- `competitions/<competition>/state.yaml`
- `memory/<competition>/rules.md`
- `memory/<competition>/trial_index.jsonl`
- `memory/<competition>/decision_log.jsonl`
- `experiments/<competition>/<trial_id>/config.yaml`
- `experiments/<competition>/<trial_id>/metrics.json`
- `experiments/<competition>/<trial_id>/local_run.log`
- job status under `jobs/<competition>/`
- optional policy file: `configs/policies/execution_policy.yaml`

## 출력

판단 결과는 다음 중 하나다.

```text
run_local
create_local_job
create_colab_job
ask_user
wait_for_metrics
blocked
```

출력은 구조화된 decision object로 남긴다.

```yaml
decision_type: execution_backend
decision: run_local
reason: Trial has a local command and no GPU/resource requirement is known.
evidence:
  default_backend: local
  metrics_exists: false
  estimated_runtime_minutes: 12
  require_gpu: false
  previous_local_failure_type: null
next_action: run-local
```

이 판단은 `memory/<competition>/decision_log.jsonl`에 기록한다.

## 기본 우선순위

### 1. 이미 metrics가 있으면 실행하지 않는다

조건:

- `experiments/<competition>/<trial_id>/metrics.json` 존재

결정:

```text
wait_for_metrics가 아니라 evaluate_metrics로 이동
```

이 경우 새 job을 만들지 않는다. 이미 결과가 있으므로 evaluation, diagnosis, memory update로 넘어간다.

### 2. 실행 command가 없고 job이 pending이면 기다린다

조건:

- `metrics.json` 없음
- 기존 job yaml이 존재
- job status가 `pending` 또는 `running`

결정:

```text
wait_for_metrics
```

단, job이 오래 멈춰 있거나 `running` 상태가 비정상적으로 길면 `ask_user` 또는 failure diagnosis로 넘긴다.

### 3. 로컬 실행 가능한 경우 local을 기본으로 한다

조건:

- `run_command`가 있음
- GPU 필수 조건이 명시되지 않음
- 예상 실행 시간이 정책 기준 이하
- 이전 로컬 실패가 resource failure가 아님

결정:

```text
run_local
```

현재 기본 정책:

```yaml
default_backend: local
local_first: true
ask_before_colab: true
```

### 4. 로컬 실행 command가 없으면 local job을 만든다

조건:

- `metrics.json` 없음
- 명시적 `run_now`가 아님
- 기존 job 없음
- Colab이 명시되지 않음

결정:

```text
create_local_job
```

이 결정은 실행을 의미하지 않는다. 사람이 command를 채우거나 다음 단계에서 runner가 사용할 수 있는 job request를 만든다.

### 5. Colab은 명시 조건이 있을 때만 사용한다

Colab을 고려하는 조건:

- config 또는 plan에 `require_gpu: true`가 있음
- 예상 실행 시간이 local 기준을 초과함
- 로컬 장치가 없음
- 이전 local run이 resource failure로 실패함
- 사용자가 명시적으로 `--backend colab`을 선택함

기본 결정:

```text
ask_user
```

사용자 승인 또는 명시 인자가 있으면:

```text
create_colab_job
```

Colab은 완전 remote server처럼 자동 조종하지 않는다. Local Main Agent가 job을 만들고, Colab Worker가 job을 읽어 실행하는 worker 구조를 유지한다.

### 6. 위험하거나 비용이 큰 실행은 사용자에게 묻는다

다음 조건이면 `ask_user`로 분기한다.

- Colab/GPU 비용이 발생할 수 있음
- 예상 실행 시간이 길다
- Kaggle submission 직전 단계다
- validation strategy 변경과 model family 변경이 동시에 발생한다
- data download 또는 대용량 파일 처리가 필요하다
- 최근 local failure 원인이 불명확하다
- 사람의 도메인 판단 없이 큰 feature/model 변경을 실행하려 한다

## 로컬 실패 원인 분류

`local_run.log`와 job returncode를 보고 실패 원인을 분류한다.

```text
code_error
missing_file
missing_dependency
data_not_found
resource_cpu_memory
resource_gpu_missing
timeout
permission_error
unknown
```

초기 rule:

- `CUDA out of memory`, `out of memory`, `MemoryError` -> `resource_cpu_memory`
- `CUDA is not available`, `No CUDA`, `GPU not found` -> `resource_gpu_missing`
- `No such file`, `FileNotFoundError` -> `missing_file`
- `ModuleNotFoundError`, `ImportError` -> `missing_dependency`
- nonzero returncode with no known pattern -> `unknown`

분류 결과가 `resource_cpu_memory` 또는 `resource_gpu_missing`이면 다음 실행에서 Colab 후보가 된다. 그래도 기본은 `ask_user` 후 Colab이다.

## Rule-Based Gate 순서

```text
1. metrics_exists?
   yes -> evaluate_metrics

2. config_valid?
   no -> blocked

3. job_already_pending_or_running?
   yes -> wait_for_metrics

4. local_previous_run_failed_due_to_resource?
   yes -> ask_user 또는 create_colab_job

5. explicit_backend_colab?
   yes -> create_colab_job

6. run_now and run_command_available?
   yes -> run_local

7. run_command_missing?
   yes -> create_local_job

8. default
   -> create_local_job
```

## LLM 호출 기준

이 정책 자체는 LLM 없이 처리한다.

LLM 또는 Main Agent 전략 판단이 필요한 경우:

- local 실패 원인이 `unknown`이고 로그 요약만으로 판단이 어렵다
- Colab 비용을 감수할 만큼 기대 이득이 있는지 판단해야 한다
- 반복 실패로 실행 방식보다 실험 전략 자체를 바꿔야 한다
- 사용자에게 보여줄 실행 선택 근거를 요약해야 한다

## 성공 기준

- 기본 실행은 local이다.
- Colab은 명시적 선택 또는 사용자 승인 뒤 사용된다.
- pending/running job을 중복 생성하지 않는다.
- 실패 원인이 다음 실행 backend 판단에 반영된다.
- 모든 중요한 실행 판단은 `decision_log.jsonl`에 남는다.
