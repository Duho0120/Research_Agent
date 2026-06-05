# Coding Request Schema

## 목적

`coding_handoff.json`과 `coding_agent_request.md`는 검증된 코드 수정 계획을 Codex/API 기반 코딩 작업자에게 전달하는 표준 계약이다.

코딩 작업자는 코드 수정만 담당한다. 이 단계에서는 학습 실행, Kaggle 제출, 보호축 변경, 허용되지 않은 파일 수정이 금지된다.

## 입력 산출물

코딩 요청은 다음 trial 산출물을 입력으로 사용할 수 있다.

- `code_patch_plan.json`
- `patch_validation.json`
- `config.yaml`
- `next_experiment.md`
- `model_candidates.json`

실제로 존재하는 파일만 `context_files`에 포함한다.

## 필수 필드

```text
schema_version
request_id
competition
trial_id
handoff_type
status
objective
strategy
pipeline_axis
context_files
target_files
allowed_write_files
create_files
forbidden_paths
config_changes
implementation_steps
validation_commands
execution_constraints
required_output
protected_axes
blocking_issues
patch_validation_status
next_action
```

## 파일 수정 규칙

- `allowed_write_files`에 포함된 파일만 수정할 수 있다.
- 존재하지 않는 파일은 `create_files`에 명시된 경우에만 생성할 수 있다.
- `target_files`에 있지만 `create_files`에 없는 누락 파일은 Patch Validator가 차단한다.
- `data/`, `submissions/`, `submission.csv`, `metrics.json`은 코딩 단계에서 수정하지 않는다.
- 보호축을 변경해야 한다면 코딩을 중단하고 새 patch plan과 승인을 요청한다.

## 실행 제약

```json
{
  "do_not_run_training": true,
  "do_not_submit": true,
  "do_not_change_protected_axes": true,
  "do_not_write_outside_allowed_files": true
}
```

코딩 작업자는 검증 명령을 다음 단계에 전달할 수 있지만, 학습이나 제출을 시작하지 않는다.

## 결과 계약

코딩 작업자는 다음 파일을 생성하는 형식으로 결과를 보고해야 한다.

```text
experiments/<competition>/<trial>/coding_result.json
experiments/<competition>/<trial>/coding_result.md
```

`coding_result.json` 필수 필드:

```text
status
summary
changed_files
validation_results
blocking_issues
```

허용 상태:

```text
completed
blocked
failed
```

결과의 다음 action은 `validate-code-change`다. 코드 검증과 제한된 수정 재시도는 이후 단계에서 담당한다.

## 결과 검증 단계

`coding_result.json`은 곧바로 실행 단계로 넘어가지 않고 `validate-coding-result` 게이트를 통과해야 한다.

검증 항목:

- `status`, `summary`, `changed_files`, `validation_results`, `blocking_issues` 필수 필드 존재 여부
- `status`가 `completed`, `blocked`, `failed` 중 하나인지 여부
- `changed_files`가 `allowed_write_files` 또는 `create_files` 안에 있는지 여부
- `data/`, `submissions/`, `submission.csv`, `metrics.json` 등 `forbidden_paths`를 건드리지 않았는지 여부

검증 결과는 다음 파일에 저장한다.

```text
experiments/<competition>/<trial>/coding_result_validation.json
experiments/<competition>/<trial>/coding_result_validation.md
```

실제 Codex/API 코딩 작업자가 붙기 전까지는 `write-code-dry-run`으로 `blocked` 상태의 placeholder 결과를 만들 수 있다. 이 dry-run은 파일을 수정하지 않고 외부 API도 호출하지 않는다.

## Code Writer Adapter

`run-code-writer`는 `coding_handoff.json`을 읽어 코딩 모델 또는 mock client에 전달한다.

운영 원칙:

- 실제 API 호출은 `--allow-api`가 있을 때만 수행한다.
- mock response file을 사용하면 외부 API 없이 같은 경로를 테스트할 수 있다.
- 모델은 로컬 파일을 직접 수정하지 않는다.
- 모델은 JSON 안에 `file_updates`를 반환하고, Python adapter가 허용된 경로만 실제로 쓴다.
- `file_updates.path`는 `allowed_write_files` 또는 `create_files` 안에 있어야 한다.
- `forbidden_paths`에 해당하는 경로는 쓰지 않는다.
- 결과 저장 후 즉시 `validate-coding-result`를 실행한다.

OpenAI Responses API를 사용할 때의 기본 endpoint는 공식 API 문서 기준 `POST https://api.openai.com/v1/responses`이며, 이 프로젝트에서는 표준 라이브러리 기반 client가 `OPENAI_API_KEY` 환경 변수를 읽는다.

## Validation Command Runner

`coding_result_validation.json`의 status가 `accepted`이면 다음 단계에서 `run-validation-commands`를 실행할 수 있다.

운영 원칙:

- `accepted`가 아니면 validation command를 실행하지 않는다.
- 실행 대상 명령은 `coding_handoff.json`의 `validation_commands`에서 읽는다.
- 각 명령의 stdout/stderr는 `validation_command_001.log` 같은 파일로 저장한다.
- 전체 결과는 `validation_run.json`과 `validation_run.md`에 저장한다.
- `run-code-writer --run-validation-commands`를 사용하면 code writer 결과가 accepted인 경우에만 이어서 validation command를 실행한다.
