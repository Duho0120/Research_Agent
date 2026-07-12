# Execution Profile Schema

Execution Profile은 특정 대회나 연구 주제를 전역 코드에 하드코딩하지 않고, 사용자가 지정한 프로젝트를 Research Agent 실행 흐름에 연결하는 표준 계약이다.

저장 위치:

```text
competitions/<workspace>/execution_profile.yaml
```

필수 항목:

```yaml
schema_version: "1.0"
competition: workspace_name
platform: kaggle | dacon | external | local_research
project_root: absolute project path
python: absolute python executable path
commands:
  test:
    - command
  train:
    - command
  predict:
    - optional command
artifacts:
  metrics:
    - relative/path/to/metrics.json
  submission:
    - relative/path/to/submission.csv
write_scope:
  allowed:
    - relative/source/path
  forbidden:
    - relative/protected/path
submission_mode: manual_external
```

선택 metrics mapping:

```yaml
metrics_contract:
  source_key: validation.macro_f1
```

외부 JSON에 표준 `cv_score`가 없을 때 `source_key`가 대표 validation 점수 위치를 지정한다. 값은 유한한 숫자여야 하며, key가 없거나 값이 잘못되면 자동 추측하지 않고 사용자 확인으로 전환한다.

원칙:

- `project_root`와 `python`은 절대경로여야 한다.
- 명령은 이 단계에서 실행하지 않고 존재 여부와 형식만 검증한다.
- artifact와 write scope 경로는 `project_root` 기준 상대경로만 허용한다.
- `..` 또는 외부 절대경로로 workspace를 벗어날 수 없다.
- metrics, submission, 원본 데이터처럼 보호할 파일은 allowed scope에 포함할 수 없다.
- 특정 대회의 모델, metric, validation, 파일명은 전역 schema에 포함하지 않는다.

검증 명령:

```powershell
python -B -m kaggle_research_agent.cli validate-execution-profile --competition <workspace>
```

검증 결과:

```text
competitions/<workspace>/execution_profile_validation.json
competitions/<workspace>/execution_profile_validation.md
```
