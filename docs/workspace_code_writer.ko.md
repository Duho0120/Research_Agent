# Workspace Code Writer

8차 구현은 `workspace_coding_handoff.json`을 받아 mock/API code writer 응답을 처리하고, Execution Profile에서 온 write scope 안의 외부 프로젝트 파일만 수정한다.

## 실행

mock 응답 사용:

```powershell
python -B -m research_agent.cli run-workspace-code-writer `
  --competition <workspace> `
  --trial trial_002 `
  --mock-response-file mock_response.json
```

API 호출 사용:

```powershell
python -B -m research_agent.cli run-workspace-code-writer `
  --competition <workspace> `
  --trial trial_002 `
  --model gpt-5 `
  --allow-api
```

결과 검증만 다시 실행:

```powershell
python -B -m research_agent.cli validate-workspace-coding-result `
  --competition <workspace> `
  --trial trial_002
```

## Path Semantics

`changed_files`와 `file_updates[].path`는 repo 상대 경로가 아니라 Execution Profile의 `project_root` 기준 상대 경로다.

예:

```json
{
  "changed_files": ["src/model.py"],
  "file_updates": [
    {
      "path": "src/model.py",
      "content": "FEATURE_FLAG = True\n"
    }
  ]
}
```

이 예시는 `<project_root>/src/model.py`를 수정한다.

## Safety Gate

다음 경로는 차단된다.

- 절대 경로
- `..` 포함 경로
- Windows drive absolute 경로
- `allowed_write_paths` 밖의 경로
- `forbidden_paths`와 일치하거나 그 하위인 경로

`outputs/metrics.json`, `outputs/submission.csv` 같은 metric/submission artifact는 7차 handoff에서 `forbidden_paths`에 포함된다.

## 산출물

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

## 1-Cycle 내 위치

```text
prepare-workspace
-> run-workspace-pipeline
-> collect-workspace-metrics
-> process-workspace-result
-> plan-next-workspace-trial
-> prepare-workspace-handoff
-> run-workspace-code-writer
```

다음 단계는 accepted code result 이후 validation command 또는 workspace pipeline 재실행으로 연결하는 것이다.
