# Workspace Coding Handoff

7차 구현은 `plan-next-workspace-trial`이 만든 다음 실험 계획을 외부 프로젝트 코드 수정 요청서로 변환한다.

이 단계는 실제 코드를 수정하지 않는다. 외부 프로젝트 파일을 읽기/쓰기하지 않고, `experiments/<workspace>/<trial>/` 아래에 coding agent가 따라야 할 요청 계약만 생성한다.

## 실행

```powershell
python -B -m research_agent.cli prepare-workspace-handoff `
  --competition <workspace> `
  --trial trial_002
```

선행 조건:

- `experiments/<workspace>/<trial>/next_experiment.md`
- `experiments/<workspace>/<trial>/continuation_context.json`
- `competitions/<workspace>/execution_profile.yaml`

## 차단 조건

다음 경우 `blocked`로 멈춘다.

- `continuation_mode == must_wait`
- `next_experiment.md` 없음
- `continuation_context.json` 없음 또는 JSON object가 아님
- Execution Profile validation 실패

## Scope Source

코딩 요청의 허용/금지 범위는 Execution Profile에서 온다.

- `write_scope.allowed` -> `allowed_write_paths`
- `write_scope.forbidden` + metrics/submission artifacts -> `forbidden_paths`
- `commands.test` -> `validation_commands`

이 단계는 임의로 target file을 추측하지 않는다.

## 산출물

```text
experiments/<workspace>/<trial>/workspace_coding_handoff.json
experiments/<workspace>/<trial>/workspace_coding_agent_request.md  # ready일 때만
memory/<workspace>/decision_log.jsonl
```

## 1-Cycle 내 위치

```text
prepare-workspace
-> run-workspace-pipeline
-> collect-workspace-metrics
-> process-workspace-result
-> plan-next-workspace-trial
-> prepare-workspace-handoff
```

다음 단계는 이 handoff를 코드 작성 에이전트/API 호출 또는 dry-run 검증으로 연결하는 것이다.
