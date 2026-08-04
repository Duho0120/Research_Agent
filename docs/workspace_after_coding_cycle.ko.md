# Workspace After-Coding Cycle

9차 구현은 accepted workspace code result 이후 실행/평가 루프로 다시 들어가는 연결 단계다.

이 단계는 새 실행 엔진을 만들지 않는다. 기존 `run-workspace-pipeline`, `collect-workspace-metrics`, `process-workspace-result`를 순서대로 호출한다.

## 실행

dry-run:

```powershell
python -B -m research_agent.cli run-workspace-after-coding `
  --competition <workspace> `
  --trial trial_002
```

실제 실행:

```powershell
python -B -m research_agent.cli run-workspace-after-coding `
  --competition <workspace> `
  --trial trial_002 `
  --run-now
```

`--run-now`가 없으면 외부 프로젝트 명령을 실행하지 않고 `workspace_run.status == planned`까지만 기록한다.

## 선행 조건

```text
experiments/<workspace>/<trial>/workspace_coding_result_validation.json
```

이 파일의 `status`가 `accepted`여야 한다. 그렇지 않으면 workspace command를 실행하지 않고 멈춘다.

## 처리 흐름

```text
workspace_coding_result_validation 확인
-> run_workspace_pipeline
-> collect_workspace_metrics
-> process_workspace_result
```

## 상태

- `ready_to_run`: code result는 accepted지만 `--run-now`가 없어 실행 대기
- `completed`: 실행, metrics 수집, result-cycle 처리까지 완료
- `blocked`: code result validation이 accepted가 아님
- `workspace_run_<status>`: workspace run이 completed가 아님
- `metrics_<status>`: metrics collection이 collected가 아님
- `result_cycle_blocked`: result-cycle 처리 실패

## 산출물

```text
experiments/<workspace>/<trial>/workspace_after_coding_cycle.json
experiments/<workspace>/<trial>/workspace_after_coding_cycle.md
experiments/<workspace>/<trial>/workspace_run.json
experiments/<workspace>/<trial>/metrics_collection.json
experiments/<workspace>/<trial>/workspace_result_cycle.json
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
-> run-workspace-code-writer
-> run-workspace-after-coding
```

이 단계까지 오면 코드 수정 이후 다시 평가와 memory 업데이트로 돌아오는 1차 실험 루프가 닫힌다.
