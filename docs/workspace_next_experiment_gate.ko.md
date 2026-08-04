# Workspace Next Experiment Gate

6차 구현은 `process-workspace-result` 이후 다음 trial 계획으로 넘어갈지 판단하는 안전 게이트다.

## 실행

```powershell
python -B -m research_agent.cli plan-next-workspace-trial `
  --competition <workspace> `
  --source-trial trial_001 `
  --next-trial trial_002
```

선행 조건:

- `experiments/<workspace>/<source-trial>/workspace_result_cycle.json` 존재
- source trial의 `metrics.json` 존재
- competition state가 준비되어 있어야 함

## Continuation Mode

- `can_continue`: Human Review 없이 다음 실험 계획 생성 가능
- `continue_with_caution`: 사용자 피드백 요청은 등록하지만, 낮은 위험의 다음 실험 계획은 계속 생성 가능
- `must_wait`: 사용자 피드백 없이는 다음 실험 계획을 만들지 않음

## Blocking Review

다음 경우에는 `must_wait`로 멈춘다.

- validation 또는 leakage 의심
- 필수 metric, label, input 정의 누락
- 안전 중요 false negative
- Human Review policy가 `urgent=true`로 표시한 경우

이 경우에도 `user_review_request.md`는 생성해서 사용자가 무엇을 판단해야 하는지 볼 수 있게 한다.

## Non-Blocking Review

대표 오류 해석, 데이터 품질 확인, 개선 방향 참고처럼 다음 실험의 전제를 즉시 깨지 않는 질문은 `continue_with_caution`으로 처리한다.

이때 생성되는 다음 trial에는 continuation metadata가 남는다.

```text
experiments/<workspace>/<next-trial>/continuation_context.json
experiments/<workspace>/<next-trial>/continuation_context.md
```

metadata에는 pending review 여부, review source trial, allowed topics, blocked topics가 기록된다.

## 산출물

source trial:

```text
experiments/<workspace>/<source-trial>/workspace_next_gate.json
experiments/<workspace>/<source-trial>/workspace_next_gate.md
experiments/<workspace>/<source-trial>/user_review_request.md  # review 대기 시
```

next trial:

```text
experiments/<workspace>/<next-trial>/next_experiment.md
experiments/<workspace>/<next-trial>/continuation_context.json
experiments/<workspace>/<next-trial>/continuation_context.md
```

memory:

```text
memory/<workspace>/decision_log.jsonl
```

## 1-Cycle 내 위치

```text
prepare-workspace
-> run-workspace-pipeline
-> collect-workspace-metrics
-> process-workspace-result
-> plan-next-workspace-trial
```

6차는 코드 수정이나 학습 실행을 직접 수행하지 않는다. 다음 trial 계획과 continuation metadata까지만 만든다.
