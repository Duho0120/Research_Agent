# Workspace Metrics Collector Design

## Goal

외부 프로젝트가 생성한 JSON metrics를 특정 대회나 metric 이름에 의존하지 않고 Research Agent의 trial 표준 `metrics.json`으로 안전하게 수집한다.

## Input Contract

- `workspace_run.json` 상태가 `completed`여야 한다.
- Execution Profile에 선언된 첫 번째 metrics artifact를 읽는다.
- metrics artifact는 JSON object여야 한다.
- 외부 원본 파일은 읽기만 하며 수정하지 않는다.

Execution Profile의 선택 확장:

```yaml
metrics_contract:
  source_key: validation.macro_f1
```

`source_key`는 JSON object 안의 점수 위치를 나타내는 dot path다.

## Score Resolution

1. 외부 JSON의 `cv_score`가 유한한 숫자면 그대로 사용한다.
2. `cv_score`가 없으면 `metrics_contract.source_key`의 값을 사용한다.
3. 둘 다 없거나 값이 숫자가 아니면 추측하지 않고 `needs_review`를 기록한다.

`bool`, NaN, infinity는 점수로 인정하지 않는다.

## Canonical Output

원본 JSON의 필드를 보존한 복사본에 아래 표준 필드를 확정하여 trial 디렉터리에 기록한다.

```json
{
  "trial_id": "trial_001",
  "metric": "macro_f1",
  "cv_score": 0.83,
  "objective": "maximize",
  "source_metrics_path": "C:/project/outputs/metrics.json"
}
```

- `metric`: 외부 JSON의 `metric`, 없으면 competition state의 metric
- `objective`: 외부 JSON의 유효한 objective, 없으면 competition state의 objective
- `trial_id`: 현재 수집 대상 trial

## Status Contract

- `collected`: 표준 `metrics.json` 생성 성공
- `needs_review`: 대표 점수 key가 없거나 값이 유효한 숫자가 아님
- `blocked`: 실행 미완료, profile 오류, artifact 누락, JSON 파싱/형식 오류

모든 결과는 `metrics_collection.json/md`와 competition decision log에 기록한다.

## Scope Boundary

4차는 metrics 수집과 표준화까지만 담당한다. `evaluate`, `diagnose`, `remember`, leaderboard 점수 수집, submission 실행은 호출하지 않는다.

## Testing

- 기존 `cv_score` 직접 수집
- nested `source_key` 매핑
- mapping 누락 시 `needs_review`
- 잘못된 JSON과 미완료 실행 차단
- CLI 연결과 원본 파일 불변성

