# Workspace Metrics Collection

4차 구현은 외부 프로젝트가 생성한 JSON metrics를 Research Agent의 trial 표준 `metrics.json`으로 수집한다. JSON은 Kaggle이나 DACON의 요구 형식이 아니라 에이전트 내부 전달 계약이다.

## 기본 흐름

```powershell
python -B -m kaggle_research_agent.cli run-workspace-pipeline `
  --competition <workspace> `
  --trial trial_001 `
  --run-now

python -B -m kaggle_research_agent.cli collect-workspace-metrics `
  --competition <workspace> `
  --trial trial_001
```

`workspace_run.json` 상태가 `completed`가 아니면 수집은 차단된다.

## 대표 점수 선택

외부 JSON에 유한한 숫자 `cv_score`가 있으면 바로 사용한다.

```json
{"cv_score": 0.83, "accuracy": 0.91}
```

다른 이름을 사용한다면 Execution Profile에 JSON dot path를 명시한다.

```yaml
metrics_contract:
  source_key: validation.macro_f1
```

```json
{"validation": {"macro_f1": 0.83}}
```

점수 key를 찾지 못하거나 값이 숫자가 아니면 자동 추측하지 않고 `needs_review`로 사용자 확인을 요청한다.

## 상태

- `collected`: trial `metrics.json` 생성 성공
- `needs_review`: 대표 점수 mapping 확인 필요
- `blocked`: 실행 미완료, profile 오류, artifact 누락, JSON 형식 오류

## 산출물

```text
experiments/<workspace>/<trial>/metrics.json
experiments/<workspace>/<trial>/metrics_collection.json
experiments/<workspace>/<trial>/metrics_collection.md
memory/<workspace>/decision_log.jsonl
```

외부 metrics 원본은 수정하지 않는다. 원본 필드는 trial metrics 복사본에 보존되고 `trial_id`, `metric`, `cv_score`, `objective`, `source_metrics_path`가 표준 필드로 확정된다.

## 플랫폼 경계

이 수집기는 Kaggle과 DACON의 로컬 학습 결과에 동일하게 사용할 수 있다. 플랫폼 leaderboard 점수와 제출 상태는 별도 submission adapter 또는 수동 기록으로 처리한다.

4차에서는 `evaluate`, `diagnose`, `remember`를 자동 호출하지 않는다.
