# Workspace Pipeline Execution

3차 구현은 검증된 Execution Profile을 실제 로컬 실행에 연결한다. 특정 대회나 모델을 전제로 하지 않으며, profile에 선언된 명령만 사용한다.

## 실행 전 확인

```powershell
python -B -m research_agent.cli validate-execution-profile `
  --competition <workspace>
```

validation 상태가 `ready`가 아니면 외부 명령은 실행되지 않는다.

## 실행 계획 기록

```powershell
python -B -m research_agent.cli run-workspace-pipeline `
  --competition <workspace> `
  --trial trial_001
```

`--run-now`가 없으면 상태는 `planned`이며 외부 프로젝트의 명령을 실행하지 않는다.

## 명시적 실행 승인

```powershell
python -B -m research_agent.cli run-workspace-pipeline `
  --competition <workspace> `
  --trial trial_001 `
  --run-now
```

`--run-now`는 현재 호출에 대한 사용자 실행 승인이다. 명령은 profile의 `project_root`에서 아래 순서로 실행된다.

```text
test -> train -> predict
```

각 그룹 안에 여러 명령이 있으면 선언 순서를 유지한다. 하나라도 실패하면 뒤의 명령과 단계는 실행하지 않는다.

## 상태

- `planned`: 승인 전 실행 계획만 기록됨
- `blocked`: Execution Profile validation 실패
- `failed`: 명령이 0이 아닌 종료 코드로 종료됨
- `incomplete_artifacts`: 명령은 성공했지만 선언된 산출물이 없음
- `completed`: 모든 명령이 성공하고 모든 선언 산출물이 존재함

## 기록 위치

```text
experiments/<workspace>/<trial>/workspace_run.json
experiments/<workspace>/<trial>/workspace_run.md
experiments/<workspace>/<trial>/workspace_logs/<stage>_<index>.log
memory/<workspace>/decision_log.jsonl
```

`workspace_run.json`에는 렌더링된 명령, 단계별 종료 코드, 로그 경로, 실패 분류, metrics/submission 산출물의 존재 여부와 크기가 저장된다.

## 이번 단계의 경계

이 단계는 pipeline을 실행하고 산출물 존재를 확인한다. `metrics.json`의 schema 검증과 점수 해석, 제출 실행, 다음 개선 계획 연결은 후속 단계에서 처리한다.
