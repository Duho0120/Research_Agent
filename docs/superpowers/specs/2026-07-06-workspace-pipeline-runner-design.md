# Workspace Pipeline Runner Design

## Goal

검증된 Execution Profile을 사용해 임의의 로컬 연구 프로젝트에서 `test -> train -> predict` 명령을 순차 실행하고, 실행 근거와 결과를 대회별 Research Agent workspace에 기록한다.

## Scope

- 특정 대회, 데이터 형식, 모델 계열에 관한 지식을 포함하지 않는다.
- `--run-now`가 없는 호출은 실행 계획만 기록하고 외부 명령을 실행하지 않는다.
- profile validation이 `ready`가 아니면 실행을 차단한다.
- 각 단계의 명령은 선언된 순서대로 실행하며 첫 실패에서 중단한다.
- stdout, stderr, 종료 코드, 실패 분류, 예상 산출물 존재 여부를 기록한다.
- metrics 내용 해석과 submission 수행은 다음 단계 범위로 남긴다.

## Architecture

새 `workspace_runner` 모듈이 profile validation과 subprocess 실행 사이의 단일 어댑터가 된다. CLI는 사용자 의도를 이 모듈에 전달할 뿐 실행 정책을 중복 구현하지 않는다. 기존 decision log를 재사용해 실행 여부와 최종 결과를 구조화한다.

## Data Flow

1. `execution_profile.yaml`을 검증하고 읽는다.
2. `{python}` placeholder를 profile의 Python 실행 경로로 치환한다.
3. 실행 승인이 없으면 `planned` 결과를 기록한다.
4. 승인이 있으면 `test`, `train`, `predict` 순서로 명령을 실행한다.
5. 각 명령의 로그를 trial 디렉터리에 저장한다.
6. 첫 실패에서 중단하고 기존 local failure classifier로 원인을 분류한다.
7. 성공 후 profile에 선언된 metrics/submission 산출물 존재 여부를 확인한다.
8. `workspace_run.json`, `workspace_run.md`, decision log를 기록한다.

## Status Contract

- `planned`: 실행 승인 전이며 외부 명령은 실행되지 않았다.
- `blocked`: profile이 유효하지 않아 실행할 수 없다.
- `failed`: 명령 하나가 0이 아닌 종료 코드로 끝났다.
- `incomplete_artifacts`: 명령은 성공했지만 선언된 산출물이 없다.
- `completed`: 모든 명령과 산출물 검사가 통과했다.

## Safety

- 명시적 `--run-now`가 실행 승인이다.
- 명령의 working directory는 profile의 `project_root`로 고정한다.
- Research Agent는 외부 프로젝트의 코드를 이 단계에서 수정하지 않는다.
- 실행 로그와 상태 파일은 Research Agent의 competition/trial 경계 안에 저장한다.

## Verification

임시 외부 프로젝트를 사용하는 단위 테스트로 dry-run, 정상 순차 실행, 실패 시 중단, invalid profile 차단, CLI 연결을 검증한다. 이후 전체 unittest, compileall, diff whitespace 검사를 수행한다.
