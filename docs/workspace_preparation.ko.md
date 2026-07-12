# Workspace Preparation

`prepare-workspace`는 특정 대회나 기존 연구 사례를 전역 로직에 반영하지 않고, 사용자가 지정한 로컬 경로 또는 연구 주제를 독립 workspace로 준비한다.

기존 프로젝트 경로를 준비하는 예시:

```powershell
python -B -m kaggle_research_agent.cli prepare-workspace `
  --competition my_workspace `
  --source-path "C:\path\to\project" `
  --topic "Predict the target from the supplied data" `
  --platform external `
  --metric accuracy `
  --objective maximize
```

연구 주제만 먼저 등록하는 예시:

```powershell
python -B -m kaggle_research_agent.cli prepare-workspace `
  --competition energy_forecasting `
  --topic "Forecast hourly energy demand"
```

자동 탐지 범위:

- `tests/`, `pytest.ini`, `pyproject.toml` 기반 test command 후보
- `train.py`, `src/train.py`, `scripts/train.py`
- `predict.py`, `inference.py`와 관례적 하위 경로
- `metrics.json`, `submission.csv`
- `src/`, `tests/`, `scripts/`와 관례적 Python entrypoint
- CSV, parquet, submission, metrics 파일의 보호 범위

상태:

- `ready`: 필수 실행 정보가 감지되고 Execution Profile 검증을 통과함
- `needs_review`: workspace는 생성됐지만 명령, artifact, 수정 범위 중 확인할 항목이 남음
- `needs_project_path`: 연구 주제만 등록됐고 실제 코드/데이터 경로가 필요함
- `blocked`: 제공된 경로가 유효하지 않음

자동 탐지는 초안 생성만 수행하며 외부 코드를 실행하거나 수정하지 않는다.

주요 산출물:

```text
competitions/<workspace>/workspace_source.json
competitions/<workspace>/workspace_inventory.json
competitions/<workspace>/workspace_preparation.json
competitions/<workspace>/execution_profile.yaml
competitions/<workspace>/execution_profile_validation.json
```
