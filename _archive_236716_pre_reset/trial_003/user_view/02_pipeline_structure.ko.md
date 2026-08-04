# trial_003 파이프라인 구조

실제 실행 코드와 실행 결과를 기준으로 재구성한 현재 유효 파이프라인입니다.

| 항목 | 값 |
|---|---|
| 기준 trial | `trial_002` |
| 평가 지표 | `R-Hit@1cm` |
| 로컬 점수 | `0.591` |
| 모델 | `RuleBasedConstantVelocity` |
| 제출 파일 | `C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\236716\outputs\submission.csv` |

## 단계별 실행 구조

### 1. 준비

- 역할: 실행에 필요한 라이브러리, 경로와 설정값을 준비합니다.
- 실제 적용:
  - 사용 라이브러리/도구: pathlib.Path.
  - workspace 기준 경로 상수 CONFIG_PATH, DATA_DIR, MODEL_PATH, SUBMISSION_PATH를 정의합니다.
- 입력: `workspace_config.json`, `Execution Profile`, `Python runtime`
- 출력: `reproducible runtime context`
- 관련 코드: `src/baseline.py`, `train_step.py`, `predict_step.py`, `test_step.py`

### 2. 데이터 로드

- 역할: 학습 및 테스트 데이터를 읽고 목표·ID 컬럼을 확인합니다.
- 실제 적용:
  - `pd.read_csv`로 train_labels.csv, sample_submission.csv를 읽습니다.
  - 학습 타깃은 `target`, 제출 ID는 `id`로 사용합니다.
- 주요 설정:
  - ID 컬럼: `id`
- 입력: `data/train.csv`, `data/test.csv`, `workspace data directory`
- 출력: `raw train/test dataframe or dataset objects`
- 관련 코드: `src/baseline.py`, `train_step.py`, `predict_step.py`, `test_step.py`, `workspace_config.json`

### 3. 전처리

- 역할: 결측치 처리, 스케일링과 범주형 인코딩을 적용합니다.
- 입력: `raw features`
- 출력: `model-ready features`
- 관련 코드: `train_step.py`

### 4. 검증 분리 / CV

- 역할: 재현 가능한 로컬 검증 데이터를 구성합니다.
- 실제 적용:
  - `metrics.json` 기준 검증 방식은 `random_holdout_by_id`입니다.
- 주요 설정:
  - 방식: `random_holdout_by_id`
  - 검증 비율: `0.1`
  - seed: `42`
  - stratify: `False`
- 입력: `training data`, `target`
- 출력: `validation score`, `fold scores or holdout score`
- 관련 코드: `src/baseline.py`, `train_step.py`, `predict_step.py`, `test_step.py`

### 5. 피처 구성

- 역할: 원본 컬럼에서 모델 입력 피처와 파생 피처를 구성합니다.
- 입력: `preprocessed features`
- 출력: `derived features or representations`
- 관련 코드: `train_step.py`, `test_step.py`

### 6. 모델 정의

- 역할: 사용할 모델과 주요 하이퍼파라미터를 정의합니다.
- 실제 적용:
  - 모델 family는 `RuleBasedConstantVelocity`입니다.
- 주요 설정:
  - estimator: `RuleBasedConstantVelocity`
- 입력: `model-ready features`
- 출력: `untrained model or estimator`
- 관련 코드: `src/baseline.py`, `test_step.py`

### 7. 손실함수와 평가 기준

- 역할: 학습 목적과 평가 지표의 관계를 명시합니다.
- 실제 적용:
  - 평가 지표는 `R-Hit@1cm`이며 목표 방향은 `maximize`입니다.
  - 명시적 loss 구현 대신 `RuleBasedConstantVelocity`의 내부 학습 목적을 사용합니다.
- 입력: `model predictions`, `target`
- 출력: `optimization objective`
- 관련 코드: `src/baseline.py`, `test_step.py`, `workspace_config.json`

### 8. 학습

- 역할: 학습 데이터로 모델을 적합합니다.
- 입력: `train split`, `model`, `objective`
- 출력: `trained model`
- 관련 코드: `src/baseline.py`, `train_step.py`, `test_step.py`, `workspace_config.json`

### 9. 로컬 평가

- 역할: 검증 데이터에서 로컬 점수를 계산합니다.
- 실제 적용:
  - `cv_score`는 0.591입니다.
- 주요 설정:
  - CV 점수: `0.591`
- 입력: `validation predictions`, `validation target`
- 출력: `metrics.json`, `cv_score`
- 관련 코드: `src/baseline.py`, `test_step.py`

### 10. 테스트 추론과 제출 파일 생성

- 역할: 테스트 예측과 제출 파일을 생성합니다.
- 실제 적용:
  - `outputs/submission.csv`를 생성합니다.
  - 제출 파일은 `id`, `target` 두 컬럼을 사용합니다.
- 주요 설정:
  - 제출 경로: `outputs/submission.csv`
  - ID 컬럼: `id`
  - 예측 컬럼: `target`
- 입력: `test data`, `trained model`
- 출력: `submission.csv or prediction artifact`
- 관련 코드: `src/baseline.py`, `train_step.py`, `predict_step.py`, `test_step.py`, `workspace_config.json`

## 재현 명령

```bat
C:\Users\ASUS\anaconda3\python.exe test_step.py
C:\Users\ASUS\anaconda3\python.exe train_step.py
C:\Users\ASUS\anaconda3\python.exe predict_step.py
```
