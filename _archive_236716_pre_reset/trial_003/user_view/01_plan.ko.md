# trial_003 실험 계획

| 항목 | 값 |
|---|---|
| 계획 유형 | `continuation_delta_plan` |
| 계획명 | trial_001로 롤백하고 submission schema를 `id,x,y,z`에 맞춤 |
| 기준 trial | `trial_002` |
| 개선축 | `submission_schema_alignment` |

## 목적

trial_001의 동작을 기준으로 모델과 데이터 분할은 변경하지 않고, 제출 파일의 예측 컬럼과 평가 경로를 `id,x,y,z` 형식에 맞춥니다.

## 왜 하는가

현재 제출 예측 컬럼이 `['target']`으로 설정되어 있어 3D 좌표 예측 형식과 일치하지 않습니다. 제출 스키마를 `id,x,y,z`로 통일하면 샘플 제출 파일 및 내부 검증기의 예상 형식에 맞출 수 있습니다.

## 그대로 유지

- Train File: `train_labels.csv`
- Sample Submission: `sample_submission.csv`
- Id Column: `id`
- Method: `random_holdout_by_id`
- Test Size: `0.1`
- Random State: `42`
- Stratify: `False`
- 전처리: `none`
- Family: `RuleBasedConstantVelocity`
- Training: `none (stateless)`
- Name: `R-Hit@1cm`
- Objective: `maximize`

## 이번 회차 변경

- **Test Inference Output**: 제출 예측 컬럼을 `['target']`에서 `['x','y','z']`로 변경합니다.
- **Test Inference Output**: `outputs/submission.csv`를 작성할 때 컬럼 순서를 정확히 `id,x,y,z`로 유지하고, `x,y,z`에 `float32`를 적용합니다.
- **Predict Step.Py**: `_read_sample_submission`이 `['id','x','y','z']` 헤더를 예상하고 전달하도록 업데이트합니다.
- **Predict Step.Py**: constant-velocity estimate에서 `['x','y','z']` 키/컬럼을 사용하는 dict/array가 반환되도록 prediction assembly를 수정합니다.
- **Predict Step.Py**: underlying predictor가 3-vector를 반환하면 `(x,y,z)`에 직접 매핑하고, 각 id에 대해 `shape == (3,)`인지 확인합니다.
- **Test Step.Py**: 평가 경로가 ground-truth와 predictions를 3D로 읽고, `R-Hit@1cm`에 대해 `_euclid`를 사용하여 Euclidean distance를 계산하도록 합니다.
- **Workspace Config.Json**: submission prediction_columns metadata를 `['x','y','z']`로 업데이트합니다.

## 성공 기준

- `outputs/submission.csv` 헤더가 정확히 `['id','x','y','z']`이고, 행 수가 sample_submission과 일치합니다.
- 모든 `x,y,z`가 유한한 실수이며, NaN/inf가 없습니다. dtype은 `float32` 또는 `float64`입니다.
- 로컬 `cv_score`가 trial_001의 동작과 일치하며, schema만 변경했을 때 성능 저하가 없습니다.
- 내부 schema validator를 통과하며, 예상하지 않은 컬럼이 없습니다.

## 실패 시 판단

- 헤더/순서 또는 dtype 검사가 실패하면 `predict_step`의 컬럼 매핑을 수정하고, model/split은 변경하지 않은 채 다시 실행합니다.
- `cv`가 예기치 않게 악화되면 코드 diff를 되돌리고 I/O만 다시 확인합니다. 이 개선축에서는 model 또는 split을 변경하지 않습니다.