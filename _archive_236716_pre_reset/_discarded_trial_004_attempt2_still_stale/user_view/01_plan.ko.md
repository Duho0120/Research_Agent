# trial_004 실험 계획

| 항목 | 값 |
|---|---|
| 계획 유형 | delta_patch |
| 계획명 | Mirror sample_submission schema in submission.csv |
| 기준 trial | trial_001 |
| 개선축 | submission_schema_alignment |

## 목적

236716에 대해 로컬 실험을 하나 생성하고 실행합니다.

## 왜 하는가

사용 가능한 Execution Profile과 로컬 metric artifact를 활용하여 실험 루프를 검증합니다.

## 그대로 유지

- `data_load`
- `preprocessing`
- `data_split_cv`
- `model_definition`
- `training`

## 이번 회차 변경

- `predict_step._read_sample_submission`: `(df, id_col=first column, pred_cols=columns[1:])`을 반환합니다.
- `predict_step.main`:
  - `submission = sample_df.copy()`를 생성합니다.
  - `predictions`를 사용해 `submission[pred_cols]`를 채웁니다.
  - `pred_cols`보다 차원이 적으면 남은 값을 `0.0`으로 채웁니다.
  - `float32`를 보장합니다.
  - `'target'`이 있으면 삭제합니다.
  - 컬럼 순서를 유지합니다.
- `test_step.main`: `sample_submission` 헤더와 동일한 채우기 로직을 사용하도록 로컬 파일 기록 방식을 맞춥니다.

### 후보

**이름:** SampleSubmission Schema Mirror

**설명:** 현재의 `id,target` 레이아웃을 `id,x,y,z`와 같이 변경하여 `submission.csv`의 컬럼과 순서가 `sample_submission.csv`와 정확히 일치하도록 합니다.

**구현 힌트:** `predict_step.py`에서 `sample_submission`을 읽어 헤더를 가져옵니다. 동일한 컬럼 순서와 자료형으로 `submission`을 구성하고, 모델 출력을 `pred_cols`에 매핑합니다. 누락된 차원은 `0.0`으로 채우고, `index=False`로 CSV를 기록합니다.

## 성공 기준

- `outputs/submission.csv`의 컬럼과 순서가 `sample_submission.csv`와 정확히 일치합니다.
- 로컬 실행 중 스키마 또는 키 오류가 발생하지 않습니다.
- 서버가 스키마 거부 없이 제출을 수락합니다.

## 실패 시 판단

제출 헤더 불일치가 계속되거나 파일 기록에 실패하면 이 후보를 실패로 판단하고, 다음 시도에서도 동일한 개선축을 유지합니다.