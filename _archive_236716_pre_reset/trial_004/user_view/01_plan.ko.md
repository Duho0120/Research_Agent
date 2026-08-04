# trial_004 실험 계획

| 항목 | 값 |
|---|---|
| 계획 유형 | `delta_patch` |
| 계획명 | sample template 병합, 열 순서 지정 및 float 정밀도를 통한 submission 정렬 |
| 기준 trial | `trial_003` |
| 개선축 | `submission_schema_alignment` |

## 목적

`236716`에 대한 로컬 실험을 하나 생성하고 실행합니다.

## 왜 하는가

사용 가능한 Execution Profile과 로컬 metric artifact를 활용하여 실험 루프가 정상적으로 작동하는지 확인합니다.

## 그대로 유지

- `data_split_cv`
- `model_definition`
- `preprocessing`

## 이번 회차 변경

- `predict_step.main`을 수정하여 다음을 수행합니다.
  - 예측값을 `sample_submission`과 병합하여 `id` 정렬을 맞춥니다.
  - 정확한 열 순서 `['id','x','y','z']`를 적용합니다.
  - 예측값을 `float32`로 변환합니다.
  - `float_format='%.6f'`, `index=False`로 CSV를 저장합니다.

### 후보 구현

- 이름: `submission_schema_alignment: MergeToTemplate | EnforceOrder+Float6`
- 설명: 예측값을 `sample_submission`에 병합하여 `id`의 포함 범위와 순서를 유지하고, 정확한 열 `[id,x,y,z]`과 소수점 6자리 형식을 적용해 submission을 작성합니다.
- 구현 힌트: `predict_step.py`의 `main()`에서 `['id','x','y','z']` 열을 사용해 `preds` DataFrame을 만든 후, `template = _read_sample_submission(); out = template[['id']].merge(preds, on='id', how='left'); out = out[['id','x','y','z']]; for c in ['x','y','z']: out[c] = out[c].astype('float32'); out.to_csv(SUBMISSION_PATH, index=False, float_format='%.6f')`를 적용합니다.

## 성공 기준

- `outputs/submission.csv`에 정확히 `id,x,y,z` 열이 이 순서대로 존재하고, 모든 테스트 `id`가 포함됩니다.
- LB 스키마 오류가 발생하지 않고 점수가 정상적으로 로드됩니다.
- 로컬 `R-Hit@1cm >= 0.591`을 달성합니다.

## 실패 시 판단

- 스키마가 여전히 맞지 않거나 행 수가 일치하지 않으면 이번 패치를 되돌립니다.
- 다음 시도에서는 `src/baseline.py`의 writer를 사용하여 열 순서와 형식을 강제합니다.