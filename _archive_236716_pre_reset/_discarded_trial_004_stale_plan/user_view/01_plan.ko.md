# trial_004 실험 계획

| 항목 | 값 |
|---|---|
| 계획 유형 | `delta_patch` |
| 계획명 | `Submission schema: mirror sample (id,x,y,z)` |
| 기준 trial | `trial_003` |
| 개선축 | `submission_schema_alignment` |

## 목적

`236716`을 대상으로 로컬 실험을 하나 생성하고 실행합니다.

## 왜 하는가

사용 가능한 Execution Profile과 로컬 metric artifact를 활용해 실험 루프가 정상적으로 작동하는지 확인합니다.

## 그대로 유지

- `data_split_cv`
- `model_definition`
- `preprocessing`
- `training`

## 이번 회차 변경

- `predict_step._read_sample_submission`을 수정하여 예상 출력 컬럼을 반환합니다. 이때 `'id'`는 제외합니다.
- `predict_step.main`에서 예측을 생성한 후, `np.asarray`를 사용해 예측 결과를 `(N, len(expected_cols))` 형태로 맞춥니다.
- 필요한 경우 누락된 차원은 채우고, 초과 차원은 잘라냅니다.
- `id`와 `expected_cols`를 사용해 submission을 구성합니다.
- `test_step.main`에서도 동일한 저장 경로를 적용하여 로컬 테스트 출력이 같은 컬럼을 사용하도록 합니다.
- 결과를 `float32`로 변환한 뒤 `outputs/submission.csv`에 저장합니다.

### 후보 구현

- 후보명: `MirrorSampleSchemaColumns`
- `sample_submission`의 헤더를 그대로 반영하여 submission을 생성합니다. 예상 헤더는 `id,x,y,z`이며, 모델 출력을 해당 컬럼에 매핑합니다.
- `predict_step.main`에서 `sample_submission`을 읽어 `expected_cols`를 가져옵니다. 이때 `'id'`는 제외합니다.
- 예측 결과를 `(N, D)` 형태로 정리하여 `expected_cols`에 매핑합니다.
- `D < len(expected_cols)`이면 누락된 차원을 `0.0`으로 채우고, `D > len(expected_cols)`이면 초과 차원을 잘라냅니다.
- `['id']+expected_cols`를 사용해 DataFrame을 작성합니다.
- `model/split/preproc`은 변경하지 않습니다.

## 성공 기준

- `outputs/submission.csv`의 컬럼이 `sample_submission`과 정확히 일치합니다: `id,x,y,z`
- 로컬 CV score 동작에 변경이 없습니다.
- Submission 파일이 schema 오류 없이 evaluator에서 승인됩니다.

## 실패 시 판단

Submission schema 불일치가 계속되거나 submission 작성 중 runtime error가 발생하면, 이 시도를 해당 개선축에서 실패한 것으로 기록합니다. 이후 동일한 개선축 내 다음 시도에서 대체 매핑 전략을 준비합니다.