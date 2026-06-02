# Pipeline Improvement Policy

이 문서는 다음 trial에서 파이프라인의 어떤 축을 개선할지 결정하는 기준을 정의한다.

핵심 원칙은 하나다. 한 trial에서는 가능한 한 하나의 주요 개선 축만 바꾸고, 원인 추적을 위해 나머지 축은 보호한다.

## 개선 축

- `validation`: data split, leakage, CV/LB 정합성
- `preprocessing`: 결측치, 정규화, modality별 전처리
- `feature_engineering`: 도메인 feature, interaction, lag, ROI 등
- `augmentation`: 이미지/텍스트/시계열/센서/좌표 증강
- `sampling`: class, group, hard example sampling
- `loss_metric_alignment`: loss, threshold, calibration, metric surrogate
- `hyperparameter`: learning rate, regularization, batch size 등
- `training_recipe`: scheduler, warmup, freeze/unfreeze, early stopping
- `model_architecture`: 같은 family 안의 구조 변경
- `model_family`: tree, CNN, transformer, graph, sequence model 등 family 변경
- `pretraining_strategy`: frozen feature, partial fine-tuning, full fine-tuning, from scratch
- `post_processing`: TTA, smoothing, NMS/WBF, rule correction
- `ensemble_submission`: ensemble, submission cadence, LB strategy
- `compute_backend`: local/Colab/GPU/시간 비용
- `error_analysis`: class/fold/group/view/scenario별 오류 분석
- `human_review`: 사람이 판단해야 하는 라벨, 도메인, 규칙, 의미 검토

## 우선순위 규칙

1. CV/LB 불일치나 leakage 의심이 있으면 `validation`을 먼저 본다.
2. 오류가 특정 class, fold, group, scenario, view, segment에 집중되면 `error_analysis`와 Human Review를 먼저 만든다.
3. 여러 번 정체되거나 예측이 기존 best와 거의 같으면 `model_family`, `model_architecture`, `pretraining_strategy`를 검토한다.
4. class imbalance나 특정 class 오류가 크면 `sampling` 또는 `loss_metric_alignment`를 검토한다.
5. threshold, calibration, metric mismatch 신호가 있으면 모델 변경보다 `loss_metric_alignment`를 먼저 본다.
6. 명확한 이슈가 없고 개선 여지가 있으면 `hyperparameter` 또는 `training_recipe`로 controlled refinement를 진행한다.

## 보호 규칙

- validation 의심이 있으면 model family, architecture, pretraining strategy를 동시에 바꾸지 않는다.
- model family를 바꿀 때는 validation과 주요 preprocessing을 고정한다.
- augmentation, loss, sampling을 동시에 크게 바꾸지 않는다.
- 사람이 봐야 할 오류 패턴이면 review pack을 먼저 만든다.

## 산출물

`plan-improvement` 또는 `plan-next` 흐름은 다음 파일을 만든다.

```text
experiments/<competition>/<trial_id>/pipeline_improvement_plan.json
experiments/<competition>/<trial_id>/pipeline_improvement_plan.md
```

주요 필드:

- `primary_axis`
- `secondary_axes`
- `protected_axes`
- `requires_human_review`
- `candidate_actions`
- `success_criteria`
- `do_not_change`
- `evidence_used`

이 산출물은 다음 trial의 `next_experiment.md`와 `code_patch_plan.md/json`이 어떤 축을 바꿀지 정하는 근거로 사용한다.
