# 개선 후보

## 1순위 후보: View-Aware Bed Exit/Wandering 분리

문제:

- C1이 가장 높은 validation error rate를 보입니다. 255개 window 중 43개 오류, error rate 0.1686입니다.
- 핵심 semantic error는 Bed Exit/Wandering 혼동입니다.
- C1에서만 Bed Exit/Wandering 혼동 window가 13개 있습니다.

제안 변경:

- Transformer와 validation split은 고정합니다.
- C1에 민감한 case를 위해 view-aware ROI geometry feature를 추가하거나 강조합니다.
- 전체 accuracy만 보지 말고 C1 error rate를 핵심 지표로 추적합니다.

먼저 시도하는 이유:

- 전체 모델 복잡도를 크게 늘리지 않고 관찰된 병목을 직접 겨냥합니다.
- Fall 성능은 이미 강하므로, 이 실험은 Fall recall을 흔들지 않는 방향이어야 합니다.

## 2순위 후보: Boundary Window Sampling

문제:

- Bed Exit과 Wandering은 전이 frame 주변에서 가장 구분이 어렵습니다.
- 여러 혼동 case가 가까운 frame range의 인접 window에서 발생합니다.

제안 변경:

- Bed Exit과 Wandering label이 바뀌는 boundary window를 제한적으로 oversampling합니다.
- 특정 시나리오를 외우지 않도록 augmentation은 약하게 유지합니다.

## 3순위 후보: Selection Metric Reweighting

문제:

- validation support에서 Fall 비중이 커서 accuracy가 높게 보일 수 있습니다.
- 실제 연구 목표는 Bed Exit/Wandering 분리와 Fall 안전성 유지에 더 가깝습니다.

제안 변경:

- checkpoint 선택 기준을 아래 조합 점수로 바꿔봅니다.
  `0.45 * BedExit_F1 + 0.35 * MacroF1 + 0.20 * FallRecall`
- raw accuracy는 보고용 지표로만 유지합니다.

## 먼저 시도하지 않을 것

- 오류 패턴을 먼저 고치기 전에 model depth나 d_model부터 키우는 것.
- threshold sweep 결과 macro F1이 낮아졌으므로, Bed Exit threshold-only 튜닝만 하는 것.
- model 변경과 validation split 변경을 동시에 하는 것.

