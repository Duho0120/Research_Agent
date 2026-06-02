# trial_002_c1_bed_wandering_focus 실험 계획

## 가설

현재 baseline은 전체적으로 고르게 실패하는 것이 아니라, 특정 조건에서 오류가 몰리고 있습니다. 특히 C1 view와 Bed Exit/Wandering 전이 구간에서 오류가 집중되므로, 다음 실험은 모델 크기를 키우기보다 **view에 민감한 전이 구간 모호성**을 줄이는 데 집중합니다.

## 제안 변경

- Transformer 구조와 validation split은 그대로 유지합니다.
- C1에서 민감하게 흔들리는 ROI geometry를 보완할 수 있도록 view-aware 또는 scenario-aware 진단 feature 경로를 추가합니다.
- Fall 처리 방식은 건드리지 않고, Bed Exit/Wandering 경계 window에 더 집중합니다.
- Bed Exit recall, Wandering precision, C1 error rate, Fall recall을 함께 추적합니다.

## Baseline에서 이 실험으로 이어지는 이유

- Baseline의 macro F1은 약 0.83이고 Fall F1은 약 0.95로 이미 높은 편입니다.
- 가장 명확하게 개선 가능한 문제는 Bed Exit/Wandering 혼동입니다.
- C1 error rate는 0.1686으로, C3의 0.0772보다 훨씬 높습니다.
- 00620_H_D_SY 시나리오에서 60개 window 중 23개 오류가 발생해, 일부 edge case에 오류가 몰리는 것으로 보입니다.

## 성공 기준

- Bed Exit F1이 0.7792보다 높아집니다.
- Macro F1이 0.8326 이상을 유지합니다.
- Fall recall이 0.96 이상을 유지합니다.
- C1 error rate가 0.15 아래로 감소합니다.
- C1에서 Bed Exit/Wandering 혼동 수가 13개 미만으로 줄어듭니다.

## 위험 요소

- C1 또는 특정 시나리오에 과도하게 맞추면 일반화 성능이 떨어질 수 있습니다.
- baseline에서 Bed Exit threshold를 낮추는 방식은 recall은 일부 개선했지만 macro F1을 낮췄으므로, threshold-only 튜닝만으로는 충분하지 않습니다.
