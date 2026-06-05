# Research Operating Protocol

이 문서는 자율 연구 에이전트가 어떤 대회나 데이터셋을 받더라도 따라야 하는 연구 진행 방식이다.

핵심 원칙:

- 바로 코드부터 수정하지 않는다.
- 먼저 metric, objective, validation, current best, public/leaderboard evidence, 실패 이력을 확인한다.
- local best와 public best를 분리해서 다룬다.
- CV 개선과 leaderboard 개선이 충돌하면 validation review를 우선한다.
- 다음 실험은 safe, main, aggressive 후보로 나누고 safe 후보를 먼저 검토한다.
- 모델 변경은 하나의 선택지일 뿐이며 validation, data, feature, calibration, post-processing, human review보다 항상 우선하지 않는다.
- 코드 작성은 research protocol, pipeline improvement plan, patch validation, coding handoff를 거친 뒤 진행한다.

표준 출력 섹션:

```text
Current State
Evidence
Risk
Candidate Actions
Recommended Next Trial
Do Not Change
Need User Check
Execution Plan
```

운영 규칙:

- Public 또는 external leaderboard에서 검증된 baseline은 anchor로 보존한다.
- local-only 개선은 leaderboard evidence가 생기기 전까지 trusted best로 승격하지 않는다.
- 작은 데이터, 적은 subject/group, CV/LB conflict, leakage warning, 높은 prediction correlation은 risk flag로 기록한다.
- 사람이 확인해야 하는 target 의미, 제출 제한, public/private split, 안전한 제출 후보 여부는 `Need User Check`에 남긴다.
