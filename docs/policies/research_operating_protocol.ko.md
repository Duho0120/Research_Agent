# Research Operating Protocol

이 문서는 특정 대회 방식에 종속되지 않는 최소 공통 연구 흐름을 정의한다.

## 전역 원칙

- 바로 코드를 수정하지 않고 현재 상태와 근거를 먼저 확인한다.
- 다음 실험에서는 주요 개선축 하나만 선택한다.
- 코드 수정 계획과 검증을 거친 뒤 실행한다.
- 원본 데이터와 보호 파일을 수정하지 않는다.
- 비용이 크거나 외부 영향을 주는 작업은 사용자 승인을 받는다.
- 사람의 판단이 필요한 경우 근거와 질문을 함께 제공한다.
- 실험 결과와 판단 근거를 구조화해 저장한다.

## 표준 출력

```text
Current State
Evidence
Issues
Candidate Actions
Recommended Action
Constraints
User Questions
Execution Plan
```

`Candidate Actions`는 기본적으로 단순 목록이며 특정 개수나 등급을 강제하지 않는다.

## 선택 확장

CV/LB 비교, leaderboard 기준선, target별 규칙, 후보 등급 같은 대회 특화 판단은 전역 규칙이 아니다.
필요한 대회에서만 다음 파일로 명시적으로 활성화한다.

```text
configs/<competition>/research_policy.yaml
```

예시:

```yaml
leaderboard_tracking:
  enabled: true
  score_field: lb_score
  affects_strategy: false
  ask_user_on_conflict: true
```

설정 파일이 없으면 Research Protocol은 leaderboard를 판단 근거로 사용하지 않는다.
