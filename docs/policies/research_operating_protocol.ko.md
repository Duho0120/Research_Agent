# Research Operating Protocol

이 문서는 자율 ML 연구 에이전트가 실험을 순차적으로 진행할 때 따르는 최소 공통 원칙을 정의한다.

## 핵심 원칙

- 첫 실험은 문제 이해, 기본 파이프라인 구성, 로컬 실행 확인에 필요한 정보를 충분히 사용한다.
- 두 번째 실험부터는 전체 재설계가 아니라 이전 기준 실험 위에 하나의 개선축만 수정한다.
- 코드는 전체 재작성보다 patch 기반 수정을 우선한다.
- 원본 데이터와 제출 파일은 코드 작성 대상에서 보호한다.
- 비용이 크거나 위험한 변경, 큰 구조 변경, 사람의 판단이 필요한 변경은 사용자 확인 대상으로 남긴다.
- 모든 trial 결과와 판단 근거는 구조화해서 다음 trial의 입력으로 사용한다.

## Best Base + Active Axis 규칙

두 번째 실험부터는 `best trial`과 `active axis`를 분리해서 다룬다.

- `best trial`은 다음 trial의 코드와 파이프라인 기준점이다.
- `active axis`는 현재 실험 중인 개선축이다.
- active axis가 있고 시도 횟수가 3회 미만이면, best trial과 active axis의 출처가 달라도 active axis를 유지한다.
- 이 경우 다음 실험은 best trial의 코드/파이프라인 위에 active axis의 새로운 후보 또는 파라미터 변형 하나만 적용한다.
- 같은 active axis에서 실패한 후보는 반복하지 않는다.
- active axis가 3회 연속 개선되지 않으면 해당 축을 보류하거나 기각하고 다음 개선축을 검토한다.

예시:

```text
trial_001: baseline, score 0.80
trial_002: model_family 개선, score 0.85 -> best
trial_003: feature_engineering 축 시도, score 0.79 -> 실패, active_axis=feature_engineering, attempt=1/3

trial_004 계획:
base code      = trial_002
active axis    = feature_engineering
next change    = trial_003 후보를 반복하지 않는 feature_engineering 후보 1개
```

## RAG 사용 원칙

RAG는 매 trial마다 많은 문서를 넣기 위한 장치가 아니라, 필요한 근거를 찾는 선택적 연구 메모리다.

- 첫 실험에서는 대회/데이터 이해를 위해 RAG context pack을 사용할 수 있다.
- active axis가 3회 미만으로 진행 중이면 planning RAG를 생략하고 decision context와 best trial 요약을 우선한다.
- 새로운 개선축을 선택해야 하거나, 기존 축이 막혔거나, 외부 자료/논문/사용자 메모가 필요한 경우에 RAG를 다시 사용한다.
- user-facing 문서는 LLM 입력보다 사람 확인용 산출물로 취급한다.

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

`Candidate Actions`는 단순 목록이며, 특정 개수나 등급을 강제하지 않는다.

## 대회별 확장

CV/LB 비교, leaderboard 기준, target별 규칙, 후보 등급 같은 특화 판단은 전역 규칙이 아니다.
필요한 대회에서만 다음 파일로 명시적으로 작성한다.

```text
configs/<competition>/research_policy.yaml
```

설정 파일이 없으면 Research Protocol은 leaderboard를 자동 판단 근거로 사용하지 않는다.
