# Workspace Result Cycle Design

## Goal

수집된 trial metrics를 기존 평가·진단·memory 흐름에 연결하되, Human Review는 긴급성 및 파이프라인 성숙도에 따라 즉시 요청하거나 보류한다.

## Result Flow

```text
metrics_collection precondition
-> duplicate-memory guard
-> evaluate
-> diagnose
-> review timing decision
-> remember
-> request_now | defer | no_review
```

객관적 실험 결과는 review timing과 무관하게 memory에 기록한다. `request_now`일 때만 다음 개선 계획을 사용자 피드백까지 중단한다.

## Review Timing

- `request_now`: 즉시 review pack 생성, 다음 계획 보류
- `defer`: 질문을 competition queue에 누적하고 다음 trial 진행 가능
- `no_review`: review 없이 다음 trial 진행 가능

즉시 요청 trigger:

- validation 또는 leakage 의심
- label/metric 의미나 필수 입력의 모호성
- 안전 중요 false negative
- 실행 지속에 필요한 blocking 정보

비긴급 trigger:

- 오류 집중 구간의 도메인 해석
- feature 의미 검토
- 반복 실패에 따른 전략 전환
- 모델·증강·전처리 방향 판단

## Pipeline Maturity

비긴급 review를 요청하려면 아래 조건이 모두 충족되어야 한다.

- Execution Profile status가 `ready`
- workspace run status가 `completed`
- metrics collection status가 `collected`
- 현재 trial을 포함한 완료 trial이 2개 이상

정상 baseline 1건만 있는 동안 비긴급 질문은 `defer`한다. 두 번째 비교 가능한 결과가 생기면 누적 질문을 현재 review pack에 합쳐 한 번에 요청한다.

## Deferred Queue

`memory/<competition>/deferred_review_queue.json`에 trigger별로 다음을 저장한다.

- trigger
- first_trial
- last_trial
- occurrences
- issues
- questions

같은 trigger는 중복 항목을 만들지 않고 횟수와 근거를 갱신한다. `request_now` review pack 생성에 성공하면 queue를 비운다.

## Idempotency

trial index에 같은 competition/trial이 이미 존재하면 `already_processed`를 반환하고 evaluation/memory를 다시 기록하지 않는다.

## Outputs

```text
experiments/<competition>/<trial>/workspace_result_cycle.json
experiments/<competition>/<trial>/workspace_result_cycle.md
memory/<competition>/deferred_review_queue.json
```

## Scope Boundary

5차는 다음 실험 계획이나 코드 작성을 실행하지 않는다. 결과 상태의 `next_action`만 `plan-next-experiment` 또는 `request-user-review`로 기록한다.

