# Workspace Result Cycle

5차 구현은 수집된 trial metrics를 `evaluate -> diagnose -> remember` 흐름에 연결하고, Human Review 요청 시점을 파이프라인 성숙도에 따라 조절한다.

## 실행

```powershell
python -B -m research_agent.cli process-workspace-result `
  --competition <workspace> `
  --trial trial_001
```

선행 `metrics_collection.json` 상태가 `collected`여야 한다.

## 처리 순서

```text
metrics collection 확인
-> 동일 trial 중복 확인
-> evaluate
-> diagnose
-> Human Review timing 판단
-> remember
```

## Human Review Timing

- `request_now`: review pack을 만들고 다음 개선 계획을 사용자 피드백까지 보류
- `defer`: 질문을 queue에 저장하고 다음 trial 진행 허용
- `no_review`: review 없이 다음 trial 진행 허용

비긴급 질문은 Execution Profile, workspace run, metrics collection이 모두 정상이고 현재 trial을 포함한 완료 trial이 2개 이상일 때 요청한다. 첫 baseline에서는 오류 집중이나 전략 해석 질문을 queue에 모은다.

다음 문제는 파이프라인 성숙도와 관계없이 즉시 요청한다.

- validation 또는 leakage 의심
- label 경계 모호성
- 안전 중요 클래스 false negative
- 필수 metric·label·input 정의 누락

## Deferred Review Queue

```text
memory/<competition>/deferred_review_queue.json
```

같은 trigger는 하나의 항목으로 합치며 `occurrences`, 최초/최근 trial, issues, questions를 갱신한다. 파이프라인이 성숙하거나 긴급 review가 발생하면 누적 내용을 현재 review pack에 포함하고 queue를 비운다.

이전 review pack이 아직 `pending_user_feedback`이면 새로운 비긴급 review pack을 만들지 않는다. 해당 질문은 queue에 보류하며, leakage 같은 긴급 trigger만 pending 상태를 우회한다.

## Memory와 Review 분리

실험 결과는 `request_now`, `defer`, `no_review` 모두 memory에 기록한다. Human Review가 필요한 경우에도 객관적인 결과를 잃지 않으며, 사용자 피드백까지 중단되는 것은 다음 개선 계획이다.

같은 trial이 이미 trial index에 있으면 `already_processed`를 반환해 memory 중복 기록을 막는다.

## 상태

- `completed`: review 없이 처리 완료
- `completed_review_deferred`: 결과 기록 완료, review는 안정화 시점까지 보류
- `awaiting_human_review`: review pack 생성 완료, 사용자 피드백 대기
- `already_processed`: 기존 memory 기록을 재사용
- `blocked`: metrics collection 선행 조건 불충족

## 산출물

```text
experiments/<competition>/<trial>/evaluation.md
experiments/<competition>/<trial>/diagnosis.md
experiments/<competition>/<trial>/workspace_result_cycle.json
experiments/<competition>/<trial>/workspace_result_cycle.md
experiments/<competition>/<trial>/review_pack/  # request_now일 때만
memory/<competition>/trial_index.jsonl
memory/<competition>/deferred_review_queue.json
```

5차는 다음 실험 계획이나 코드 작성을 직접 실행하지 않는다.
