# Diagnose And Human Review Policy

## 목적

이 문서는 trial 진단 결과를 바탕으로 언제 사람에게 의견을 물어볼지 정의한다.

Human Review는 이미지/영상/ROI 대회에만 쓰는 예외 기능이 아니다. 모든 Kaggle 대회에서 에이전트가 불확실성, 도메인 판단, 큰 전략 전환, 제출 위험을 사용자에게 설명하고 의견을 받기 위한 범용 의사결정 노드다.

## 기본 원칙

- metrics만으로 명확한 판단은 rule-based로 처리한다.
- 도메인 의미 해석이 필요한 경우 사람에게 묻는다.
- 큰 비용, 큰 코드 변경, 제출 전략 변경은 사용자 확인을 우선한다.
- 사용자의 판단은 `memory/<competition>/user_feedback.jsonl`에 저장한다.
- 사용자 판단이 다음 trial 계획에 반영되면 `decision_log.jsonl`에 `user_input_used: true`로 남긴다.

## 진단 입력

진단과 review 판단은 다음 자료를 사용한다.

- `metrics.json`
- `evaluation.md`
- `diagnosis.md`
- `confusion_matrix`, `classification_report`, `segment_errors` 같은 metric artifact
- `submission_result.md`
- `memory/<competition>/trial_index.jsonl`
- `memory/<competition>/decision_log.jsonl`
- `memory/<competition>/user_feedback.jsonl`
- `competitions/<competition>/data_notes.md`
- optional policy file: `configs/policies/human_review_policy.yaml`

## Human Review 결정값

```text
no_review
request_review
request_approval
prepare_review_pack
blocked_until_feedback
```

의미:

- `no_review`: 자동으로 다음 단계 진행
- `request_review`: 사용자 의견을 받아 다음 전략에 반영
- `request_approval`: 비용, 제출, 큰 변경 전 승인 필요
- `prepare_review_pack`: 사람이 볼 자료를 먼저 생성
- `blocked_until_feedback`: 사용자 판단 없이는 진행 위험이 큼

## 공통 트리거

다음 중 하나라도 강하게 만족하면 human review를 고려한다.

### 1. 특정 구간에 오류가 집중됨

예:

- 특정 scenario/view/group/fold에서 오류율이 높다
- 특정 class pair 혼동이 반복된다
- 특정 data source 또는 time period에서만 실패한다

결정:

```text
prepare_review_pack -> request_review
```

### 2. 라벨 경계가 애매함

예:

- Bed Exit/Wandering 전이 구간
- positive/negative label 기준이 사람마다 다를 수 있음
- NLP label ambiguity
- time-series anomaly boundary

결정:

```text
request_review
```

### 3. Validation 또는 leakage 의심

예:

- CV는 개선됐지만 LB가 악화됨
- public LB와 local CV가 반복적으로 어긋남
- feature leakage warning 존재
- group split 또는 time split이 의심됨

결정:

```text
request_review 또는 request_approval
```

### 4. 안전 중요 클래스의 false negative

예:

- Fall false negative
- fraud/malware/disease 같은 위험 클래스 recall 하락

결정:

```text
blocked_until_feedback
```

단, competition 목적과 metric이 안전 recall보다 leaderboard score를 우선하는지 사용자가 판단해야 한다.

### 5. 큰 전략 전환

예:

- model family 변경
- architecture redesign
- validation strategy 변경
- SOTA solution 도입
- ensemble/blending 구조 도입
- 대용량 Colab/GPU 실행

결정:

```text
request_approval
```

### 6. 반복 실패

조건:

- 최근 N개 trial이 개선되지 않음
- 같은 방향의 작은 변경이 seed noise 안에 머묾
- current focus가 포화된 것으로 보임

결정:

```text
request_review
```

질문:

- 현재 방향을 계속 갈지
- validation을 재검토할지
- 모델 계열을 바꿀지
- 외부 solution/SOTA를 조사할지

## 대회 유형별 예시

### 이미지/영상

Review 대상:

- 오분류 sample image/frame
- Grad-CAM 또는 attention visualization
- ROI/box/keypoint overlay
- 라벨 경계 사례

질문:

- 사람이 보기에도 label이 맞는가?
- 모델이 보는 feature가 실제 의미와 맞는가?
- ROI나 crop이 중요한 정보를 잘라내는가?

### Tabular

Review 대상:

- feature importance
- suspicious high-leakage features
- validation fold별 score
- train/test distribution shift

질문:

- 이 feature가 대회 시점에 실제로 사용 가능한 정보인가?
- group/time split이 더 적절한가?
- LB를 믿을지 CV를 믿을지 판단 근거가 있는가?

### Time Series

Review 대상:

- error concentration by time
- window boundary samples
- seasonality/anomaly segments
- train/test temporal split

질문:

- windowing이 label 의미를 왜곡하는가?
- 미래 정보가 섞였을 가능성이 있는가?
- 특정 기간만 과적합하고 있는가?

### NLP

Review 대상:

- ambiguous text samples
- class confusion examples
- prompt/template variants
- tokenization edge cases

질문:

- 라벨 기준이 일관적인가?
- 모델이 domain-specific 표현을 놓치고 있는가?
- prompt 또는 preprocessing을 바꿔야 하는가?

## Review 요청에 포함할 질문 유형

질문은 막연하면 안 된다. 아래 유형 중 하나 이상으로 구체화한다.

```text
label_question
validation_question
feature_question
execution_approval
submission_approval
strategy_shift_question
visual_semantic_question
data_quality_question
```

질문 예:

- 이 split은 대회 데이터 구조상 안전한가?
- 이 feature는 test 시점에 사용할 수 있는 정보인가?
- 이 오분류 sample은 실제로도 애매한가?
- Colab으로 2시간 실행할 만큼 이 변경의 기대 이득이 있는가?
- public LB가 하락했는데 이 trial을 버릴지 더 검증할지?

## Review 이후 처리

사용자 피드백은 다음 위치에 저장한다.

```text
memory/<competition>/user_feedback.jsonl
experiments/<competition>/<trial_id>/user_review_response.md
```

기본 필드:

```yaml
time:
competition:
trial_id:
topic:
question:
user_feedback:
decision:
follow_up_action:
```

decision 후보:

```text
continue
change_validation
change_feature
change_model_family
prepare_sota_research
skip_submission
approve_submission
approve_colab
stop_trial
```

## Decision Log 연동

Human Review 관련 판단은 `decision_log.jsonl`에 남긴다.

```yaml
decision_type: human_review
decision: request_review
reason: Segment errors are concentrated in C1 Bed Exit/Wandering cases.
evidence:
  segment: C1
  confused_pair: Bed Exit/Wandering
  error_count: 13
user_input_used: false
next_action: prepare_review_pack
```

피드백 반영 후:

```yaml
decision_type: strategy_revision
decision: change_feature
reason: User confirmed ROI definition may be causing semantic mismatch.
user_input_used: true
next_action: plan_roi_feature_revision
```

## LLM 호출 기준

LLM은 다음 경우에만 호출한다.

- 진단 결과를 사용자에게 보여줄 review 요약으로 압축해야 함
- 여러 오류 패턴 중 사람이 볼 샘플을 고르는 기준을 설명해야 함
- 사용자 피드백을 다음 trial 전략으로 바꾸는 판단이 필요함
- 큰 전략 전환 후보를 비교해야 함

LLM에 raw training log 전체를 넣지 않는다. 필요한 경우 Python/tool이 먼저 summary artifact를 만든다.

## 성공 기준

- 사람에게 묻는 이유가 명확하다.
- 사람에게 보여줄 자료와 질문이 구체적이다.
- 피드백이 memory에 남고 다음 trial에 반영된다.
- Human Review가 자율성을 방해하는 예외가 아니라 안정성을 높이는 정식 branch로 작동한다.
# Review Timing

Human Review는 모든 trial마다 요청하지 않는다.

- `request_now`: leakage, label boundary, safety false negative, 필수 정보 누락 또는 성숙한 파이프라인의 review trigger
- `defer`: 비긴급 trigger가 있지만 비교 가능한 완료 trial이 부족함
- `no_review`: 사람의 판단이 필요한 trigger가 없음

비긴급 review의 기본 성숙도는 Execution Profile ready, workspace run completed, metrics collected, 완료 trial 2개 이상이다. 보류된 trigger는 competition별 queue에 누적해 성숙 시점의 review pack에 합친다.

기존 review pack이 사용자 피드백 대기 중이면 후속 비긴급 요청은 다시 보내지 않는다. 긴급 trigger만 이 대기 상태를 우회한다.
