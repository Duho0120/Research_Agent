# Review Pack Schema

## 목적

이 문서는 Human Review가 필요할 때 에이전트가 사용자에게 보여줄 자료, 질문, 답변 저장 형식을 정의한다.

Review Pack은 “사람에게 그냥 물어보기”가 아니라, 사람이 짧은 시간 안에 판단할 수 있도록 증거를 정리한 패키지다.

## 생성 위치

기본 위치:

```text
experiments/<competition>/<trial_id>/review_pack/
```

기존 호환 파일:

```text
experiments/<competition>/<trial_id>/user_review_request.md
experiments/<competition>/<trial_id>/user_review_response.md
```

`user_review_request.md`는 사람이 바로 읽는 단일 요약 파일로 유지하고, `review_pack/`은 자료 묶음의 표준 위치로 사용한다.

## 표준 폴더 구조

```text
review_pack/
  manifest.json
  summary.ko.md
  questions.ko.md
  cases.jsonl
  metrics_snapshot.json
  artifact_index.jsonl
  human_feedback.md
  human_feedback.json
  assets/
    images/
    frames/
    plots/
    tables/
```

필수 파일:

- `manifest.json`
- `summary.ko.md`
- `questions.ko.md`
- `cases.jsonl`
- `metrics_snapshot.json`

선택 파일:

- `artifact_index.jsonl`
- `human_feedback.md`
- `human_feedback.json`
- `assets/*`

## manifest.json

Review Pack의 메타데이터다.

```json
{
  "schema_version": "1.0",
  "competition": "patient_action_skeleton",
  "trial_id": "trial_002_c1_bed_wandering_focus",
  "created_at": "2026-06-01T00:00:00+00:00",
  "review_type": "visual_semantic_question",
  "priority": "high",
  "status": "pending_user_feedback",
  "source_files": [
    "metrics.json",
    "evaluation.md",
    "diagnosis.md"
  ],
  "questions_file": "questions.ko.md",
  "cases_file": "cases.jsonl",
  "metrics_file": "metrics_snapshot.json"
}
```

필드:

```text
schema_version
competition
trial_id
created_at
review_type
priority
status
source_files
questions_file
cases_file
metrics_file
```

`review_type` 후보:

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

`priority` 후보:

```text
low
medium
high
blocking
```

`status` 후보:

```text
pending_user_feedback
feedback_recorded
applied_to_strategy
closed
```

## summary.ko.md

사용자가 가장 먼저 읽는 요약이다.

포함할 내용:

- 왜 review가 필요한가
- 현재 trial의 핵심 결과
- 문제가 집중된 영역
- 에이전트가 확신하는 부분
- 에이전트가 확신하지 못하는 부분
- 사용자가 판단해주면 다음 trial에 어떻게 반영되는가

권장 형식:

```markdown
# Review Summary

## 왜 확인이 필요한가

...

## 핵심 증거

- ...

## 에이전트 판단

- ...

## 사용자가 봐야 할 부분

- ...

## 다음 결정 후보

- continue
- change_validation
- change_feature
- skip_submission
```

## questions.ko.md

사용자에게 묻는 질문 목록이다. 질문은 선택 가능한 결정으로 연결되어야 한다.

권장 형식:

```markdown
# Review Questions

## Q1. C1 view의 Bed Exit/Wandering 혼동이 실제 라벨 경계 문제로 보이나요?

- 관련 cases: case_001, case_002, case_003
- 판단 옵션:
  - label_boundary_is_ambiguous
  - roi_feature_mismatch
  - model_error_without_label_issue

## Q2. 다음 trial에서 ROI feature를 수정해도 될까요?

- 판단 옵션:
  - approve_feature_revision
  - keep_current_feature
  - inspect_more_samples
```

질문은 3개 이하를 권장한다. 질문이 많으면 사람이 판단하기 어렵고 token 비용도 커진다.

## cases.jsonl

사람이 볼 개별 case 목록이다.

각 줄은 하나의 JSON object다.

```json
{
  "case_id": "case_001",
  "source": "validation",
  "sample_id": "00620_H_D_SY",
  "fold": 3,
  "group": "C1",
  "true_label": "Bed Exit",
  "pred_label": "Wandering",
  "confidence": 0.62,
  "error_type": "class_confusion",
  "reason_for_review": "C1 Bed Exit/Wandering confusion is concentrated in this scenario.",
  "artifacts": [
    "assets/frames/case_001_frame_030.png",
    "assets/plots/case_001_skeleton_overlay.png"
  ],
  "question_refs": ["Q1"]
}
```

공통 필드:

```text
case_id
source
sample_id
fold
group
true_label
pred_label
confidence
error_type
reason_for_review
artifacts
question_refs
```

대회 유형에 따라 추가 필드를 허용한다.

이미지/영상:

```text
frame_range
view
roi_id
image_path
overlay_path
```

Tabular:

```text
row_id
important_features
feature_values
leakage_suspects
```

Time Series:

```text
time_range
window_start
window_end
series_id
```

NLP:

```text
text_id
text_excerpt
tokenization_notes
```

## metrics_snapshot.json

Review Pack 생성 시점의 metric 요약이다. LLM이나 사용자에게 raw log를 보여주지 않기 위해 먼저 구조화한다.

```json
{
  "competition": "patient_action_skeleton",
  "trial_id": "trial_002_c1_bed_wandering_focus",
  "objective": "maximize",
  "cv_score": 0.7926,
  "lb_score": null,
  "best_cv_before": 0.7926,
  "metrics": {
    "macro_f1": 0.8326,
    "fall_recall": 0.96,
    "bed_exit_f1": 0.7792
  },
  "concentration": {
    "view": "C1",
    "error_rate": 0.1686,
    "confused_pair": "Bed Exit/Wandering",
    "confusion_count": 13
  },
  "diagnosis": {
    "needs_user_review": true,
    "strategy_recommendation": "continue_refinement"
  }
}
```

## artifact_index.jsonl

이미지, 표, plot, frame 같은 자료의 인덱스다.

```json
{
  "artifact_id": "artifact_001",
  "case_id": "case_001",
  "type": "image",
  "path": "assets/frames/case_001_frame_030.png",
  "caption": "Frame around Bed Exit/Wandering boundary.",
  "created_by": "review_pack_generator"
}
```

## human_feedback.md

사람이 읽고 작성하기 쉬운 응답 파일이다.

```markdown
# Human Feedback

## Overall Decision

change_feature

## Answers

### Q1

ROI boundary looks too narrow. The sample is closer to Bed Exit than Wandering.

### Q2

Approve feature revision. Keep validation split unchanged.

## Follow-up Action

Plan a ROI feature revision trial focused on C1 boundary cases.
```

## human_feedback.json

에이전트가 읽는 구조화된 응답 파일이다.

```json
{
  "competition": "patient_action_skeleton",
  "trial_id": "trial_002_c1_bed_wandering_focus",
  "review_pack_id": "patient_action_skeleton_trial_002_c1_review_pack_001",
  "reviewed_at": "2026-06-01T00:00:00+00:00",
  "overall_decision": "change_feature",
  "answers": [
    {
      "question_id": "Q1",
      "decision": "roi_feature_mismatch",
      "notes": "ROI boundary is too narrow in C1."
    }
  ],
  "follow_up_action": "Plan ROI feature revision trial.",
  "approved_actions": ["change_feature"],
  "rejected_actions": ["change_validation"]
}
```

## Decision Log 연동

Review Pack 생성 시:

```yaml
decision_type: human_review
decision: prepare_review_pack
reason: Diagnosis requires domain judgment.
next_action: request_user_review
```

사용자 피드백 반영 시:

```yaml
decision_type: human_feedback
decision: change_feature
reason: User reviewed cases and approved ROI feature revision.
user_input_used: true
next_action: plan_next_trial
```

## Token 절약 원칙

- raw training log를 review pack에 넣지 않는다.
- 먼저 Python/tool로 metric과 case를 요약한다.
- LLM에는 `summary.ko.md`, `questions.ko.md`, `metrics_snapshot.json` 정도만 전달한다.
- 이미지/영상은 필요한 sample만 선택한다.
- review 질문은 3개 이하로 유지한다.

## 생성 기준

Review Pack은 다음 조건일 때 생성한다.

- `diagnosis.needs_user_review == true`
- `human_review_policy`가 `prepare_review_pack`을 반환함
- 사람이 볼 artifact가 존재하거나 생성 가능함
- 사용자 승인 또는 판단 없이는 다음 행동의 위험이 큼

## 성공 기준

- 사용자가 어떤 자료를 봐야 하는지 즉시 알 수 있다.
- 질문과 case가 연결되어 있다.
- 사람의 답변이 구조화되어 memory와 다음 trial에 반영된다.
- Review Pack이 대회 종류에 의존하지 않고 확장 가능하다.
