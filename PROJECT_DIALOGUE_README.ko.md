# 프로젝트 핵심 대화 README

이 문서는 구현 설명서가 아니라, 프로젝트를 진행하면서 사용자가 물어본 질문과 요청한 내용, 그리고 그때 정리된 핵심 답변을 빠르게 다시 보기 위한 대화형 README입니다.

오랜만에 프로젝트를 다시 열었을 때 다음을 복기하는 용도입니다.

- 내가 무엇을 중요하게 생각했는지
- 어떤 방향으로 구현하기로 했는지
- 아직 반영하지 말라고 한 것은 무엇인지
- 사람이 개입해야 하는 지점은 어디인지
- 다음에 이어서 논의해야 할 질문은 무엇인지

## 1. 처음 이해한 프로젝트 방향

사용자 질문/요청:

> 이미지의 Autonomous Kaggle Research Agent 구조를 보고 이해한 대로 정리해달라.

핵심 정리:

- Kaggle 대회를 대상으로 실험 계획, 학습, 평가, 제출, 기록, 다음 전략 수립을 반복하는 자율 연구 에이전트 구조다.
- 단순 AutoML이 아니라, 연구 노트와 장기 기억을 가진 실험 연구자에 가깝다.
- 주요 루프는 `Plan -> Train -> Evaluate -> Decide -> Memory Update`다.

프로젝트에 남은 의미:

- 이 프로젝트의 목표는 모델 하나를 잘 만드는 것이 아니라, 실험 연구 과정을 구조화하고 자동화하는 것이다.

## 2. 에이전트 지식과 필요한 도구

사용자 질문/요청:

> 딥러닝은 알지만 AI 에이전트 지식은 부족하다. 구현에 필요한 툴이나 지식을 알려달라.

핵심 정리:

- 새로 배워야 할 중심은 모델 학습보다 `agent loop`, `tool calling`, `memory`, `planning`, `reflection`, `evaluation`이다.
- 필요한 도구는 LLM API, Kaggle API/MCP, 실험 추적 저장소, 파일 기반 memory, 실행 환경, 나중에 LangGraph다.
- 초기에는 완전 자율 연구자보다 실험을 읽고 다음 계획을 제안하는 assistant부터 만드는 것이 좋다.

프로젝트에 남은 의미:

- 구조가 먼저고, 자동화는 나중이다.
- memory와 state를 명시적으로 설계해야 한다.

## 3. Assistant 구성 방식

사용자 질문/요청:

> 먼저 assistant 구성 방법을 알려달라.

핵심 정리:

- 첫 assistant는 대회 정보 읽기, 이전 trial 읽기, 다음 실험 계획 작성, 결과 reflection 작성만 해도 충분하다.
- `planner.py`, `reflector.py`, `memory.py`, `prompts.py` 같은 작은 구성으로 시작한다.
- 모든 raw log를 LLM context에 넣지 말고, 요약된 trial 기록과 metrics를 주는 방식이 좋다.

프로젝트에 남은 의미:

- 초기 구현 범위는 `Planner`, `Evaluator`, `Memory` 중심으로 제한했다.

## 4. Main Agent와 Sub Agent 구조

사용자 질문/요청:

> 메인 에이전트가 전체 흐름을 알고 있고 서브에이전트들에게 명령을 할당하는 방식이면 어떤 구조가 좋을까?

핵심 정리:

- Supervisor/Worker multi-agent 구조가 적합하다.
- Main Agent는 전체 상태, 전략, 판단을 담당한다.
- Sub Agent는 Planner, Engineer, Evaluator, Memory, Diagnoser처럼 좁은 전문 작업을 맡는다.
- Tool은 실제 실행을 담당한다.

프로젝트에 남은 의미:

```text
Main Agent = 판단과 통제
Sub Agent = 전문 작업
Tool = 실제 실행
Memory = 다음 판단의 근거
```

## 5. 완전 자율 연구자 구조

사용자 질문/요청:

> assistant를 넘어서 스스로 리더보드 결과를 보고 전략을 세우고 개선하고 제출하는 완전 자율 연구자를 만들려면?

핵심 정리:

- 전체 루프는 `Load State -> Plan -> Implement -> Train -> Evaluate -> Submit -> Read Leaderboard -> Diagnose -> Memory Update -> Next Strategy`다.
- 자율 연구자를 어렵게 만드는 핵심은 학습 실행이 아니라 판단 품질이다.
- CV/LB 괴리, seed noise, leakage, public LB overfitting, submission 낭비를 조심해야 한다.

프로젝트에 남은 의미:

- 지금은 완전 자율이 아니라 단계별로 자율성을 올리는 방식으로 진행한다.

## 6. Colab 사용 방식

사용자 질문/요청:

> 로컬에 쓸 만한 GPU가 없어서 Colab을 사용했는데, 프로그램이 Colab 환경을 사용할 수 있게 연결할 수 있을까?

핵심 정리:

- 가능하지만 Colab을 완전한 remote GPU server처럼 조종하는 것은 안정적이지 않다.
- 추천 방식은 Local Main Agent + Colab Worker Notebook이다.
- 로컬은 job.yaml을 만들고, Colab worker가 job을 실행하고 결과를 저장한다.

프로젝트에 남은 의미:

- Colab은 worker로 설계한다.
- 하지만 이후 사용자는 로컬에서 가능한 작업은 로컬 우선으로 하고 싶다고 요청했다.

현재 반영 상태:

- 초기에는 Colab job skeleton을 만들었다.
- 이후 기본 backend를 `local`로 바꾸고, Colab은 `--backend colab`일 때만 사용하도록 수정했다.

## 7. 전체 구조 요약

사용자 질문/요청:

> 전체 틀을 한눈에 보이게 요약해달라.

핵심 정리:

```text
Main Research Agent
  -> Planner
  -> Engineer
  -> Job Manager
  -> Worker(Local/Colab)
  -> Evaluator
  -> Submission Tool
  -> Diagnoser
  -> Memory
```

프로젝트에 남은 의미:

- 이 구조를 장기 목표로 두되, 지금은 Level 0/1/2/4까지만 구현했다.

## 8. LangGraph와 MCP에 대한 질문

사용자 질문/요청:

> LangGraph와 Kaggle/Colab MCP도 연동해야 하는가?

핵심 정리:

- LangGraph는 전체 흐름을 제어하는 상태 머신/뇌 역할이다.
- MCP는 Kaggle, Colab, 파일, memory 같은 외부 도구와 연결되는 손발 역할이다.
- 처음부터 MCP로 묶을 필요는 없고, local tool wrapper로 시작해도 된다.

프로젝트에 남은 의미:

- 현재는 LangGraph/MCP 미적용.
- 함수 구조는 나중에 LangGraph node로 옮기기 쉽게 분리되어 있다.

중요 요청:

- 사용자가 “아직 코드를 반영하지마!”라고 명시했으므로, LangGraph 관련 구현은 아직 하지 않는다.

## 9. 프로젝트 진행 순서

사용자 질문/요청:

> 이 프로젝트를 이해하고 기획 후 실행 단계로 들어가려면 어떤 작업 순서를 지켜야 할까?

핵심 정리:

```text
0. 목표 정의
1. 사용 시나리오 정의
2. 자율성 범위 결정
3. 전체 아키텍처 설계
4. 상태/메모리 설계
5. 실험 단위 설계
6. 도구 연동 설계
7. 안전장치/판단 규칙 설계
8. MVP 범위 확정
9. 구현 계획 수립
10. 작은 대회/작은 시나리오로 검증
```

프로젝트에 남은 의미:

- 사용자는 프로젝트 결정권자이며, 구현자는 사용자가 맥락을 이해하고 판단할 수 있도록 도와야 한다.

## 10. 단계별 구현 범위

사용자 질문/요청:

> 단계별로 어떻게 나누어지는가? 몇 단계까지 바로 만들 수 있는가?

핵심 정리:

- Level 0: 프로젝트 골격
- Level 1: Research Assistant
- Level 2: Colab 학습 연동 skeleton
- Level 3: Kaggle 제출 연동
- Level 4: Config 기반 자동 실험
- Level 5: Code Editing Agent
- Level 6: 완전 자율 연구 루프

현재 구현:

- Level 0 완료
- Level 1 완료
- Level 2 skeleton 완료
- Level 4 기본 구조 완료

아직 미구현:

- Level 3 Kaggle 제출
- Level 5 코드 수정 Agent
- Level 6 완전 자율 루프

## 11. 구현 후 검토

사용자 요청:

> 0, 1, 2, 4까지만 구현해달라. 구현 후 검토만 할 것이다.

핵심 결과:

- `C:\Users\ASUS\Desktop\Research_Agent`에 프로젝트 생성
- 기본 CLI와 agent 함수 구현
- demo trial로 plan/config/job/evaluate/remember 흐름 검증
- 이후 `cycle` 명령과 local backend도 추가

현재 상태:

- 구현 후 검토 가능한 1차 골격이 있다.
- 실제 목표를 잡고 목표 달성까지 이어가는 것은 다음 단계다.

## 12. 환자 행동 인식 노트북 테스트

사용자 질문/요청:

> 작업하던 파일을 보내면 가능한 시나리오를 실행해볼 수 있는가?

제공 파일:

```text
C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\Notebooks\V07_COM_Lean_Method_4-Torch_XPU.ipynb
```

핵심 분석:

- 30프레임 skeleton 시계열 모델
- Normal / Bed Exit / Wandering / Fall 4-class 분류
- Transformer + ROI feature + LDAM + DRW + aux head 구조
- Confusion matrix와 classification report가 저장되어 있음

기록한 baseline:

- `trial_001_v07_baseline`

생성한 다음 계획:

- `trial_002_c1_bed_wandering_focus`

핵심 병목:

- Bed Exit/Wandering 혼동
- C1 view에서 높은 error rate
- 특정 시나리오 `00620_H_D_SY`에 오류 집중

## 13. 새 작업 파일 생성 여부에 대한 정리

사용자 질문:

> 방금 내가 준 작업 파일을 에이전트가 새로운 시도 방안을 만들었고 새로운 작업 파일을 만든 것으로 알고 있어. 맞아?

정확한 답:

- 새로운 학습 노트북이나 실제 학습 코드를 만든 것은 아니다.
- 만든 것은 실험 계획 파일, config, metrics 기록, job 요청이다.

생성된 핵심 계획 파일:

```text
experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/plan.md
experiments/patient_action_skeleton/trial_002_c1_bed_wandering_focus/improvement_candidates.md
```

향후 실제로 해야 할 일:

- 원본 노트북 보존
- trial_002용 새 notebook 또는 `train_patient_action_skeleton.py` 생성
- 개선 아이디어를 코드에 반영
- 로컬 실행
- metrics를 다시 기록

## 14. 문서 언어 정책

사용자 요청:

> `.md` 파일은 한글로 보고 싶다. 영어본도 남겨둬야 할지 고민된다.

정리된 정책:

- 사람이 읽는 기본 문서는 한글
- 에이전트/LLM 참고용 영어본은 `.en.md`로 보존
- config, metrics, schema, code는 영어 유지

현재 반영:

```text
plan.md
plan.ko.md
plan.en.md
improvement_candidates.md
improvement_candidates.ko.md
improvement_candidates.en.md
```

## 15. Human-in-the-loop의 중요성

사용자 핵심 의견:

> 이미지 기반 모델이라면 모델이 어디에 집중하는지, 실제 이미지가 어떤지 판단해야 할 때 사람의 육안이 필요하다.

환자 행동 인식 프로젝트에서의 실제 경험:

- 원본 데이터는 이미지/영상 기반이었다.
- 사용자가 직접 GUI 라벨링 도구를 만들고 라벨링했다.
- ROI feature를 추가할 때도 GUI 도구를 만들고 직접 침대 라인을 잡았다.
- 특정 오탐을 직접 추론 결과로 확인해보니, 의자 때문에 ROI 영역이 좁아지고 골반 좌표가 ROI 밖으로 넘어가는 문제가 있었다.
- 이 경우 모델 혼동은 단순 성능 문제가 아니라 ROI 정의/라벨 기준/시각적 맥락 문제일 수 있다.

핵심 결론:

- Human Review는 예외 처리가 아니라 LangGraph의 정식 branch로 들어가야 한다.
- 에이전트는 사람이 봐야 할 순간을 감지하고 review pack을 만들어야 한다.
- 사람의 판단은 memory에 기록되고 다음 trial 전략에 반영되어야 한다.

추천 graph branch:

```text
diagnose_errors
  ↓
need_human_review?
  ├─ no  → plan_next_trial
  └─ yes → prepare_review_pack
              ↓
            request_human_review
              ↓
            ingest_human_feedback
              ↓
            update_memory
              ↓
            revise_strategy
```

## 16. 카카오톡/사람 개입 채널

사용자 질문:

> 자율 연구자에게 작업 중 아이디어나 주의사항을 카카오톡으로 제시하고 싶다.

정리된 방향:

- 카카오톡은 우선 알림 채널로 사용하는 것이 현실적이다.
- 사용자의 입력은 카카오톡 직접 수신보다 링크 기반 inbox가 안정적이다.
- 중요한 판단 지점에서 에이전트가 알림을 보내고, 사용자는 승인/주의사항/아이디어를 남긴다.

향후 구조:

```text
Agent detects decision point
  ↓
Kakao notification
  ↓
User opens input/approval link
  ↓
human_notes.jsonl
  ↓
Main Agent reads before next cycle
```

## 17. 지금까지 별도로 만든 기록 문서

현재 프로젝트에는 README 외에 다음 기록 문서가 있다.

```text
PROJECT_CONVERSATION_NOTES.ko.md
  - 프로젝트 철학, 설계 원칙, 사용자의 핵심 관점 정리

PROJECT_CHANGELOG.ko.md
  - 포트폴리오용 날짜/시간별 변경 이력

PROJECT_DIALOGUE_README.ko.md
  - 사용자가 물어본 질문과 요청사항 중심의 대화형 복기 문서
```

## 18. 아직 반영하지 말라고 한 것

명시적으로 아직 코드 반영하지 않기로 한 것:

- LangGraph 전환 코드
- Main graph node 구현
- Human Review graph branch 구현
- 카카오톡 연동 코드
- Claude/Codex 멀티모델 라우팅
- 실제 코드 수정 Agent

현재는 설계와 기록까지이며, 구현은 사용자가 승인한 뒤 진행한다.

## 19. 다음에 이어서 볼 질문

다음에 다시 프로젝트를 열면 아래 질문부터 이어가면 좋다.

- LangGraph state schema를 어떻게 정의할까?
- Main Agent가 local/colab/ask_user를 판단하는 정책은 어떻게 둘까?
- Human Review Pack의 표준 폴더 구조는 어떻게 만들까?
- 환자 행동 인식 trial_002를 실제 코드/노트북으로 어떻게 분리할까?
- `train_patient_action_skeleton.py`를 만들 것인가, 새 `.ipynb`를 만들 것인가?
- 사람이 남긴 관찰을 `human_observations.jsonl`로 어떻게 저장할까?
- 카카오톡 알림은 언제, 어떤 메시지 형식으로 보낼까?

## 20. 비용 효율적 자율 연구 정책 1차 설계

사용자 요청:

> 자율 연구자가 토큰을 효율적으로 활용하고 안정적으로 작동하려면, LLM은 중요한 판단 지점에서만 호출하고 반복 작업은 Python/rule/tool이 처리하는 구조가 좋겠다.

정리된 방향:

- `Execution decision policy`: `local / colab / ask_user / wait_for_metrics` 판단 기준을 먼저 문서화한다.
- `Diagnose + Human Review policy`: 어떤 오류 패턴에서 사람에게 물어볼지 문서화한다.
- `Review Pack schema`: 사람이 볼 자료, 질문, 답변 저장 형식을 문서화한다.

생성한 문서:

```text
docs/policies/execution_decision_policy.ko.md
docs/policies/human_review_policy.ko.md
docs/policies/review_pack_schema.ko.md
```

다음 구현 후보:

- `token_policy.yaml`
- `execution_policy.yaml`
- `human_review_policy.yaml`
- policy loader
- rule-based gate
- LLM 호출/비호출 decision log
- review pack generator
- local failure classifier

현재 구현:

- 정책 yaml 3종을 `configs/policies/`에 추가했다.
- `research_agent/policies.py`에서 정책을 로딩한다.
- `research_agent/agents/policy_gate.py`에서 execution decision, human review decision, token gate, local failure classification을 처리한다.
- `research_agent/agents/review_pack.py`에서 표준 review pack 파일을 생성한다.
- `run_cycle()`에 execution decision과 human review policy 판단을 연결했다.

추가 구현:

- `decide-llm` CLI를 추가해 LLM 호출/비호출 판단을 `decision_log.jsonl`에 남긴다.
- `request-review` CLI가 Human Review policy를 평가하고 필요하면 `review_pack/`까지 생성한다.
- 테스트는 `69 tests OK` 상태다.

4차 구현:

- `record-feedback`가 review pack 상태를 `feedback_recorded`로 갱신한다.
- `review_pack/human_feedback.md`와 `review_pack/human_feedback.json`을 생성한다.
- 사용자 피드백은 `decision_log.jsonl`에 `user_input_used=true`로 기록된다.
- `plan-next`가 최신 user feedback을 읽어 다음 strategy에 반영한다.
- 테스트는 `72 tests OK` 상태다.
