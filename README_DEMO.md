# 2주차 데모 실행 가이드 - LangGraph/RAG 1차 실험 루프

이 문서는 데모 시연자가 CMD에서 그대로 따라 실행할 수 있도록 정리한 가이드입니다.

이번 데모의 목표는 **에이전트가 대회 내용을 이해하고, 1차 실험 계획을 세우고, 파이프라인 코드를 작성한 뒤, 로컬에서 실행해 점수와 산출물을 남기는 것**을 보여주는 것입니다.

## 0. 데모에서 보여줄 핵심

이번 데모는 대화형 챗봇 시연이 아닙니다. 다음 흐름이 실제로 한 번 돌아가는지 보여줍니다.

```text
대회/데이터 맥락 로드
-> RAG context pack 구성
-> LangGraph 노드 흐름 시작
-> 실험 계획 생성
-> 파이프라인 코드 작성
-> 로컬 실행
-> 결과 기록
-> 사용자 확인용 산출물 생성
```

LangGraph는 각 단계를 노드로 연결해 실행 흐름을 관리합니다.

RAG는 대회 정보, 데이터 파일 목록, 이전 실험 메모, 파이프라인 요약 같은 자료를 필요한 단계에만 압축해서 전달합니다. 1차 실험에서는 이전 실험이 없으므로, 대회/데이터 정보 중심의 context pack이 사용됩니다.

## 1. 이번 데모 범위

포함:

- 1차 실험 1회 실행
- 대회 링크와 수동 제공 데이터 기반 context 생성
- LLM 기반 실험 계획 생성
- LLM 기반 파이프라인 코드 작성
- 로컬 실행
- 로컬 점수 기록
- 사용자 확인용 계획서와 파이프라인 요약 생성

제외:

- Kaggle 실제 제출
- Leaderboard 점수 조회
- 2차 이후 자동 개선 루프
- Human Review
- Slack/카카오톡/웹 대시보드 연동

주의: 실행 중 `submission.csv`가 생성될 수 있지만, 이것은 **제출 준비 파일**입니다. 데모 가이드에서는 Kaggle에 실제 업로드하지 않습니다.

## 2. CMD 열기

CMD를 열고 프로젝트 폴더로 이동하세요.

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
```

다른 위치에 압축을 풀었다면 해당 폴더로 이동하면 됩니다.

예:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent_Submission
```

## 3. OpenAI API Key 확인

API Key를 이미 환경 변수로 등록했다면 아래 명령으로 확인합니다.

```bat
if defined OPENAI_API_KEY (echo OPENAI_API_KEY set) else (echo OPENAI_API_KEY missing)
```

`OPENAI_API_KEY missing`이 나오면 현재 CMD 창에서만 임시로 설정할 수 있습니다.

```bat
set OPENAI_API_KEY=여기에_OpenAI_API_Key
```

다시 확인합니다.

```bat
if defined OPENAI_API_KEY (echo OPENAI_API_KEY set) else (echo OPENAI_API_KEY missing)
```

이번 데모에서 실제 API 호출을 승인하면 F-02, F-03 단계에서 고비용 LLM을 사용합니다.

- F-02: 실험 계획 생성
- F-03: 파이프라인 코드 작성

나머지 단계는 규칙 기반 함수와 로컬 실행으로 처리됩니다.

## 4. 데모 가이드 실행

아래 명령을 실행합니다.

```bat
python -m research_agent.cli demo-guide
```

그러면 이런 메뉴가 나옵니다.

```text
Autonomous ML Research Agent

현재 등록된 실험:
1. titanic          status: ...
2. demo            status: ...

무엇을 하시겠습니까?
1. 새 실험 시작
2. 기존 실험 사이클 실행
3. 해당 실험 현황보기
4. 종료
```

새 실험을 시작하려면 `1`을 입력합니다.

## 5. Titanic 실험 정보 입력

대회 링크를 물어보면 아래 링크를 입력합니다.

```text
https://www.kaggle.com/competitions/titanic/overview
```

이후 기본값이 대괄호로 나오면 맞는 경우 그냥 Enter를 누르면 됩니다.

예:

```text
competition [titanic]:
platform [kaggle]:
topic [Titanic - Machine Learning from Disaster]:
metric [accuracy]:
objective [maximize]:
target_column [Survived]:
id_column [PassengerId]:
required_data_files [train.csv, test.csv]:
```

기존 `titanic` 실험이 이미 있어서 새 데모를 깨끗하게 보고 싶다면 competition 이름만 바꿔도 됩니다.

예:

```text
competition [titanic]: titanic-demo
```

보충 설명을 물으면 아래처럼 짧게 입력하면 충분합니다.

```text
Predict passenger survival. train.csv contains Survived as target. test.csv has no target. Submission should contain PassengerId and Survived. Evaluation metric is accuracy.
```

이 설명은 링크 내용을 완전히 읽지 못했을 때를 대비한 안전 장치입니다.

## 6. 데이터 넣기

에이전트가 데이터 폴더를 안내합니다.

예:

```text
데이터 파일을 아래 폴더에 넣어주세요:
C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\titanic\data\
```

Kaggle에서 받은 Titanic 폴더 안의 파일을 그대로 넣어도 됩니다.

필수 파일:

```text
train.csv
test.csv
```

같이 넣어도 되는 참고 파일:

```text
gender_submission.csv
```

파일을 넣은 뒤 CMD로 돌아와서 Enter를 누르면 1회 실험 사이클이 시작됩니다.

## 7. API 호출 승인

실행 직전에 아래와 비슷한 질문이 나옵니다.

```text
데모 LLM provider: openai
데모 LLM model   : gpt-5.5
실제 OpenAI API를 호출할까요? 비용이 발생할 수 있습니다. [y/N]:
```

실제 데모에서 LLM이 계획과 코드를 생성하는 것을 보여주려면 `y`를 입력합니다.

```text
y
```

## 8. 실행 중 보이는 진행 상황

CMD에는 이런 식으로 단계별 진행 상황이 표시됩니다.

```text
[진행] 1/5 F-01 대회/데이터 맥락 확인 - 설정, 데이터 목록, 이전 실험 기록을 읽는 중
[완료] 1/5 F-01 대회/데이터 맥락 확인 - 준비 완료: platform=kaggle, metric=accuracy, objective=maximize

[진행] 2/5 F-02 실험 계획 생성 - LLM으로 첫 실험 계획 1개를 만드는 중
[완료] 2/5 F-02 실험 계획 생성 - 계획 생성 완료

[진행] 3/5 F-03 파이프라인 코드 작성 - 계획을 코드 변경으로 변환하는 중
[완료] 3/5 F-03 파이프라인 코드 작성 - 코드 변경 완료: ...

[진행] 4/5 F-04 로컬 실행 - 로컬에서 테스트/학습/예측 명령을 실행하는 중
[완료] 4/5 F-04 로컬 실행 - 로컬 실행 완료

[진행] 5/5 F-06 결과 기록 - 점수와 산출물 경로를 저장하는 중
[완료] 5/5 F-06 결과 기록 - 결과 저장 완료, 확인 폴더=runs/titanic/trial_001

[완료] 5/5 done 1회 실험 종료 - 1회 실험 사이클이 완료되었습니다.
```

현재 데모는 1차 실험까지만 보여줍니다.

## 9. 발표 때 보여줄 사용자용 산출물

사용자가 보면 되는 폴더는 아래입니다.

```text
C:\Users\ASUS\Desktop\Research_Agent\runs\titanic\trial_001\
```

competition 이름을 `titanic-demo`로 입력했다면 경로는 아래처럼 바뀝니다.

```text
C:\Users\ASUS\Desktop\Research_Agent\runs\titanic-demo\trial_001\
```

발표 때는 이 파일들을 우선 보여주면 됩니다.

```text
01_plan.ko.md
02_pipeline_structure.ko.md
04_result.ko.md
```

각 파일의 역할:

- `01_plan.ko.md`: 이번 회차에 무엇을 실험할지 정리한 실험 계획서
- `02_pipeline_structure.ko.md`: 데이터 로드, 전처리, 분할, 모델, 평가, 예측 파일 생성까지의 파이프라인 요약
- `04_result.ko.md`: 로컬 실행 결과와 점수

보조로 보여줄 수 있는 파일:

- `03_code_pipeline.ko.md`: 코드 파일별 역할 요약
- `05_paths.ko.md`: 원본 산출물, 코드, 로그 위치
- `code\`: 실제 생성된 파이프라인 코드 복사본

## 10. LangGraph/RAG 사용 증거로 보여줄 파일

LangGraph와 RAG가 내부에서 사용되었음을 보여주고 싶다면 아래 내부 파일도 확인할 수 있습니다.

```text
experiments\titanic\trial_001\graph_state.json
experiments\titanic\trial_001\node_events.jsonl
experiments\titanic\trial_001\context_pack_experiment_planning.md
experiments\titanic\trial_001\context_pack_workspace_code_writing.md
```

각 파일의 의미:

- `graph_state.json`: LangGraph 실행 상태와 노드별 결과
- `node_events.jsonl`: 노드 진행 이벤트 기록
- `context_pack_experiment_planning.md`: 실험 계획 노드에 전달된 RAG context
- `context_pack_workspace_code_writing.md`: 코드 작성 노드에 전달된 RAG context

발표에서는 사용자용 산출물을 먼저 보여주고, 시간이 남으면 이 파일들로 “내부적으로 LangGraph/RAG가 흐름과 맥락을 관리한다”는 점을 보여주면 됩니다.

## 11. 상태만 다시 보고 싶을 때

다른 CMD에서 상태를 확인하고 싶으면 아래 명령을 실행합니다.

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
python -m research_agent.cli watch-demo-cycle --competition titanic --trial trial_001
```

계속 따라보려면:

```bat
python -m research_agent.cli watch-demo-cycle --competition titanic --trial trial_001 --follow
```

## 12. 발표용 짧은 설명

데모 설명은 아래 흐름으로 말하면 됩니다.

```text
이 프로젝트는 협업형 자율 ML 연구 에이전트입니다.
사용자가 대회 링크와 데이터를 제공하면, 에이전트가 LangGraph 기반 노드 흐름으로 1차 실험을 수행합니다.
대회/데이터 맥락은 RAG context pack으로 정리되어 계획 생성과 코드 작성 단계에 전달됩니다.
LLM은 실험 계획과 코드 작성처럼 판단이 필요한 단계에만 사용하고, 실행과 결과 기록은 규칙 기반으로 처리합니다.
이번 데모에서는 실제 제출은 하지 않고, 로컬 실행 점수와 실험 계획서, 파이프라인 요약 산출물까지 확인합니다.
```

## 13. 주의 사항

- 데모 범위에서는 `submit-trial` 명령을 실행하지 않습니다.
- Kaggle 웹사이트에 직접 파일을 업로드하지 않습니다.
- 기존 `titanic` 실험이 남아 있으면 새 이름을 사용하세요. 예: `titanic-demo`, `titanic-lg-rag-demo`
- 1차 실험 결과만 보여주는 데모라면 `runs\<competition>\trial_001\` 폴더만 확인하면 충분합니다.

