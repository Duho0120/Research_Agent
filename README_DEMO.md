# 2주차 프로토타입 데모 실행 가이드

이 문서는 데모 시연자가 CMD에서 그대로 따라 실행할 수 있도록 정리한 가이드입니다.

데모 목표는 다음 흐름이 실제로 연결되어 작동하는지 보여주는 것입니다.

```text
CMD 실행
-> 새 실험 등록
-> 데이터 폴더 안내
-> 데이터 준비
-> LLM이 실험 계획 생성
-> LLM이 파이프라인 코드 작성
-> 로컬 실행
-> 결과 기록
-> runs/titanic/trial_001에서 사용자용 산출물 확인
```

데모 기준 실험 이름은 `titanic`입니다.

## 1. CMD 열기

CMD를 열고 프로젝트 폴더로 이동하세요.

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
```

압축 파일을 다른 위치에 풀었다면, 해당 폴더로 이동하면 됩니다.

예:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent_Submission
```

실제 LLM API 호출로 데모를 진행하려면 OpenAI API Key를 설정합니다.

```bat
set OPENAI_API_KEY=여기에_OpenAI_API_Key
```

설정 여부를 확인하려면:

```bat
if defined OPENAI_API_KEY (echo OPENAI_API_KEY set) else (echo OPENAI_API_KEY missing)
```

이번 데모에서 API 호출을 승인하면, 실험 계획 생성과 파이프라인 코드 작성 단계에서 고비용 LLM을 사용합니다. 따라서 API Key가 설정되어 있어야 하고, 실제 API 비용이 발생할 수 있습니다.

## 2. 데모 가이드 실행

아래 명령어를 실행합니다.

```bat
python -m research_agent.cli demo-guide
```

그러면 이런 메뉴가 나옵니다.

```text
Autonomous ML Research Agent

현재 등록된 실험:
1. titanic          status: ...
2. demo             status: ...

무엇을 하시겠습니까?
1. 새 실험 시작
2. 기존 실험 사이클 실행
3. 해당 실험 현황보기
4. 종료
```

새 실험을 시작하려면 `1`을 입력합니다.

## 3. Titanic 새 실험 시작

대회 링크를 물어보면 아래 링크를 입력합니다.

```text
https://www.kaggle.com/competitions/titanic/overview
```

이후 기본값이 대괄호로 나오면, 맞는 경우 그냥 Enter를 누르면 됩니다.

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

보충 설명을 물으면 아래처럼 짧게 넣으면 충분합니다.

```text
Predict passenger survival. train.csv contains Survived as target. test.csv has no target. Submission should contain PassengerId and Survived. Evaluation metric is accuracy.
```

## 4. 데이터 넣기

에이전트가 데이터 폴더를 안내합니다.

예:

```text
데이터 파일을 아래 폴더에 넣어주세요:
C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\titanic\data\
```

압축을 다른 위치에 풀었다면 해당 프로젝트 폴더 아래의 경로를 사용하면 됩니다.

```text
demo_workspaces\titanic\data\
```

Kaggle에서 받은 Titanic 폴더 안 파일을 그대로 넣어도 됩니다.

필수 파일:

```text
train.csv
test.csv
```

같이 있어도 되는 참고 파일:

```text
gender_submission.csv
```

즉, 필수는 `train.csv`, `test.csv`이고, `gender_submission.csv`는 제출 형식 참고 파일로 같이 있어도 괜찮습니다.

파일을 넣은 뒤 CMD로 돌아와서 Enter를 누르면 1회 실험 사이클을 시작할 수 있습니다.

## 5. API 호출 승인

데이터를 넣고 Enter를 누르면, 데모 실행 전에 API 호출 여부를 묻습니다.

예:

```text
데모 LLM provider: openai
데모 LLM model   : gpt-5.5
실제 OpenAI API를 호출할까요? 비용이 발생할 수 있습니다. [y/N]:
```

실제 데모에서 LLM이 계획을 세우고 코드를 작성하는 흐름을 보여주려면 `y`를 입력합니다.

```text
y
```

`y`를 입력하면 다음 두 단계에서 고비용 LLM이 호출됩니다.

```text
F-02 실험 계획 생성
F-03 파이프라인 코드 작성
```

나머지 단계는 규칙 기반 함수와 로컬 실행으로 처리됩니다.

```text
F-01 대회/데이터 맥락 확인
F-04 로컬 실행
F-06 결과 기록
```

즉, 데모의 비용 구조는 “중요한 판단과 코드 작성만 고비용 LLM을 사용하고, 반복 실행과 기록은 일반 코드가 처리한다”는 설계를 보여줍니다.

## 6. 실행 중 보이는 진행 상황

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

현재 데모 범위는 1회 실험 루프입니다. 2회차 이후의 자가 개선, 제출, 휴먼 리뷰, 장기 자동 루프는 데모 범위에서 제외되어 있습니다.

## 7. 결과 확인 위치

사용자가 보면 되는 폴더는 여기입니다.

```text
C:\Users\ASUS\Desktop\Research_Agent\runs\titanic\trial_001\
```

압축을 다른 위치에 풀었다면 아래 상대 경로를 확인하면 됩니다.

```text
runs\titanic\trial_001\
```

주요 파일:

```text
README.ko.md
01_plan.ko.md
02_pipeline_structure.ko.md
03_code_pipeline.ko.md
04_result.ko.md
05_paths.ko.md
code\
```

특히 먼저 볼 파일은 이 세 개입니다.

```text
01_plan.ko.md              실험 계획
03_code_pipeline.ko.md     작성된 파이프라인 설명
04_result.ko.md            실행 결과
```

추가로 파이프라인 단계별 적용 내용을 자세히 보려면 아래 파일을 확인합니다.

```text
02_pipeline_structure.ko.md
```

실제 제출 파일은 실행 후 아래 경로에 생성됩니다.

```text
demo_workspaces\titanic\outputs\submission.csv
```

## 8. 상태만 다시 보고 싶을 때

다른 CMD에서 상태를 확인하고 싶으면:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
python -m research_agent.cli watch-demo-cycle --competition titanic --trial trial_001
```

계속 따라보려면:

```bat
python -m research_agent.cli watch-demo-cycle --competition titanic --trial trial_001 --follow
```

압축을 다른 위치에 풀었다면 첫 줄의 `cd` 경로만 바꾸면 됩니다.

## 9. 전체 흐름 요약

데모 흐름은 아래처럼 보면 됩니다.

```text
CMD 실행
-> 새 실험 등록
-> 데이터 폴더 안내
-> 데이터 넣고 Enter
-> API 호출 승인에서 y 입력
-> 고비용 LLM이 계획 생성
-> 고비용 LLM이 코드 작성
-> 로컬 실행
-> 결과 기록
-> runs/titanic/trial_001에서 사용자용 산출물 확인
```

이 데모는 전체 자율 연구 에이전트 프로젝트 중 “대회 하나에 대해 1회 실험 루프가 실제로 돌아가는지”를 보여주기 위한 제한 실행 모드입니다.
