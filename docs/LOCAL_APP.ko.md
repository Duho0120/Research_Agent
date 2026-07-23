# 로컬 데스크톱 앱 실행 가이드

이 앱은 웹 서버를 띄우지 않고, 현재 PC에서 바로 실행되는 Tkinter 기반 로컬 데스크톱 앱이다.

현재 UI는 다크 테마와 카드형 패널을 사용한다. 실험 목록, 사용자 요청, trial 목록, 산출물 미리보기를 한 창에서 확인하는 관찰용 앱 v1이다.

## 실행

프로젝트 폴더에서 실행한다.

```cmd
cd /d C:\Users\ASUS\Desktop\Research_Agent
python -m research_agent.app
```

파일시스템과 SQLite 동기화를 생략하고 빠르게 열고 싶다면:

```cmd
python -m research_agent.app --no-sync
```

## 현재 제공 기능

- 전체 실험 목록 확인
- 실험별 trial 목록 확인
- trial별 사용자용 산출물 목록 확인
- trial 핵심 요약을 먼저 확인
- 실험 계획서, 파이프라인 구조도, 결과 파일을 앱에서 읽기 좋은 형태로 미리보기
- 다음 trial 실행 가능 상태 미리보기
- 대기 중인 사용자 요청 확인
- 사용자 요청에 답변 기록
- 특정 trial에 사용자 인사이트 직접 추가

## 산출물 읽기 방식

Trial을 선택하면 오른쪽 화면에 먼저 `0. 핵심 요약`이 표시된다.

추천 확인 순서:

1. 핵심 요약
2. 실험 계획서
3. 파이프라인 구조도
4. 실행 결과
5. 판단 카드

Markdown 원문은 앱 안에서 제목, 표, 불릿을 단순한 읽기 화면으로 변환해서 보여준다. 원문 파일을 그대로 확인하고 싶으면 `파일 열기` 버튼을 사용한다.

## 설계 의도

로컬 앱은 CLI 출력 문자열을 파싱하지 않는다. `kaggle_research_agent.interface_contract`의 공통 operation을 호출해서 상태를 가져온다.

따라서 나중에 웹 UI나 Slack/카카오톡형 메신저를 추가하더라도 동일한 operation을 재사용할 수 있다.

## 아직 포함하지 않은 것

- 앱 내부에서 trial을 직접 실행하는 버튼
- 실행 중인 trial의 실시간 로그 스트리밍
- 제출 승인 게이트 화면
- Slack/카카오톡 연동

이 기능들은 다음 단계에서 `run_next_trial`, approval gate, message adapter operation을 추가한 뒤 연결하는 것이 안정적이다.
