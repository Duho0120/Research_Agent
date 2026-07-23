# CLI ML 엔지니어 점검 체크리스트

## 현재 좋은 점

- [ ] 선택된 실험의 상태가 한 화면에 보인다.
  - 현재 trial, 다음 trial, 최근 완료 trial, 베스트 trial
- [ ] local score와 submit score가 함께 표시된다.
- [ ] 피드백 요청 개수가 메뉴에 표시된다.
- [ ] 자동 실험 시작, 중단 요청, 상태 새로고침이 CLI에서 가능하다.
- [ ] 사용자 인사이트를 다음 trial에 남길 수 있다.
- [ ] SQLite DB 요약과 trial 상세 조회가 가능하다.
- [x] 폴더/DB 메뉴에서 베스트 산출물, 최근 제출 파일, 실행 워크스페이스를 구분해 열 수 있다.
  - 실행 워크스페이스와 실험 전체 기록 폴더의 용도를 안내한다.
  - 내부 기록은 현재/최근/베스트/trial 직접 입력으로 선택해 열 수 있다.
  - 제출 파일 폴더는 최근/베스트/최신 outputs/trial 직접 입력으로 선택해 열 수 있다.

## 부족한 점

- [ ] Trial 비교 정보가 부족하다.
  - base trial
  - local delta
  - submit delta
  - decision
  - model
  - feature set
  - validation method
  - submission ref
  - token/cost

- [ ] 다음 실험의 기준 base trial이 홈 화면에 명확히 보이지 않는다.
  - 예: `다음 실험 기준 base: trial_003`

- [ ] 자동 실험 시작 전 preflight가 부족하다.
  - OpenAI API 사용 가능 여부
  - Kaggle 인증 여부
  - 데이터 파일 존재 여부
  - execution_profile 유효성
  - 제출 가능 모드 여부

- [ ] 진행 로그가 ML workflow 단계로 보이지 않는다.
  - 예: 계획 생성 중, 코드 수정 중, 로컬 실행 중, 제출 중, 점수 기록 중

- [x] CLI 안에서 사용자용 산출물 미리보기가 가능하다.
  - 베스트 trial 계획서 요약
  - 파이프라인 구조 요약
  - 점수표
  - decision/reason

- [ ] 에이전트 질문 답변의 근거 확인이 약하다.
  - 사용한 파일 목록
  - 근거 파일 열기
  - 특정 trial 기준 질문

- [ ] manual/import trial의 코드 재현성이 약하다.
  - 예: `trial_003`은 공식 DB와 산출물에는 들어왔지만, code snapshot/decision/card 연결은 더 보강 필요

## 다음 개선 우선순위

- [x] 홈 화면에 `다음 실험 기준 base trial` 표시
- [x] `Trial 비교표` 메뉴 추가
  - 예:
    ```text
    trial | base | local | submit | delta | axis | decision | best
    ```
- [ ] `자동 실험 시작 전 점검` 추가
- [x] `베스트/최근/직접 입력 trial 요약 보기` 추가
- [ ] manual/import trial의 코드 스냅샷 연결
- [ ] 진행 로그를 단계형으로 표시

## 목표 방향

- [ ] CLI를 단순 실행 콘솔에서 실험 의사결정 콘솔로 발전시킨다.
- [ ] 사용자가 CLI 안에서 다음 질문에 바로 답할 수 있게 한다.
  - 왜 이 trial이 베스트인가?
  - 다음 trial은 무엇을 base로 삼는가?
  - 어떤 개선축이 성공/실패했는가?
  - 지금 실행해도 되는 상태인가?
  - 산출물과 실제 코드는 어디에 있는가?
