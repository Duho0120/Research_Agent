# Model Policy

## 목적

`model_policy.yaml`은 자율 연구 루프에서 어떤 LLM 호출을 고비용 모델에 맡기고, 어떤 호출을 저비용 모델에 맡길지 정한다.

데모 범위에서는 LLM 호출을 최소화하며, 실제 고가치 판단 지점인 실험 계획과 코드 작성만 고비용 모델을 사용한다.

## 기본 정책

- 고비용 모델: Anthropic Claude Sonnet 5 (`claude-sonnet-5`)
- 저비용 모델: OpenAI GPT-5.6 Luna (`gpt-5.6-luna`)

## 고비용 호출

- `experiment_planning`
- `code_writing`
- `workspace_code_writing`
- `complex_diagnosis`
- `research_strategy`

## 저비용 호출

- `status_summary`
- `log_summary`
- `short_note_rewrite`
- `simple_context_summary`

## 데모 적용

`demo-one-cycle`은 기본적으로 `configs/policies/model_policy.yaml`을 읽는다.

- F-02 실험 계획: `experiment_planning` -> Claude Sonnet 5
- F-03 코드 작성: `workspace_code_writing` -> Claude Sonnet 5
- CMD 상태 요약이나 짧은 로그 요약처럼 가벼운 작업은 향후 `gpt-5.6-luna` 슬롯을 사용한다.

`--model`, `--provider` 옵션을 주면 데모의 고비용 호출 모델/provider만 임시로 덮어쓸 수 있다.

## API 키

- Anthropic 호출: `ANTHROPIC_API_KEY`
- OpenAI 호출: `OPENAI_API_KEY`

mock 응답 파일을 사용하는 데모에서는 실제 API 키가 없어도 동일한 모델 정책과 토큰 로그 구조를 확인할 수 있다.
