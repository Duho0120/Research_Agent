# 2026-07-13 Demo Model Policy 적용

## 요약

- 데모용 LLM 모델 선택을 코드 하드코딩이 아니라 `configs/policies/model_policy.yaml`에서 읽도록 정리했다.
- 고비용 호출(`experiment_planning`, `code_writing`, `workspace_code_writing`)은 Anthropic Claude Sonnet 5 (`claude-sonnet-5`)로 설정했다.
- 저비용 호출(`status_summary`, `log_summary`, `short_note_rewrite`, `simple_context_summary`)은 OpenAI GPT-5.6 Luna (`gpt-5.6-luna`)로 설정했다.
- `demo-one-cycle`은 기본적으로 이 정책을 사용하고, 필요하면 `--model`, `--provider`로 고비용 호출만 임시 override할 수 있다.
- Anthropic Messages API 어댑터를 추가해 실제 API 실행 시 `ANTHROPIC_API_KEY`를 사용하도록 했다.
- mock 응답을 쓰는 데모에서도 token usage 로그에는 정책상 provider/model이 남도록 수정했다.

## 주요 파일

- `configs/policies/model_policy.yaml`
- `docs/policies/model_policy.ko.md`
- `research_agent/policies.py`
- `research_agent/demo_one_cycle.py`
- `research_agent/agents/code_writer_adapter.py`
- `research_agent/workspace_code_writer.py`
- `tests/test_demo_one_cycle.py`
- `tests/test_code_writer_adapter.py`
- `tests/test_policy_gate.py`

## 검증

```powershell
python -B -m unittest discover -s tests -v
python -B -m compileall -q research_agent tests
git diff --check
```

결과:

- `205 tests`, `OK`
