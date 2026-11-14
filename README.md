# LLM Agent Bedrock

LLM 에이전트를 바닥부터 직접 구현하는 실습

## 버전 관리

| 버전 | 날짜 | 내용 |
|---|---|---|
| 0.1.0 | 2026-11-14 | LLM Provider 정의, 최소 단위 동작 |

## 구성

```
configs/models.yaml          모델 별칭과 active 모델
src/agent_labs/types.py      공통 요청·응답 타입 (Message, Request, Response, Usage)
src/agent_labs/errors.py     공통 예외 (AuthError, RateLimitError, TransientError, ContextLengthError)
src/agent_labs/config.py     .env와 models.yaml 로드, make(별칭), llm
src/agent_labs/providers/    Claude, GPT, OpenRouter 어댑터
```

## 설치

```bash
py -m venv .venv                     # macOS/Linux: python3 -m venv .venv
.venv\Scripts\activate               # macOS/Linux: source .venv/bin/activate
pip install -e .
copy .env.example .env               # macOS/Linux: cp .env.example .env, 이후 쓸 provider 키 입력
```

- `configs/models.yaml`의 `active`가 기본 모델

## 사용

| 기능 | 방법 |
|---|---|
| 툴 호출 | `Request(tools=[ToolSpec(...)])` → `resp.tool_calls`, 결과는 `Message("tool", ..., tool_call_id=...)` |
| 히스토리 이어 붙이기 | `resp.to_message()` (provider 원본 블록 보존) |
| 구조화 출력 | `Request(output_schema=JSON Schema)` → `resp.text`가 스키마를 따르는 JSON |
| 스트리밍 | `stream(req, on_text)`, 첫 조각 전 실패만 재시도 |
| 재시도 | 429, 5xx, 네트워크 오류는 지수 백오프와 지터로 재시도 |
| 사용량 | `resp.usage` (입력, 출력, 캐시 읽기·쓰기 토큰) |

## provider

| provider | API | 비고 |
|---|---|---|
| `Claude` | Anthropic Messages | 툴 결과를 user 메시지의 `tool_result` 블록으로 묶음 |
| `GPT` | OpenAI Responses | 툴은 strict 스키마, `max_output_tokens`에 추론 토큰 포함 |
| `OpenRouter` | Chat Completions | 여러 회사 모델을 한 키로, 구조화 출력은 지원하는 업스트림으로만 전송 |