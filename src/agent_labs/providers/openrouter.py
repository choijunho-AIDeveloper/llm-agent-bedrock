"""OpenRouter Chat Completions API 어댑터 (OpenAI 호환 엔드포인트)"""

import json
import os
from contextlib import contextmanager

import openai

from ..errors import AuthError, ContextLengthError, RateLimitError, TransientError
from ..types import Response, ToolCall, Usage
from .base import LLMProvider

BASE_URL = "https://openrouter.ai/api/v1"

# Chat Completions finish_reason -> Anthropic 기준 정규화
FINISH_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

class OpenRouter(LLMProvider):
    def __init__(self, model="openrouter/auto", api_key=None, **kw):
        super().__init__(model=model, **kw) # 모델, 재시도 등을 base.py에 위임
        self._client = openai.OpenAI(
            base_url=BASE_URL, # SDK는 OpenAI 것을 그대로 쓰고 주소만 교체
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        )

    def _to_payload(self, req) -> dict:
        """공통 Request -> Chat Completions Payload 변환"""
        msgs = [] # system도 messages 배열 안에 그대로 들어감

        for m in req.messages:
            if m.role == "tool": # 툴 실행 결과 (호출 1건당 메시지 1개)
                content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
                if m.is_error: # is_error 필드가 없어서 본문으로 실패를 알림
                    content = f"[tool error] {content}"
                msgs.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
            elif m.role == "assistant" and m.raw_blocks is not None: # 이전 응답이 툴 콜링일 때
                msgs.append(m.raw_blocks) # tool_calls/reasoning 포함 원본 assistant 메시지
            else: # 시스템/사용자/일반 assistant 메시지
                msgs.append({"role": m.role, "content": m.content})

        payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": req.max_tokens,
        }

        if req.temperature is not None: # None이면 모델 기본값
            payload["temperature"] = req.temperature
        if req.tools: # 툴을 쓰는 요청
            payload["tools"] = [
                {
                    "type": "function",
                    "function": { # Chat Completions는 function 객체로 한 번 더 감쌈
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in req.tools
            ]
        if req.tool_choice: # Anthropic 형식 -> OpenAI 형식
            payload["tool_choice"] = self._tool_choice(req.tool_choice)
        if req.output_schema: # 구조화 출력 (지원 여부는 모델마다 다름)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": req.output_schema, "strict": True},
            }
            payload["extra_body"] = {"provider": {"require_parameters": True}} # 스키마를 지원하는 업스트림으로만 보냄 (OpenRouter 전용 필드)

        return payload # 완성된 요청

    @staticmethod
    def _tool_choice(choice: dict):
        """{"type": auto|any|none|tool} -> auto | required | none | {"type": "function", "function": {...}}"""
        kind = choice.get("type")
        if kind == "any":
            return "required"
        if kind == "tool":
            return {"type": "function", "function": {"name": choice["name"]}}
        return kind # auto, none은 이름이 같음

    def _invoke(self, req):
        """OpenRouter 요청"""
        with self._errors():
            raw = self._client.chat.completions.create(**self._to_payload(req)) # 실제 http 요청
        return self._check(raw)

    def _stream(self, req, on_text):
        """스트리밍 요청: 텍스트 조각마다 on_text, 끝나면 완성된 응답"""
        payload = {**self._to_payload(req), "stream_options": {"include_usage": True}} # 마지막 조각에 사용량 포함
        with self._errors(), self._client.chat.completions.stream(**payload) as stream:
            for event in stream:
                if event.type == "content.delta" and event.delta:
                    on_text(event.delta)
            raw = stream.get_final_completion()
        return self._check(raw)

    @staticmethod
    def _check(raw):
        if not raw.choices: # 업스트림 실패를 200 + error 본문으로 돌려주는 경우가 있음
            raise TransientError(f"empty choices from OpenRouter: {getattr(raw, 'error', None)}")
        return raw

    @contextmanager
    def _errors(self):
        """SDK 예외 -> 공통 예외"""
        try:
            yield
        except openai.RateLimitError as e: # 429
            raise RateLimitError(str(e)) from e
        except openai.APIConnectionError as e: # 네트워크, 타임아웃
            raise TransientError(str(e)) from e
        except openai.AuthenticationError as e: # 401, 키 문제
            raise AuthError(str(e)) from e
        except openai.BadRequestError as e: # 400
            if "context length" in str(e).lower(): # 업스트림마다 코드가 달라 메시지로 판별
                raise ContextLengthError(str(e)) from e
            raise
        except openai.APIStatusError as e: # 나머지 (402 크레딧 부족 등)
            if e.status_code >= 500:
                raise TransientError(str(e)) from e
            raise

    def _parse(self, raw) -> Response:
        """답변 프롬프트 구조화"""
        choice = raw.choices[0]
        msg = choice.message
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                args=json.loads(tc.function.arguments or "{}"), # 인자가 JSON 문자열로 옴
            )
            for tc in (msg.tool_calls or [])
        ]
        usage = raw.usage # 일부 업스트림은 usage를 안 줌
        replay = {"role": "assistant", **msg.model_dump(exclude_none=True)} # 스트리밍에서 role이 빠지는 업스트림 대비
        for tc in replay.get("tool_calls", []):
            tc.pop("index", None) # 스트리밍 전용 필드라 다시 보내면 거절될 수 있음

        return Response(
            text=msg.content or "",
            tool_calls=calls,
            finish_reason=FINISH_REASONS.get(choice.finish_reason, choice.finish_reason), # Anthropic 기준으로 정규화
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                cache_read_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
            ),
            raw_blocks=replay, # assistant 메시지로 돌려주기 위한 원본 보존 (reasoning 포함)
            raw=raw, # 디버깅용 원본 전체
        )
