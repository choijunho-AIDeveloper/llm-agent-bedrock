"""OpenAI Responses API 어댑터"""

import json
from contextlib import contextmanager

import openai

from ..errors import AuthError, ContextLengthError, RateLimitError, TransientError
from ..types import Response, ToolCall, Usage
from .base import LLMProvider

class GPT(LLMProvider):
    def __init__(self, model="gpt-5.6", **kw):
        super().__init__(model=model, **kw) # 모델, 재시도 등을 base.py에 위임
        self._client = openai.OpenAI() # OPENAI_API_KEY 환경변수 사용

    def _to_payload(self, req) -> dict:
        """공통 Request -> OpenAI Responses Payload 변환"""
        system = [m.content for m in req.messages if m.role == "system"] # 시스템 프롬프트
        items = [] # Responses API는 messages 대신 input item 배열

        for m in req.messages:
            if m.role == "system": # instructions로 따로 보냄
                continue
            if m.role == "tool": # 툴 실행 결과
                output = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
                if m.is_error: # is_error 필드가 없어서 본문으로 실패를 알림
                    output = f"[tool error] {output}"
                items.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": output,
                })
            elif m.role == "assistant" and m.raw_blocks is not None: # 이전 응답이 툴 콜링일 때
                items.extend(m.raw_blocks) # reasoning/function_call item까지 원본 그대로 이어붙임
            else: # 사용자/일반 assistant 메시지
                items.append({"role": m.role, "content": m.content})

        payload = {
            "model": self.model,
            "input": items,
            "max_output_tokens": req.max_tokens, # Responses API는 max_tokens가 아닌 max_output_tokens
        }

        if req.temperature is not None: # 추론 모델은 temperature 미지원이라 지정했을 때만 보냄
            payload["temperature"] = req.temperature
        if system: # 시스템 메시지가 있을 경우
            payload["instructions"] = "\n\n".join(system)
        if req.tools: # 툴을 쓰는 요청
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters, # OpenAI는 input_schema가 아닌 parameters 필드
                    "strict": True, # 스키마 강제 (모든 필드 required + additionalProperties: false 필요)
                }
                for t in req.tools
            ]
        if req.tool_choice: # Anthropic 형식 -> OpenAI 형식
            payload["tool_choice"] = self._tool_choice(req.tool_choice)
        if req.output_schema: # 구조화 출력 (strict는 모든 필드 required + additionalProperties: false 필요)
            payload["text"] = {"format": {"type": "json_schema", "name": "output", "schema": req.output_schema, "strict": True}}

        return payload # 완성된 요청

    @staticmethod
    def _tool_choice(choice: dict):
        """{"type": auto|any|none|tool} -> auto | required | none | {"type": "function", "name": ...}"""
        kind = choice.get("type")
        if kind == "any":
            return "required"
        if kind == "tool":
            return {"type": "function", "name": choice["name"]}
        return kind # auto, none은 이름이 같음

    def _invoke(self, req):
        """GPT 요청"""
        with self._errors():
            return self._client.responses.create(**self._to_payload(req)) # 실제 http 요청

    def _stream(self, req, on_text):
        """스트리밍 요청: 텍스트 조각마다 on_text, 끝나면 완성된 응답"""
        with self._errors(), self._client.responses.stream(**self._to_payload(req)) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    on_text(event.delta)
            return stream.get_final_response()

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
            if getattr(e, "code", None) == "context_length_exceeded": # 컨텍스트 초과
                raise ContextLengthError(str(e)) from e
            raise
        except openai.APIStatusError as e: # 나머지
            if e.status_code >= 500:
                raise TransientError(str(e)) from e
            raise

    def _parse(self, raw) -> Response:
        """답변 프롬프트 구조화"""
        text, calls = [], []
        for item in raw.output:
            if item.type == "message": # 일반 텍스트 (content 안에 여러 파트)
                for c in item.content:
                    if c.type == "output_text":
                        text.append(c.text)
            elif item.type == "function_call": # 툴 호출 요청
                calls.append(ToolCall(
                    id=item.call_id, # 결과를 돌려줄 때 쓰는 id는 item.id가 아닌 call_id
                    name=item.name,
                    args=json.loads(item.arguments or "{}"), # 인자가 JSON 문자열로 옴
                ))

        return Response(
            text="".join(text),
            tool_calls=calls,
            finish_reason=self._finish_reason(raw, calls), # Anthropic 기준으로 정규화
            usage=Usage(
                input_tokens=raw.usage.input_tokens,
                output_tokens=raw.usage.output_tokens,
                cache_read_tokens=getattr(raw.usage.input_tokens_details, "cached_tokens", 0) or 0, # 자동 캐싱
            ),
            raw_blocks=raw.output, # 다음 요청 input으로 돌려주기 위한 원본 보존
            raw=raw, # 디버깅용 원본 전체
        )

    @staticmethod
    def _finish_reason(raw, calls) -> str:
        """Responses API는 stop_reason이 없어서 status/output으로 추론"""
        if calls:
            return "tool_use"
        if raw.status == "incomplete": # 중간에 끊긴 경우
            reason = getattr(raw.incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                return "max_tokens"
            if reason == "content_filter":
                return "refusal"
            return reason or "incomplete"
        return "end_turn"
