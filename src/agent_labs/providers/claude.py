"""Anthropic Response API 어댑터"""

from contextlib import contextmanager

import anthropic

from ..errors import AuthError, ContextLengthError, RateLimitError, TransientError
from ..types import Response, ToolCall, Usage
from .base import LLMProvider

class Claude(LLMProvider):
    def __init__(self, model="claude-sonnet-5", **kw):
        super().__init__(model=model, **kw) # 모델, 재시도 등을 base.py에 위임
        self._client = anthropic.Anthropic()

    def _to_payload(self, req) -> dict:
        """공통 Request -> Anthropic Payload 변환"""
        system = [m.content for m in req.messages if m.role == "system"] # 시스템 프롬프트
        msgs, i = [], 0 # msgs: 최종 메시지 배열, i: 수동 인덱스
        turns = [m for m in req.messages if m.role != "system"] # 시스템 외 프롬프트

        while i < len(turns): 
            m = turns[i] # 현재 메시지
            if m.role == "tool":
                results = []
                while i < len(turns) and turns[i].role == "tool":
                    t = turns[i]
                    block = {"type": "tool_result", "tool_use_id": t.tool_call_id}
                    if t.content is not None: # 컨텐츠 내용 추가(빈 컨텐츠 허용)
                        block["content"] = t.content
                    if t.is_error: # 툴 콜링 실패
                        block["is_error"] = True
                    results.append(block)
                    i += 1
                msgs.append({"role": "user", "content": results})
                continue

            if m.role == "assistant" and m.raw_blocks is not None: # 이전 응답이 툴 콜링일 때
                msgs.append({"role": "assistant", "content": m.raw_blocks})
            else: # 사용자 메시지
                msgs.append({"role": m.role, "content": m.content})
            i += 1

        payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": req.max_tokens,
        }

        # req.temperature는 무시: anthropic SDK 2.x / Claude 5 계열은 temperature 파라미터 자체가 없음
        if system: # 시스템 메시지가 있을 경우
            payload["system"] = [{"type": "text", "text": s} for s in system]
        if req.tools: # 툴을 쓰는 요청
            payload["tools"] = [
                {
                    "name": t.name, # 모델이 호출할 때 쓰는 이름
                    "description": t.description, # 언제 쓸지 판단하는 근거
                    "input_schema": t.parameters, # 앤트로픽은 parameter가 아닌 input_schema 필드로 받음
                }
                for t in req.tools
            ]
        if req.tool_choice: # auto/any/tool/none
            payload["tool_choice"] = req.tool_choice
        if req.output_schema: # 구조화 출력
            payload["output_config"] = {"format": {"type": "json_schema", "schema": req.output_schema}}

        return payload # 완성된 요청

    def _invoke(self, req):
        """Claude 요청"""
        with self._errors():
            return self._client.messages.create(**self._to_payload(req)) # 실제 http 요청

    def _stream(self, req, on_text):
        """스트리밍 요청: 텍스트 조각마다 on_text, 끝나면 완성된 메시지"""
        with self._errors(), self._client.messages.stream(**self._to_payload(req)) as stream:
            for text in stream.text_stream:
                on_text(text)
            return stream.get_final_message()

    @contextmanager
    def _errors(self):
        """SDK 예외 -> 공통 예외"""
        try:
            yield
        except anthropic.RateLimitError as e: # 429
            raise RateLimitError(str(e)) from e
        except anthropic.APIConnectionError as e: # 네트워크
            raise TransientError(str(e)) from e
        except anthropic.AuthenticationError as e: # 401, 키 문제
            raise AuthError(str(e)) from e
        except anthropic.APIStatusError as e: # 나머지
            if e.status_code >= 500:
                raise TransientError(str(e)) from e
            if e.status_code == 400 and "prompt is too long" in str(e).lower():
                raise ContextLengthError(str(e)) from e
            raise

    def _parse(self, raw) -> Response:
        """답변 프롬프트 구조화"""
        text, calls = [], []
        for block in raw.content:
            if block.type == "text": # 일반 텍스트
                text.append(block.text)
            elif block.type == "tool_use": # 툴 호출 요청
                calls.append(ToolCall(id=block.id, name=block.name, args=block.input))

        cache_read = raw.usage.cache_read_input_tokens or 0
        cache_write = raw.usage.cache_creation_input_tokens or 0
        return Response(
            text="".join(text),
            tool_calls=calls,
            finish_reason=raw.stop_reason, # end_turn | tool_use | max_tokens | stop_sequence | refusal | pause_turn
            usage=Usage(
                input_tokens=raw.usage.input_tokens + cache_read + cache_write, # Claude의 input_tokens는 캐시 분을 뺀 값
                output_tokens=raw.usage.output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            raw_blocks=raw.content, # assistant 메시지로 돌려주기 위한 원본 보존
            raw=raw, # 디버깅용 원본 전체
        )