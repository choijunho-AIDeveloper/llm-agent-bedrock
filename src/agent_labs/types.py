"""provider 공통 요청/응답 타입"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: Any = None # 보통 str, provider 고유 블록 리스트도 허용
    tool_call_id: str | None = None # role == "tool"일 때 어떤 호출에 대한 결과인지
    is_error: bool = False # role == "tool"일 때 툴 실행 실패 여부
    raw_blocks: Any = None # role == "assistant"일 때 provider 원본 응답 (툴 콜링 히스토리 재전송용)

    def __str__(self) -> str:
        head = f"[{self.role}{' 오류' if self.is_error else ''}]"
        lines = [f"{head} {self.content}" if self.content else head]
        lines += [f"  → {name}({args})" for _, name, args in tool_calls_in(self.raw_blocks)]
        return "\n".join(lines)


@dataclass
class ToolSpec:
    name: str # 모델이 호출할 때 쓰는 이름
    description: str # 언제 쓸지 판단하는 근거
    parameters: dict # JSON Schema


@dataclass
class Request:
    messages: list[Message]
    tools: list[ToolSpec] | None = None
    tool_choice: dict | None = None # Anthropic 형식 {"type": "auto" | "any" | "none" | "tool", "name"?: str}
    max_tokens: int = 4096
    temperature: float | None = None # None이면 provider 기본값 사용
    output_schema: dict | None = None # JSON Schema를 주면 응답 text가 이 스키마를 따르는 JSON


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Usage:
    input_tokens: int = 0 # 캐시에서 읽은 토큰까지 포함한 전체 입력
    output_tokens: int = 0
    cache_read_tokens: int = 0 # 입력 중 캐시에서 읽은 토큰
    cache_write_tokens: int = 0 # 입력 중 캐시에 새로 쓴 토큰 (Claude만 따로 과금)


@dataclass
class Response:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None # end_turn | tool_use | max_tokens | refusal ... (Anthropic 기준으로 정규화)
    usage: Usage = field(default_factory=Usage)
    raw_blocks: Any = None # 다음 요청의 assistant 메시지로 그대로 돌려줄 provider 원본
    raw: Any = None # 디버깅용 원본 전체

    def __str__(self) -> str:
        lines = [self.text] if self.text else []
        lines += [f"→ {c.name}({c.args})" for c in self.tool_calls]
        lines.append(f"({self.finish_reason}, 입력 {self.usage.input_tokens:,} / 출력 {self.usage.output_tokens:,} 토큰)")
        return "\n".join(lines)

    def to_message(self) -> Message:
        """히스토리에 붙일 assistant 메시지로 변환"""
        return Message(role="assistant", content=self.text, raw_blocks=self.raw_blocks)


def tool_calls_in(raw_blocks) -> list[tuple]:
    """provider별 raw_blocks에서 (id, 이름, 인자) 추출"""
    if raw_blocks is None:
        return []
    if isinstance(raw_blocks, dict): # OpenRouter: assistant 메시지 dict
        return [
            (tc["id"], tc["function"]["name"], tc["function"]["arguments"])
            for tc in raw_blocks.get("tool_calls", [])
        ]
    calls = []
    for b in raw_blocks: # Claude content block, GPT output item (객체 또는 dict)
        get = b.get if isinstance(b, dict) else (lambda k, b=b: getattr(b, k, None))
        if get("type") == "tool_use":
            calls.append((get("id"), get("name"), get("input")))
        elif get("type") == "function_call":
            calls.append((get("call_id"), get("name"), get("arguments")))
    return calls
