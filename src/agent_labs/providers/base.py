"""provider 공통 베이스: 재시도 루프 + 어댑터 인터페이스"""

import random
import time
from abc import ABC, abstractmethod

from ..errors import RETRYABLE
from ..types import Request, Response


class LLMProvider(ABC):
    def __init__(self, model: str, max_retries: int = 3, backoff: float = 1.0, max_backoff: float = 30.0):
        self.model = model
        self.max_retries = max_retries # 첫 시도 이후 추가 시도 횟수
        self.backoff = backoff # 첫 재시도 대기(초), 이후 2배씩 증가
        self.max_backoff = max_backoff

    def generate(self, req: Request) -> Response:
        """요청 1회 (일시적 오류는 지수 백오프로 재시도)"""
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._invoke(req)
                break
            except RETRYABLE:
                if attempt == self.max_retries: # 마지막 시도까지 실패
                    raise
                delay = min(self.backoff * 2 ** attempt, self.max_backoff)
                time.sleep(delay * random.uniform(0.5, 1.0)) # 지터로 동시 재시도 분산
        return self._parse(raw)

    def stream(self, req: Request, on_text) -> Response:
        """조각이 올 때마다 on_text(조각) 호출, 끝나면 Response (첫 조각 전 실패만 재시도)"""
        started = False

        def emit(chunk):
            nonlocal started
            started = True
            on_text(chunk)

        for attempt in range(self.max_retries + 1):
            try:
                return self._parse(self._stream(req, emit))
            except RETRYABLE:
                if started or attempt == self.max_retries: # 이미 보여 준 조각이 있으면 이어 붙일 수 없음
                    raise
                delay = min(self.backoff * 2 ** attempt, self.max_backoff)
                time.sleep(delay * random.uniform(0.5, 1.0))

    def _stream(self, req: Request, on_text):
        """스트리밍 미지원 provider는 한 번에 받아 통째로 넘김"""
        raw = self._invoke(req)
        on_text(self._parse(raw).text)
        return raw

    @abstractmethod
    def _to_payload(self, req: Request) -> dict:
        """공통 Request -> provider Payload 변환"""

    @abstractmethod
    def _invoke(self, req: Request):
        """실제 API 호출, SDK 예외를 ..errors 타입으로 변환"""

    @abstractmethod
    def _parse(self, raw) -> Response:
        """provider 원본 응답 -> 공통 Response"""

    def __repr__(self):
        return f"{type(self).__name__}(model={self.model!r})"
