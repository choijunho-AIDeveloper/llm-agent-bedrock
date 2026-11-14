"""provider 공통 예외 계층 (SDK별 예외를 이 타입들로 변환해서 올림)"""


class LLMError(Exception):
    """모든 provider 예외의 부모"""


class AuthError(LLMError):
    """401/403, API 키 누락·오류 (재시도 무의미)"""


class RateLimitError(LLMError):
    """429, 요청 한도 초과 (백오프 후 재시도)"""


class TransientError(LLMError):
    """5xx, 네트워크 끊김, 타임아웃 등 일시적 오류 (재시도)"""


class ContextLengthError(LLMError):
    """입력이 컨텍스트 윈도우를 초과 (재시도 무의미, 메시지 축약 필요)"""


RETRYABLE = (RateLimitError, TransientError) # base.py 재시도 대상
