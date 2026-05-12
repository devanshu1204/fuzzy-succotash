import time
import random
from typing import Any

from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError

from config.settings import LITELLM_API_KEY, LITELLM_PROXY_URL, MODEL_NAME

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_INITIAL_DELAY = 1.0
_MAX_DELAY = 60.0


class LLMWithRetry:
    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def _invoke_with_retry(self, fn, *args, **kwargs):
        delay = _INITIAL_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except RateLimitError:
                pass
            except APIError as e:
                if e.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError("LLM max retries exceeded")
            jitter = random.uniform(0, delay * 0.2)
            time.sleep(min(delay + jitter, _MAX_DELAY))
            delay = min(delay * 2, _MAX_DELAY)

    def invoke(self, *args, **kwargs):
        return self._invoke_with_retry(self._llm.invoke, *args, **kwargs)

    def bind_tools(self, tools, **kwargs) -> "LLMWithRetry":
        return LLMWithRetry(self._llm.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema, **kwargs) -> "LLMWithRetry":
        return LLMWithRetry(self._llm.with_structured_output(schema, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


llm = LLMWithRetry(
    ChatOpenAI(
        base_url=LITELLM_PROXY_URL,
        api_key=LITELLM_API_KEY,
        model=MODEL_NAME,
    )
)
