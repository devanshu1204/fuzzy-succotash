import asyncio
import logging
import random
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._client_utils import StreamChunkTimeoutError
from openai import APIError, RateLimitError
from pydantic import ValidationError

from config.settings import LITELLM_API_KEY, LITELLM_PROXY_URL, MODEL_NAME

log = logging.getLogger(__name__)

# Bedrock/Anthropic streams for large aggregations can have multi-minute gaps
# between chunks (especially when the model is producing huge JSON payloads
# for sections with 40+ child segments). The default 120s is too tight.
_STREAM_CHUNK_TIMEOUT_S = 600.0

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_INITIAL_DELAY = 1.0
_MAX_DELAY = 60.0


def _format_validation_error(e: ValidationError, limit: int = 3) -> str:
    errs = e.errors()
    parts = []
    for err in errs[:limit]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', '')}")
    suffix = f" (+{len(errs) - limit} more)" if len(errs) > limit else ""
    return f"{len(errs)} error(s) [{'; '.join(parts)}]{suffix}"


class LLMWithRetry:
    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def _invoke_with_retry(self, fn, *args, **kwargs):
        delay = _INITIAL_DELAY
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:
                last_exc = e
            except APIError as e:
                if e.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                last_exc = e
            except ValidationError as e:
                last_exc = e
                log.warning(
                    f"Structured-output validation failed on attempt "
                    f"{attempt + 1}/{_MAX_RETRIES}: {_format_validation_error(e)}; retrying"
                )
            except StreamChunkTimeoutError as e:
                last_exc = e
                log.warning(
                    f"LLM stream stalled on attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"after {getattr(e, 'chunks_received', '?')} chunks; retrying"
                )
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError("LLM max retries exceeded") from last_exc
            jitter = random.uniform(0, delay * 0.2)
            time.sleep(min(delay + jitter, _MAX_DELAY))
            delay = min(delay * 2, _MAX_DELAY)

    async def _ainvoke_with_retry(self, fn, *args, **kwargs):
        delay = _INITIAL_DELAY
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await fn(*args, **kwargs)
            except RateLimitError as e:
                last_exc = e
            except APIError as e:
                if e.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                last_exc = e
            except ValidationError as e:
                last_exc = e
                log.warning(
                    f"Structured-output validation failed on attempt "
                    f"{attempt + 1}/{_MAX_RETRIES}: {_format_validation_error(e)}; retrying"
                )
            except StreamChunkTimeoutError as e:
                last_exc = e
                log.warning(
                    f"LLM stream stalled on attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"after {getattr(e, 'chunks_received', '?')} chunks; retrying"
                )
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError("LLM max retries exceeded") from last_exc
            jitter = random.uniform(0, delay * 0.2)
            await asyncio.sleep(min(delay + jitter, _MAX_DELAY))
            delay = min(delay * 2, _MAX_DELAY)

    def invoke(self, *args, **kwargs):
        return self._invoke_with_retry(self._llm.invoke, *args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self._ainvoke_with_retry(self._llm.ainvoke, *args, **kwargs)

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
        stream_chunk_timeout=_STREAM_CHUNK_TIMEOUT_S,
    )
)
