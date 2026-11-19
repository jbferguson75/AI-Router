"""Shared Ollama client with bounded requests and transient-error retries."""

import logging
import time

import ollama

from config import OLLAMA_MAX_RETRIES, OLLAMA_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)
client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)


def _call_with_retry(method, **kwargs):
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            return method(**kwargs)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = (
                status_code is None
                or status_code == 429
                or status_code >= 500
            )
            if not retryable or attempt >= OLLAMA_MAX_RETRIES:
                raise

            delay = min(2 ** attempt, 8)
            logger.warning(
                "Ollama request failed; retrying in %ss (%s/%s): %s",
                delay,
                attempt + 1,
                OLLAMA_MAX_RETRIES,
                exc
            )
            time.sleep(delay)


def chat(**kwargs):
    return _call_with_retry(client.chat, **kwargs)


def embed(**kwargs):
    return _call_with_retry(client.embed, **kwargs)
