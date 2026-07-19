"""Thin wrapper around the OpenAI SDK pointed at DashScope's OpenAI-compatible endpoint."""

import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0

_REGIONS = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_region = os.environ.get("DASHSCOPE_REGION", "intl")
BASE_URL = _REGIONS.get(_region, _REGIONS["intl"])

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def call(model: str, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", **kwargs):
    """Call a Qwen model via DashScope's OpenAI-compatible chat completions endpoint.

    Retries transient errors (connection issues, timeouts, rate limits, 5xx) with
    exponential backoff. Returns the raw ChatCompletion response.
    """
    client = get_client()
    request_kwargs = {"model": model, "messages": messages, **kwargs}
    if tools is not None:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = tool_choice

    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**request_kwargs)
        except RETRYABLE_EXCEPTIONS:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
