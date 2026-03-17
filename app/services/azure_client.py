from __future__ import annotations

import os
import time
from typing import Optional

import requests
from config.exceptions import PipelineError
from utils.logging import get_logger
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


class AzureClient:
    """Azure OpenAI client for LLM requests."""

    def __init__(
        self,
        api_key: str,
        deployment: str,
        endpoint: str,
        api_version: str,
        system_message: str,
        max_rpm: int = 30,
    ):
        self.api_key = api_key
        self.deployment = deployment
        self.endpoint = endpoint
        self.api_version = api_version
        self.system_message = system_message
        self.full_endpoint = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )
        self._limiter = RateLimiter(max_rpm=max_rpm)
        self._max_retries_429 = 5

    @classmethod
    def from_env(cls, system_message: str, max_rpm: int = 30) -> Optional["AzureClient"]:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if all([api_key, deployment, endpoint, api_version]):
            return cls(api_key, deployment, endpoint, api_version, system_message, max_rpm=max_rpm)  # type: ignore
        return None

    def send(self, prompt: str, *, timeout: int = 60) -> tuple[str, dict]:
        """Send prompt to LLM and return response with usage stats.
        
        Returns:
            tuple of (response_message, usage_dict)
            usage_dict contains: prompt_tokens, completion_tokens, total_tokens
        
        Raises:
            PipelineError on failure.
        """
        if not self.full_endpoint or not self.api_key:
            raise PipelineError("AzureClient not configured")

        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
        }

        logger.info("Sending request to LLM...")
        logger.debug(
            "Endpoint: %s, Prompt length: %d chars", self.full_endpoint, len(prompt)
        )

        # Rate-limit: wait for next available slot
        self._limiter.acquire()

        response = self._send_with_retry(headers, payload, timeout)

        try:
            data = response.json()
        except Exception as e:
            raise PipelineError(f"Failed to parse LLM response: {e}") from e

        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        if usage:
            logger.info(
                "Tokens used: %d (prompt=%d, completion=%d)",
                usage.get("total_tokens", 0),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        if message.strip().startswith("["):
            row_count = message.count("row_id")
            logger.info("Classified %d rows", row_count)
        else:
            logger.debug("Response: %s...", message[:100] if message else "(empty)")

        return message, usage

    def _send_with_retry(
        self, headers: dict, payload: dict, timeout: int
    ) -> requests.Response:
        """Send HTTP request with retry on 429 (rate-limited)."""
        for attempt in range(1, self._max_retries_429 + 1):
            try:
                response = requests.post(
                    self.full_endpoint, headers=headers, json=payload, timeout=timeout
                )
            except Exception as e:
                logger.error("Request failed: %s", e)
                raise PipelineError(f"LLM request failed: {e}") from e

            if response.status_code != 429:
                # Not throttled — check for other errors
                if response.status_code != 200:
                    error_msg = response.text[:500]
                    try:
                        error_msg = str(response.json())
                    except Exception:
                        pass
                    logger.error("LLM error [%d]: %s", response.status_code, error_msg)
                    raise PipelineError(f"LLM error [{response.status_code}]: {error_msg}")
                return response

            # 429 — honour Retry-After header or use exponential backoff
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = min(2 ** attempt, 60)

            logger.warning(
                "Rate-limited (429). Retry %d/%d in %.1fs",
                attempt, self._max_retries_429, wait,
            )
            time.sleep(wait)

        raise PipelineError(
            f"Azure OpenAI rate-limited after {self._max_retries_429} retries"
        )

    @property
    def rate_limiter_stats(self) -> dict:
        """Expose rate-limiter statistics for end-of-run reporting."""
        return self._limiter.stats


__all__ = ["AzureClient"]
