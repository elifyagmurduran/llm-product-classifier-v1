"""Simple token-bucket rate limiter for Azure OpenAI requests.

Ensures the pipeline stays under a configured requests-per-minute (RPM)
limit, preventing Azure throttling (HTTP 429) that causes expensive
retry storms and overall slower throughput.

Usage:
    from utils.rate_limiter import RateLimiter

    limiter = RateLimiter(max_rpm=30)
    limiter.acquire()          # blocks until a request slot is available
    response = client.send(...)
"""
from __future__ import annotations

import time
from utils.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Token-bucket rate limiter scoped to requests per minute."""

    def __init__(self, max_rpm: int = 30):
        self.max_rpm = max_rpm
        self.interval = 60.0 / max_rpm  # seconds between requests
        self._last_request: float = 0.0
        self._total_wait: float = 0.0
        self._total_requests: int = 0

    def acquire(self) -> None:
        """Wait until the next request slot is available."""
        now = time.monotonic()
        elapsed = now - self._last_request
        wait = self.interval - elapsed

        if wait > 0:
            logger.debug("Rate limiter: waiting %.2fs before next request", wait)
            time.sleep(wait)
            self._total_wait += wait

        self._last_request = time.monotonic()
        self._total_requests += 1

    @property
    def stats(self) -> dict:
        """Return cumulative rate-limiter statistics."""
        return {
            "total_requests": self._total_requests,
            "total_wait_seconds": round(self._total_wait, 2),
            "avg_wait_seconds": round(
                self._total_wait / self._total_requests, 2
            ) if self._total_requests > 0 else 0.0,
        }


__all__ = ["RateLimiter"]
