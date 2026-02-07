"""
Rate Limiter
============

Simple rate limiter for API calls to prevent hitting rate limits.

H4 FIX: Used by broker implementations to throttle API requests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Thread-safe implementation that tracks request timestamps
    and blocks when the rate limit would be exceeded.

    Example:
        >>> limiter = RateLimiter(requests_per_minute=60)
        >>> limiter.acquire()  # Blocks if rate limit exceeded
        >>> # Make API call
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        requests_per_second: float | None = None,
        burst_size: int | None = None,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute (default: 60)
            requests_per_second: Max requests per second (overrides per_minute)
            burst_size: Max burst size (default: 10% of per-minute rate)
        """
        if requests_per_second is not None:
            self._rate_per_second = float(requests_per_second)
        elif requests_per_minute is not None:
            self._rate_per_second = float(requests_per_minute) / 60.0
        else:
            self._rate_per_second = 1.0  # Default: 1 req/sec

        self._min_interval = 1.0 / self._rate_per_second
        self._burst_size = burst_size or max(1, int(self._rate_per_second * 6))  # 10% of per-minute

        # Track request timestamps
        self._timestamps: Deque[float] = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0

        logger.debug(
            f"RateLimiter: {self._rate_per_second:.2f} req/s, "
            f"burst={self._burst_size}"
        )

    def acquire(self, timeout: float | None = None) -> bool:
        """
        Acquire permission to make a request.

        Blocks if rate limit would be exceeded.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.monotonic()

        with self._lock:
            while True:
                now = time.monotonic()

                # Check timeout
                if timeout is not None and (now - start_time) >= timeout:
                    logger.warning("Rate limiter timeout")
                    return False

                # Clean old timestamps (older than 60 seconds)
                cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()

                # Check if we can proceed
                if len(self._timestamps) < self._burst_size:
                    # Burst capacity available
                    self._timestamps.append(now)
                    self._last_request_time = now
                    return True

                # Check minimum interval from last request
                time_since_last = now - self._last_request_time
                if time_since_last >= self._min_interval:
                    self._timestamps.append(now)
                    self._last_request_time = now
                    return True

                # Need to wait
                sleep_time = self._min_interval - time_since_last
                if timeout is not None:
                    remaining = timeout - (now - start_time)
                    if sleep_time > remaining:
                        sleep_time = remaining

                # Release lock while sleeping
                self._lock.release()
                try:
                    time.sleep(min(sleep_time, 0.1))  # Sleep in small increments
                finally:
                    self._lock.acquire()

    def try_acquire(self) -> bool:
        """
        Try to acquire without blocking.

        Returns:
            True if acquired, False if would exceed rate limit
        """
        return self.acquire(timeout=0.0)

    def get_wait_time(self) -> float:
        """
        Get estimated wait time until next request allowed.

        Returns:
            Seconds until next request allowed (0 if can proceed now)
        """
        with self._lock:
            now = time.monotonic()

            # Clean old timestamps
            cutoff = now - 60.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) < self._burst_size:
                return 0.0

            time_since_last = now - self._last_request_time
            if time_since_last >= self._min_interval:
                return 0.0

            return self._min_interval - time_since_last

    def get_current_rate(self) -> float:
        """
        Get current request rate (requests per second over last minute).

        Returns:
            Current rate in requests per second
        """
        with self._lock:
            now = time.monotonic()

            # Clean old timestamps
            cutoff = now - 60.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if not self._timestamps:
                return 0.0

            # Calculate rate over the actual time window
            time_window = now - self._timestamps[0]
            if time_window <= 0:
                return 0.0

            return len(self._timestamps) / time_window

    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self._lock:
            self._timestamps.clear()
            self._last_request_time = 0.0

    @property
    def max_rate_per_second(self) -> float:
        """Get configured max rate per second."""
        return self._rate_per_second

    @property
    def burst_size(self) -> int:
        """Get burst size."""
        return self._burst_size
