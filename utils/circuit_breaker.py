"""简单的熔断器实现（支持同步和异步）"""
import logging
import threading
import time
import asyncio
from enum import Enum
from typing import Callable, Optional, TypeVar, Any, Awaitable

logger = logging.getLogger("safeguard_system")

T = TypeVar("T")


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """熔断器打开异常"""
    pass


class CircuitBreaker:
    """
    简单的熔断器实现，支持同步 call 和异步 async_call。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        half_open_max_calls: int = 3,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        # 同时支持线程安全和异步环境下的状态一致性
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _before_call(self):
        """调用前的状态检查逻辑"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and (time.time() - self._last_failure_time) > self.timeout:
                    logger.info("CircuitBreaker[%s]: OPEN -> HALF_OPEN (timeout expired)", self.name)
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    raise CircuitBreakerOpen(f"Circuit breaker [{self.name}] is OPEN")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(f"Circuit breaker [{self.name}] HALF_OPEN max calls exceeded")
                self._half_open_calls += 1

    def call(self, func: Callable[[], T]) -> T:
        """同步调用"""
        self._before_call()
        try:
            result = func()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    async def async_call(self, func: Callable[[], Awaitable[T]]) -> T:
        """异步调用"""
        self._before_call()
        try:
            result = await func()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("CircuitBreaker[%s]: HALF_OPEN -> CLOSED (success)", self.name)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                logger.warning("CircuitBreaker[%s]: HALF_OPEN -> OPEN (failure)", self.name)
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning("CircuitBreaker[%s]: CLOSED -> OPEN (failures=%d)", self.name, self._failure_count)
                    self._state = CircuitState.OPEN

    def reset(self):
        with self._lock:
            logger.info("CircuitBreaker[%s]: Manual reset", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
