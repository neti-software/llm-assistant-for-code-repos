import asyncio
import os
import time
from datetime import datetime
from functools import wraps
from typing import Callable, Any, Dict, List

from src.utils.logger import logger


class ExecutionProfiler:
    """Singleton profiler for measuring function execution times."""

    _instance = None

    def __new__(cls, enable: bool = True, save_logs: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, enable: bool = True, save_logs: bool = False):
        if self._initialized:
            return
        self._execution_times: Dict[str, List[float]] = {}
        self._decorated_functions: List[str] = []
        self._initialized = True

    # ---------------- Decorator ----------------
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            full_name = self._get_full_function_name(func, args)
            self._execution_times.setdefault(full_name, []).append(elapsed_time)
            return result

        self._store_function_name(func)
        return wrapper

    # ---------------- Helpers ----------------
    def _get_full_function_name(self, func: Callable, args: tuple) -> str:
        """Return class.method or function name."""
        if args and hasattr(args[0], "__class__"):
            class_name = func.__qualname__.split(".")[0]
            return f"{class_name}.{func.__name__}"
        if "." in func.__qualname__:
            class_name, _ = func.__qualname__.rsplit(".", 1)
            return f"{class_name}.{func.__name__}"
        return func.__name__

    def _store_function_name(self, func: Callable) -> None:
        full_name = self._get_full_function_name(func, [])
        if full_name not in self._decorated_functions:
            self._decorated_functions.append(full_name)

    # ---------------- Reporting ----------------
    def print_info(self) -> None:
        for func_name, times in self._execution_times.items():
            total_time = sum(times)
            count = len(times)
            if count > 1:
                mean_time = total_time / count
                logger.debug(
                    f"{self.__class__.__name__} - {func_name}: "
                    f"called {count} times, total {total_time:.4f}s, mean {mean_time:.4f}s"
                )
            else:
                logger.debug(
                    f"{self.__class__.__name__} - {func_name}: total {total_time:.4f}s"
                )

    # ---------------- Maintenance ----------------
    def clean(self) -> None:
        """Reset profiler state."""
        self._execution_times.clear()
        self._decorated_functions.clear()


# ---------------- Other decorators ----------------
def time_it(func: Callable) -> Callable:
    """Measure execution time of sync/async functions and log it."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Executed {func.__name__}: {elapsed:.5f} sec")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Executed {func.__name__}: {elapsed:.5f} sec")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Global singleton instance
execution_profiler = ExecutionProfiler()
