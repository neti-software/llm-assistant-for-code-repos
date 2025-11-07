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

    def __init__(self):
        if self._initialized:
            return
        self._execution_times: Dict[str, List[float]] = {}
        self._decorated_functions: List[str] = []
        self._events: List[Dict[str, Any]] = []
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
    def record_event(self, name: str, **metadata: Any) -> None:
        """Record a profiling event with arbitrary metadata."""

        self._events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "metadata": metadata,
            }
        )

    def iter_events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def clean(self) -> None:
        """Reset profiler state."""
        self._execution_times.clear()
        self._decorated_functions.clear()
        self._events.clear()


# Global context for tracking progress
_progress_context = {}

def set_progress_context(repo_name: str, current: int, total: int) -> None:
    """Set the current progress context for logging."""
    _progress_context["repo_name"] = repo_name
    _progress_context["current"] = current
    _progress_context["total"] = total

def clear_progress_context() -> None:
    """Clear the progress context."""
    _progress_context.clear()

def _get_progress_suffix() -> str:
    """Get a progress suffix for logging if context is set."""
    if _progress_context:
        repo = _progress_context.get("repo_name", "")
        current = _progress_context.get("current", 0)
        total = _progress_context.get("total", 0)
        if current > 0 and total > 0:
            return f" [{current}/{total} {repo}]"
    return ""

# ---------------- Other decorators ----------------
def time_it(func: Callable) -> Callable:
    """Measure execution time of sync/async functions and log it."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        progress_suffix = _get_progress_suffix()
        logger.info(f"Executed {func.__name__}: {elapsed:.5f} sec{progress_suffix}")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        progress_suffix = _get_progress_suffix()
        logger.info(f"Executed {func.__name__}: {elapsed:.5f} sec{progress_suffix}")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Global singleton instance
execution_profiler = ExecutionProfiler()
