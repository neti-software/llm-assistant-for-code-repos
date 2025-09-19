#!/usr/bin/env python3
"""Convenience entry point to run the LangGraph-focused test suites."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEST_PATHS = [
    "tests/langgraph",
    "tests/integration",
]


def main() -> int:
    repo_root = Path(__file__).parent
    cmd = [sys.executable, "-m", "pytest", *TEST_PATHS]
    process = subprocess.run(cmd, cwd=repo_root)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
