#!/usr/bin/env python3
"""
Quick test script to demonstrate the new progress logging.
This simulates embedding operations with progress context.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.profiler import time_it, set_progress_context, clear_progress_context
from src.utils.logger import logger


@time_it
def simulate_embed_operation(batch_size: int) -> list:
    """Simulate an embedding operation."""
    time.sleep(0.5)  # Simulate API call
    return [[0.1 * i for _ in range(100)] for i in range(batch_size)]


def main():
    repos = ["repo_1", "repo_2", "repo_3"]
    batches_per_repo = 3
    
    logger.info("=" * 60)
    logger.info("Starting progress logging demo")
    logger.info("=" * 60)
    
    for repo_idx, repo_name in enumerate(repos, start=1):
        set_progress_context(repo_name, repo_idx, len(repos))
        logger.info(f"\n📦 Processing repository: {repo_name}")
        
        for batch_idx in range(1, batches_per_repo + 1):
            batch_size = 32
            simulate_embed_operation(batch_size)
        
    clear_progress_context()
    logger.info("\n✅ Demo completed!")


if __name__ == "__main__":
    main()

