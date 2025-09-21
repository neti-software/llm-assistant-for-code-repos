#!/usr/bin/env python3
"""Test script to see what the LLM returns for task planning."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_module.llm_builder import build_llm

def test_llm_response():
    """Test what the LLM returns for the task planning prompt."""

    llm_config = {
        "type": "cloud",
        "provider": "openrouter",
        "model": "openrouter/sonoma-sky-alpha",
        "context_length": 2000000,
        "max_completion_tokens": 4000,
        "tool_choice": "auto",
        "path_to_api_key": "configs/openrouter_key.yaml",
        "path_to_prompt_config": "configs/prompt_config.yaml",
        "base_url": "https://openrouter.ai/api/v1"
    }

    try:
        llm = build_llm(llm_config)

        prompt = """Analyze this user question and determine what types of research tasks are needed to provide a comprehensive answer.

User Question: What is IPNI in the Filecoin ecosystem and is it centralized?

Available Task Types:
1. repo_research - Gather repository-level context, documentation, and relevant code snippets
2. code_context - Collect specific code implementations, functions, or classes
3. documentation_search - Find README files, API docs, or help documentation
4. configuration_analysis - Analyze configuration files, settings, or deployment files

Consider:
- What specific information does the user need?
- What types of evidence would be most helpful?
- Are they asking about implementation details, configuration, or general concepts?
- Do they need code examples or just conceptual explanations?

Respond with a JSON array of tasks. Each task should have:
- type: one of the available task types
- description: clear description of what to research
- priority: integer (1-10, higher = more important)

Example response:
[
  {
    "type": "repo_research",
    "description": "Search for existing implementations and usage patterns",
    "priority": 8
  }
]
"""

        print("Calling LLM for task planning...")
        want_tool, response = llm.generate(prompt)

        print(f"Want tool: {want_tool}")
        print(f"Response length: {len(response)}")
        print(f"Response: {response}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_response()
