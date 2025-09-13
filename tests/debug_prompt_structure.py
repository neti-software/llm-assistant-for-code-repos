#!/usr/bin/env python3
"""
Debug script to see exactly what PromptLayer is sending
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm_module.llm_builder import build_llm
from src.utils.helper import load_yaml

print("🔍 DEBUGGING PROMPTLAYER PROMPT STRUCTURE")
print("=" * 50)

llm_config = load_yaml('configs/llm_config.yaml')
llm = build_llm(llm_config)

# Check what prompt name is being used
if hasattr(llm, 'get_promptlayer_status'):
    status = llm.get_promptlayer_status()
    print(f"Prompt name: {status.get('prompt_name', 'unknown')}")

print("\nTesting simple question to see debug output...")
try:
    want_tool, resp = llm.generate("test")
    print(f"Want tool: {want_tool}")
    print(f"Response: {resp}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("EXPECTED RESULTS:")
print("- Number of messages: 1 (system only)")
print("- No missing variable warnings")
print("- Non-empty response")
