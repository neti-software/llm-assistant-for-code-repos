#!/usr/bin/env python3
"""
Test script for the new 'neti' PromptLayer prompt
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm_module.llm_builder import build_llm
from src.utils.helper import load_yaml

def test_prompt_loading():
    """Test that the new prompt loads correctly"""
    print("🧪 TEST 1: Prompt Loading")
    print("-" * 40)
    
    llm_config = load_yaml('configs/llm_config.yaml')
    llm = build_llm(llm_config)
    
    # Check status
    if hasattr(llm, 'get_promptlayer_status'):
        status = llm.get_promptlayer_status()
        prompt_name = status.get('prompt_name', 'unknown')
        active = status.get('active', False)
        
        print(f"Prompt name: {prompt_name}")
        print(f"PromptLayer active: {active}")
        
        if prompt_name == 'neti' and active:
            print("✅ SUCCESS: New 'neti' prompt loaded correctly!")
            return True, llm
        else:
            print(f"❌ FAILED: Expected 'neti', got '{prompt_name}'")
            return False, llm
    else:
        print("❌ FAILED: No status method available")
        return False, llm

def test_repository_questions(llm):
    """Test repository assistant behavior"""
    print("\n🧪 TEST 2: Repository Assistant Behavior")
    print("-" * 40)
    
    test_cases = [
        "search for authentication code in repositories",
        "find login functionality", 
        "what files contain database connections",
        "show me the project structure"
    ]
    
    results = []
    
    for i, question in enumerate(test_cases, 1):
        print(f"\nTest {i}: {question}")
        
        try:
            want_tool, resp = llm.generate(question)
            
            print(f"  Want tool: {want_tool}")
            
            if want_tool and isinstance(resp, dict):
                tool_name = resp.get('action', 'unknown')
                print(f"  Tool: {tool_name}")
                print(f"  ✅ SUCCESS: Correctly requested tool call!")
                results.append(True)
                break  # One successful tool call is enough to prove it works
            else:
                content = resp.get('content', '') if isinstance(resp, dict) else str(resp)
                print(f"  Response: {content[:100]}...")
                
                if not content.strip():
                    print(f"  ❌ FAILED: Empty response")
                    results.append(False)
                elif 'prompt' in content.lower() and 'engineering' in content.lower():
                    print(f"  ❌ FAILED: Still getting prompt engineering advice")
                    results.append(False)
                elif any(word in content.lower() for word in ['uncertain', 'search', 'evidence', 'repository']):
                    print(f"  ✅ SUCCESS: Repository assistant language detected!")
                    results.append(True)
                else:
                    print(f"  ⚠️  UNCLEAR: Got response but unsure if correct")
                    results.append(False)
                    
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append(False)
    
    return any(results)

def test_non_repo_question(llm):
    """Test with a non-repository question to see behavior"""
    print("\n🧪 TEST 3: Non-Repository Question")
    print("-" * 40)
    
    question = "what is the capital of France"
    print(f"Question: {question}")
    
    try:
        want_tool, resp = llm.generate(question)
        
        print(f"Want tool: {want_tool}")
        content = resp.get('content', '') if isinstance(resp, dict) else str(resp)
        print(f"Response: {content[:150]}...")
        
        # For non-repo questions, it should either:
        # 1. Say UNCERTAIN, or 
        # 2. Give a direct answer, or
        # 3. Try to use tools anyway (also valid behavior)
        
        if 'uncertain' in content.lower():
            print("✅ SUCCESS: Correctly identified as outside scope")
            return True
        elif want_tool:
            print("✅ SUCCESS: Trying to use tools (acceptable behavior)")
            return True
        elif content.strip() and 'prompt' not in content.lower():
            print("✅ SUCCESS: Gave a reasonable response")
            return True
        else:
            print("⚠️  UNCLEAR: Response behavior unclear")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 TESTING NEW 'NETI' PROMPTLAYER PROMPT")
    print("=" * 50)
    
    # Test 1: Prompt loading
    loading_success, llm = test_prompt_loading()
    
    if not loading_success:
        print("\n❌ OVERALL RESULT: FAILED - Prompt not loading correctly")
        return
    
    # Test 2: Repository questions
    repo_success = test_repository_questions(llm)
    
    # Test 3: Non-repository question
    non_repo_success = test_non_repo_question(llm)
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    print("=" * 50)
    print(f"✅ Prompt Loading: {'PASS' if loading_success else 'FAIL'}")
    print(f"✅ Repository Behavior: {'PASS' if repo_success else 'FAIL'}")
    print(f"✅ General Behavior: {'PASS' if non_repo_success else 'FAIL'}")
    
    if loading_success and repo_success:
        print("\n🎉 OVERALL: SUCCESS! The 'neti' prompt is working correctly!")
        print("🎯 PromptLayer setup is now complete and functional!")
    else:
        print("\n⚠️  OVERALL: Issues detected. May need further debugging.")

if __name__ == "__main__":
    main()
