#!/usr/bin/env python3
"""Test script for improved verifier response generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langgraph.agents.verifier import VerifierAgent
from langgraph.state_models import ConversationState, EvidenceItem
from llm_module.llm_builder import build_llm
from utils.helper import load_yaml


def test_verifier_with_filecoin_question():
    """Test the improved verifier with the Filecoin DC question."""

    # Load LLM config
    llm_config = load_yaml("configs/llm_config.yaml")

    # Build LLM
    try:
        llm = build_llm(llm_config)
        print(f"[TEST] Built LLM: {llm}")
    except Exception as e:
        print(f"[TEST] Failed to build LLM: {e}")
        print("[TEST] Testing with None LLM (will fallback to rule-based)")
        llm = None

    # Create verifier
    verifier = VerifierAgent(llm=llm)

    # Create sample evidence similar to what was collected
    evidence_items = [
        EvidenceItem(
            source_path="filecoin-docs/basics/how-storage-works/filecoin-plus.md",
            confidence=0.9,
            summary="Filecoin Plus program explanation including DataCap tokens and verified deals",
            snippet="DataCap is a token paid to storage providers as part of a deal in which the client and the data they are storing is verified by a Filecoin Plus allocator.",
            citations=["filecoin-docs/basics/how-storage-works/filecoin-plus.md"]
        ),
        EvidenceItem(
            source_path="filecoin-docs/storage-providers/filecoin-deals/verified-deals.md",
            confidence=0.85,
            summary="Detailed explanation of verified deals and DataCap allocation",
            snippet="A deal becomes verified after the data owner completes a verification process where allocators assess the client's use case.",
            citations=["filecoin-docs/storage-providers/filecoin-deals/verified-deals.md"]
        )
    ]

    # Create conversation state
    state = ConversationState()
    state.evidence_store = evidence_items

    # Test question
    question = "What is DC, what is a verified deal, and what is the journey of the data cap token from the root key holder to the storage provider (SP)?"

    print(f"[TEST] Testing question: {question}")
    print(f"[TEST] Evidence items: {len(evidence_items)}")

    # Run verifier
    try:
        report = verifier.evaluate(state, question=question)

        print(f"[TEST] Coverage score: {report.coverage_score:.3f}")
        print(f"[TEST] Missing items: {report.missing_items}")
        print(f"[TEST] Citations: {len(report.citations)}")
        print("[TEST] Response text:")
        print("-" * 50)
        print(report.response_text)
        print("-" * 50)

        print("[TEST] SUCCESS: Verifier generated structured response!")

    except Exception as e:
        print(f"[TEST] ERROR: Verifier failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_verifier_with_filecoin_question()
