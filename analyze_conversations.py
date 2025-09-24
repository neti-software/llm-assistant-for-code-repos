#!/usr/bin/env python3
"""Analyze and rank conversation quality from saved chat history files."""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
from src.conversation.conversation_history import ConversationHistory

def analyze_single_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single conversation file."""
    return ConversationHistory.rank_conversation_quality(file_path)

def analyze_directory(directory: str) -> List[Dict[str, Any]]:
    """Analyze all conversation files in a directory."""
    results = []
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return results

    json_files = list(directory_path.glob("*.json"))

    for file_path in json_files:
        result = analyze_single_file(str(file_path))
        results.append(result)

    return results

def print_analysis(results: List[Dict[str, Any]]) -> None:
    """Print formatted analysis results."""
    if not results:
        print("No conversation files found to analyze.")
        return

    print("\n" + "="*80)
    print("CONVERSATION QUALITY ANALYSIS")
    print("="*80)

    # Sort by score (highest first)
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. {Path(result['filename']).name}")
        print(f"   Grade: {result.get('grade', 'Unknown')} ({result.get('percentage', 0):.1f}%)")
        print(f"   Score: {result.get('score', 0)}/{result.get('max_score', 10)}")

        if "error" in result:
            print(f"   Error: {result['error']}")
            continue

        criteria = result.get("criteria", {})
        print("   Criteria met:")
        for criterion, value in criteria.items():
            if isinstance(value, bool):
                status = "✓" if value else "✗"
                print(f"     {status} {criterion.replace('_', ' ').title()}")
            elif isinstance(value, int):
                if criterion == "evidence_count":
                    print(f"     📊 Evidence items: {value}")
                elif criterion == "tool_calls_count":
                    print(f"     🔧 Tool calls: {value}")
                elif criterion == "iterations":
                    print(f"     🔄 Iterations: {value}")

        # Show additional metrics if available
        if "metrics" in result:
            metrics = result["metrics"]
            if "total_time" in metrics:
                print(f"     ⏱️  Execution time: {metrics['total_time']:.2f}s")
            if "total_iterations" in metrics:
                print(f"     🔄 Total iterations: {metrics['total_iterations']}")

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_score = sum(r.get("score", 0) for r in valid_results) / len(valid_results)
        avg_percentage = sum(r.get("percentage", 0) for r in valid_results) / len(valid_results)

        print(f"Total conversations analyzed: {len(results)}")
        print(f"Valid conversations: {len(valid_results)}")
        print(f"Average score: {avg_score:.1f}/10 ({avg_percentage:.1f}%)")

        grade_counts = {}
        for result in valid_results:
            grade = result.get("grade", "Unknown").split()[0]  # Get just the letter grade
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        print("Grade distribution:")
        for grade in sorted(grade_counts.keys()):
            count = grade_counts[grade]
            percentage = (count / len(valid_results)) * 100
            print(f"  {grade}: {count} ({percentage:.1f}%)")
    else:
        print("No valid conversations found to analyze.")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_conversations.py <file_or_directory>")
        print("  python analyze_conversations.py logs/chat_history/  # Analyze all files in directory")
        print("  python analyze_conversations.py logs/chat_history/2025-09-21_17-01-08.json  # Analyze single file")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isfile(target):
        # Analyze single file
        result = analyze_single_file(target)
        print_analysis([result])
    elif os.path.isdir(target):
        # Analyze all files in directory
        results = analyze_directory(target)
        print_analysis(results)
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
