from src.langgraph.agents.repo_intelligence import RepoIntelligenceAgent

# Test the conversion method directly
agent = RepoIntelligenceAgent([])

# Mock search results in the actual format
mock_results = [
    {
        "project": "filplus-registry",
        "path_to_file": "filplus-registry/src/hooks/useApplicationActions.ts",
        "value": "const datacap = activeRequest?.['Allocation Amount']",
        "score": 0.5273,
        "start_line": 123,
        "end_line": 123
    },
    {
        "project": "filplus-utils",
        "path_to_file": "filplus-utils/README.md",
        "value": "This repository contains utilities for Filecoin+ DataCap management",
        "score": 0.4456,
        "start_line": 1,
        "end_line": 1
    }
]

print("Testing conversion of mock search results...")
evidence_items = agent._convert_results("rag_search", mock_results, {})

print(f"Converted {len(evidence_items)} evidence items:")
for i, item in enumerate(evidence_items):
    print(f"Evidence {i+1}:")
    print(f"  Source: {item.source_path}")
    print(f"  Snippet: {item.snippet}")
    print(f"  Metadata: {item.metadata}")
    print()
