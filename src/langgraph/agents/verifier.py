"""Verifier agent responsible for drafting final text and coverage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..state_models import ConversationState, EvidenceItem


@dataclass
class VerifierReport:
    response_text: str
    coverage_score: float
    missing_items: List[str]
    citations: List[str]


class VerifierAgent:
    def __init__(self, coverage_threshold: float = 0.7, llm=None) -> None:
        self.coverage_threshold = coverage_threshold
        self.llm = llm

    def evaluate(self, state: ConversationState, *, question: str) -> VerifierReport:
        evidence = state.evidence_store
        if not evidence:
            return VerifierReport(
                response_text=f"Insufficient evidence to answer: {question}",
                coverage_score=0.0,
                missing_items=[f"Collect supporting context for: {question}"],
                citations=[],
            )

        # Use LLM-based analysis if available
        print(f"[DEBUG] VerifierAgent: llm={self.llm is not None}")
        if self.llm:
            print(f"[DEBUG] VerifierAgent: Using LLM-based analysis")
            return self._evaluate_with_llm(question, evidence, state)

        # Fallback to rule-based analysis
        print(f"[DEBUG] VerifierAgent: Using rule-based analysis fallback")
        return self._evaluate_with_rules(question, evidence)

    def _evaluate_with_llm(self, question: str, evidence: List[EvidenceItem], state: ConversationState) -> VerifierReport:
        """Use LLM to provide intelligent coverage analysis and response synthesis."""
        try:
            # Prepare evidence for LLM analysis
            evidence_context = self._format_evidence_for_llm(evidence)

            # Create a structured prompt that requests JSON response
            prompt = f"""You are an expert Evidence Analysis and Coverage Assessment Agent. Your role is to:

## Primary Goals:
1. **Analyze all collected evidence** for relevance and completeness
2. **Assess coverage quality** based on the original question
3. **Identify information gaps** that need to be filled
4. **Synthesize findings** into a comprehensive response
5. **Provide actionable recommendations** for missing information

## Analysis Framework:

### Evidence Quality Assessment:
- **Relevance**: How directly does this evidence address the question?
- **Depth**: Does it provide sufficient detail or just surface-level info?
- **Credibility**: Is the source authoritative and trustworthy?
- **Freshness**: Is the information current and applicable?

### Coverage Analysis:
- **Comprehensiveness**: Are all aspects of the question covered?
- **Detail Level**: Is there sufficient technical depth where needed?
- **Practical Value**: Does it provide actionable information?

### Gap Identification:
- **Missing Technical Details**: Implementation specifics, code examples, configuration steps
- **Missing Conceptual Context**: Background information, related concepts, alternatives
- **Missing Practical Guidance**: Setup instructions, usage examples, troubleshooting

## Evidence Collected:
{evidence_context}

## Original Question:
{question}

## Analysis Instructions:
1. Carefully review ALL evidence provided
2. Assess how well it answers the question
3. Identify specific information that's missing
4. Provide a comprehensive synthesized response
5. Recommend what additional evidence would improve coverage

## Response Format:
You MUST respond with a valid JSON object containing these exact fields:
{{
    "coverage_score": <float 0.0-1.0>,
    "response_text": "<comprehensive answer>",
    "missing_items": ["<specific gap 1>", "<specific gap 2>"],
    "citations": ["<source 1>", "<source 2>"]
}}

Make sure the response_text provides a comprehensive, well-structured answer to the original question based on all the evidence."""

            # Use structured generation with JSON schema
            json_schema = {
                "type": "object",
                "properties": {
                    "coverage_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "response_text": {"type": "string"},
                    "missing_items": {"type": "array", "items": {"type": "string"}},
                    "citations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["coverage_score", "response_text", "missing_items", "citations"]
            }

            # Prepare response format for structured generation
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "verifier_analysis",
                    "schema": json_schema,
                    "strict": True
                }
            }

            if not self.llm:
                raise RuntimeError("LLM not available for structured generation")
            response = self.llm.generate(prompt, response_format=response_format)

            # Parse the LLM response (expecting JSON structure)
            return self._parse_llm_analysis_response(response, question, evidence)

        except Exception as e:
            print(f"[DEBUG] VerifierAgent: LLM analysis failed: {e}")
            # Fallback to rule-based analysis
            return self._evaluate_with_rules(question, evidence)

    def _evaluate_with_rules(self, question: str, evidence: List[EvidenceItem]) -> VerifierReport:
        """Fallback rule-based coverage evaluation with better synthesis."""
        if not evidence:
            return VerifierReport(
                response_text=f"Insufficient evidence to answer: {question}",
                coverage_score=0.0,
                missing_items=[f"Collect supporting context for: {question}"],
                citations=[],
            )

        # Extract key information from evidence
        key_concepts = self._extract_key_concepts(evidence)
        citations = []
        for item in evidence:
            citations.extend(item.citations)

        # Synthesize a basic answer
        response_text = self._synthesize_basic_answer(question, key_concepts, evidence)

        deduped_citations = list(dict.fromkeys(citations))
        coverage_score = self._estimate_coverage(evidence)

        missing_items: List[str] = []
        if coverage_score < self.coverage_threshold:
            missing_items.append("Additional evidence or LLM analysis needed for comprehensive answer.")

        return VerifierReport(
            response_text=response_text,
            coverage_score=coverage_score,
            missing_items=missing_items,
            citations=deduped_citations,
        )

    def _extract_key_concepts(self, evidence: List[EvidenceItem]) -> Dict[str, List[str]]:
        """Extract key concepts from evidence for basic synthesis."""
        concepts = {
            "definitions": [],
            "processes": [],
            "examples": [],
            "sources": []
        }

        for item in evidence:
            # Use full_content from the agent
            content = item.full_content
            if content:
                content_lower = content.lower()
                if any(keyword in content_lower for keyword in ["is a", "are", "defined as", "refers to"]):
                    concepts["definitions"].append(content)
                elif any(keyword in content_lower for keyword in ["process", "workflow", "journey", "steps", "how to"]):
                    concepts["processes"].append(content)
                elif any(keyword in content_lower for keyword in ["example", "for example", "such as", "like"]):
                    concepts["examples"].append(content)

            if item.source_path:
                concepts["sources"].append(item.source_path)

        return concepts

    def _synthesize_basic_answer(self, question: str, concepts: Dict[str, List[str]], evidence: List[EvidenceItem]) -> str:
        """Synthesize a basic answer from extracted concepts."""
        lines = []
        lines.append(f"Based on the collected evidence, here's an analysis of: {question}")
        lines.append("")

        # Add definitions if found
        if concepts["definitions"]:
            lines.append("## Key Definitions:")
            for definition in concepts["definitions"][:3]:  # Limit to 3 definitions
                lines.append(f"- {definition}")
            lines.append("")

        # Add processes if found
        if concepts["processes"]:
            lines.append("## Processes and Workflows:")
            for process in concepts["processes"][:3]:  # Limit to 3 processes
                lines.append(f"- {process}")
            lines.append("")

        # Add examples if found
        if concepts["examples"]:
            lines.append("## Examples and Applications:")
            for example in concepts["examples"][:2]:  # Limit to 2 examples
                lines.append(f"- {example}")
            lines.append("")

        # Add evidence content
        lines.append("## Supporting Evidence:")
        for i, item in enumerate(evidence[:5], 1):  # Limit to 5 items
            # Use full_content from the agent
            content = item.full_content
            if content:
                # Truncate if too long
                truncated_content = content.splitlines()[0][:200] + "..." if len(content) > 200 else content
                lines.append(f"{i}. {truncated_content}")
                if item.source_path:
                    lines.append(f"   *Source: {item.source_path}*")

        return "\n".join(lines)

    def apply_report(self, state: ConversationState, report: VerifierReport) -> None:
        state.control_flags.last_verifier_report = {
            "coverage_score": report.coverage_score,
            "missing_items": list(report.missing_items),
            "citations": list(report.citations),
        }

    def _format_evidence_for_llm(self, evidence: List[EvidenceItem]) -> str:
        """Format evidence collection for LLM analysis."""
        evidence_parts = []
        for i, item in enumerate(evidence, 1):
            part = f"""
                ### Evidence Item {i}:
                - **Source**: {item.source_path or 'Unknown'}
                - **Confidence**: {item.confidence or 0:.3f}
                - **Content**: {item.full_content}
                - **Citations**: {', '.join(item.citations) if item.citations else 'None'}
                - **Metadata**: {item.metadata}
            """
            evidence_parts.append(part)

        return "\n".join(evidence_parts)

    def _parse_llm_analysis_response(self, response_content: str, question: str, evidence: List[EvidenceItem]) -> VerifierReport:
        """Parse LLM analysis response into VerifierReport structure."""
        try:
            import json

            # Try to parse the entire response as JSON first
            try:
                analysis_data = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    # Ultimate fallback
                    raise ValueError("No JSON found in response")

            # Validate required fields
            required_fields = ["coverage_score", "response_text", "missing_items", "citations"]
            for field in required_fields:
                if field not in analysis_data:
                    raise ValueError(f"Missing required field: {field}")

            return VerifierReport(
                response_text=analysis_data["response_text"],
                coverage_score=min(1.0, max(0.0, analysis_data["coverage_score"])),
                missing_items=analysis_data["missing_items"],
                citations=analysis_data["citations"]
            )

        except Exception as e:
            print(f"[DEBUG] VerifierAgent: Failed to parse LLM response: {e}")
            print(f"[DEBUG] VerifierAgent: Raw response: {response_content[:500]}")
            # Ultimate fallback to rule-based analysis
            return self._evaluate_with_rules(question, evidence)

    def _estimate_coverage(self, evidence: List[EvidenceItem]) -> float:
        """Simple rule-based coverage estimation for fallback."""
        high_confidence_items = [item for item in evidence if (item.confidence or 0) >= 0.7]
        if not evidence:
            return 0.0
        return min(1.0, len(high_confidence_items) / max(1, len(evidence)))
