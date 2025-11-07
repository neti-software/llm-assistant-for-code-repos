"""Verifier agent responsible for drafting final text and coverage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any

from ..state_models import ConversationState, EvidenceItem


@dataclass
class VerifierReport:
    response_text: str
    coverage_score: float
    missing_items: List[str]
    citations: List[str]


class VerifierAgent:
    def __init__(self, coverage_threshold: float = 0.7, llm: Optional[Any] = None) -> None:
        self.coverage_threshold = coverage_threshold
        self.llm = llm

    def evaluate(self, state: ConversationState, *, question: str) -> VerifierReport:
        if not self.llm:
            raise RuntimeError("VerifierAgent requires an LLM configuration. None was provided.")

        evidence = state.evidence_store
        if not evidence:
            return VerifierReport(
                response_text=f"Insufficient evidence to answer: {question}",
                coverage_score=0.0,
                missing_items=[f"Collect supporting context for: {question}"],
                citations=[],
            )

        print(f"[DEBUG] VerifierAgent: Using LLM-based analysis")
        return self._evaluate_with_llm(question, evidence, state)

    def _evaluate_with_llm(self, question: str, evidence: List[EvidenceItem], state: ConversationState) -> VerifierReport:
        """Use LLM to provide intelligent coverage analysis and response synthesis."""
        assert self.llm is not None, "LLM must be configured"
        
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

        want_tool, raw_response = self.llm.generate(prompt, response_format=response_format)

        if want_tool:
            raise RuntimeError("VerifierAgent LLM unexpectedly requested a tool call during verification.")

        if isinstance(raw_response, dict):
            if "content" in raw_response and raw_response["content"] is not None:
                response_text = raw_response["content"]
            elif "response" in raw_response and raw_response["response"] is not None:
                response_text = raw_response["response"]
            else:
                # Some clients may nest the message payload
                message = raw_response.get("message")
                if isinstance(message, dict) and "content" in message:
                    response_text = message["content"]
                else:
                    response_text = str(raw_response)
        else:
            response_text = str(raw_response)

        # Parse the LLM response (expecting JSON structure)
        return self._parse_llm_analysis_response(response_text, question, evidence)



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
        import json
        import re

        # Try to parse the entire response as JSON first
        try:
            analysis_data = json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON found in LLM response: {response_content[:500]}")

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

