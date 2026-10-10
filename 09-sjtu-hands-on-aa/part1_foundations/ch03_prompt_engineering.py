#!/usr/bin/env python3
"""
Chapter 3: Prompt Engineering & Reasoning Elicitation (提示工程与推理激发)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 1: Foundations

Implements:
1. System + Few-Shot + Chain-of-Thought (CoT) Prompt Builder (`PromptTemplateEngine`)
2. Self-Consistency Majority Voting Aggregator (`SelfConsistencyVoter`) with confidence entropy
3. Structured JSON Output Parser & Automatic Repair Loop (`StructuredJSONParser`)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FewShotExample:
    question: str
    reasoning: str
    answer: str


class PromptTemplateEngine:
    """Constructs Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompts."""

    def __init__(
        self,
        system_role: str = "You are a rigorous mathematical and logical reasoning assistant.",
        few_shot_examples: Optional[List[FewShotExample]] = None,
    ) -> None:
        self.system_role = system_role
        self.examples = few_shot_examples or []

    def build_prompt(self, query: str, use_cot: bool = True) -> str:
        blocks: List[str] = [f"[SYSTEM]\n{self.system_role}"]
        for idx, ex in enumerate(self.examples, start=1):
            if use_cot:
                blocks.append(
                    f"[EXAMPLE {idx}]\nQ: {ex.question}\nThought: {ex.reasoning}\nAnswer: {ex.answer}"
                )
            else:
                blocks.append(f"[EXAMPLE {idx}]\nQ: {ex.question}\nAnswer: {ex.answer}")

        cot_directive = "\nLet's think step by step before giving the final Answer:" if use_cot else ""
        blocks.append(f"[USER]\nQ: {query}{cot_directive}")
        return "\n\n".join(blocks)


class SelfConsistencyVoter:
    """
    Implements Wang et al. (2023) Self-Consistency over Chain-of-Thought trajectories:
    Samples N reasoning paths, extracts normalized final answers, and performs majority voting.
    """

    ANSWER_PATTERN = re.compile(r"Answer\s*:\s*([^\n]+)", re.IGNORECASE)

    @classmethod
    def extract_answer(cls, raw_completion: str) -> str:
        match = cls.ANSWER_PATTERN.search(raw_completion)
        if match:
            return match.group(1).strip().rstrip(".")
        lines = [ln.strip() for ln in raw_completion.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    @classmethod
    def aggregate(cls, completions: List[str]) -> Dict[str, Any]:
        extracted = [cls.extract_answer(c) for c in completions if cls.extract_answer(c)]
        if not extracted:
            return {"winner": "", "confidence": 0.0, "entropy": 0.0, "vote_counts": {}}

        counts = Counter(extracted)
        total = len(extracted)
        winner, top_count = counts.most_common(1)[0]
        confidence = top_count / total

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            "winner": winner,
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "vote_counts": dict(counts),
            "num_samples": total,
        }


class StructuredJSONParser:
    """
    Extracts and validates JSON payloads from LLM responses (supporting markdown code fences,
    trailing commas, and schema validation with automatic repair feedback).
    """

    CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def __init__(self, required_keys: Dict[str, type]) -> None:
        self.required_keys = required_keys

    def parse(self, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        candidate = text.strip()
        fence_match = self.CODE_BLOCK_RE.search(candidate)
        if fence_match:
            candidate = fence_match.group(1)
        else:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start : end + 1]

        # Repair common trailing commas before '}' or ']'
        candidate = re.sub(r",\s*([\}\]])", r"\1", candidate)

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, f"JSONDecodeError: {exc.msg} at pos {exc.pos}"

        if not isinstance(data, dict):
            return None, "Expected top-level JSON object (dict)."

        for key, expected_type in self.required_keys.items():
            if key not in data:
                return None, f"Missing required field '{key}'."
            if not isinstance(data[key], expected_type):
                return (
                    None,
                    f"Field '{key}' expected type {expected_type.__name__}, got {type(data[key]).__name__}.",
                )

        return data, None


def run_chapter_demo() -> Dict[str, Any]:
    engine = PromptTemplateEngine(
        few_shot_examples=[
            FewShotExample(
                question="A server has 4 GPUs each with 80GB HBM. If KV cache uses 40% of total HBM, how many GB remain?",
                reasoning="Total HBM = 4 * 80 = 320 GB. KV cache uses 0.4 * 320 = 128 GB. Remaining = 320 - 128 = 192 GB.",
                answer="192",
            )
        ]
    )
    prompt = engine.build_prompt(
        "A cluster has 8 nodes with 4 GPUs each. If 6 GPUs are offline, how many GPUs are active?",
        use_cot=True,
    )

    sampled_cot_paths = [
        "Thought: Total GPUs = 8 * 4 = 32. Subtract 6 offline GPUs: 32 - 6 = 26.\nAnswer: 26",
        "Thought: 8 nodes * 4 GPUs/node = 32 GPUs. 32 - 6 = 26 active GPUs.\nAnswer: 26",
        "Thought: 8 * 4 = 32, minus 6 is 24.\nAnswer: 24",
        "Thought: Active GPUs = 32 - 6 = 26.\nAnswer: 26",
        "Thought: 8*4=32, 32-6=26.\nAnswer: 26",
    ]
    sc_result = SelfConsistencyVoter.aggregate(sampled_cot_paths)

    parser = StructuredJSONParser({"action": str, "TargetTemp": int})
    raw_llm_output = 'Here is the action:\n```json\n{"action": "SET_TEMP", "TargetTemp": 22,}\n```'
    parsed_json, err = parser.parse(raw_llm_output)

    return {
        "prompt_length": len(prompt),
        "self_consistency": sc_result,
        "parsed_json": parsed_json,
        "parse_error": err,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 03 Demo] Self-Consistency Winner={res['self_consistency']['winner']} (conf={res['self_consistency']['confidence']}), Parsed JSON={res['parsed_json']}")
