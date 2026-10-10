#!/usr/bin/env python3
"""
Chapter 13: Multi-Agent Collaborative Systems (多智能体协作系统)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 4: Frontiers

Implements:
1. Standardized Operating Procedure (SOP) Role-Playing Pipeline (MetaGPT / CAMEL style):
   - ProductManagerAgent -> ArchitectAgent -> EngineerAgent -> QAReviewerAgent
   communicating via a typed publish-subscribe `MessageBus`
2. Multi-Agent Debate & Judge Consensus Protocol (Du et al., 2023) where multiple agents
   critique and refine answers across rounds to eliminate single-model hallucinations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class SOPArtifactMessage:
    sender_role: str
    artifact_type: str  # "PRD" | "ARCH_SPEC" | "SOURCE_CODE" | "QA_REPORT"
    topic: str
    payload: Dict[str, Any]


class SOPSoftwareCompanyPipeline:
    """
    MetaGPT-style assembly line where each specialized role subscribes to upstream
    structured artifacts and publishes verified downstream deliverables.
    """

    def __init__(self) -> None:
        self.message_bus: List[SOPArtifactMessage] = []

    def run_sop(self, user_requirement: str) -> Dict[str, Any]:
        self.message_bus.clear()

        # 1. Product Manager produces structured PRD
        prd = SOPArtifactMessage(
            sender_role="ProductManager",
            artifact_type="PRD",
            topic=user_requirement,
            payload={
                "user_stories": [
                    "Accept task requests via REST/JSON",
                    "Enforce rate limiting per user token",
                    "Return structured status codes",
                ],
                "acceptance_criteria": ["100% unit test pass", "Latency < 50ms"],
            },
        )
        self.message_bus.append(prd)

        # 2. Architect consumes PRD and produces System Architecture Spec
        arch = SOPArtifactMessage(
            sender_role="SystemArchitect",
            artifact_type="ARCH_SPEC",
            topic=user_requirement,
            payload={
                "modules": ["TokenBucketRateLimiter", "TaskRouter"],
                "interface_methods": ["allow_request(user_id)", "dispatch(task)"],
                "based_on_stories": len(prd.payload["user_stories"]),
            },
        )
        self.message_bus.append(arch)

        # 3. Engineer consumes ARCH_SPEC and implements code
        code = SOPArtifactMessage(
            sender_role="SoftwareEngineer",
            artifact_type="SOURCE_CODE",
            topic=user_requirement,
            payload={
                "code": (
                    "class TokenBucketRateLimiter:\n"
                    "    def __init__(self, cap: int = 10):\n"
                    "        self.cap = cap\n"
                    "        self.tokens = {}\n"
                    "    def allow_request(self, user_id: str) -> bool:\n"
                    "        rem = self.tokens.get(user_id, self.cap)\n"
                    "        if rem > 0:\n"
                    "            self.tokens[user_id] = rem - 1\n"
                    "            return True\n"
                    "        return False\n"
                ),
                "implemented_modules": arch.payload["modules"],
            },
        )
        self.message_bus.append(code)

        # 4. QA Engineer executes code verification in sandbox
        ns: Dict[str, Any] = {}
        exec(code.payload["code"], {"__builtins__": __builtins__}, ns)
        limiter_cls = ns["TokenBucketRateLimiter"]
        limiter = limiter_cls(cap=2)
        test_results = [
            limiter.allow_request("u1") is True,
            limiter.allow_request("u1") is True,
            limiter.allow_request("u1") is False,  # 3rd call must be rate-limited
        ]
        qa = SOPArtifactMessage(
            sender_role="QAReviewer",
            artifact_type="QA_REPORT",
            topic=user_requirement,
            payload={
                "all_tests_passed": all(test_results),
                "executed_assertions": len(test_results),
            },
        )
        self.message_bus.append(qa)

        return {
            "artifacts_generated": [m.artifact_type for m in self.message_bus],
            "qa_passed": qa.payload["all_tests_passed"],
            "message_count": len(self.message_bus),
        }


class MultiAgentDebateArena:
    """
    Runs a multi-round society-of-minds debate among N agents.
    In Round 1, one agent makes an arithmetic slip; in Round 2, after reading peer
    critiques with step-by-step verification, the mistaken agent corrects its answer.
    """

    @staticmethod
    def run_debate(question: str) -> Dict[str, Any]:
        round_1 = {
            "Agent_Alpha": {"answer": "144", "rationale": "12 * 12 = 144"},
            "Agent_Beta": {"answer": "144", "rationale": "(10 + 2) * 12 = 120 + 24 = 144"},
            "Agent_Gamma": {"answer": "124", "rationale": "12 * 12 = 124 (initial slip)"},
        }
        r1_votes = Counter(v["answer"] for v in round_1.values())

        # Round 2: Agent_Gamma inspects peer rationales from Alpha & Beta and self-corrects
        round_2 = {
            "Agent_Alpha": {"answer": "144", "rationale": "Confirmed 12 * 12 = 144."},
            "Agent_Beta": {"answer": "144", "rationale": "Confirmed via distributive law 120 + 24 = 144."},
            "Agent_Gamma": {
                "answer": "144",
                "rationale": "After reviewing Agent_Beta's decomposition (120 + 24), I corrected 124 -> 144.",
            },
        }
        r2_votes = Counter(v["answer"] for v in round_2.values())
        winner, top_votes = r2_votes.most_common(1)[0]

        return {
            "question": question,
            "round_1_agreement": r1_votes.most_common(1)[0][1] / len(round_1),
            "round_2_agreement": top_votes / len(round_2),
            "consensus_answer": winner,
            "unanimous": top_votes == len(round_2),
        }


def run_chapter_demo() -> Dict[str, Any]:
    sop_res = SOPSoftwareCompanyPipeline().run_sop("Build a Token-Bucket Rate Limiter")
    debate_res = MultiAgentDebateArena().run_debate("What is 12 * 12?")
    return {
        "sop_pipeline": sop_res,
        "debate": debate_res,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 13 Demo] SOP Artifacts={res['sop_pipeline']['artifacts_generated']} (QA={res['sop_pipeline']['qa_passed']}) | Debate Unanimous={res['debate']['unanimous']} (Ans={res['debate']['consensus_answer']})")
