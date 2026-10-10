#!/usr/bin/env python3
"""
Chapter 4: Agent Evaluation & Benchmarking (智能体评测方法与基准)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 1: Foundations

Implements:
1. Unbiased `pass@k` estimator (Chen et al., Codex formulation)
2. Tool-Call Trajectory Precision, Recall, F1 & Exact Sequence Match Evaluator
3. Process-Level Step Progress Score (PRM-style milestone completion)
4. Token & Dollar Cost / Latency Budget Tracker
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes the unbiased estimator of pass@k:
        pass@k = 1 - C(n - c, k) / C(n, k)
    where n = total samples generated, c = number of correct samples, k = budget.
    """
    if n < k or k <= 0:
        raise ValueError(f"Invalid arguments: require n >= k > 0, got n={n}, k={k}")
    if n - c < k:
        return 1.0
    # Compute product_{i=0}^{k-1} (n - c - i) / (n - i) for numerical stability
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return round(1.0 - prod, 6)


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        sorted_items = sorted((k, repr(v)) for k, v in self.arguments.items())
        return f"{self.tool_name}({','.join(f'{k}={v}' for k, v in sorted_items)})"


class TrajectoryEvaluator:
    """Evaluates predicted agent tool-call trajectories against gold reference traces."""

    @staticmethod
    def evaluate_tool_calls(
        predicted: Sequence[ToolInvocation],
        gold: Sequence[ToolInvocation],
    ) -> Dict[str, float]:
        pred_sigs = [t.signature() for t in predicted]
        gold_sigs = [t.signature() for t in gold]

        pred_set = set(pred_sigs)
        gold_set = set(gold_sigs)

        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
        recall = tp / len(gold_set) if gold_set else 1.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        exact_order_match = 1.0 if pred_sigs == gold_sigs else 0.0

        # Subsequence milestone progress (order-preserving LCS length / len(gold))
        m, n = len(pred_sigs), len(gold_sigs)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_sigs[i - 1] == gold_sigs[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        progress_score = dp[m][n] / n if n > 0 else 1.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "exact_match": exact_order_match,
            "milestone_progress": round(progress_score, 4),
        }


class AgentCostTracker:
    """Tracks multi-turn token consumption, KV cache hit savings, and API dollar cost."""

    def __init__(
        self,
        input_cost_per_1m: float = 2.50,
        cached_input_cost_per_1m: float = 0.25,
        output_cost_per_1m: float = 10.00,
    ) -> None:
        self.input_cost_per_1m = input_cost_per_1m
        self.cached_input_cost_per_1m = cached_input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.turns: List[Dict[str, float]] = []

    def record_turn(
        self,
        uncached_prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        cost = (
            (uncached_prompt_tokens / 1e6) * self.input_cost_per_1m
            + (cached_prompt_tokens / 1e6) * self.cached_input_cost_per_1m
            + (completion_tokens / 1e6) * self.output_cost_per_1m
        )
        self.turns.append(
            {
                "uncached_prompt_tokens": uncached_prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost,
            }
        )

    def summary(self) -> Dict[str, float]:
        total_uncached = sum(t["uncached_prompt_tokens"] for t in self.turns)
        total_cached = sum(t["cached_prompt_tokens"] for t in self.turns)
        total_comp = sum(t["completion_tokens"] for t in self.turns)
        total_prompt = total_uncached + total_cached
        kv_reuse_ratio = total_cached / total_prompt if total_prompt > 0 else 0.0
        return {
            "num_turns": len(self.turns),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
            "kv_cache_reuse_ratio": round(kv_reuse_ratio, 4),
            "total_latency_ms": round(sum(t["latency_ms"] for t in self.turns), 2),
            "total_cost_usd": round(sum(t["cost_usd"] for t in self.turns), 6),
        }


def run_chapter_demo() -> Dict[str, Any]:
    pass_1 = compute_pass_at_k(n=10, c=3, k=1)
    pass_5 = compute_pass_at_k(n=10, c=3, k=5)

    gold_trace = [
        ToolInvocation("search_db", {"query": "user_42"}),
        ToolInvocation("check_balance", {"account_id": "ACC_42"}),
        ToolInvocation("send_receipt", {"account_id": "ACC_42", "status": "ok"}),
    ]
    pred_trace = [
        ToolInvocation("search_db", {"query": "user_42"}),
        ToolInvocation("log_audit", {"event": "lookup"}),
        ToolInvocation("check_balance", {"account_id": "ACC_42"}),
        ToolInvocation("send_receipt", {"account_id": "ACC_42", "status": "ok"}),
    ]
    eval_metrics = TrajectoryEvaluator.evaluate_tool_calls(pred_trace, gold_trace)

    tracker = AgentCostTracker()
    tracker.record_turn(uncached_prompt_tokens=1000, cached_prompt_tokens=0, completion_tokens=150, latency_ms=420.0)
    tracker.record_turn(uncached_prompt_tokens=200, cached_prompt_tokens=1150, completion_tokens=120, latency_ms=310.0)
    tracker.record_turn(uncached_prompt_tokens=180, cached_prompt_tokens=1470, completion_tokens=90, latency_ms=280.0)

    return {
        "pass_at_1": pass_1,
        "pass_at_5": pass_5,
        "trajectory_metrics": eval_metrics,
        "cost_summary": tracker.summary(),
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 04 Demo] pass@1={res['pass_at_1']}, pass@5={res['pass_at_5']}, F1={res['trajectory_metrics']['f1']}, MilestoneProgress={res['trajectory_metrics']['milestone_progress']}")
