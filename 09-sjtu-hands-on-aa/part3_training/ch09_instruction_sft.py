#!/usr/bin/env python3
"""
Chapter 9: Agent Instruction Tuning & Trajectory Loss Masking (指令微调与智能体轨迹SFT)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 3: Agent Post-Training

Implements:
1. Multi-Turn Agent Trajectory Tokenizer & Loss Mask Builder (`AgentTrajectoryCollator`):
   - Assigns `IGNORE_INDEX = -100` to `system`, `user`, and `tool_observation` tokens
   - Retains valid target IDs ONLY on `assistant` (Thought + Tool Call + Final Answer) tokens
2. Masked Causal Cross-Entropy Loss & Analytical Logit Gradient (`compute_masked_sft_loss`)
   preventing the agent from memorizing environment outputs or user prompts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrajectoryTurn:
    role: str  # "system" | "user" | "assistant" | "tool"
    token_ids: List[int]


class AgentTrajectoryCollator:
    """
    Converts a multi-turn agent trajectory (System -> User -> Assistant Thought/Action ->
    Tool Observation -> Assistant Final Answer) into `input_ids` and `labels` with
    `IGNORE_INDEX = -100` applied to all non-assistant spans.
    """

    @staticmethod
    def collate(turns: List[TrajectoryTurn]) -> Dict[str, Any]:
        input_ids: List[int] = []
        labels: List[int] = []
        role_spans: List[Tuple[str, int, int]] = []

        for turn in turns:
            start = len(input_ids)
            input_ids.extend(turn.token_ids)
            end = len(input_ids)
            role_spans.append((turn.role, start, end))

            if turn.role == "assistant":
                labels.extend(turn.token_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(turn.token_ids))

        # Causal shift: logits[t] predicts labels[t + 1]
        shifted_logits_positions = len(input_ids) - 1
        shifted_labels = labels[1:]
        active_token_count = sum(1 for y in shifted_labels if y != IGNORE_INDEX)

        return {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
            "shifted_labels": np.array(shifted_labels, dtype=np.int64),
            "active_token_count": active_token_count,
            "total_tokens": shifted_logits_positions,
            "role_spans": role_spans,
        }


def compute_masked_sft_loss(
    logits: np.ndarray, labels: np.ndarray, ignore_index: int = IGNORE_INDEX
) -> Tuple[float, float, np.ndarray]:
    """
    Computes causal language modeling cross-entropy loss and gradient w.r.t. `logits[:-1]`
    masked by `ignore_index`.

    Args:
        logits: shape (T, V)
        labels: shape (T,) unshifted token labels (with `ignore_index` on non-assistant tokens)

    Returns:
        (loss, perplexity, grad_shifted_logits)
    """
    shifted_logits = logits[:-1]  # (T-1, V)
    shifted_labels = labels[1:]   # (T-1,)

    # Stable log-softmax
    max_l = np.max(shifted_logits, axis=-1, keepdims=True)
    exp_l = np.exp(shifted_logits - max_l)
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    log_probs = (shifted_logits - max_l) - np.log(np.sum(exp_l, axis=-1, keepdims=True))

    valid_mask = shifted_labels != ignore_index
    num_valid = int(np.sum(valid_mask))
    if num_valid == 0:
        return 0.0, 1.0, np.zeros_like(shifted_logits)

    losses: List[float] = []
    grad = np.zeros_like(shifted_logits, dtype=np.float64)

    for t in range(len(shifted_labels)):
        if valid_mask[t]:
            target_id = int(shifted_labels[t])
            losses.append(-float(log_probs[t, target_id]))
            grad[t] = probs[t]
            grad[t, target_id] -= 1.0
            grad[t] /= num_valid

    mean_loss = sum(losses) / num_valid
    ppl = math.exp(min(20.0, mean_loss))
    return round(mean_loss, 6), round(ppl, 6), grad


def run_chapter_demo() -> Dict[str, Any]:
    trajectory = [
        TrajectoryTurn("system", [1, 2, 3]),
        TrajectoryTurn("user", [4, 5, 6, 7]),
        TrajectoryTurn("assistant", [8, 9, 10, 11]),  # Thought + ToolCall
        TrajectoryTurn("tool", [12, 13, 14, 15, 16]),  # Environment Observation (masked!)
        TrajectoryTurn("assistant", [17, 18, 19]),     # Final Answer
    ]
    batch = AgentTrajectoryCollator.collate(trajectory)

    rng = np.random.default_rng(7)
    T = len(batch["input_ids"])
    V = 32
    logits = rng.normal(0.0, 1.0, size=(T, V))
    loss, ppl, grad = compute_masked_sft_loss(logits, batch["labels"])

    # Verify gradient is strictly zero on masked tool/user positions
    masked_positions_grad_norm = float(
        np.linalg.norm(grad[batch["shifted_labels"] == IGNORE_INDEX])
    )
    active_positions_grad_norm = float(
        np.linalg.norm(grad[batch["shifted_labels"] != IGNORE_INDEX])
    )

    return {
        "total_shifted_tokens": batch["total_tokens"],
        "active_assistant_tokens": batch["active_token_count"],
        "sft_loss": loss,
        "perplexity": ppl,
        "masked_grad_norm": masked_positions_grad_norm,
        "active_grad_norm": round(active_positions_grad_norm, 6),
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 09 Demo] ActiveTokens={res['active_assistant_tokens']}/{res['total_shifted_tokens']}, Loss={res['sft_loss']}, MaskedGradNorm={res['masked_grad_norm']}")
