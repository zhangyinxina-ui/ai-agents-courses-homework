"""CMU 11-768 Assignment 3 Preview & Reference Implementation:
Agent Trajectory Filtering (Rejection Sampling SFT) and Multi-Turn GRPO Training.

Connects Lecture 13 (Agentic Evolution & Continual Learning) with the Harbor
verifier environments built in Assignment 2:
  1. Collect multi-turn trajectories from an environment verifier (reward in [0, 1]).
  2. Filter trajectories via Rejection Sampling / Expert Iteration for SFT, applying
     strict Assistant-only action token masking (masking out system/user/tool observation tokens).
  3. Compute Group Relative Policy Optimization (GRPO) advantages and clipped policy
     loss over multi-turn agent trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


IGNORE_INDEX = -100


@dataclass
class Turn:
    """A single message turn in a multi-turn agent trajectory."""

    role: str  # "system", "user", "assistant", or "tool"
    content: str
    token_ids: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)
    ref_logprobs: list[float] = field(default_factory=list)


@dataclass
class AgentTrajectory:
    """A complete rollout of an agent on a task verified by a Harbor-style environment."""

    task_id: str
    run_id: str
    turns: list[Turn]
    verifier_reward: float  # 1.0 if all verifier tests pass, 0.0 otherwise
    num_tool_errors: int = 0
    elapsed_seconds: float = 0.0

    @property
    def num_assistant_steps(self) -> int:
        return sum(1 for t in self.turns if t.role == "assistant")

    def shaped_reward(self, step_penalty: float = 0.01, error_penalty: float = 0.05) -> float:
        """Compute shaped scalar return: verifier reward minus efficiency penalties on passing runs."""
        if self.verifier_reward <= 0.0:
            return 0.0
        penalty = step_penalty * max(0, self.num_assistant_steps - 1) + error_penalty * self.num_tool_errors
        return max(0.1, self.verifier_reward - penalty)


def build_action_masked_sft_example(trajectory: AgentTrajectory) -> dict[str, list[int]]:
    """Build (input_ids, labels, action_mask) where ONLY assistant turns receive loss gradients.

    System prompts, user instructions, and environment tool observations are masked with
    IGNORE_INDEX (-100) so the policy only learns to generate agent reasoning and tool calls,
    never hallucinating environment observations.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    action_mask: list[int] = []

    for turn in trajectory.turns:
        ids = list(turn.token_ids)
        input_ids.extend(ids)
        if turn.role == "assistant":
            labels.extend(ids)
            action_mask.extend([1] * len(ids))
        else:
            labels.extend([IGNORE_INDEX] * len(ids))
            action_mask.extend([0] * len(ids))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "action_mask": action_mask,
    }


def filter_trajectories_for_sft(
    trajectories: list[AgentTrajectory],
    *,
    min_reward: float = 1.0,
    max_assistant_steps: int = 25,
    max_tool_errors: int = 2,
    top_k_per_task: int = 2,
) -> list[AgentTrajectory]:
    """Rejection Sampling / Expert Iteration filter for multi-turn agent trajectories.

    Keeps only trajectories that:
      1. Pass the environment verifier (`verifier_reward >= min_reward`),
      2. Stay within step and tool-error budgets (`num_assistant_steps <= max_assistant_steps`,
         `num_tool_errors <= max_tool_errors`),
      3. Are deduplicated by assistant action sequence, keeping up to `top_k_per_task`
         highest-shaped-reward trajectories per task.
    """
    by_task: dict[str, list[AgentTrajectory]] = {}
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()

    for traj in trajectories:
        if traj.verifier_reward < min_reward:
            continue
        if traj.num_assistant_steps == 0 or traj.num_assistant_steps > max_assistant_steps:
            continue
        if traj.num_tool_errors > max_tool_errors:
            continue
        assistant_sig = tuple(t.content.strip() for t in traj.turns if t.role == "assistant")
        dedup_key = (traj.task_id, assistant_sig)
        if dedup_key in seen_signatures:
            continue
        seen_signatures.add(dedup_key)
        by_task.setdefault(traj.task_id, []).append(traj)

    selected: list[AgentTrajectory] = []
    for task_id in sorted(by_task):
        ranked = sorted(
            by_task[task_id],
            key=lambda t: (-t.shaped_reward(), t.num_assistant_steps, t.run_id),
        )
        selected.extend(ranked[:top_k_per_task])
    return selected


def compute_grpo_advantages(
    group_trajectories: list[AgentTrajectory],
    *,
    eps: float = 1e-6,
) -> list[float]:
    """Compute group-normalized advantages across G rollouts for the same task."""
    if not group_trajectories:
        return []
    returns = [t.shaped_reward() for t in group_trajectories]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(var_r)
    if std_r < eps:
        return [0.0 for _ in returns]
    return [(r - mean_r) / (std_r + eps) for r in returns]


def compute_grpo_trajectory_loss(
    trajectory: AgentTrajectory,
    new_assistant_logprobs: list[list[float]],
    advantage: float,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> dict[str, float]:
    """Compute token-level masked GRPO policy loss and Schulman KL penalty for one trajectory.

    Args:
        trajectory: Rollout containing assistant turns with `old_logprobs` and `ref_logprobs`.
        new_assistant_logprobs: Current policy log-probabilities for each assistant turn's tokens.
        advantage: Scalar group-normalized advantage A_i for this trajectory.
        clip_eps: PPO/GRPO probability ratio clipping epsilon.
        kl_beta: Coefficient on the unbiased per-token KL divergence estimator.
    """
    assistant_turns = [t for t in trajectory.turns if t.role == "assistant"]
    if len(assistant_turns) != len(new_assistant_logprobs):
        raise ValueError("Mismatch between assistant turns and provided new_assistant_logprobs")

    total_surrogate = 0.0
    total_kl = 0.0
    total_tokens = 0

    for turn, cur_lps in zip(assistant_turns, new_assistant_logprobs):
        if len(cur_lps) != len(turn.old_logprobs) or len(cur_lps) != len(turn.ref_logprobs):
            raise ValueError("Token length mismatch in assistant turn logprobs")
        for lp_new, lp_old, lp_ref in zip(cur_lps, turn.old_logprobs, turn.ref_logprobs):
            ratio = math.exp(lp_new - lp_old)
            clipped_ratio = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
            surr1 = ratio * advantage
            surr2 = clipped_ratio * advantage
            total_surrogate += min(surr1, surr2)

            # Unbiased positive KL estimator: exp(lp_ref - lp_new) - (lp_ref - lp_new) - 1
            diff = lp_ref - lp_new
            kl_token = math.exp(diff) - diff - 1.0
            total_kl += kl_token
            total_tokens += 1

    if total_tokens == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "num_action_tokens": 0.0}

    mean_surrogate = total_surrogate / total_tokens
    mean_kl = total_kl / total_tokens
    loss = -mean_surrogate + kl_beta * mean_kl
    return {
        "loss": loss,
        "policy_loss": -mean_surrogate,
        "kl": mean_kl,
        "num_action_tokens": float(total_tokens),
    }


def _self_test() -> None:
    # Construct a group of 3 rollouts on task_01
    t_fast_pass = AgentTrajectory(
        task_id="task_01",
        run_id="task_01__rollout_1",
        verifier_reward=1.0,
        num_tool_errors=0,
        turns=[
            Turn(role="system", content="You are an agent.", token_ids=[1, 2]),
            Turn(role="user", content="Plot cluster_jobs.csv", token_ids=[3, 4, 5]),
            Turn(
                role="assistant",
                content="python3 plot.py",
                token_ids=[10, 11, 12],
                old_logprobs=[-0.5, -0.4, -0.3],
                ref_logprobs=[-0.6, -0.4, -0.35],
            ),
            Turn(role="tool", content="returncode=0", token_ids=[20, 21]),
        ],
    )
    t_slow_pass = AgentTrajectory(
        task_id="task_01",
        run_id="task_01__rollout_2",
        verifier_reward=1.0,
        num_tool_errors=1,
        turns=[
            Turn(role="system", content="You are an agent.", token_ids=[1, 2]),
            Turn(role="user", content="Plot cluster_jobs.csv", token_ids=[3, 4, 5]),
            Turn(
                role="assistant",
                content="python3 bad.py",
                token_ids=[13, 14],
                old_logprobs=[-0.8, -0.7],
                ref_logprobs=[-0.8, -0.7],
            ),
            Turn(role="tool", content="returncode=1", token_ids=[22, 23]),
            Turn(
                role="assistant",
                content="python3 plot.py",
                token_ids=[10, 11, 12],
                old_logprobs=[-0.5, -0.4, -0.3],
                ref_logprobs=[-0.6, -0.4, -0.35],
            ),
        ],
    )
    t_fail = AgentTrajectory(
        task_id="task_01",
        run_id="task_01__rollout_3",
        verifier_reward=0.0,
        num_tool_errors=2,
        turns=[
            Turn(role="system", content="You are an agent.", token_ids=[1, 2]),
            Turn(role="user", content="Plot cluster_jobs.csv", token_ids=[3, 4, 5]),
            Turn(
                role="assistant",
                content="exit 1",
                token_ids=[15, 16],
                old_logprobs=[-1.0, -1.1],
                ref_logprobs=[-0.9, -1.0],
            ),
        ],
    )

    group = [t_fast_pass, t_slow_pass, t_fail]
    sft_kept = filter_trajectories_for_sft(group, top_k_per_task=1)
    assert len(sft_kept) == 1 and sft_kept[0].run_id == "task_01__rollout_1"

    masked = build_action_masked_sft_example(t_fast_pass)
    assert masked["action_mask"] == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert masked["labels"] == [-100, -100, -100, -100, -100, 10, 11, 12, -100, -100]

    advs = compute_grpo_advantages(group)
    assert len(advs) == 3 and advs[0] > advs[1] > advs[2]
    assert abs(sum(advs)) < 1e-5

    metrics = compute_grpo_trajectory_loss(
        t_fast_pass,
        new_assistant_logprobs=[[-0.45, -0.38, -0.29]],
        advantage=advs[0],
    )
    assert metrics["num_action_tokens"] == 3.0
    assert metrics["kl"] >= 0.0
    print("Assignment 3 reference self-test passed:", metrics)


if __name__ == "__main__":
    _self_test()
