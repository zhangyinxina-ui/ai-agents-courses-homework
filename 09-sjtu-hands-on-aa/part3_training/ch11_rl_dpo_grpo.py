#!/usr/bin/env python3
"""
Chapter 11: Agent Reinforcement Learning — DPO & GRPO (智能体强化学习：DPO与GRPO)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 3: Agent Post-Training

Implements:
1. Direct Preference Optimization (`compute_dpo_loss`, Rafailov et al., 2023):
     L_DPO = -E[ log sigma( beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x))) ) ]
   along with implicit reward margins `r_w - r_l`.
2. Group Relative Policy Optimization (`compute_grpo_objective`, Shao et al., DeepSeekMath / DeepSeek-R1):
   - Critic-free group-normalized advantage across G sampled trajectories for prompt q:
       A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
   - Clipped surrogate objective + Schulman unbiased non-negative KL penalty:
       D_KL(pi_theta || pi_ref) = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 / (1 + exp(-x))) = -softplus(-x)."""
    return -np.logaddexp(0.0, -x)


def compute_dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float = 0.1,
) -> Dict[str, Any]:
    """
    Computes DPO loss, implicit chosen/rejected rewards, preference accuracy, and reward margin.
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)
    losses = -log_sigmoid(logits)

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    reward_margins = chosen_rewards - rejected_rewards
    reward_accuracy = float(np.mean(chosen_rewards > rejected_rewards))

    return {
        "loss": round(float(np.mean(losses)), 6),
        "reward_margin_mean": round(float(np.mean(reward_margins)), 6),
        "reward_accuracy": round(reward_accuracy, 4),
        "chosen_rewards": chosen_rewards.tolist(),
        "rejected_rewards": rejected_rewards.tolist(),
    }


def compute_grpo_objective(
    group_rewards: np.ndarray,
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Computes the Group Relative Policy Optimization (GRPO) loss for a group of G
    agent rollouts sampled from the same prompt.

    Args:
        group_rewards: shape (G,) scalar verifiable rewards for each trajectory in the group
        new_log_probs: shape (G, T) current policy token log-probabilities
        old_log_probs: shape (G, T) rollout policy token log-probabilities
        ref_log_probs: shape (G, T) reference model token log-probabilities
    """
    # 1. Group-relative advantage normalization (eliminates separate Value Critic network)
    r_mean = float(np.mean(group_rewards))
    r_std = float(np.std(group_rewards))
    advantages = (group_rewards - r_mean) / (r_std + eps)  # (G,)
    adv_expanded = advantages[:, None]  # (G, 1) broadcast across token steps T

    # 2. Importance sampling ratio pi_theta / pi_old
    ratios = np.exp(new_log_probs - old_log_probs)  # (G, T)
    clipped_ratios = np.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps)

    surr1 = ratios * adv_expanded
    surr2 = clipped_ratios * adv_expanded
    policy_surrogate = np.minimum(surr1, surr2)  # (G, T)

    # 3. Unbiased non-negative per-token KL estimator: exp(log_ref - log_new) - (log_ref - log_new) - 1
    log_ref_over_new = ref_log_probs - new_log_probs
    per_token_kl = np.exp(log_ref_over_new) - log_ref_over_new - 1.0

    # 4. GRPO objective maximized: E[ policy_surrogate - kl_beta * per_token_kl ]
    #    Loss to minimize is the negative objective:
    grpo_loss = -float(np.mean(policy_surrogate - kl_beta * per_token_kl))

    return {
        "grpo_loss": round(grpo_loss, 6),
        "advantages": [round(float(a), 6) for a in advantages],
        "advantage_mean": round(float(np.mean(advantages)), 6),
        "mean_kl": round(float(np.mean(per_token_kl)), 6),
        "mean_ratio": round(float(np.mean(ratios)), 6),
    }


def run_chapter_demo() -> Dict[str, Any]:
    # 1. DPO demo on 4 preference pairs where policy prefers chosen trajectories
    pi_w = np.array([-4.2, -3.8, -5.1, -2.9])
    pi_l = np.array([-6.5, -5.9, -7.2, -5.0])
    ref_w = np.array([-5.0, -4.5, -5.8, -3.5])
    ref_l = np.array([-5.2, -4.8, -6.0, -3.9])
    dpo_res = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.2)

    # 2. GRPO demo on G=4 agent rollouts of length T=6
    rewards = np.array([1.0, 0.2, -0.5, 1.5])
    old_lp = np.full((4, 6), -1.2)
    # Policy slightly increases probability on positive-reward rollouts (idx 0 and 3)
    new_lp = old_lp + np.array([[0.08], [-0.02], [-0.10], [0.12]])
    ref_lp = np.full((4, 6), -1.25)
    grpo_res = compute_grpo_objective(rewards, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_beta=0.04)

    return {
        "dpo": dpo_res,
        "grpo": grpo_res,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 11 Demo] DPO Loss={res['dpo']['loss']} (Margin={res['dpo']['reward_margin_mean']}, Acc={res['dpo']['reward_accuracy']}) | GRPO Loss={res['grpo']['grpo_loss']}, MeanKL={res['grpo']['mean_kl']}")
