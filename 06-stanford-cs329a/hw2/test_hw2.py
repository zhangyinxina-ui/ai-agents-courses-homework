"""
Unit tests for Stanford CS329A HW2: LATS, ADaPT, STaR & GRPO.
"""

import math
import pytest

from lats_and_grpo import (
    ADaPTPlanner,
    GameOf24Environment,
    LATSAgent,
    LATSNode,
    compute_grpo_advantages,
    compute_grpo_loss,
    run_hw2_benchmarks,
    schulman_k3_kl_divergence,
    star_filter_and_rationalize,
)


def test_lats_uct_and_backpropagation():
    root = LATSNode(state="[3, 3, 8, 8]")
    c1 = LATSNode(state="[3, 8, 24]", action="3 * 8 = 24", parent=root)
    c2 = LATSNode(state="[6, 8, 8]", action="3 + 3 = 6", parent=root)
    root.children = [c1, c2]

    # Unvisited children have +inf UCT score
    assert math.isinf(c1.uct_score())

    agent = LATSAgent(exploration_weight=1.414)
    agent.backpropagate(c1, 0.8)
    agent.backpropagate(c2, 0.2)

    assert root.visits == 2
    assert math.isclose(root.q_value, 0.5, rel_tol=1e-9)
    assert c1.q_value > c2.q_value
    assert agent.select(root) is c1


def test_lats_solves_game_of_24_instances():
    for nums in ([4, 7, 8, 8], [1, 5, 5, 5], [2, 3, 4, 6], [3, 3, 8, 8]):
        env = GameOf24Environment(nums)
        agent = LATSAgent(max_iterations=45, branching_factor=10)
        res = agent.search(env)
        assert res["solved"], f"Failed to solve Game of 24 for {nums}"
        assert len(res["actions"]) == 3
        assert res["final_state"] == "[24]"


def test_adapt_as_needed_decomposition():
    call_log = []

    def atomic_exec(task: str, state: dict):
        call_log.append(task)
        if task == "simple_task":
            return True, "done_simple"
        if task == "leaf_a":
            return True, "done_a"
        if task == "leaf_b_fail":
            return False, "fail_b"
        if task == "leaf_b_backup":
            return True, "done_b_backup"
        return False, "complex"

    def decomposer(task: str):
        if task == "composite_root":
            return "AND", ["leaf_a", "resilient_b"]
        if task == "resilient_b":
            return "OR", ["leaf_b_fail", "leaf_b_backup"]
        return "AND", []

    planner = ADaPTPlanner(atomic_exec, decomposer, max_depth=3)

    # Atomic task must succeed immediately without invoking decomposition
    simple_node = planner.run("simple_task", {})
    assert simple_node.succeeded and simple_node.atomic_succeeded
    assert simple_node.children == []

    # Composite task decomposes on failure and handles AND/OR branching
    root_node = planner.run("composite_root", {})
    assert root_node.succeeded
    assert not root_node.atomic_succeeded
    assert root_node.operator == "AND"
    assert len(root_node.children) == 2
    assert root_node.children[1].operator == "OR"
    assert "done_b_backup" in root_node.output


def test_star_filter_and_rationalize():
    dataset = [
        ("q1", "What is 7 * 8?", "56"),
        ("q2", "What is 12 * 12?", "144"),
        ("q3", "Unruly puzzle", "999"),
    ]

    def forward_gen(q: str):
        if "7 * 8" in q:
            return ("Step 1: 7 * 8 = 56.", "56")
        return ("Step 1: Wrong guess.", "0")

    def hint_rat(q: str, gt: str):
        if "12 * 12" in q:
            return (f"Step 1: 12 * 10 = 120, 12 * 2 = 24, 120 + 24 = {gt}.", gt)
        return ("", "wrong")

    res = star_filter_and_rationalize(dataset, forward_gen, hint_rat)
    assert res["num_direct"] == 1
    assert res["num_rationalized"] == 1
    assert res["num_rejected"] == 1
    assert len(res["sft_corpus"]) == 2
    assert math.isclose(res["retention_rate"], 2.0 / 3.0)


def test_grpo_advantages_kl_and_loss():
    rewards = [1.0, 0.0, 1.0, 0.0]
    advs = compute_grpo_advantages(rewards)
    # Mean of normalized advantages must be 0, std must be ~1
    assert math.isclose(sum(advs), 0.0, abs_tol=1e-7)
    assert math.isclose(advs[0], 1.0, rel_tol=1e-6)
    assert math.isclose(advs[1], -1.0, rel_tol=1e-6)

    # Identical rewards yield 0 advantage
    assert compute_grpo_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    # Schulman K3 KL divergence is 0 when pi_new == pi_ref, and strictly > 0 when different
    kls_zero = schulman_k3_kl_divergence([-1.2, -0.8], [-1.2, -0.8])
    assert all(math.isclose(k, 0.0, abs_tol=1e-12) for k in kls_zero)

    kls_pos = schulman_k3_kl_divergence([-0.5, -1.5], [-1.0, -1.0])
    assert all(k > 0.0 for k in kls_pos)

    # Clipped loss check when ratio exceeds 1 + clip_eps
    loss_out = compute_grpo_loss(
        log_probs_new=[-0.2, -2.0, -0.3, -1.8],
        log_probs_old=[-0.8, -2.0, -0.3, -1.8],  # first sample has rho = exp(0.6) ~ 1.822 > 1.2
        log_probs_ref=[-0.8, -2.0, -0.3, -1.8],
        rewards=[1.0, 0.0, 0.5, 0.0],
        clip_eps=0.2,
        beta_kl=0.04,
    )
    assert loss_out.clipped_fraction == 0.25
    assert loss_out.kl_penalty > 0.0

    # End-to-end HW2 benchmark check
    bench = run_hw2_benchmarks()
    assert bench["grpo_history"][-1]["optimal_prob"] > bench["grpo_history"][0]["optimal_prob"]
