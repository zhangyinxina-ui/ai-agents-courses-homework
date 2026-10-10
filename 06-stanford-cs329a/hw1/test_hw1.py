"""
Unit tests for Stanford CS329A HW1: Test-Time Compute Scaling, Verifiers & Self-Debugging.
"""

import math
import pytest

from test_time_scaling import (
    CODING_BENCHMARK,
    MATH_BENCHMARK,
    ReasoningCandidate,
    SelfDebuggingAgent,
    VerifierModel,
    best_of_n,
    estimate_pass_at_k,
    estimate_pass_at_k_batch,
    execute_python_with_tests,
    extract_boxed_or_last_number,
    majority_voting,
    normalize_answer,
    run_test_time_scaling_experiments,
)


def test_estimate_pass_at_k_exact_identities():
    # When c == 0, pass@k is always 0
    assert estimate_pass_at_k(10, 0, 1) == 0.0
    assert estimate_pass_at_k(10, 0, 5) == 0.0

    # When n - c < k, at least one correct sample is guaranteed
    assert estimate_pass_at_k(10, 8, 3) == 1.0
    assert estimate_pass_at_k(5, 5, 5) == 1.0

    # Analytical check: n=10, c=3, k=2 -> 1 - C(7, 2)/C(10, 2) = 1 - 21/45 = 24/45 = 8/15
    assert math.isclose(estimate_pass_at_k(10, 3, 2), 8.0 / 15.0, rel_tol=1e-9)

    # Monotonicity in k: pass@1 <= pass@2 <= pass@4 <= pass@8
    vals = [estimate_pass_at_k(16, 4, k) for k in (1, 2, 4, 8)]
    for a, b in zip(vals, vals[1:]):
        assert a < b

    # Batch estimator
    batch_score = estimate_pass_at_k_batch([10, 10], [0, 10], k=2)
    assert math.isclose(batch_score, 0.5, rel_tol=1e-9)

    with pytest.raises(ValueError):
        estimate_pass_at_k(5, 2, 0)
    with pytest.raises(ValueError):
        estimate_pass_at_k(5, 2, 6)


def test_answer_normalization_and_majority_voting():
    assert normalize_answer(r"\boxed{1,200.0}") == "1200"
    assert extract_boxed_or_last_number("Step 1: 5+5=10. Therefore \\boxed{10}") == "10"
    assert extract_boxed_or_last_number("We compute 3 * 7 = 21. Final answer: 21") == "21"

    cands = [
        ReasoningCandidate(steps=["1+1=2"], final_answer="204", logprob=-1.0),
        ReasoningCandidate(steps=["bad"], final_answer="194", logprob=-0.5),
        ReasoningCandidate(steps=["good"], final_answer="\\boxed{204}", logprob=-0.8),
    ]
    winner, counts = majority_voting(cands)
    assert winner == "204"
    assert counts["204"] == 2
    assert counts["194"] == 1


def test_prm_vs_majority_voting_on_coherent_distractors():
    verifier = VerifierModel()
    # Suppose 2 candidates agree on wrong answer "194" with arithmetic flaws,
    # and 1 candidate has the verified trace for "204"
    flawed_1 = ReasoningCandidate(
        steps=["Step 1: 18 * 6 = 98.", "Step 2: 24 * 4 = 96.", "Step 3: 98 + 96 = 194."],
        final_answer="194",
    )
    flawed_2 = ReasoningCandidate(
        steps=["Step 1: 18 * 6 = 108.", "Step 2: 24 * 4 = 86.", "Step 3: 108 + 86 = 194."],
        final_answer="194",
    )
    correct_1 = ReasoningCandidate(
        steps=[
            "Step 1: 18 * 6 = 108.",
            "Step 2: 24 * 4 = 96.",
            "Step 3: 108 + 96 = 204. Therefore 204.",
        ],
        final_answer="204",
    )
    pool = [flawed_1, flawed_2, correct_1]

    # Majority voting picks 194 (2 votes vs 1 vote)
    maj_ans, _ = majority_voting(pool)
    assert maj_ans == "194"

    # PRM (min aggregation) detects the arithmetic error in both flawed candidates (score 0.08)
    # and selects "204" (score >= 0.90) in both Best-of-N and Verifier-Weighted Voting!
    prm_bon_ans, scores = best_of_n(pool, verifier, mode="prm_min", weighted_voting=False)
    assert prm_bon_ans == "204"
    assert scores[2] > 0.85 and scores[0] < 0.15 and scores[1] < 0.15

    prm_w_ans, _ = best_of_n(pool, verifier, mode="prm_min", weighted_voting=True)
    assert prm_w_ans == "204"


def test_self_debugging_agent_repairs_all_benchmark_tasks():
    agent = SelfDebuggingAgent(max_turns=4)
    for task in CODING_BENCHMARK:
        # Initial buggy code must fail at least one unit test
        initial_res = execute_python_with_tests(task.buggy_initial_code, task.unit_tests)
        assert not initial_res.passed

        # Self-debugging loop must repair all 5 benchmark coding problems within 4 turns
        traj = agent.solve(task.task_id, task.prompt, task.buggy_initial_code, task.unit_tests)
        assert traj.solved, f"Task {task.task_id} failed to repair within 4 turns"
        assert 2 <= traj.turns_used <= 4


def test_scaling_experiment_pipeline():
    results = run_test_time_scaling_experiments(n_values=(1, 4, 16), num_trials=15, seed=123)
    m1 = results["scaling_metrics"][1]
    m16 = results["scaling_metrics"][16]

    # Scaling from N=1 to N=16 must substantially improve accuracy across all methods
    assert m16["pass_at_n"] > m1["pass_at_n"]
    assert m16["prm_weighted_voting"] >= m16["majority_voting"]
    assert m16["prm_best_of_n"] > m1["prm_best_of_n"]
    assert results["debug_turn_accuracy"][4] == 1.0
