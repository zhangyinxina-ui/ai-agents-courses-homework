"""
Unit tests for UC Berkeley CS294-196 Labs:
  - Lean 4 Autoformalization & Iterative Proof Repair (`lean4_formal_verification_lab.py`)
  - Foundation Labs 1, 2, and 3 (`foundation_labs_1_to_3.py`)
"""

import pytest

from foundation_labs_1_to_3 import (
    MultiAgentReflexionSystem,
    ReActRestaurantAgent,
    TieredMemoryAgent,
)
from lean4_formal_verification_lab import (
    BENCHMARK_SPECS,
    Autoformalizer,
    Lean4ProofRepairAgent,
    Lean4Verifier,
    check_no_forbidden_cheat_tokens,
    check_theorem_header_preserved,
    parse_lean4_diagnostics,
    run_lean4_verification_benchmark,
)


def test_lean4_diagnostic_parser_and_anti_cheat_guardrails():
    sample_err = (
        "<input>:4:3: error: tactic 'rfl' failed\n"
        "n : Nat\n"
        "⊢ double n = n + n\n"
    )
    diags = parse_lean4_diagnostics(sample_err)
    assert len(diags) == 1
    assert diags[0].line == 4
    assert diags[0].column == 3
    assert diags[0].severity == "error"
    assert diags[0].unsolved_goal == "double n = n + n"

    # Anti-cheating check must catch `sorry` and `admit`, but ignore `-- sorry` in comments
    ok_comment, _ = check_no_forbidden_cheat_tokens("theorem t : 1 = 1 := by\n  -- no sorry here\n  rfl")
    assert ok_comment is True

    ok_cheat, tok = check_no_forbidden_cheat_tokens("theorem t : 1 = 1 := by sorry")
    assert ok_cheat is False
    assert tok == "sorry"

    # Header preservation check
    header = "theorem add_comm_reassoc (a b c : Nat) : (a + b) + c = (a + c) + b := by"
    assert check_theorem_header_preserved(header, header + "\n  omega") is True
    assert check_theorem_header_preserved(header, "theorem add_comm_reassoc : True := by trivial") is False


def test_lean4_autoformalizer_and_iterative_proof_repair():
    verifier = Lean4Verifier(prefer_subprocess=False)
    agent = Lean4ProofRepairAgent(verifier=verifier, max_turns=4)

    # All 4 benchmark specs start with flawed/incomplete proofs that must fail on turn 1
    # and get repaired to 100% verified proofs within 4 turns
    trajectories = run_lean4_verification_benchmark()
    assert len(trajectories) == len(BENCHMARK_SPECS)

    for traj in trajectories:
        assert traj.attempts[0].verification.verified is False
        assert traj.solved is True, f"Failed to repair Lean 4 proof for {traj.task_id}"
        assert 2 <= traj.turns_used <= 4
        assert traj.attempts[-1].verification.verified is True
        assert not traj.attempts[-1].verification.has_forbidden_tokens
        assert traj.attempts[-1].verification.spec_preserved


def test_lab1_react_restaurant_agent():
    agent = ReActRestaurantAgent()
    chez = agent.evaluate_restaurant("How good is Chez Panisse?")
    mediocre = agent.evaluate_restaurant("Evaluate Mediocre Diner")

    assert len(chez.trajectory) == 3
    assert chez.trajectory[0].action_name == "fetch_reviews"
    assert chez.trajectory[1].action_name == "score_review_sentiments"
    assert chez.trajectory[2].action_name == "compute_bayesian_rating"
    assert chez.bayesian_weighted_score >= 4.0
    assert mediocre.bayesian_weighted_score < 2.5
    assert chez.bayesian_weighted_score > mediocre.bayesian_weighted_score


def test_lab2_tiered_memory_memgpt_agent():
    agent = TieredMemoryAgent(max_fifo_turns=3)
    agent.archival_insert(
        "doc_omega",
        "The omega tactic in Lean 4 decides Presburger linear integer and natural number arithmetic.",
    )

    agent.chat("Hello, I am allergic to shellfish and I prefer Lean4 for verification.")
    agent.chat("My project budget is $500.")
    agent.chat("We are testing memory eviction.")
    res = agent.chat("Which tactic in Lean 4 decides Presburger linear arithmetic?")

    # Core memory must persist extracted user constraints even after early turns are evicted from FIFO
    assert res["core_memory"].get("dietary_restriction") == "allergic to shellfish"
    assert res["core_memory"].get("preferred_stack").lower() == "lean4"
    assert res["core_memory"].get("max_budget_usd") == "500"
    assert res["fifo_size"] <= 3
    assert res["page_fault_count"] >= 4
    assert any("omega" in p.text for p in res["retrieved_passages"])


def test_lab3_multi_agent_reflexion_actor_critic():
    system = MultiAgentReflexionSystem(max_rounds=4)
    rounds = system.run_collaboration("Safely migrate users table")

    # Round 1 initial draft has 4 defects and must be rejected by the Critic
    assert rounds[0].critic_review.approved is False
    assert len(rounds[0].critic_review.defects) == 4

    # Round 2 incorporates the Critic's episodic reflection and passes all 4 invariants
    assert rounds[-1].critic_review.approved is True
    assert rounds[-1].critic_review.score == 1.0
    assert rounds[-1].actor_draft["canary_percent"] <= 10
    assert "BEGIN;" in rounds[-1].actor_draft["sql_migration"]
    assert "-- ROLLBACK:" in rounds[-1].actor_draft["sql_migration"]
