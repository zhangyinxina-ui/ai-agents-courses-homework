#!/usr/bin/env python3
"""
Comprehensive Test Suite for All 15 Chapters of SJTU《动手学大模型智能体》(Hands-on AI Agent)
Verifies 100% pass rate across:
  - Part 1: Foundations (Chapters 1–4)
  - Part 2: Architectures (Chapters 5–8)
  - Part 3: Training (Chapters 9–11)
  - Part 4: Frontiers (Chapters 12–15)
"""

from __future__ import annotations

import numpy as np
import pytest

from part1_foundations import (
    ch01_agent_intro,
    ch02_generative_lm,
    ch03_prompt_engineering,
    ch04_agent_evaluation,
)
from part2_architectures import (
    ch05_react_and_plan_solve,
    ch06_memory_and_rag,
    ch07_tool_calling,
    ch08_tree_search_tot_mcts,
)
from part3_training import (
    ch09_instruction_sft,
    ch10_lora_and_quantization,
    ch11_rl_dpo_grpo,
)
from part4_frontiers import (
    ch12_multimodal_gui_agent,
    ch13_multi_agent_systems,
    ch14_agent_safety,
    ch15_agent_protocols,
)


def test_ch01_agent_intro() -> None:
    res = ch01_agent_intro.run_chapter_demo()
    assert res["success"] is True
    assert res["steps"] == 6
    assert res["total_reward"] > 7.0
    assert res["trajectory"][-1]["action"] == "FINISH"


def test_ch02_generative_lm() -> None:
    res = ch02_generative_lm.run_chapter_demo()
    assert len(res["generated_sequence"]) == 8
    assert res["kv_cache_final_len"] == 7
    assert res["max_cached_vs_full_logit_diff"] < 1e-10
    assert res["top_k3_nonzero_count"] == 3
    assert abs(res["top_p07_prob_sum"] - 1.0) < 1e-9


def test_ch03_prompt_engineering() -> None:
    res = ch03_prompt_engineering.run_chapter_demo()
    assert res["prompt_length"] > 100
    assert res["self_consistency"]["winner"] == "26"
    assert res["self_consistency"]["confidence"] == 0.8
    assert res["parsed_json"] == {"action": "SET_TEMP", "TargetTemp": 22}
    assert res["parse_error"] is None


def test_ch04_agent_evaluation() -> None:
    res = ch04_agent_evaluation.run_chapter_demo()
    assert abs(res["pass_at_1"] - 0.3) < 1e-6
    assert res["pass_at_5"] > res["pass_at_1"]
    assert res["trajectory_metrics"]["recall"] == 1.0
    assert res["trajectory_metrics"]["milestone_progress"] == 1.0
    assert res["cost_summary"]["kv_cache_reuse_ratio"] > 0.60


def test_ch05_react_and_plan_solve() -> None:
    res = ch05_react_and_plan_solve.run_chapter_demo()
    assert res["answers_match"] is True
    assert res["react"]["answer"] == "260"
    assert res["plan_and_solve"]["answer"] == "260"
    assert res["plan_and_solve"]["llm_calls"] < res["react"]["llm_calls"]


def test_ch06_memory_and_rag() -> None:
    res = ch06_memory_and_rag.run_chapter_demo()
    assert res["working_memory"]["active_count"] == 3
    assert "Summarized" in res["working_memory"]["summary"]
    assert res["episodic_top_id"] == "D1"
    assert res["hybrid_rag_top_id"] == "D1"


def test_ch07_tool_calling() -> None:
    res = ch07_tool_calling.run_chapter_demo()
    assert res["attempt1_failed_as_expected"] is True
    assert "float" in (res["attempt1_recovery_hint"] or "")
    assert res["attempt2_success"] is True
    assert res["attempt2_output"]["utilization_pct"] == 80.0


def test_ch08_tree_search_tot_mcts() -> None:
    res = ch08_tree_search_tot_mcts.run_chapter_demo()
    assert res["tot"]["solved"] is True
    assert res["mcts"]["solved"] is True
    assert len(res["tot"]["history"]) == 3
    assert len(res["mcts"]["history"]) == 3


def test_ch09_instruction_sft() -> None:
    res = ch09_instruction_sft.run_chapter_demo()
    assert res["active_assistant_tokens"] == 7
    assert res["sft_loss"] > 0.0
    assert res["masked_grad_norm"] == 0.0
    assert res["active_grad_norm"] > 0.0


def test_ch10_lora_and_quantization() -> None:
    res = ch10_lora_and_quantization.run_chapter_demo()
    assert res["param_stats"]["compression_ratio"] == 8.0
    assert res["init_zero_delta_diff"] == 0.0
    assert res["merge_equivalence_max_diff"] < 1e-12
    assert res["int8_more_accurate_than_int4"] is True


def test_ch11_rl_dpo_grpo() -> None:
    res = ch11_rl_dpo_grpo.run_chapter_demo()
    assert res["dpo"]["reward_accuracy"] == 1.0
    assert res["dpo"]["reward_margin_mean"] > 0.0
    assert abs(res["grpo"]["advantage_mean"]) < 1e-6
    assert res["grpo"]["mean_kl"] >= 0.0


def test_ch12_multimodal_gui_agent() -> None:
    res = ch12_multimodal_gui_agent.run_chapter_demo()
    assert res["grounding_iou"] > 0.70
    assert res["origin_pixel_center"] == (672, 194)
    assert res["task_success"] is True


def test_ch13_multi_agent_systems() -> None:
    res = ch13_multi_agent_systems.run_chapter_demo()
    assert res["sop_pipeline"]["artifacts_generated"] == [
        "PRD",
        "ARCH_SPEC",
        "SOURCE_CODE",
        "QA_REPORT",
    ]
    assert res["sop_pipeline"]["qa_passed"] is True
    assert res["debate"]["unanimous"] is True
    assert res["debate"]["consensus_answer"] == "144"


def test_ch14_agent_safety() -> None:
    res = ch14_agent_safety.run_chapter_demo()
    assert res["injection_blocked"] is True
    assert res["pii_redaction_count"] == 3
    assert "[REDACTED_API_KEY]" in res["sanitized_log"]
    assert res["rbac_denial"] is True
    assert res["hitl_gate_triggered"] is True
    assert res["hitl_approved_allowed"] is True
    assert res["sql_injection_blocked"] is True


def test_ch15_agent_protocols() -> None:
    res = ch15_agent_protocols.run_chapter_demo()
    assert res["mcp_protocol_version"] == "2025-03-26"
    assert "estimate_ppa" in res["mcp_listed_tools"]
    assert "RV64I" in res["mcp_resource_text"]
    assert res["a2a_discovered_agent"] == "hw-eda-specialist"
    assert res["a2a_task_status"] == "COMPLETED"
    assert "Area=268.8um2" in res["a2a_artifact"]
