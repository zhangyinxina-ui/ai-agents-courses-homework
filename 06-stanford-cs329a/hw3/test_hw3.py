"""
Unit tests for Stanford CS329A HW3: Search-o1 & KernelOptimizationAgent.
"""

import pytest

from search_and_kernel_agent import (
    BEGIN_QUERY_TOKEN,
    BEGIN_RESULT_TOKEN,
    END_QUERY_TOKEN,
    END_RESULT_TOKEN,
    KernelOptimizationAgent,
    SearchO1Agent,
    build_kernel_benchmark_suite,
)


def test_search_o1_token_parsing_and_reason_in_documents():
    agent = SearchO1Agent()
    trace = f"Step 1: Let us look up Chinchilla -> {BEGIN_QUERY_TOKEN}Chinchilla D = 20 N{END_QUERY_TOKEN}"
    assert agent.extract_search_query(trace) == "Chinchilla D = 20 N"

    resp = agent.answer(
        "According to the Chinchilla compute-optimal scaling law, how many training tokens should be used for a 35B parameter model?"
    )
    assert resp.final_answer == "700B"
    assert BEGIN_QUERY_TOKEN in resp.reasoning_trace
    assert END_QUERY_TOKEN in resp.reasoning_trace
    assert BEGIN_RESULT_TOKEN in resp.reasoning_trace
    assert END_RESULT_TOKEN in resp.reasoning_trace
    # Reason-in-documents compression must reduce raw document length
    assert 0.1 < resp.compression_ratio < 0.85


def test_search_o1_kv_cache_and_flashattention_queries():
    agent = SearchO1Agent()
    resp_kv = agent.answer(
        "What is the Transformer KV-cache memory for L=32, G_kv=8, d_h=128, S=4096 in FP16 when batch size B is doubled from 1 to 2?"
    )
    assert resp_kv.final_answer == "1024 MiB"

    resp_fa = agent.answer(
        "What HBM memory complexity and peak A100 BF16 throughput does FlashAttention-2 achieve?"
    )
    assert "O(N)" in resp_fa.final_answer and "225 TFLOPS" in resp_fa.final_answer


def test_kernel_optimization_rejects_incorrect_fast_candidates_and_speeds_up_valid_ones():
    agent = KernelOptimizationAgent(warmup_runs=1, timed_runs=5)
    suite = build_kernel_benchmark_suite()

    # Check Kernel 1 specifically: mutation 0 is a buggy prefix sum (must be rejected),
    # mutation 1 is a valid O(N + Q) prefix sum (must be accepted with speedup > 1.5x)
    k1_report = agent.optimize_kernel(suite[0])
    assert k1_report.history[0].eval_result.compiled is True
    assert k1_report.history[0].eval_result.is_correct is False
    assert k1_report.history[0].eval_result.speedup_over_ref == 0.0

    assert k1_report.history[1].eval_result.is_correct is True
    assert k1_report.best_speedup > 2.0

    # Check all 4 kernels achieve verified speedups > 1.15x with max_abs_error <= 1e-5
    for task in suite:
        rep = agent.optimize_kernel(task)
        assert rep.best_speedup > 1.15, f"Kernel {task.kernel_id} did not achieve speedup"
        valid_steps = [s for s in rep.history if s.eval_result.is_correct]
        assert len(valid_steps) >= 1
        assert all(s.eval_result.max_abs_error <= task.atol for s in valid_steps)
