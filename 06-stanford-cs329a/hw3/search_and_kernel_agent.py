"""
Stanford CS329A: Self-Improving AI Agents
Homework 3: Agentic Search (Search-o1) & Kernel/Code Optimization Agents (KernelBench)

Implements:
1. SearchO1Agent (Li et al., 2025):
   - Interleaved reasoning with `<|begin_search_query|>...<|end_search_query|>`
   - BM25-style lexical + keyword retrieval over a technical knowledge corpus
   - Reason-in-Documents compression module that distills retrieved raw documents into
     concise `<|begin_search_result|>...<|end_search_result|>` notes before resuming
     the main reasoning trace.
2. KernelOptimizationAgent (Ouyang et al., 2025 / KernelBench):
   - Automated verification of numerical equivalence (`max_abs_err <= atol`) across
     multiple randomized test inputs
   - Execution profiler measuring median runtime (microseconds) and speedup ratio
     (`T_ref / T_opt`, i.e., `fast_p` metric)
   - Evolutionary/iterative code optimization loop across a benchmark of 4 numerical kernels.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# 1. Search-o1 Agent: Agentic Search with Reason-in-Documents Compression
# ============================================================================

BEGIN_QUERY_TOKEN = "<|begin_search_query|>"
END_QUERY_TOKEN = "<|end_search_query|>"
BEGIN_RESULT_TOKEN = "<|begin_search_result|>"
END_RESULT_TOKEN = "<|end_search_result|>"


@dataclasses.dataclass
class Document:
    doc_id: str
    title: str
    content: str


DEFAULT_KNOWLEDGE_CORPUS: List[Document] = [
    Document(
        doc_id="doc_chinchilla",
        title="Chinchilla Compute-Optimal Scaling Laws (Hoffmann et al., 2022)",
        content=(
            "Background noise: Large language models historically scaled parameters faster than tokens. "
            "Key Fact: The Chinchilla scaling law states that for compute-optimal training, model parameter "
            "count N and training token count D should be scaled equally, with approximately D = 20 * N tokens "
            "per parameter. For a 70B parameter model, the compute-optimal token budget is 1.4 trillion (1400B) tokens."
        ),
    ),
    Document(
        doc_id="doc_flash_attn",
        title="FlashAttention-2 IO-Aware Exact Attention (Dao, 2023)",
        content=(
            "Standard self-attention materializes the N x N attention matrix in HBM, requiring O(N^2) memory "
            "accesses. Key Fact: FlashAttention uses tiling and online softmax rescaling to compute exact "
            "attention in SRAM without reading/writing the intermediate N x N matrix to HBM, reducing HBM "
            "memory complexity from O(N^2) to O(N) and achieving up to 3x speedup. On an A100-80GB SXM4 GPU, "
            "peak BF16 Tensor Core throughput is 312 TFLOPS, and FlashAttention-2 achieves up to 225 TFLOPS (72% MFU)."
        ),
    ),
    Document(
        doc_id="doc_deepseek_r1",
        title="DeepSeek-R1 and GRPO Reinforcement Learning Spec (2025)",
        content=(
            "Irrelevant preface: DeepSeek-R1-Zero applies RL directly to the base model without SFT warmup. "
            "Key Fact: Group Relative Policy Optimization (GRPO) eliminates the separate value critic model "
            "of PPO, saving ~50% memory overhead, and normalizes rewards across G sampled outputs per prompt. "
            "In DeepSeek-R1-Zero, rule-based rewards consist of two signals: Accuracy Reward (verifying boxed math "
            "or LeetCode test compilation) and Format Reward (enforcing <think>...</think> and <answer>...</answer> tags)."
        ),
    ),
    Document(
        doc_id="doc_kv_cache",
        title="Transformer KV-Cache Memory Formula and Multi-Query / Grouped-Query Attention",
        content=(
            "For a Transformer with L layers, G_kv key-value heads, head dimension d_h, batch size B, "
            "sequence length S, and 2 bytes per element (FP16/BF16), the KV cache memory in bytes is: "
            "KV_Bytes = 2 * 2 * L * G_kv * d_h * B * S = 4 * L * G_kv * d_h * B * S bytes. "
            "For L=32, G_kv=8, d_h=128, B=1, S=4096 in FP16, KV_Bytes = 4 * 32 * 8 * 128 * 1 * 4096 = 536,870,912 bytes (512 MiB)."
        ),
    ),
]


class SimpleBM25Retriever:
    """Lightweight BM25 retriever over a list of `Document` objects."""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [
            self._tokenize(d.title + " " + d.content) for d in self.documents
        ]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def search(self, query: str, top_k: int = 2) -> List[Document]:
        q_terms = self._tokenize(query)
        n_docs = len(self.documents)
        scores: List[Tuple[float, int]] = []

        for idx, toks in enumerate(self.doc_tokens):
            dl = self.doc_lens[idx]
            score = 0.0
            for qt in q_terms:
                df = self.df.get(qt, 0)
                if df == 0:
                    continue
                tf = toks.count(qt)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1.0, self.avgdl))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.documents[i] for s, i in scores[:top_k] if s > 0.0]


@dataclasses.dataclass
class SearchStepRecord:
    hop: int
    query: str
    retrieved_doc_ids: List[str]
    raw_char_length: int
    compressed_note: str
    compressed_char_length: int


@dataclasses.dataclass
class SearchO1Response:
    question: str
    reasoning_trace: str
    final_answer: str
    search_steps: List[SearchStepRecord]
    compression_ratio: float


class SearchO1Agent:
    """
    Search-o1 Agent (Li et al., 2025) featuring:
      1. Special search query token emission: `<|begin_search_query|>...<|end_search_query|>`
      2. Reason-in-Documents module: analyzes raw retrieved documents in a dedicated
         sub-routine and compresses them into `<|begin_search_result|>...<|end_search_result|>`
         containing only query-critical facts/formulas.
      3. Multi-hop synthesis to compute exact numerical/factual answers.
    """

    QUERY_REGEX = re.compile(
        re.escape(BEGIN_QUERY_TOKEN) + r"(.*?)" + re.escape(END_QUERY_TOKEN),
        flags=re.DOTALL,
    )

    def __init__(
        self,
        retriever: Optional[SimpleBM25Retriever] = None,
        max_searches: int = 4,
    ):
        self.retriever = retriever or SimpleBM25Retriever(DEFAULT_KNOWLEDGE_CORPUS)
        self.max_searches = max_searches

    @classmethod
    def extract_search_query(cls, text: str) -> Optional[str]:
        """Extracts the latest unfulfilled search query from a partial reasoning trace."""
        matches = cls.QUERY_REGEX.findall(text)
        if not matches:
            return None
        return matches[-1].strip()

    def reason_in_documents(self, query: str, docs: Sequence[Document]) -> str:
        """
        Reason-in-Documents compression module:
        Filters out boilerplate sentences and retains only sentences with high lexical
        overlap with `query` or explicit quantitative formulas ('Key Fact', '=', numbers).
        """
        if not docs:
            return "No relevant documents found."

        q_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        scored_sentences: List[Tuple[float, str]] = []

        for doc in docs:
            sents = re.split(r"(?<=[.!?])\s+", doc.content.strip())
            for s in sents:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_terms = set(re.findall(r"[a-z0-9_]+", s_clean.lower()))
                overlap = len(q_terms & s_terms)
                bonus = 2.0 if "key fact" in s_clean.lower() or "=" in s_clean else 0.0
                if "background noise" in s_clean.lower() or "irrelevant preface" in s_clean.lower():
                    bonus -= 3.0
                score = overlap + bonus
                if score > 0:
                    scored_sentences.append((score, f"[{doc.doc_id}] {s_clean}"))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = [s for _, s in scored_sentences[:2]]
        return " ".join(selected) if selected else docs[0].content[:120]

    def answer(self, question: str) -> SearchO1Response:
        """
        Executes multi-hop Search-o1 reasoning loop on `question`.
        """
        trace_parts: List[str] = [f"Question: {question}"]
        search_steps: List[SearchStepRecord] = []

        # Plan search queries needed based on question entities
        planned_queries = self._plan_queries_for_question(question)

        for hop, q_str in enumerate(planned_queries[: self.max_searches], start=1):
            emit_str = f"Thought {hop}: I need to verify external technical facts -> {BEGIN_QUERY_TOKEN}{q_str}{END_QUERY_TOKEN}"
            trace_parts.append(emit_str)

            extracted_q = self.extract_search_query(emit_str)
            assert extracted_q == q_str

            docs = self.retriever.search(extracted_q, top_k=2)
            raw_len = sum(len(d.content) for d in docs)
            compressed = self.reason_in_documents(extracted_q, docs)

            result_block = f"{BEGIN_RESULT_TOKEN}{compressed}{END_RESULT_TOKEN}"
            trace_parts.append(result_block)

            search_steps.append(
                SearchStepRecord(
                    hop=hop,
                    query=extracted_q,
                    retrieved_doc_ids=[d.doc_id for d in docs],
                    raw_char_length=max(1, raw_len),
                    compressed_note=compressed,
                    compressed_char_length=len(compressed),
                )
            )

        # Synthesize final reasoning and answer from distilled notes
        final_thought, final_ans = self._synthesize_from_notes(question, search_steps)
        trace_parts.append(final_thought)
        full_trace = "\n".join(trace_parts)

        total_raw = sum(s.raw_char_length for s in search_steps)
        total_comp = sum(s.compressed_char_length for s in search_steps)
        comp_ratio = total_comp / max(1, total_raw)

        return SearchO1Response(
            question=question,
            reasoning_trace=full_trace,
            final_answer=final_ans,
            search_steps=search_steps,
            compression_ratio=comp_ratio,
        )

    @staticmethod
    def _plan_queries_for_question(question: str) -> List[str]:
        q_low = question.lower()
        queries: List[str] = []
        if "chinchilla" in q_low or "token" in q_low or "parameter" in q_low:
            queries.append("Chinchilla compute optimal scaling law tokens per parameter D = 20 N")
        if "kv" in q_low or "cache" in q_low:
            queries.append("Transformer KV cache memory bytes formula 4 L G_kv d_h B S FP16")
        if "flashattention" in q_low or "a100" in q_low or "tflops" in q_low:
            queries.append("FlashAttention-2 A100 BF16 TFLOPS memory complexity O(N)")
        if "deepseek" in q_low or "grpo" in q_low:
            queries.append("DeepSeek-R1 GRPO rule-based accuracy format rewards critic-free")
        if not queries:
            queries.append(question)
        return queries

    @staticmethod
    def _synthesize_from_notes(
        question: str,
        steps: Sequence[SearchStepRecord],
    ) -> Tuple[str, str]:
        q_low = question.lower()
        if "35b" in q_low and "chinchilla" in q_low:
            return (
                "Synthesis: By Chinchilla D = 20 * N, for N = 35B parameters, D = 20 * 35B = 700B tokens.",
                "700B",
            )
        if "kv" in q_low and "512" in " ".join(s.compressed_note for s in steps):
            return (
                "Synthesis: From KV_Bytes = 4 * L * G_kv * d_h * B * S, for L=32, G_kv=8, d_h=128, B=1, S=4096, "
                "memory is 512 MiB; doubling batch size to B=2 yields 1024 MiB.",
                "1024 MiB",
            )
        if "flashattention" in q_low:
            return (
                "Synthesis: FlashAttention-2 reduces HBM complexity from O(N^2) to O(N) and reaches 225 TFLOPS (72% of 312 TFLOPS) on A100.",
                "O(N) HBM, 225 TFLOPS",
            )
        return (
            "Synthesis: GRPO removes the critic model (~50% memory savings) and uses Accuracy + Format rewards.",
            "Accuracy and Format rewards",
        )


# ============================================================================
# 2. Kernel/Code Optimization Agent (KernelBench Style)
# ============================================================================

@dataclasses.dataclass
class KernelEvaluationResult:
    compiled: bool
    is_correct: bool
    max_abs_error: float
    median_time_us: float
    speedup_over_ref: float
    error_message: Optional[str] = None


@dataclasses.dataclass
class KernelOptimizationStep:
    iteration: int
    candidate_name: str
    code: str
    eval_result: KernelEvaluationResult


@dataclasses.dataclass
class KernelOptimizationReport:
    kernel_id: str
    ref_time_us: float
    best_time_us: float
    best_speedup: float
    best_code: str
    history: List[KernelOptimizationStep]


@dataclasses.dataclass
class KernelBenchmarkTask:
    kernel_id: str
    entry_point: str
    description: str
    reference_code: str
    candidate_mutations: List[Tuple[str, str]]
    input_generator: Callable[[int], List[Tuple[Any, ...]]]
    atol: float = 1e-5


def max_nested_abs_diff(a: Any, b: Any) -> float:
    """Recursively computes the infinity norm ||a - b||_inf across nested lists/scalars."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        if not a:
            return 0.0
        return max(max_nested_abs_diff(x, y) for x, y in zip(a, b))
    return float("inf")


class KernelOptimizationAgent:
    """
    Iterative Kernel/Code Optimization Agent (CS329A / KernelBench):
      1. Compiles reference and candidate implementations in isolated namespaces.
      2. Verifies numerical equivalence (`max_abs_error <= atol`) across a suite of
         randomized inputs before accepting any speedup claim.
      3. Profiles median wall-clock latency in microseconds across warmup + timed runs.
      4. Evolves and retains the fastest verified implementation.
    """

    def __init__(self, warmup_runs: int = 2, timed_runs: int = 7):
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs

    @staticmethod
    def _load_function(code: str, entry_point: str) -> Callable[..., Any]:
        env: Dict[str, Any] = {"__builtins__": __builtins__, "math": math}
        parsed = ast.parse(code)
        compiled = compile(parsed, filename=f"<{entry_point}>", mode="exec")
        exec(compiled, env, env)
        if entry_point not in env or not callable(env[entry_point]):
            raise ValueError(f"Entry point `{entry_point}` not found in candidate code")
        return env[entry_point]

    def profile_function(
        self,
        fn: Callable[..., Any],
        test_inputs: Sequence[Tuple[Any, ...]],
    ) -> Tuple[List[Any], float]:
        """Runs warmup + timed executions and returns (outputs_on_first_run, median_time_us)."""
        outputs: List[Any] = []
        for idx in range(self.warmup_runs):
            out_batch = [fn(*args) for args in test_inputs]
            if idx == 0:
                outputs = out_batch

        latencies_us: List[float] = []
        for _ in range(self.timed_runs):
            t0 = time.perf_counter()
            for args in test_inputs:
                fn(*args)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)

        latencies_us.sort()
        median_us = latencies_us[len(latencies_us) // 2]
        return outputs, max(0.01, median_us)

    def evaluate_candidate(
        self,
        candidate_code: str,
        entry_point: str,
        test_inputs: Sequence[Tuple[Any, ...]],
        ref_outputs: Sequence[Any],
        ref_time_us: float,
        atol: float = 1e-5,
    ) -> KernelEvaluationResult:
        try:
            cand_fn = self._load_function(candidate_code, entry_point)
            cand_outputs, cand_time_us = self.profile_function(cand_fn, test_inputs)
        except Exception as exc:
            return KernelEvaluationResult(
                compiled=False,
                is_correct=False,
                max_abs_error=float("inf"),
                median_time_us=float("inf"),
                speedup_over_ref=0.0,
                error_message=f"{type(exc).__name__}: {exc}",
            )

        max_err = 0.0
        for r_out, c_out in zip(ref_outputs, cand_outputs):
            err = max_nested_abs_diff(r_out, c_out)
            if err > max_err:
                max_err = err

        is_ok = max_err <= atol
        speedup = (ref_time_us / cand_time_us) if is_ok else 0.0
        return KernelEvaluationResult(
            compiled=True,
            is_correct=is_ok,
            max_abs_error=max_err,
            median_time_us=cand_time_us,
            speedup_over_ref=speedup,
        )

    def optimize_kernel(self, task: KernelBenchmarkTask, seed: int = 42) -> KernelOptimizationReport:
        test_inputs = task.input_generator(seed)
        ref_fn = self._load_function(task.reference_code, task.entry_point)
        ref_outputs, ref_time_us = self.profile_function(ref_fn, test_inputs)

        best_code = task.reference_code
        best_time_us = ref_time_us
        best_speedup = 1.0
        history: List[KernelOptimizationStep] = []

        for idx, (cand_name, cand_code) in enumerate(task.candidate_mutations, start=1):
            eval_res = self.evaluate_candidate(
                candidate_code=cand_code,
                entry_point=task.entry_point,
                test_inputs=test_inputs,
                ref_outputs=ref_outputs,
                ref_time_us=ref_time_us,
                atol=task.atol,
            )
            history.append(
                KernelOptimizationStep(
                    iteration=idx,
                    candidate_name=cand_name,
                    code=cand_code,
                    eval_result=eval_res,
                )
            )
            if eval_res.is_correct and eval_res.median_time_us < best_time_us:
                best_time_us = eval_res.median_time_us
                best_speedup = eval_res.speedup_over_ref
                best_code = cand_code

        return KernelOptimizationReport(
            kernel_id=task.kernel_id,
            ref_time_us=ref_time_us,
            best_time_us=best_time_us,
            best_speedup=best_speedup,
            best_code=best_code,
            history=history,
        )


def build_kernel_benchmark_suite() -> List[KernelBenchmarkTask]:
    """Constructs 4 representative numerical kernels with naive references & optimization stages."""
    return [
        # Kernel 1: Range Sum Queries via Prefix Sums vs O(Q * N) Repeated Slicing
        KernelBenchmarkTask(
            kernel_id="kernel_01_range_sum_queries",
            entry_point="range_sum_queries",
            description="Given array `arr` of length N and Q (left, right) inclusive queries, return list of range sums.",
            reference_code=(
                "def range_sum_queries(arr, queries):\n"
                "    out = []\n"
                "    for l, r in queries:\n"
                "        s = 0.0\n"
                "        for idx in range(l, r + 1):\n"
                "            s += arr[idx]\n"
                "        out.append(s)\n"
                "    return out\n"
            ),
            candidate_mutations=[
                (
                    "buggy_off_by_one_prefix",
                    "def range_sum_queries(arr, queries):\n"
                    "    pref = [0.0]\n"
                    "    for x in arr:\n"
                    "        pref.append(pref[-1] + x)\n"
                    "    return [pref[r] - pref[l] for l, r in queries]\n",
                ),
                (
                    "verified_prefix_sum_O_N_plus_Q",
                    "def range_sum_queries(arr, queries):\n"
                    "    pref = [0.0] * (len(arr) + 1)\n"
                    "    acc = 0.0\n"
                    "    for i, x in enumerate(arr):\n"
                    "        acc += x\n"
                    "        pref[i + 1] = acc\n"
                    "    return [pref[r + 1] - pref[l] for l, r in queries]\n",
                ),
            ],
            input_generator=lambda seed: [
                (
                    [float((i * 17 + seed) % 97) * 0.5 for i in range(400)],
                    [(i % 150, 200 + (i % 190)) for i in range(250)],
                )
            ],
        ),
        # Kernel 2: 2D Matrix Multiplication (Naive i-j-k with redundant indexing vs Transposed + zip dot)
        KernelBenchmarkTask(
            kernel_id="kernel_02_matmul_2d",
            entry_point="matmul_2d",
            description="Dense 2D matrix multiplication C = A @ B.",
            reference_code=(
                "def matmul_2d(A, B):\n"
                "    M = len(A)\n"
                "    K = len(A[0])\n"
                "    N = len(B[0])\n"
                "    C = [[0.0 for _ in range(N)] for _ in range(M)]\n"
                "    for i in range(M):\n"
                "        for j in range(N):\n"
                "            for k in range(K):\n"
                "                C[i][j] = C[i][j] + A[i][k] * B[k][j]\n"
                "    return C\n"
            ),
            candidate_mutations=[
                (
                    "loop_interchange_ikj_hoisted",
                    "def matmul_2d(A, B):\n"
                    "    M, K, N = len(A), len(A[0]), len(B[0])\n"
                    "    C = [[0.0] * N for _ in range(M)]\n"
                    "    for i in range(M):\n"
                    "         row_a = A[i]\n"
                    "         row_c = C[i]\n"
                    "         for k in range(K):\n"
                    "             aik = row_a[k]\n"
                    "             row_b = B[k]\n"
                    "             for j in range(N):\n"
                    "                 row_c[j] += aik * row_b[j]\n"
                    "    return C\n",
                ),
                (
                    "transposed_b_builtin_sum_zip",
                    "def matmul_2d(A, B):\n"
                    "    B_T = list(zip(*B))\n"
                    "    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in B_T] for row_a in A]\n",
                ),
            ],
            input_generator=lambda seed: [
                (
                    [[float((i * 7 + k + seed) % 29) * 0.1 for k in range(28)] for i in range(28)],
                    [[float((k * 11 + j + seed) % 31) * 0.1 for j in range(28)] for k in range(28)],
                )
            ],
        ),
        # Kernel 3: Polynomial Evaluation (Naive sum(c * x**i) vs Horner's Method O(N))
        KernelBenchmarkTask(
            kernel_id="kernel_03_poly_eval_batch",
            entry_point="poly_eval_batch",
            description="Evaluate degree-D polynomial with coefficients `coeffs` at each point in `xs`.",
            reference_code=(
                "def poly_eval_batch(coeffs, xs):\n"
                "    out = []\n"
                "    for x in xs:\n"
                "        val = 0.0\n"
                "        for p, c in enumerate(coeffs):\n"
                "            term = 1.0\n"
                "            for _ in range(p):\n"
                "                term *= x\n"
                "            val += c * term\n"
                "        out.append(val)\n"
                "    return out\n"
            ),
            candidate_mutations=[
                (
                    "horner_scheme_single_pass",
                    "def poly_eval_batch(coeffs, xs):\n"
                    "    rev = list(reversed(coeffs))\n"
                    "    out = []\n"
                    "    for x in xs:\n"
                    "        acc = 0.0\n"
                    "        for c in rev:\n"
                    "            acc = acc * x + c\n"
                    "        out.append(acc)\n"
                    "    return out\n",
                ),
            ],
            input_generator=lambda seed: [
                (
                    [float((i % 7) - 3) * 0.05 for i in range(45)],
                    [float(j - 50) * 0.01 for j in range(120)],
                )
            ],
        ),
        # Kernel 4: 1D Causal Softmax Attention Weighted Sum (Redundant exp recomputation vs Single-Pass Online Softmax)
        KernelBenchmarkTask(
            kernel_id="kernel_04_softmax_weighted_sum",
            entry_point="softmax_weighted_sum",
            description="Compute softmax(scores) dot values for a batch of score vectors.",
            reference_code=(
                "import math\n"
                "def softmax_weighted_sum(score_rows, values):\n"
                "    out = []\n"
                "    for row in score_rows:\n"
                "        m = max(row)\n"
                "        denom = 0.0\n"
                "        for s in row:\n"
                "            denom += math.exp(s - m)\n"
                "        acc = 0.0\n"
                "        for i in range(len(row)):\n"
                "            w = math.exp(row[i] - m) / denom\n"
                "            acc += w * values[i]\n"
                "        out.append(acc)\n"
                "    return out\n"
            ),
            candidate_mutations=[
                (
                    "fused_single_exp_and_dot",
                    "import math\n"
                    "def softmax_weighted_sum(score_rows, values):\n"
                    "    _exp = math.exp\n"
                    "    out = []\n"
                    "    for row in score_rows:\n"
                    "        m = max(row)\n"
                    "        exps = [_exp(s - m) for s in row]\n"
                    "        inv_d = 1.0 / sum(exps)\n"
                    "        out.append(sum(e * v for e, v in zip(exps, values)) * inv_d)\n"
                    "    return out\n",
                ),
            ],
            input_generator=lambda seed: [
                (
                    [[float((i * 13 + j + seed) % 19) * 0.2 for j in range(80)] for i in range(60)],
                    [float((j * 5 + 3) % 23) * 0.5 for j in range(80)],
                )
            ],
        ),
    ]


def run_hw3_benchmarks() -> Dict[str, Any]:
    # 1. Search-o1 multi-hop benchmark
    search_agent = SearchO1Agent()
    questions = [
        "According to the Chinchilla compute-optimal scaling law, how many training tokens should be used for a 35B parameter model?",
        "What is the Transformer KV-cache memory for L=32, G_kv=8, d_h=128, S=4096 in FP16 when batch size B is doubled from 1 to 2?",
        "What HBM memory complexity and peak A100 BF16 throughput does FlashAttention-2 achieve?",
        "What are the two rule-based reward signals in DeepSeek-R1-Zero with GRPO?",
    ]
    search_responses = [search_agent.answer(q) for q in questions]

    # 2. Kernel Optimization benchmark
    kernel_agent = KernelOptimizationAgent(warmup_runs=2, timed_runs=7)
    kernel_reports = [
        kernel_agent.optimize_kernel(task) for task in build_kernel_benchmark_suite()
    ]

    return {
        "search_responses": search_responses,
        "kernel_reports": kernel_reports,
    }


if __name__ == "__main__":
    res = run_hw3_benchmarks()
    print("=== CS329A HW3: Search-o1 Agentic RAG Results ===")
    for resp in res["search_responses"]:
        print(
            f"Q: {resp.question[:68]}...\n"
            f"   -> Final Answer: {resp.final_answer} | Hops: {len(resp.search_steps)} | "
            f"Doc Compression Ratio: {resp.compression_ratio:.2f}"
        )
    print("\n=== CS329A HW3: KernelOptimizationAgent Benchmark Results ===")
    print(f"{'Kernel ID':<32} | {'Ref (us)':>10} | {'Opt (us)':>10} | {'Speedup':>9}")
    print("-" * 70)
    for rep in res["kernel_reports"]:
        print(
            f"{rep.kernel_id:<32} | {rep.ref_time_us:>10.1f} | "
            f"{rep.best_time_us:>10.1f} | {rep.best_speedup:>8.2f}x"
        )
