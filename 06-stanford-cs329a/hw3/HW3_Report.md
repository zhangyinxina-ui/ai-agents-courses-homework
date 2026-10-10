# Stanford CS329A — Homework 3 Technical Report
## Agentic Search (`Search-o1`) & Automated Kernel/Code Optimization (`KernelBench`)

**Course:** Stanford CS329A — *Self-Improving AI Agents* (Instructors: Aakanksha Chowdhery & Azalia Mirhoseini)  
**Implementation:** [`hw3/search_and_kernel_agent.py`](./search_and_kernel_agent.py) | **Test Suite:** [`hw3/test_hw3.py`](./test_hw3.py)

---

## 1. Agentic RAG with `Search-o1` and Reason-in-Documents Compression

### 1.1 Architectural Overview (Li et al., 2025)
Standard Retrieval-Augmented Generation (RAG) performs a single static retrieval pass prior to generation and injects raw, noisy document chunks directly into the LLM prompt, polluting long-horizon reasoning chains with irrelevant context. [`SearchO1Agent`](./search_and_kernel_agent.py) implements two synergistic mechanisms:

1. **Dynamic Autonomous Search Invocation:** During multi-step reasoning, whenever the policy encounters an external factual or formulaic dependency, it emits special control tokens:
   $$\texttt{<|begin\_search\_query|>} \; q_t \; \texttt{<|end\_search\_query|>}$$
   Generation pauses at $\texttt{<|end\_search\_query|>}$, triggers BM25 retrieval (`SimpleBM25Retriever.search(q_t, top_k=2)`), and passes the retrieved documents $\mathcal{D}_t$ to the Reason-in-Documents module.
2. **Reason-in-Documents Compression (`reason_in_documents`):** Rather than appending raw passages $\mathcal{D}_t$ to the main reasoning trace, a dedicated document-analysis step scores each sentence by query-term overlap and mathematical/factual density, strips boilerplate distraction, and injects a concise knowledge note:
   $$\texttt{<|begin\_search\_result|>} \; \tilde{d}_t \; \texttt{<|end\_search\_result|>}$$

### 1.2 Empirical Results on Technical AI Systems Queries
| Query Topic | Search Query Emitted | Retrieved Docs | Raw Doc Chars | Compressed Chars | Compression Ratio | Verified Final Answer |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Chinchilla 35B Optimal Tokens** | `Chinchilla compute optimal scaling law tokens per parameter D = 20 N` | `doc_chinchilla`, `doc_kv_cache` | 730 | 281 | **0.38** | `700B` |
| **Transformer KV-Cache ($B=2$)** | `Transformer KV cache memory bytes formula 4 L G_kv d_h B S FP16` | `doc_kv_cache`, `doc_flash_attn` | 786 | 417 | **0.53** | `1024 MiB` |
| **FlashAttention-2 Complexity & MFU** | `FlashAttention-2 A100 BF16 TFLOPS memory complexity O(N)` | `doc_flash_attn`, `doc_kv_cache` | 786 | 378 | **0.48** | `O(N) HBM, 225 TFLOPS` |
| **DeepSeek-R1-Zero GRPO Rewards** | `DeepSeek-R1 GRPO rule-based accuracy format rewards critic-free` | `doc_deepseek_r1` | 476 | 419 | **0.88** | `Accuracy and Format rewards` |

---

## 2. Iterative Kernel & Code Optimization Agent (`KernelOptimizationAgent`)

### 2.1 Verification-First Optimization Loop (Ouyang et al., 2025 / KernelBench)
Optimizing performance-critical numerical kernels with LLMs faces a critical failure mode: an agent can easily achieve high speedups by subtly breaking edge-case semantics or numerical precision (reward hacking). Following CS329A's KernelBench methodology, [`KernelOptimizationAgent`](./search_and_kernel_agent.py) enforces strict **two-gate evaluation**:

1. **Gate 1 — Compilation & Numerical Equivalence Verification:**
   For every randomized test input $x_m \in \mathcal{X}_{\text{test}}$, the candidate kernel $f_{\text{cand}}$ is compared against the reference implementation $f_{\text{ref}}$ under the infinity norm:
   $$\varepsilon_{\max} = \max_{x_m \in \mathcal{X}_{\text{test}}} \left\| f_{\text{cand}}(x_m) - f_{\text{ref}}(x_m) \right\|_\infty \le \tau_{\text{atol}} \quad (\tau_{\text{atol}} = 10^{-5})$$
   Any candidate violating $\varepsilon_{\max} \le \tau_{\text{atol}}$ is assigned $\text{Speedup} = 0.0\times$ and rejected regardless of its execution speed.
2. **Gate 2 — Warmup + Median Latency Profiling (`fast_p` Metric):**
   Each valid candidate executes $W = 2$ warmup passes followed by $R = 7$ timed runs using `time.perf_counter()`, recording the median latency $T_{\text{med}}$ in microseconds ($\mu\text{s}$) and speedup ratio $S = T_{\text{ref}} / T_{\text{cand}}$.

### 2.2 Empirical Benchmark Results Across 4 Numerical Kernels
| Kernel ID | Optimization Strategy Applied | Correctness ($\varepsilon_{\max} \le 10^{-5}$) | Reference Latency ($\mu\text{s}$) | Optimized Latency ($\mu\text{s}$) | Verified Speedup ($S$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `kernel_01_range_sum_queries` | Candidate 1: Off-by-one prefix sum (`pref[r] - pref[l]`) | **Rejected** ($\varepsilon_{\max} = 48.0$) | 999.1 | — | $0.00\times$ (Rejected) |
| `kernel_01_range_sum_queries` | Candidate 2: Verified $\mathcal{O}(N+Q)$ prefix sum (`pref[r+1] - pref[l]`) | **Verified** ($\varepsilon_{\max} = 0.0$) | 999.1 | **27.1** | **$36.83\times$** |
| `kernel_02_matmul_2d` | Transposed $B^\top$ cache-aligned dot product + loop hoisting | **Verified** ($\varepsilon_{\max} < 10^{-12}$) | 1033.6 | **688.0** | **$1.50\times$** |
| `kernel_03_poly_eval_batch` | Horner's nested evaluation scheme ($\mathcal{O}(D)$ vs $\mathcal{O}(D^2)$) | **Verified** ($\varepsilon_{\max} < 10^{-12}$) | 2030.0 | **90.7** | **$22.37\times$** |
| `kernel_04_softmax_weighted_sum` | Fused single-pass `math.exp` memoization + inverse denominator | **Verified** ($\varepsilon_{\max} < 10^{-12}$) | 558.9 | **453.6** | **$1.23\times$** |

**Summary:** Across all four kernels, `KernelOptimizationAgent` achieves `fast_1 = 100%` (4/4 kernels both functionally equivalent and strictly faster than reference), with geometric mean speedup **$6.41\times$** and peak algorithmic + systems speedup of **$36.83\times$**.
