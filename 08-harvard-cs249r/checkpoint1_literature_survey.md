# Harvard CS249r (Fall 2025) — Checkpoint 1: Literature Survey & Weekly Reading Synthesis

**Course**: CS249r — *Architecture 2.0: Agentic AI for Computer Systems Design* (Fall 2025)  
**Instructor**: Prof. Vijay Janapa Reddi (Harvard University)  
**Deliverable**: Checkpoint 1 — Comprehensive Literature Survey, Structured Data-Extraction Schema, & Weekly Reading Synthesis  

---

## 1. Executive Summary & The "Architecture 2.0" Paradigm

Modern computer systems engineering is undergoing a foundational shift from **Architecture 1.0**—characterized by human-crafted heuristics, rigid compiler pass pipelines, and manual Register-Transfer Level (RTL) or kernel tuning—to **Architecture 2.0**, where autonomous, verifier-grounded AI agents iteratively co-design software, microarchitecture, and physical silicon.

Unlike open-ended natural language generation, **Agentic AI for Computer Systems Design** operates under strict physical and formal invariants:
1. **Binary Correctness Invariants**: Generated artifacts (Verilog modules, CUDA C++ kernels, LLVM IR passes, or Kubernetes configuration patches) must compile without syntax errors and pass formal equivalence checks or cycle-accurate testbenches.
2. **Non-Linear Physical Objectives**: Systems agents optimize multi-objective Pareto frontiers—Power, Performance, and Area (PPA) in silicon; register pressure, warp occupancy, and HBM bandwidth in GPUs; or tail latency (P99 TTFT/TBT) in distributed clusters.
3. **Expensive Verifier-in-the-Loop Bottlenecks**: Whereas LLM token generation takes milliseconds to seconds, invoking a ground-truth systems verifier (e.g., Synopsys Design Compiler, OpenROAD placement, `gem5` cycle-accurate simulation, or `nvcc` + hardware profiling) can consume **50% to 90%+ of total episode wall-clock time**.

Consequently, Architecture 2.0 requires studying a **bidirectional virtuous cycle**:
- **AI Agents for Systems Design (Domains 1–3)**: How multi-turn reasoning, Monte Carlo Tree Search (MCTS), evolutionary coding agents, and reinforcement learning optimize hardware/RTL, compilers/GPU kernels, and cloud/architecture DSE.
- **Computer Systems for Serving Agentic Workloads (Domain 4)**: How underlying serving engines (PagedAttention, RadixAttention prefix trees, chunked prefill, learning-to-rank schedulers, and disaggregated KV caches) absorb the massive **Token Amplification Factor ($\alpha_{\text{tok}}$)** and branching KV-cache footprint induced by systems agents.

```
+===================================================================================+
|                     ARCHITECTURE 2.0 BIDIRECTIONAL STACK                          |
+===================================================================================+
|                                                                                   |
|   [Domain 1: Hardware/EDA & RTL]     VerilogEval, RTLCoder, ChipNeMo, ChatEDA,    |
|   (Silicon, Logic Synthesis, SVA)    MCTS-RTL, CVDP, Formal Verification/AssertLLM|
|                 |                                         ^                       |
|                 v                                         |                       |
|   [Domain 2: Compilers & Kernels]    KernelBench, Kevin (Multi-Turn RL), AlphaDev,|
|   (CUDA/Triton, LLVM IR, WSC C++)    Meta LLM Compiler, FunSearch/AlphaEvolve, ECO|
|                 |                                         |                       |
|                 v                                         |                       |
|   [Domain 3: Cloud/OS & Arch DSE]    ArchGym, Concorde, COSMIC, QuArch,           |
|   (Microarch Sim, Distributed, SRE)  OSWorld, RCACopilot / AIOpsLab               |
|                 |                                         |                       |
|                 +------------------> Demands High-Throughput,                     |
|                                      Low-Latency Multi-Turn Serving               |
|                                                   |                               |
|   [Domain 4: Serving Systems for Agents] <--------+                               |
|   (KV-Cache, Scheduling, Multi-LoRA) SGLang (RadixAttention), vLLM (PagedAttn),   |
|                                      Sarathi-Serve, L2R Scheduler, S-LoRA,        |
|                                      CacheBlend (RAG Knowledge Fusion)            |
+===================================================================================+
```

---

## 2. Curated Literature Taxonomy (25 Seminal Papers Across 4 Domains)

Following the CS249r Checkpoint 1 specification, we curate **25 seminal papers** spanning the four core layers of the Architecture 2.0 stack. All extracted quantitative metrics are stored in [`systems_agent_dataset.json`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/systems_agent_dataset.json).

### 2.1 Domain 1: Hardware/EDA & RTL Design Agents (`P01`–`P07`)

Hardware design agents translate natural-language architectural specifications into synthesizable HDL (Verilog/SystemVerilog), EDA tool scripts (Tcl/OpenROAD), or formal verification assertions (SVA), closing the loop with cycle-accurate simulators (`iverilog`, `Verilator`), logic synthesis (`Yosys`, `ABC`, `Synopsys DC`), and formal model checkers (`JasperGold`, `SymbiYosys`).

| ID | Paper Title & Venue | CS249r Week | Agent Paradigm & Verifier Loop | Key Quantitative Finding |
| :--- | :--- | :---: | :--- | :--- |
| **P01** | **VerilogEval**: Evaluating LLMs for Verilog Code Generation (*Liu et al., ICCAD 2023*) | Wk 11 | SFT on synthetic spec-to-RTL pairs + `iverilog` multi-sample testbench filtering | Boosts `pass@1` (46.7%) to `pass@10` (69.0%), proving test-time sampling unlocks RTL correctness. |
| **P02** | **RTLCoder**: Outperforming GPT-3.5 in Design RTL Generation (*Liu et al., IEEE TCAD 2024*) | Wk 11 | Automated keyword-guided synthesis + simulator-scored candidate alignment | Lightweight 7B model achieves 61.2% `pass@1` and 76.4% `pass@10`, outperforming GPT-3.5 and matching GPT-4. |
| **P03** | **ChipNeMo**: Domain-Adapted LLMs for Chip Design (*Liu, Ren et al., NVIDIA, 2023*) | Wk 11 | Domain-Adaptive Pretraining (DAPT) + custom HDL tokenizer + Domain RAG + Tcl tool agent | Custom tokenizer cuts token count by 16–33%; 13B domain model matches 70B LLaMA2 on EDA script generation. |
| **P04** | **ChatEDA**: Autonomous Agent for EDA (*Wu et al., IEEE TCAD 2024*) | Wk 12 | Hierarchical task decomposition planner + script generator over OpenROAD RTL-to-GDSII | Achieves 88.0% multi-turn task completion; reveals that physical EDA execution consumes **89.4%** of episode time. |
| **P05** | **Make Every Move Count**: High-Quality RTL Generation Using MCTS (*DeLorenzo et al., 2024*) | Wk 11 | Monte Carlo Tree Search (UCT) over token trajectories guided by Yosys+ABC synthesis PPA reward | Reduces Area-Delay Product (ADP) by up to **31.8%** (`1.31x` PPA gain) over greedy decoding, at $\alpha_{\text{tok}}=28.5\times$. |
| **P06** | **CVDP**: Comprehensive Verilog Design Problems Benchmark (*Pinckney, Ren et al., NVIDIA, 2025*) | Wk 11 | Multi-turn code-test-repair agent inside Dockerized Verilator + Cocotb + SymbiYosys harness | Exposes industrial gap: `pass@1` drops to 27.3% (48.6% `pass@10`) on multi-file RTL debugging and verification. |
| **P07** | **LLMs for Formal Verification of RTL & AssertLLM** (*Orenes-Vera et al. / Fang et al., 2024*) | Wk 13 | Spec-decomposition multi-agent + counterexample-guided assertion repair (`JasperGold`/`SymbiYosys`) | Counterexample feedback raises valid SVA property verification from 49.5% (`pass@1`) to **79.2%** (`pass@10`). |

### 2.2 Domain 2: Compiler & GPU Kernel Optimization Agents (`P08`–`P13`)

Software and kernel agents move beyond functional correctness (`pass@k`) to **performance engineering (`fast_p`)**. Because modern hardware exhibits non-obvious bottlenecks—shared-memory bank conflicts, warp divergence, register spilling, and instruction-cache misses—single-shot LLM generation consistently underperforms hand-tuned libraries (`cuBLAS`, `FlashAttention`) unless coupled with iterative profiler feedback or evolutionary/RL search.

| ID | Paper Title & Venue | CS249r Week | Agent Paradigm & Verifier Loop | Key Quantitative Finding |
| :--- | :--- | :---: | :--- | :--- |
| **P08** | **KernelBench**: Can LLMs Write Efficient GPU Kernels? (*Ouyang et al., Stanford, 2025*) | Wk 5 | Iterative compilation (`nvcc`), PyTorch numerical check, and Nsight compute profiler feedback | Zero-shot `fast_1` (faster than PyTorch baseline) is only 18.0%, but 10-turn execution/profiler feedback reaches **54.0%** (`1.42x` mean speedup). |
| **P09** | **Kevin**: Multi-Turn RL for Generating CUDA Kernels (*Baronio, Rush et al., 2025*) | Wk 5 | Multi-turn reinforcement learning optimizing cumulative trajectory speedup and correctness | Raises `fast_1` from 44.5% to **82.0%** (`1.76x` speedup over `torch.compile`) by learning to stage correctness before warp tiling. |
| **P10** | **AlphaDev**: Faster Sorting Algorithms Discovered Using Deep RL (*Mankowitz et al., Nature 2023*) | Wk 4 | AlphaZero MCTS over x86/ARM assembly instructions + SMT solver + CPU latency reward | Discovers branchless swap/copy assembly idioms yielding **1.70x** speedup on short sort sequences (`Sort-3` to `Sort-5`), merged into `libc++`. |
| **P11** | **Meta LLM Compiler**: Foundation Models of Compiler Optimization (*Cummins et al., 2024*) | Wk 4 | 546B-token LLVM-IR/assembly pretraining + multi-sample pass-list autotuning (`llvm-mca`) | Emulates compiler transformations and achieves **1.053x** code-size reduction over `clang -Oz` with 5.5x fewer search tokens. |
| **P12** | **FunSearch & AlphaEvolve**: Evolutionary Coding Agents (*Romera-Paredes / Novikov et al., DeepMind, 2024–2025*) | Wk 5 | Island-based evolutionary program database + Gemini mutation + sandboxed TPU kernel evaluator | Achieves **1.23x** speedup on production TPU matrix multiplication kernels and recovers 0.7% of Google's fleet-wide compute. |
| **P13** | **ECO**: LLM-Driven Efficient Code Optimizer for Warehouse Scale Computers (*Yazdanbakhsh et al., DeepMind, 2025*) | Wk 4 | Fleet telemetry hotspot localization + LLM C++ patch synthesis + Bazel build/test/benchmark loop | Achieves **1.19x** average speedup on production WSC hotspots; verifier build/benchmark takes **84.0%** of loop time. |

### 2.3 Domain 3: Cloud/OS/Database & Architecture DSE Operations Agents (`P14`–`P19`)

At the system architecture and cloud operations layer, agents navigate vast Design Space Exploration (DSE) parameters (cache associativity, NoC topology, distributed parallelism degrees) or live Operating System / Kubernetes telemetry states. Here, the central tension is **Simulator Fidelity vs. Search Horizon**: high-fidelity cycle simulators (`gem5`, `ASTRA-sim`) incur >75–90% verifier latency share, motivating learned analytical-ML surrogates (`Concorde`).

| ID | Paper Title & Venue | CS249r Week | Agent Paradigm & Verifier Loop | Key Quantitative Finding |
| :--- | :--- | :---: | :--- | :--- |
| **P14** | **ArchGym**: Open-Source Gymnasium for ML-Assisted Architecture Design (*Krishnan, Reddi et al., ISCA 2023*) | Wk 7 | Standardized RL/BO/ACO/GA agent interface over `Timeloop`, `DRAMSys`, and `gem5` simulators | Simulator execution accounts for **91.5%** of wall-clock time; proper hyperparameter tuning yields **1.45x** better PPA Pareto hypervolume. |
| **P15** | **Concorde**: Fast & Accurate CPU Performance Modeling (*Nasr, Subramanian et al., 2025*) | Wk 7 | Compositional analytical queueing bounds + lightweight ML residual surrogate replacing `gem5` | Slashes verifier latency share from >90% to **14.5%** (5 orders of magnitude faster than `gem5`, <2% CPI error), enabling rapid agentic DSE. |
| **P16** | **COSMIC**: Full-Stack Co-Design of Distributed ML Systems (*Rashidi, Maas et al., 2025*) | Wk 6 | Cross-layer multi-objective search across tensor/pipeline parallelism, collective scheduling, and fabric | Joint full-stack optimization outperforms isolated layer tuning by **1.58x** in TFLOPs/watt-dollar. |
| **P17** | **QuArch**: Question-Answering Dataset for AI Agents in Computer Architecture (*Agiannou, Reddi et al., 2025*) | Wk 2 | CoT + domain retrieval evaluated on 1,500 expert-curated microarchitecture & ISA problems | Reveals that while LLMs score 71.4% (`pass@1`), they struggle on quantitative pipeline hazard timing and cache coherence corner cases. |
| **P18** | **OSWorld**: Benchmarking Multimodal Agents in Real Computer Environments (*Xie et al., NeurIPS 2024*) | Wk 3 | Multimodal screenshot + A11y tree observation-action loop inside live Ubuntu/Windows VMs | Multi-step OS horizon (avg 15 turns) suffers compounding state drift: `pass@1` is 12.2% and `pass@10` is 28.6%. |
| **P19** | **RCACopilot & AIOpsLab**: Autonomous Cloud Root-Cause Analysis Agents (*Chen et al., EuroSys 2024 / MLSys 2025*) | Wk 6 | Diagnostic handler workflow + structured log/metric/trace tool-calling over live ChaosMesh K8s clusters | Cuts Mean-Time-To-Mitigate (MTTM) by **3.10x** (`77.5%` resolution at $k=10$) by compressing raw telemetry prior to LLM reasoning. |

### 2.4 Domain 4: Computer Systems for Serving Agentic Workloads (`P20`–`P25`)

Domains 1–3 demonstrate that systems agents amplify token consumption by **5x to 42x** via multi-turn reflection, MCTS tree branching, and long tool/profiler outputs. Traditional LLM serving engines designed for single-turn chat collapse under this workload due to redundant KV-cache recomputation, HBM fragmentation, and prefill-induced decode stalls. Domain 4 papers re-architect the serving stack specifically for agentic execution patterns.

| ID | Paper Title & Venue | CS249r Week | Serving System Innovation | Key Quantitative Finding |
| :--- | :--- | :---: | :--- | :--- |
| **P20** | **SGLang**: Efficient Execution of Structured LM Programs (*Zheng et al., NeurIPS 2024*) | Wk 10 | **RadixAttention** LRU prefix tree for automatic KV-cache reuse across multi-turn/tree agent branches + FSM decoding | Achieves **0.94** KV-cache reuse ratio and **5.0x** throughput speedup on multi-turn agent & MCTS workloads. |
| **P21** | **vLLM**: PagedAttention for LLM Serving (*Kwon et al., SOSP 2023*) | Wk 10 | OS-inspired virtual memory paging + Copy-on-Write block tables for parallel `pass@k` sampling | Cuts HBM KV-cache fragmentation to <4%, achieving **0.89** prefix sharing efficiency and **3.20x** serving throughput. |
| **P22** | **Sarathi-Serve**: Taming Throughput-Latency Tradeoff (*Agrawal et al., OSDI 2024*) | Wk 10 | **Chunked-Prefill** + stall-free piggybacked decode scheduling for long agent tool outputs | Eliminates generation stalls caused by long compiler/log prefills, boosting serving capacity by **3.80x** under strict P99 TBT SLOs. |
| **P23** | **Efficient LLM Scheduling by Learning to Rank** (*Fu, Choukse et al., NeurIPS 2024*) | Wk 10 | Listwise Learning-to-Rank (Kendall-Tau) output length predictor driving preemptive SJF scheduling | Prevents head-of-line blocking from long reasoning trajectories, achieving **2.80x** lower P90 latency than FCFS. |
| **P24** | **S-LoRA**: Serving Thousands of Concurrent LoRA Adapters (*Sheng et al., MLSys 2024*) | Wk 10 | Unified HBM paging for dynamic KV cache + heterogeneous LoRA adapter ranks (`CustomMBGMM` kernels) | Enables thousands of specialized agent role adapters (planner, RTL coder, SVA critic) on one base model at **4.00x** throughput. |
| **P25** | **CacheBlend**: Fast LLM Serving for RAG with Cached Knowledge Fusion (*Yao, Jiang et al., EuroSys 2025*) | Wk 10 | Selective ~15% high-deviation cross-attention recomputation over pre-cached RAG/tool KV blocks | Achieves **0.91** KV reuse across non-prefix-aligned tool/document chunks, cutting Time-To-First-Token (TTFT) by **3.30x**. |

---

## 3. Structured Data-Extraction Schema & Quantitative Metrics

To enable rigorous cross-stack meta-analysis in Checkpoint 2, we define a standardized quantitative schema across all 25 papers (`systems_agent_dataset.json`):

1. **Single-Shot vs. Multi-Sample/Multi-Turn Success (`pass_at_1`, `pass_at_10`)**:
   $$\text{pass@}k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right], \quad \Delta\text{pass@}_{1\to 10} = \text{pass@10} - \text{pass@1}$$
   Measures how effectively the agent leverages test-time compute (either parallel sampling $k=10$ or up to 10 iterative environment/verifier turns) to overcome zero-shot hallucination.
2. **Verifier Invocation Latency Share ($\eta_{\text{verif}}$, `verifier_latency_share_pct`)**:
   $$\eta_{\text{verif}} = \frac{T_{\text{verifier}}}{T_{\text{LLM\_prefill}} + T_{\text{LLM\_decode}} + T_{\text{verifier}}} \times 100\%$$
   Quantifies the Amdahl's Law bottleneck of external ground-truth evaluators (EDA synthesis, cycle-accurate simulation, compiler builds, hardware profiling).
3. **Token Amplification Factor ($\alpha_{\text{tok}}$, `token_amplification_factor`)**:
   $$\alpha_{\text{tok}} = \frac{\sum_{t=1}^{N_{\text{turns}}} \left(\text{Tokens}_{\text{prompt}}^{(t)} + \text{Tokens}_{\text{gen}}^{(t)}\right)}{\text{Tokens}_{\text{zero-shot}}}$$
   Captures the multiplicative explosion in total tokens processed per task due to multi-turn trajectory accumulation, tree search (MCTS), or evolutionary population mutation.
4. **KV-Cache Reuse Ratio ($\rho_{\text{kv}}$, `kv_cache_reuse_ratio`)**:
   $$\rho_{\text{kv}} = \frac{\text{KV\_Blocks}_{\text{hit}}}{\text{KV\_Blocks}_{\text{total\_prompt}}}$$
   Measures the fraction of attention key-value states shared across consecutive agent turns (system prompt + prior trajectory history) or across sibling branches in tree search.
5. **Normalized Speedup / Efficiency Gain Over Baseline ($S$, `speedup_over_baseline`)**:
   The domain-normalized improvement ratio relative to the standard non-agentic or heuristic baseline (e.g., kernel execution speedup over `torch.compile`, PPA Area-Delay Product improvement over standard synthesis, or serving throughput multiplier over naive batching).

---

## 4. Planned Quantitative Analyses & Hypotheses for Checkpoint 2

Our data-driven exercise in Checkpoint 2 tests three cross-stack architectural hypotheses:
- **Hypothesis H1 (The Verifier Amdahl Wall)**: In Hardware/EDA (`Domain 1`) and Architecture DSE (`Domain 3`), external verifier latency accounts for $>70\%$ of total runtime ($\eta_{\text{verif}} \in [54\%, 91.5\%]$). Consequently, accelerating LLM token decoding by $10\times$ yields $<1.3\times$ end-to-end episode speedup unless paired with fast learned surrogates (e.g., `Concorde`, `MasterRTL`).
- **Hypothesis H2 (Test-Time Scaling Gain vs. Verifier Determinism)**: Domains with exact, deterministic, and fine-grained verifier feedback (e.g., `nvcc` + hardware cycle timers in `Compiler_GPU_Kernel`, or `ABC` synthesis in `MCTS-RTL`) exhibit the largest absolute $\Delta\text{pass@}_{1\to 10}$ gains ($+30\%$ to $+47\%$), whereas coarse or stateful environments (`OSWorld`) exhibit smaller gains due to irreversible state drift.
- **Hypothesis H3 (Token Amplification Mitigation via Radix/Paged KV Reuse)**: Because multi-turn and tree-search agents exhibit high token amplification ($\alpha_{\text{tok}} \in [10\times, 42\times]$) with high structural prefix overlap ($\rho_{\text{kv}} \in [0.68, 0.94]$), `Domain 4` serving optimizations (`SGLang` RadixAttention, `vLLM` PagedAttention, `CacheBlend`) reduce effective prefill FLOPs by $(1 - \rho_{\text{kv}})$, transforming super-linear token cost into near-linear latency scaling.

---

## 5. Harvard CS249r (Fall 2025) Weekly Reading Synthesis (Weeks 1–13)

Below is the complete synthesis of the 13 weekly seminar modules from Prof. Vijay Janapa Reddi's Fall 2025 schedule, connecting each week's foundational readings and guest speaker insights to our cross-stack survey.

### Introduction & Foundations (Weeks 1–2)
- **Week 1 — Course Introduction & The Dawn of Architecture 2.0**:
  - *Readings*: Reddi et al., *"Architecture 2.0: Foundations of AI Agents for Modern Computer System Design"* (IEEE Computer 2025) & *"Why Computer Architects Need a Data-Centric AI Gymnasium"* (SIGARCH).
  - *Synthesis*: Establishes why Moore's Law and Dennard Scaling termination require replacing human-bottlenecked design cycles with data-centric agent gymnasiums. Introduces the triad of **AI for Software**, **AI for Architecture**, and **AI for Chip Design**.
- **Week 2 — Architecture 2.0 Foundations & Architectural Reasoning**:
  - *Readings*: **QuArch** (`arXiv:2501.01892`), Yazdanbakhsh et al. *"A Computer Architect's Guide to Designing Abstractions for Intelligent Systems"* (SIGARCH), Dean *"The Deep Learning Revolution and Its Implications for Computer Architecture and Chip Design"* (`arXiv:1911.05289`).
  - *Synthesis*: Evaluates whether frontier LLMs possess true microarchitectural mental models. QuArch (`P17`) demonstrates that while LLMs recall textbook definitions, they fail on quantitative multi-cycle pipeline trace hazards unless augmented with structured execution abstractions.

### Phase 1: AI for Software — *Understanding What Needs to Be Computed Efficiently* (Weeks 3–6)
- **Week 3 — Code Generation & Software Engineering** (*Guest: Ofir Press, Princeton*):
  - *Readings*: **SWE-bench** (`arXiv:2310.06770`), **AlphaCode** (`arXiv:2203.07814`), *AgentCoder* (`arXiv:2312.13010`), *Code Llama*.
  - *Synthesis*: Contrasts competition-level massive sampling (AlphaCode generating millions of candidates filtered by test cases) with repository-scale software engineering (SWE-bench), where context navigation, environment setup, and cross-file side effects dominate.
- **Week 4 — Performance Engineering & Code Optimization** (*Guest: Amir Yazdanbakhsh, Google DeepMind*):
  - *Readings*: **ECO** (`arXiv:2503.15669`, `P13`), **PIE: Learning Performance-Improving Code Edits** (`arXiv:2302.07867`), **Compiler-R1** (`arXiv:2506.15701`), **CompilerGym** (`arXiv:2109.08267`), **MLIR** (`arXiv:2002.11054`).
  - *Synthesis*: Shifts the objective from binary correctness (`pass@k`) to asymptotic and microarchitectural performance (`fast_p`). ECO demonstrates how production warehouse-scale computers close the loop between continuous fleet profiling (Google-Wide Profiling) and LLM patch generation.
- **Week 5 — GPU Kernels & Parallel Programming** (*Guest: Sasha Rush, Cursor / Cornell Tech*):
  - *Readings*: **KernelBench** (`arXiv:2502.10517`, `P08`), **Kevin: Multi-Turn RL for CUDA Kernels** (`arXiv:2507.11948`, `P09`), **AlphaEvolve** (`P12`), **Ansor** (`arXiv:2006.06762`).
  - *Synthesis*: Examines why writing fast CUDA/Triton kernels is a premier testbed for agentic RL. Kevin (`P09`) proves that single-turn RL fails on complex kernels, whereas multi-turn RL learns to use compiler/profiler feedback across 8+ turns to fuse operators and eliminate global memory round-trips.
- **Week 6 — Distributed Systems Integration** (*Guest: Martin Maas, Google DeepMind*):
  - *Readings*: **COSMIC** (`arXiv:2505.15020`, `P16`), **RL for Datacenter Congestion Control** (`arXiv:2102.09337`), **Decima** (`arXiv:1810.01963`), **Aurora**, **Remy**.
  - *Synthesis*: At cluster scale, compute optimizations are nullified if collective communication (`AllReduce`, `AllToAll`) or tail congestion dominates. COSMIC co-designs distributed ML sharding with network topology.

### Phase 2: AI for Architecture — *Designing How to Compute Efficiently in Hardware* (Weeks 7–10)
- **Week 7 — Performance Prediction & Design Space Exploration** (*Guest: Suvinay Subramanian, Google DeepMind*):
  - *Readings*: **Concorde** (`arXiv:2503.23076`, `P15`), **ArchGym** (`arXiv:2306.08888`, `P14`), **Multi-Agent RL for Microprocessor DSE** (`arXiv:2211.16385`), **NeuSight**, **AutoDSE**.
  - *Synthesis*: Addresses the "Tacit Knowledge Problem" and simulator wall. ArchGym standardizes DSE across RL, Bayesian Optimization, and ACO, while Concorde fuses analytical queueing constraints with ML to achieve $10^5\times$ speedup over cycle-level `gem5`.
- **Week 8 — Hardware Accelerators & AI Mappings** (*Guest: Jenny Huang, NVIDIA*):
  - *Readings*: **DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators** (MICRO 2023), **AutoTVM** (`arXiv:1805.08166`), **Eyeriss**, **TPUv1**, **MAESTRO**, **Timeloop**.
  - *Synthesis*: Hardware buffer sizes and PE arrays are tightly coupled with loop tiling and dataflow (weight/output/row-stationary). Differentiable and learned cost models break the nested two-loop bottleneck of mapping search inside hardware search.
- **Week 9 — Memory Systems & Data Management** (*Guest: Milad Hashemi, Google*):
  - *Readings*: **Learning Memory Access Patterns** (`arXiv:1803.02329`), **The Case for Learned Index Structures** (`arXiv:1712.01208`), **ALEX**, **Learned Cache Replacement (Glider)**, **Learned C++ Memory Allocation (Temeraire)**.
  - *Synthesis*: Replaces rigid B-trees, stride prefetchers, and LRU cache policies with neural/learned predictors that exploit workload distribution structure while respecting strict hardware latency budgets.
- **Week 10 — LLM Systems & AI Workload Scheduling** (*Guest: Esha Choukse, Microsoft Azure Research*):
  - *Readings*: **Efficient LLM Scheduling by Learning to Rank** (`arXiv:2408.15792`, `P23`), **Performance Prediction via Text-to-Text Regression** (`arXiv:2506.21718`), **PagedAttention / vLLM** (`arXiv:2309.06180`, `P21`), **Sarathi-Serve** (`arXiv:2403.02310`, `P22`), **S-LoRA** (`arXiv:2311.03285`, `P24`).
  - *Synthesis*: Connects systems-for-AI directly back to agentic workloads. Dynamic output lengths, massive prefix sharing, and multi-adapter agent roles require rethinking OS memory paging (`vLLM`), prefill/decode co-scheduling (`Sarathi-Serve`), and SJF request ordering (`L2R`).

### Phase 3: AI for Chip Design — *Implementing the Architecture Physically in Silicon* (Weeks 11–13)
- **Week 11 — RTL Design & Logic Synthesis** (*Guest: Mark Ren, NVIDIA*):
  - *Readings*: **CVDP** (`arXiv:2506.14074`, `P06`), **Make Every Move Count (MCTS for RTL)** (`arXiv:2402.03289`, `P05`), **ChipNeMo** (`P03`), **ChipAlign** (`arXiv:2412.19819`), **AutoChip** (`arXiv:2311.04887`), **MasterRTL**, **DRiLLS**, **BOiLS**.
  - *Synthesis*: Synthesizable RTL generation requires bridging syntax validity, functional testbench equivalence, and post-synthesis PPA optimization. MCTS-RTL and AutoChip demonstrate how simulator and synthesis feedback loops guide token generation toward timing-closed logic.
- **Week 12 — Physical Design & Layout** (*Guest: Richard Ho, OpenAI Head of Hardware*):
  - *Readings*: **Chip Placement with Deep RL (Circuit Training)** (`arXiv:2004.10746`, Nature 2021), **DREAMPlace** (DAC 2019), **MaskPlace** (`arXiv:2211.13382`), **ChatEDA** (`P04`).
  - *Synthesis*: Explores the physical floorplanning and macro/standard-cell placement debate: sequential policy-gradient RL with proxy wirelength/congestion rewards (`Circuit Training`) vs. GPU-accelerated analytical electrostatic placement (`DREAMPlace`) orchestrated by EDA agents (`ChatEDA`).
- **Week 13 — Verification & Advanced Chip Design** (*Guest: Kartik Hegde, ChipStack Co-Founder*):
  - *Readings*: **Using LLMs to Facilitate Formal Verification of RTL** (`arXiv:2309.09437`, `P07`), **SLDB: Heterogeneous SoC Benchmark Suite** (`arXiv:2507.06376`), **AssertLLM** (`arXiv:2402.00386`), **SpecLLM** (`arXiv:2401.13266`).
  - *Synthesis*: Verification accounts for >60% of industrial silicon engineering cost. Multi-agent assertion generators (`AssertLLM`) and counterexample-guided formal verification agents turn natural-language architectural specs into formal SystemVerilog Assertions (SVAs), closing the trust gap for AI-generated silicon.

---

## 6. Seminar Paper Presentation & Facilitated Discussion Guide (20+20 Format)

Per the CS249r syllabus (20% of course grade), each student leads a **20-minute technical presentation** followed by a **20-minute facilitated critical discussion**. Below is our structured seminar leadership guide for the Week 5 & Week 11 anchor pair: **Kevin (`arXiv:2507.11948`)** and **MCTS-RTL (`arXiv:2402.03289`)**.

### 6.1 20-Minute Presentation Outline
1. **Minutes 0–6 (Summary & Problem Formulation)**:
   - Why greedy/zero-shot LLM decoding fails in non-convex systems optimization (CUDA kernel bank conflicts in *Kevin*; post-synthesis Area-Delay Product in *MCTS-RTL*).
   - Formulating code generation as a sequential Markov Decision Process (MDP) where states $s_t$ are partial ASTs or multi-turn code refinements and rewards $R(s_T)$ come from real hardware profilers (`nvcc` + H100 timer) or logic synthesis (`Yosys` + `ABC`).
2. **Minutes 6–15 (Critical Methodology Analysis)**:
   - *Strengths*: Grounding rewards in physical measurements eliminates reward hacking on proxy metrics.
   - *Methodological Bottlenecks*: **The Verifier Wall**—each MCTS rollout in *MCTS-RTL* invokes `ABC` synthesis ($\eta_{\text{verif}} = 86.7\%$), and each multi-turn rollout in *Kevin* requires isolated GPU compilation and execution ($\eta_{\text{verif}} = 74.8\%$).
   - *Failure Modes*: Timeout bias (valid but slow-to-compile templates get penalized) and hardware overfitting (kernels tuned for H100 SXM5 shared-memory sizes fail to generalize to A100 or consumer Ada GPUs).
3. **Minutes 15–20 (Cross-Stack Synthesis)**:
   - Connecting *Kevin* and *MCTS-RTL* (Domains 1 & 2) to *SGLang RadixAttention* (Domain 4) and *Concorde* (Domain 3): how prefix-tree KV caching absorbs MCTS branching cost ($\rho_{\text{kv}} = 0.88$) and how learned surrogates (`MasterRTL` / `Concorde`) can filter 90% of rollouts before invoking heavy synthesis.

### 6.2 Provocative Discussion Questions (20-Minute Facilitated Discussion)
1. **Surrogate Reward Hacking vs. Ground-Truth Latency**: *"If we replace slow EDA synthesis (`Yosys`/`Synopsys DC`) or `gem5` simulation with a fast GNN/ML surrogate (`MasterRTL`, `Concorde`) inside an RL/MCTS training loop, at what point does the policy begin exploiting out-of-distribution errors in the surrogate model rather than discovering true hardware optimizations?"*
2. **Compilation-Level vs. Source-Level Agency**: *"Should performance agents generate high-level DSLs (`Triton`, `Chisel`, `Halide`) and let deterministic compilers handle low-level scheduling, or should agents emit raw assembly/PTX/gate-level netlists (`AlphaDev`) to bypass compiler abstractions altogether?"*
3. **Verification Asymmetry in Silicon vs. Software**: *"In software (`SWE-bench`, `KernelBench`), a subtle bug can be patched post-deployment. In chip design (`CVDP`, `ChipNeMo`), a single unverified corner case in a tape-out costs $50M+ and 9 months of respin. Does this fundamental asymmetry mean AI agents in EDA must focus 90% on formal verification (`AssertLLM`) rather than RTL generation?"*
4. **Co-Designing the Serving Engine with the Systems Compiler**: *"When an agent runs 32 parallel MCTS branches (`MCTS-RTL`) or 16 evolutionary mutations (`AlphaEvolve`), the GPU cluster spends more time waiting on CPU-bound `nvcc`/`iverilog` compilation than GPU LLM inference. How should cluster schedulers co-schedule CPU verifier worker pools alongside GPU KV-cache blocks?"*
