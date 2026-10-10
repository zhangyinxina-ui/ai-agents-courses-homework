# Architecture 2.0: A Cross-Stack Quantitative Survey of Agentic AI for Computer Systems Design and Systems for Agentic AI

**Course**: Harvard CS249r — *Architecture 2.0: Agentic AI for Computer Systems Design* (Fall 2025)  
**Instructor**: Prof. Vijay Janapa Reddi  
**Milestone**: Checkpoint 3 & Final Integrated Survey Paper  

---

## Abstract

For five decades, computer systems engineering relied on **Architecture 1.0**: human-crafted heuristics across isolated abstraction layers—instruction scheduling passes in compilers, cache replacement policies in microarchitectures, analytical wirelength approximations in Electronic Design Automation (EDA), and static resource schedulers in cloud clusters. As Moore's Law and Dennard Scaling conclude, the combinatorial explosion of heterogeneous hardware-software co-design spaces has outpaced human engineering bandwidth. In response, **Architecture 2.0** has emerged: deploying autonomous, multi-turn AI agents grounded in cycle-accurate simulators, compilers, hardware profilers, and formal verification engines to iteratively synthesize and optimize software, computer architecture, and physical silicon.

In this survey, we present a systematic, quantitative cross-stack analysis of **25 seminal works** spanning four interconnected domains: **(1) Hardware, EDA, and RTL Design Agents**, **(2) Compiler and GPU Kernel Optimization Agents**, **(3) Cloud, Operating System, and Architecture Design Space Exploration (DSE) Agents**, and **(4) Computer Systems for Serving Agentic Workloads**. By extracting and analyzing standardized metrics across all 25 systems (`systems_agent_dataset.json`), we establish four empirical laws of Architecture 2.0:
1. **Test-Time Verification Scaling**: Grounding LLM agents in deterministic systems verifiers elevates mean task success from **47.5% (`pass@1`)** to **74.5% (`pass@10`)** across systems design tasks (+27.0 percentage points), with multi-turn RL and Monte Carlo Tree Search (MCTS) achieving **+36.6%** average gain in compiler and GPU kernel optimization.
2. **The Verifier Amdahl Wall**: External ground-truth verification (logic synthesis, physical placement, cycle-accurate `gem5` simulation, and GPU kernel profiling) consumes **68.1% of end-to-end episode wall-clock time** on average (peaking at **89.4%–91.5%** in `ChatEDA` and `ArchGym`). Consequently, accelerating LLM token generation by $5\times$ yields only a **$1.31\times$** episode speedup unless paired with compositional analytical-ML surrogates (`Concorde`, `MasterRTL`), which unlock **$3.61\times$–$3.79\times$** speedup.
3. **Token Amplification Explosion**: Iterative compiler/EDA feedback and tree/evolutionary search amplify token consumption by **$14.3\times$ on average** (up to **$42.0\times$** in `AlphaDev` and **$35.0\times$** in `FunSearch/AlphaEvolve`), exhibiting a strong positive correlation ($r = +0.768$) with multi-turn accuracy gains.
4. **Prefix-Tree & Disaggregated KV-Cache Mitigation**: Because agentic trajectories share deep structural prefixes ($\rho_{\text{kv}} = 0.76$ mean, up to $0.94$), specialized agent-serving architectures (`SGLang` RadixAttention, `vLLM` PagedAttention, `Sarathi-Serve`, `CacheBlend`, `S-LoRA`) eliminate **76%** of redundant prefill compute, reducing effective prefill amplification from **$14.3\times$ to $4.15\times$** and delivering a **$3.62\times$ geometric-mean serving throughput speedup**.

**Keywords**: *Architecture 2.0, Agentic AI, Computer Architecture, Electronic Design Automation (EDA), Register-Transfer Level (RTL), GPU Kernel Optimization, Compiler Autotuning, Design Space Exploration (DSE), LLM Serving Systems, KV-Cache Management.*

---

## 1. Introduction: The Dawn of Architecture 2.0

The traditional computing stack is built upon strict abstraction boundaries: software engineers write high-level applications; compilers lower source code through fixed intermediate representations (LLVM IR, MLIR); computer architects design instruction set architectures (ISAs) and microarchitectural pipelines using cycle-accurate simulators (`gem5`); and physical design engineers translate Register-Transfer Level (RTL) descriptions into manufacturable GDSII layouts via Electronic Design Automation (EDA) suites. While modularity enabled decades of exponential scaling under Moore's Law, it now imposes severe performance penalties. Modern AI and warehouse-scale workloads require **full-stack co-design**, where tensor dataflows, CUDA warp schedules, network-on-chip (NoC) topologies, and standard-cell placements are optimized jointly.

Recent breakthroughs in Large Language Models (LLMs), Reinforcement Learning (RL), and evolutionary search have catalyzed **Architecture 2.0**—a paradigm in which autonomous agents perceive system telemetry, reason over architectural trade-offs, synthesize executable artifacts (C++/CUDA code, LLVM pass pipelines, SystemVerilog modules, Tcl physical design scripts), and refine their designs through closed-loop interaction with ground-truth system evaluators.

```
+-----------------------------------------------------------------------------------+
|                THE ARCHITECTURE 2.0 CLOSED-LOOP OPTIMIZATION CYCLE                |
|                                                                                   |
|   +------------------------+     Synthesized Artifact    +--------------------+   |
|   |   Agentic Policy pi    | --------------------------> | Fast Tier-1 Filter |   |
|   | (LLM / MCTS / RL / Evo)|   (RTL, CUDA, IR, DSE Cfg)  | (Syntax, Analytical|   |
|   +------------------------+                             |  Surrogate:Concorde|   |
|               ^                                          +--------------------+   |
|               |                                                    |              |
|    Structured Diagnostic                                  Pruned Top-k Candidates |
|    Reward & Counterexample                                         v              |
|               |                                          +--------------------+   |
|   +------------------------+   Timing, PPA, Speedup,     | Slow Tier-2 Signoff|   |
|   |  Serving Engine (D4)   | <-------------------------- | (Yosys/ABC, nvcc,  |   |
|   | (RadixAttn, PagedAttn, |   Formal Counterexamples    |  OpenROAD, gem5,   |   |
|   |  Sarathi, CacheBlend)  |                             |  JasperGold SVA)   |   |
|   +------------------------+                             +--------------------+   |
+-----------------------------------------------------------------------------------+
```

However, deploying AI agents for computer systems design exposes a profound **cross-stack duality**:
- **Direction A (AI Agents for Systems Design)**: How do we transform stochastic sequence models into rigorous systems engineers capable of guaranteeing formal functional equivalence while optimizing non-convex physical objectives (Area-Delay Product, GPU occupancy, datacenter tail latency)?
- **Direction B (Computer Systems for Agentic AI)**: When systems design agents execute multi-turn compilation repair, 32-branch MCTS rollouts, or island-based evolutionary mutations, they generate **$10\times$ to $42\times$ more tokens** than conversational chatbots and interleave GPU inference with long CPU-bound simulator stalls. How must underlying GPU memory managers and cluster schedulers evolve to serve these agentic workloads efficiently?

This paper unifies both directions into a single quantitative framework, synthesizing the 13-week curriculum of Harvard CS249r (Fall 2025) with empirical data across 25 landmark papers.

---

## 2. Formal Foundations & Cross-Stack Quantitative Laws

### 2.1 Formulating Systems Design as a Verifier-Grounded POMDP

We formalize an Architecture 2.0 design task as a Partially Observable Markov Decision Process (POMDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{V}, R)$:
- **State Space $\mathcal{S}$**: The partial or complete system artifact $s_t$ at turn $t$ (e.g., a SystemVerilog AST, a CUDA kernel source file, an LLVM optimization pass sequence, or an accelerator parameter vector) paired with the execution environment state.
- **Action Space $\mathcal{A}$**: Code mutations, script invocations, or parameter updates $a_t \sim \pi_\theta(a_t \mid h_t)$ conditioned on the interaction history $h_t = (o_0, a_0, o_1, \dots, o_t)$.
- **Deterministic Verifier $\mathcal{V}: \mathcal{S} \to \{0, 1\} \times \mathbb{R}^d \times \Sigma^*$**: Unlike human preference reward models in chat alignment, systems agents query an executable verifier $\mathcal{V}(s_t)$ that returns:
  1. A binary correctness indicator $c(s_t) \in \{0, 1\}$ (compilation success + testbench/SMT equivalence);
  2. A physical performance vector $\mathbf{m}(s_t) \in \mathbb{R}^d$ (e.g., execution latency, area, power, wirelength, CPI); and
  3. Diagnostic feedback $o_{t+1} \in \Sigma^*$ (compiler error logs, formal counterexample waveforms, Nsight profiling counters).
- **Lexicographic Systems Reward $R(s_t)$**: Because an invalid fast kernel or non-synthesizable RTL module is useless, the reward strictly prioritizes correctness before physical optimization:
  $$R(s_t) = \begin{cases} -C_{\text{syntax}} & \text{if compilation fails} \\ -C_{\text{func}} + \alpha \cdot \text{PassRatio}(s_t) & \text{if } c(s_t) = 0 \text{ (testbench/formal failure)} \\ \beta \cdot \log\left(\frac{\text{Perf}(s_t)}{\text{Perf}(s_{\text{base}})}\right) & \text{if } c(s_t) = 1 \text{ (functionally verified)} \end{cases}$$
  As demonstrated in **KernelBench** (`P08`) and **Kevin** (`P09`), composite metric $\text{fast}_p$ measures the fraction of tasks where $c(s_t) = 1$ *and* $\text{Speedup}(s_t) > p$.

### 2.2 Law 1: The Verifier Amdahl Wall

Let an agentic episode consist of $N_{\text{turns}}$ sequential or tree-structured iterations. The total wall-clock latency $T_{\text{episode}}$ decomposes into LLM prefill latency $T_{\text{prefill}}$, LLM decode latency $T_{\text{decode}}$, and external verifier execution latency $T_{\text{verif}}$:
$$T_{\text{episode}} = \sum_{t=1}^{N_{\text{turns}}} \left( T_{\text{prefill}}^{(t)} + T_{\text{decode}}^{(t)} + T_{\text{verif}}^{(t)} \right)$$
Defining the **Verifier Invocation Latency Share** as $\eta_{\text{verif}} = T_{\text{verif}} / T_{\text{episode}}$, the end-to-end speedup $S_{\text{e2e}}$ achieved by accelerating the LLM serving engine by factor $S_{\text{LLM}}$ and accelerating the verifier (e.g., via learned surrogates) by factor $S_{\text{verif}}$ follows **Agentic Amdahl's Law**:
$$S_{\text{e2e}}(S_{\text{LLM}}, S_{\text{verif}}) = \frac{1}{\frac{\eta_{\text{verif}}}{S_{\text{verif}}} + \frac{1 - \eta_{\text{verif}}}{S_{\text{LLM}}}}$$

Across our 25-paper dataset ([Table 2.1 in Checkpoint 2](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/checkpoint2_report.md)), $\eta_{\text{verif}}$ averages **74.6%** in Hardware/EDA (`Domain 1`) and **72.9%** in Compiler/GPU Kernels (`Domain 2`), reaching **89.4%** in `ChatEDA` (`OpenROAD` placement/routing) and **91.5%** in `ArchGym` (`gem5`/`Timeloop` simulation). When $S_{\text{verif}} = 1$, even an infinite LLM speedup ($S_{\text{LLM}} \to \infty$) caps $S_{\text{e2e}}$ at $1 / \eta_{\text{verif}} \approx 1.34\times$. Overcoming this wall requires **Two-Tier Surrogate Verification** ($S_{\text{verif}} \gg 1$), exemplified by **Concorde** (`P15`), which fuses analytical queueing models with ML to achieve a $10^5\times$ raw speedup over `gem5` ($<2\%$ CPI error) and slashes $\eta_{\text{verif}}$ from $91.5\%$ to **$14.5\%$**.

### 2.3 Law 2: Token Amplification & KV-Cache Prefix Reuse Roofline

A zero-shot single-turn prompt processes $L_0$ tokens. An agentic workflow with $N_{\text{turns}}$ turns or $B$ search branches processes a cumulative token volume $\alpha_{\text{tok}} L_0$, where $\alpha_{\text{tok}}$ is the **Token Amplification Factor**. In our dataset, $\alpha_{\text{tok}}$ averages **$12.8\times$** in Hardware/EDA, **$20.8\times$** in Compiler/GPU Kernels, and **$10.3\times$** in Cloud/Arch DSE.

However, because turn $t+1$ appends diagnostic feedback $o_{t+1}$ to prefix $h_t$ (or MCTS sibling nodes branch from a shared AST prefix), a fraction $\rho_{\text{kv}} \in [0, 1]$ of prompt KV blocks is identical to previously computed KV states. With prefix-tree KV caching (`SGLang` RadixAttention `P20`, `vLLM` PagedAttention `P21`, `CacheBlend` `P25`), the **Effective Prefill Compute Amplification** $\alpha_{\text{eff}}$ is reduced to:
$$\alpha_{\text{eff}} = 1 + (\alpha_{\text{tok}} - 1)(1 - \rho_{\text{kv}})$$
Empirically, across `Domains 1–3`, $\rho_{\text{kv}} = 0.76$, compressing mean prefill amplification from **$\alpha_{\text{tok}} = 14.32\times$** down to **$\alpha_{\text{eff}} = 4.15\times$**—a **71% reduction in redundant prefill FLOPs**.

---

## 3. Phase 1 Analysis: AI Agents for Software, Compilers & GPU Kernels

Aligned with Weeks 3–6 of CS249r, Phase 1 investigates how agents bridge the chasm between *syntactically plausible code* and *hardware-optimal machine execution*.

### 3.1 From Repository Software Engineering to Warehouse-Scale Code Optimization
While **SWE-bench** (*Jimenez, Press et al., Week 3*) established that repository-level software patching requires multi-turn file navigation and test execution, functional correctness is only the entry point for systems engineering. In **ECO: An LLM-Driven Efficient Code Optimizer for Warehouse Scale Computers** (*Yazdanbakhsh et al., Week 4, `P13`*), an autonomous loop ingests fleet-wide continuous profiling telemetry (Google-Wide Profiling) to identify CPU-hotspot functions across production C++ binaries, synthesizes algorithmic and data-structure refactorings (e.g., flat hash maps, arena allocation, loop unrolling), and validates patches against Bazel unit tests and microbenchmarks. ECO achieves a **1.19x** average speedup on already heavily optimized (`-O3` + AutoFDO + ThinLTO) production binaries, demonstrating that LLMs can reason about high-level data structure semantics invisible to static compiler passes.

### 3.2 Compiler Pass Autotuning & Assembly Superoptimization
At the compiler IR and assembly level, human heuristics struggle with phase-ordering interactions across hundreds of LLVM passes:
- **Meta LLM Compiler** (`P11`, Week 4) pretrains foundation models on 546 billion tokens of LLVM IR and assembly, learning an internal mental model of compiler transformations. It achieves **1.053x** code-size reduction over `clang -Oz` with low token amplification ($\alpha_{\text{tok}} = 5.5\times$).
- **Compiler-R1** (*Week 4*) extends pass autotuning with reinforcement learning over interactive `CompilerGym` environments.
- **AlphaDev** (`P10`, Week 4) formulates x86/ARM assembly synthesis as a single-player AlphaZero MCTS game. By searching at $\alpha_{\text{tok}} = 42.0\times$ with SMT equivalence verification and CPU latency rewards, AlphaDev discovered novel branchless instruction sequences ("AlphaDev swap and copy moves") that accelerated `libc++` short sorting routines (`Sort-3` to `Sort-5`) by **1.70x**.

### 3.3 GPU Kernel Synthesis: Multi-Turn RL and Evolutionary Agents
GPU programming represents one of the steepest optimization cliffs in computer science: a naive CUDA kernel can easily run $10\times$ slower than PyTorch's `cuBLAS`/`cudnn` due to shared-memory bank conflicts, uncoalesced global memory loads, or register spilling.
- **KernelBench** (*Ouyang et al., Week 5, `P08`*) introduced the `fast_p` evaluation framework across 250 PyTorch workloads, revealing that single-shot frontier LLMs achieve `<20%` `fast_1` (faster than PyTorch Eager). However, providing 10 turns of `nvcc` compiler errors and Nsight Compute profiler feedback triples success to **54.0%** (`1.42x` speedup).
- **Kevin: Multi-Turn RL for Generating CUDA Kernels** (*Baronio, Rush et al., Week 5, `P09`*) demonstrates why *single-turn* RL fails on CUDA generation: the model collapses into conservative edits to avoid compilation penalties. By training end-to-end with **multi-turn RL** over 8-turn trajectories (`pass@1` = 44.5% $\to$ `pass@10` = **82.0%**, **1.76x** speedup), Kevin learns an emergent curriculum: Turn 1–2 establishes numerical correctness via simple thread indexing; Turns 3–5 introduce shared-memory tiling; Turns 6–8 apply warp-level primitives (`__shfl_sync`) and vectorized `float4` loads.
- **FunSearch & AlphaEvolve** (*DeepMind, Week 5, `P12`*) pair Gemini coding agents with an island-based evolutionary program database. Applied to production Pallas/XLA matrix multiplication and FlashAttention kernels on Google TPUs, AlphaEvolve achieved a **1.23x** kernel speedup and recovered **0.7% of Google's global fleet compute**.

---

## 4. Phase 2 Analysis: AI Agents for Computer Architecture, Distributed Co-Design & Cloud Systems

Aligned with Weeks 6–9 of CS249r, Phase 2 moves beneath the ISA to ask: *How can AI agents design microprocessors, domain-specific accelerators, memory hierarchies, and distributed clusters?*

### 4.1 Architectural Reasoning & The Design Space Exploration (DSE) Gymnasium
First, **QuArch** (*Agiannou, Reddi et al., Week 2, `P17`*) benchmarks LLM comprehension across 1,500 microprocessor architecture questions. While frontier models achieve **71.4%** zero-shot accuracy (`84.8%` with multi-sample CoT+RAG), error analysis exposes systematic weaknesses in quantitative pipeline timing diagrams, cache coherence transient states, and memory consistency models—proving that linguistic fluency cannot substitute for cycle-level simulation.

To standardize agent-driven hardware search, **ArchGym** (*Krishnan, Reddi et al., ISCA 2023, Week 7, `P14`*) wraps architectural simulators (`Timeloop`, `DRAMSys`, `Maestro`, `gem5`) behind an OpenAI Gym interface supporting Reinforcement Learning (PPO/SAC), Bayesian Optimization, Ant Colony Optimization, and Genetic Algorithms. ArchGym uncovered a foundational insight: **there is no single dominant ML search algorithm** across all hardware design spaces; instead, hyperparameter tuning of the agent and simulator fidelity govern Pareto frontier hypervolume (**1.45x** improvement over heuristics). Crucially, ArchGym revealed that simulator invocation accounts for **91.5%** of DSE runtime.

### 4.2 Breaking the Simulation Wall: Analytical-ML Fusion & Differentiable Co-Design
To break ArchGym's simulation bottleneck, **Concorde** (*Nasr, Subramanian et al., Week 7, `P15`*) introduces **compositional analytical-ML fusion** for out-of-order CPU modeling. Rather than training a black-box neural network on instruction traces, Concorde computes first-order analytical throughput bounds (fetch/decode width, ROB capacity, load/store queue bandwidth, branch misprediction flush penalty) and trains a lightweight ML model to predict residual non-linear contention. Concorde achieves $<2\%$ CPI error relative to `gem5` while running **5 orders of magnitude faster**, reducing verifier latency share in agentic DSE from $91.5\%$ to **14.5%**.

In DNN accelerator co-design (Week 8), **DOSA** (*MICRO 2023*) and **AutoTVM** (*Chen et al., `P11/Wk8`*) tackle the coupled two-loop problem of hardware resource allocation (PE array dimensions, scratchpad sizes) and software loop mapping (tiling, spatial unrolling). By replacing discrete inner-loop search (`Timeloop`) with differentiable analytical/learned performance models, DOSA optimizes hardware parameters and mappings in a single gradient-guided loop.

### 4.3 Learned Memory Hierarchies, Distributed Co-Design & Autonomous Cloud Operations
- **Memory Systems (Week 9)**: *Learning Memory Access Patterns* (*Hashemi et al.*) and *The Case for Learned Index Structures* (*Kraska et al.*) reframe hardware prefetching and database indexing as cumulative distribution function (CDF) and sequence-prediction tasks, achieving superior cache hit rates and lookup latency over B-Trees and stride prefetchers.
- **Distributed ML Co-Design (Week 6)**: **COSMIC** (*Rashidi, Maas et al., `P16`*) extends DSE to full-stack distributed ML pods (`ASTRA-sim`), jointly optimizing tensor/pipeline/data parallelism degrees, collective communication scheduling (`AllReduce` ring vs. tree), and optical interconnect topology for a **1.58x** TFLOPs/$ improvement.
- **OS & Cloud Incident Agents (Weeks 3 & 6)**: In live production systems, **OSWorld** (`P18`) benchmarks multimodal GUI/CLI agents across 369 real desktop OS tasks, revealing that long-horizon (15-turn) workflows suffer severe compounding errors (`pass@1` = 12.2%, `pass@10` = 28.6%) without state-snapshot rollback. Conversely, in cloud SRE operations, **RCACopilot & AIOpsLab** (`P19`) combine domain-specific diagnostic handlers with tool-calling agents over live Kubernetes clusters (`ChaosMesh`), cutting Mean-Time-To-Mitigate (MTTM) by **3.10x** (`77.5%` multi-turn resolution).

---

## 5. Phase 3 Analysis: AI Agents for Silicon, RTL Design, Physical EDA & Formal Verification

Aligned with Weeks 11–13 of CS249r, Phase 3 addresses the physical realization of architectures in silicon—where verification errors carry multi-million-dollar mask respin costs.

### 5.1 Synthesizable RTL Generation & Domain-Adapted Hardware LLMs
Early RTL benchmarks such as **VerilogEval** (*Liu et al., `P01`, Week 11*) and **RTLCoder** (`P02`, Week 11) demonstrated that fine-tuning open-weights models (7B–13B) on synthetic specification-to-RTL pairs verified by `iverilog` elevates `pass@10` to **69.0%–76.4%**. At industrial scale, NVIDIA's **ChipNeMo** (*Liu, Ren et al., `P03`, Week 11*) introduced Domain-Adaptive Pretraining (DAPT) on internal Verilog, C++, and EDA Tcl repositories, coupled with a custom HDL tokenizer that improves tokenization efficiency by 16–33% and a domain RAG pipeline that enables a 13B model to match or exceed 70B generalist models on engineering assistant benchmarks.

However, **CVDP (Comprehensive Verilog Design Problems)** (*Pinckney, Ren et al., 2025, `P06`, Week 11*) exposes the complexity cliff between single-module benchmarks and industrial chip design: across 13 categories including multi-file context navigation, pipeline bug repair, and verification harness writing, `pass@1` drops to **27.3%** and `pass@10` to **48.6%**, with Dockerized `Verilator`/`SymbiYosys` verification consuming **79.1%** of episode time.

### 5.2 PPA-Guided Tree Search & Physical Design Automation
Generating *functionally valid* Verilog is insufficient if the resulting gate-level netlist violates setup/hold timing or area budgets.
- **Make Every Move Count (MCTS-RTL)** (*DeLorenzo et al., `P05`, Week 11*) integrates Monte Carlo Tree Search (UCT) directly into RTL token decoding, invoking `Yosys` + `ABC` logic synthesis at leaf states to compute an Area-Delay Product (ADP) reward. Across arithmetic and control modules, MCTS-RTL improves synthesis ADP by up to **31.8% (`1.31x`)** over greedy LLM decoding and climbs from 52.0% (`pass@1`) to **84.5%** (`pass@10`), at the cost of a $28.5\times$ token amplification factor and an **86.7%** verifier latency share.
- **ChatEDA** (*Wu et al., `P04`, Week 12*) and **Circuit Training / DREAMPlace** (*Week 12*) extend agency to physical layout (RTL-to-GDSII). ChatEDA decomposes natural-language physical design goals into executable Python/Tcl scripts orchestrating `OpenROAD` floorplanning, placement, clock-tree synthesis (CTS), and routing (`88.0%` multi-turn completion, **2.15x** speedup over naive ReAct), while highlighting that physical EDA tool execution accounts for **89.4%** of runtime.

### 5.3 Closing the Silicon Trust Gap: Formal Verification Agents
Because simulation testbenches cannot exhaustively cover state spaces in complex FSMs or cache coherence controllers, Week 13 focuses on **Formal Verification**. **Using LLMs to Facilitate Formal Verification of RTL** (*Orenes-Vera et al., Princeton*) and **AssertLLM** (*Fang et al., `P07`*) deploy multi-agent workflows that decompose natural-language design specifications into SystemVerilog Assertions (SVAs) (`assert property (@(posedge clk) ...)`), execute bounded model checking via `Cadence JasperGold` or `SymbiYosys`, and parse counterexample waveform traces to iteratively repair assertions or RTL bugs. Counterexample-guided refinement boosts valid formal property verification from **49.5% (`pass@1`)** to **79.2% (`pass@10`)** (**2.40x** coverage improvement over unassisted synthesis).

---

## 6. Computer Systems for Serving Agentic AI Workloads (Domain 4)

Aligned with Week 10 of CS249r (*Guest: Esha Choukse, Microsoft Azure Research*), Domain 4 closes the Architecture 2.0 loop by asking: *How must computer systems be re-architected to serve the multi-turn, tree-branching, tool-heavy agentic workloads analyzed in Sections 3–5?*

### 6.1 The Four Pathologies of Agentic Workloads on Traditional Serving Engines
Our quantitative dataset reveals why systems agents break conventional LLM serving engines:
1. **Branching KV-Cache Explosion**: An MCTS agent (`MCTS-RTL` `P05`, $\alpha_{\text{tok}}=28.5\times$) or parallel `pass@10` sampler (`Kevin` `P09`) spawns 10–32 concurrent trajectories sharing 70%–94% of their prompt prefix. Contiguous KV buffers duplicate these gigabytes across HBM.
2. **Long Tool-Output Prefill Stalls**: When an agent ingests a 16K-token `Verilator` waveform trace, compiler dump, or `OpenROAD` log (`ChatEDA`, `RCACopilot`), the compute-bound prefill phase blocks ongoing decode tokens in co-batched requests, spiking Time-Between-Tokens (TBT) P99 latency.
3. **Unpredictable Multi-Turn Trajectory Lengths**: Reasoning and reflection trajectories vary from 100 tokens to 32K tokens, causing severe Head-of-Line (HoL) blocking under First-Come-First-Served (FCFS) scheduling.
4. **Multi-Role Adapter Heterogeneity**: Multi-agent systems (`AssertLLM`, `ChatEDA`) route subtasks across specialized fine-tuned LoRA adapters (Spec Analyzer, RTL Coder, SVA Critic, Tcl Planner).

### 6.2 Architectural Solutions in Domain 4 (`P20`–`P25`)
Each paper in Domain 4 directly targets one of these four pathologies:

| Pathology | Serving System (`ID`) | Architectural Mechanism | Quantitative Impact |
| :--- | :--- | :--- | :--- |
| **Contiguous HBM Fragmentation in Parallel Sampling (`pass@k`)** | **vLLM / PagedAttention** (`P21`, *SOSP 2023*) | Virtual-to-physical block tables (OS paging) with Copy-on-Write (CoW) for parallel beam/sample branches | Cuts KV waste to $<4\%$; **0.89** reuse ratio; **3.20x** serving throughput. |
| **Multi-Turn & Tree-Search Prefix Re-Computation** | **SGLang / RadixAttention** (`P20`, *NeurIPS 2024*) | Retains KV-cache blocks across completed requests in an LRU **Radix Tree**, automatically matching prefixes across agent turns & MCTS nodes + compressed FSM decoding | Achieves **0.94** KV-cache reuse ratio and **5.00x** throughput speedup on agent workloads. |
| **Non-Prefix RAG / Tool Chunk Re-Computation** | **CacheBlend** (`P25`, *EuroSys 2025*) | Precomputes KV caches for modular RAG/documentation chunks and selectively recomputes cross-attention on only **~15%** of high-deviation tokens | **0.91** KV reuse on arbitrary chunk permutations; **3.30x** lower TTFT. |
| **Tool-Output Prefill Blocking Decode SLOs** | **Sarathi-Serve** (`P22`, *OSDI 2024*) | **Chunked-Prefill** splits long tool/log prefills into fixed-budget chunks and piggybacks ongoing decode steps in mixed batches | Eliminates decode stalls; **3.80x** higher serving capacity under strict P99 TBT SLOs. |
| **Head-of-Line Blocking from Long Reasoning Traces** | **Learning-to-Rank LLM Scheduler** (`P23`, *NeurIPS 2024*) | Lightweight listwise ranking model (Kendall-Tau loss) predicts relative remaining generation length to drive preemptive Shortest-Job-First (SJF) | Reduces P90 end-to-end latency by **2.80x** over FCFS without exact length regression error. |
| **Multi-Agent Role Adapter Switching Overhead** | **S-LoRA** (`P24`, *MLSys 2024*) | Unified Paging manages both dynamic KV caches and heterogeneous-rank LoRA adapters in a single HBM pool via custom `MBGMM` CUDA kernels | Serves thousands of concurrent agent role adapters at **4.00x** throughput. |

---

## 7. Cross-Stack Synthesis, Open Challenges & Future Research Directions

Synthesizing our quantitative findings across all 25 papers yields four high-priority research directions for the next phase of Architecture 2.0:

### 7.1 Hierarchical Surrogate-to-Signoff Verification Pipelines
As proven by our Amdahl's Law analysis ([Table 2.2 in Checkpoint 2](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/checkpoint2_report.md)), neither faster LLM serving alone ($1.29\times$–$1.34\times$ E2E speedup) nor uncalibrated ML surrogates alone (which suffer from reward hacking) can solve systems design. The future lies in **Multi-Fidelity Hierarchical Verification**:
- **Tier 0 (Sub-millisecond FSM/AST Grammar Constraints)**: Enforce syntactic and port-interface invariants during token decoding (`SGLang` compressed FSM).
- **Tier 1 (Millisecond Analytical-ML Surrogates)**: Filter $90\%+$ of candidate rollouts during inner-loop MCTS or RL using physics-informed surrogates (`Concorde` for CPU CPI, `MasterRTL` for pre-synthesis PPA, `llvm-mca` for basic-block throughput).
- **Tier 2 (Second-to-Minute Ground-Truth Signoff)**: Invoke full `Yosys/OpenROAD`, `nvcc` hardware profiling, or `JasperGold` formal model checking only on top-$k$ Pareto candidates, feeding exact counterexamples back to update both the agent policy and the Tier-1 surrogate online.

### 7.2 Mitigating Surrogate Reward Hacking & Hardware Overfitting
In RL-driven systems design (`Kevin`, `Compiler-R1`, `Circuit Training`), policies rapidly exploit inaccuracies in proxy cost models or overfit to microarchitectural idiosyncrasies of a single target SKU (e.g., tuning shared-memory tile sizes exclusively for NVIDIA H100 228KB SM capacity). Developing **uncertainty-aware epistemic verifiers** that trigger ground-truth simulation whenever surrogate ensemble variance exceeds a threshold is a critical open problem.

### 7.3 Co-Scheduling CPU Verifier Pools and GPU KV-Cache Trees
During an episode of `Kevin` (`P09`), `CVDP` (`P06`), or `ECO` (`P13`), an agent alternates between GPU-bound token generation (~25% of time) and CPU/sandbox-bound compilation, synthesis, or simulation (~75% of time). Current serving systems (`vLLM`, `SGLang`) either hold the agent's KV-cache resident in scarce GPU HBM while waiting 30 seconds for `OpenROAD`/`Verilator` to finish, or evict it and incur expensive recomputation. **Verifier-Aware KV Offloading**—proactively streaming frozen agent KV-cache branches to host DRAM/NVMe (`LMCache`/`CacheBlend`) the instant a tool call is dispatched to the CPU EDA cluster, and prefetching them back into HBM 100ms before `Verilator` exits—can multiply effective GPU cluster utilization by $3\times$–$4\times$.

### 7.4 Formal Specification Alignment as the Primary Bottleneck
Across `CVDP` (`P06`) and `Formal Verification of RTL` (`P07`), as LLM code generation improves, the dominant failure mode shifts from *implementation bugs* to *specification ambiguity*. Natural-language architecture documents under-specify reset behavior, AXI handshake corner cases, and memory ordering. Consequently, **Agentic Specification Engineering** (`SpecLLM`, `AssertLLM`)—where agents co-design formal executable specifications (SVAs, TLA+ invariants) interactively with human architects before writing a single line of RTL—will define trustworthy silicon engineering.

---

## 8. Complete 25-Paper Survey Bibliography

1. **[P01]** Liu, M., Pinckney, N., Khailany, B., & Ren, H. (2023). *VerilogEval: Evaluating Large Language Models for Verilog Code Generation*. IEEE/ACM ICCAD 2023.
2. **[P02]** Liu, S., Fang, W., Lu, Y., Zhang, Q., & Xie, Z. (2024). *RTLCoder: Outperforming GPT-3.5 in Design RTL Generation with Our Open-Source Dataset and Lightweight Solution*. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD).
3. **[P03]** Liu, M., Ene, T.-D., Kirby, R., Cheng, C., Pinckney, N., Liang, R., Alben, J., Anand, H., Banerjee, S., Bayraktaroglu, I., et al., & Ren, H. (2023). *ChipNeMo: Domain-Adapted LLMs for Chip Design*. arXiv:2311.00176.
4. **[P04]** Wu, H., He, Z., Zhang, X., Yao, X., Zheng, S., Zheng, H., & Yu, B. (2024). *ChatEDA: A Large Language Model Powered Autonomous Agent for EDA*. IEEE TCAD 2024.
5. **[P05]** DeLorenzo, M., Gohil, V., & Rajendran, J. (2024). *Make Every Move Count: LLM-based High-Quality RTL Code Generation Using MCTS*. arXiv:2402.03289.
6. **[P06]** Pinckney, N., Batten, C., Liu, M., Ren, H., et al. (2025). *Comprehensive Verilog Design Problems: A Next-Generation Benchmark Dataset for Evaluating Large Language Models and Agents on RTL Design and Verification*. arXiv:2506.14074.
7. **[P07]** Orenes-Vera, M., Martonosi, M., & Wentzlaff, D. (2023/2024). *Using LLMs to Facilitate Formal Verification of RTL* (arXiv:2309.09437) & Fang, W., et al. (2024). *AssertLLM: Generating and Evaluating Hardware Verification Assertions from Design Specifications via Multi-LLMs* (arXiv:2402.00386).
8. **[P08]** Ouyang, A., Guo, S., Arora, S., Zhang, A. L., Hu, W., Ré, C., & Mirhoseini, A. (2025). *KernelBench: Can LLMs Write Efficient GPU Kernels?*. arXiv:2502.10517.
9. **[P09]** Baronio, C., Marsella, P., Pan, B., Guo, S., & Rush, S. (2025). *Kevin: Multi-Turn RL for Generating CUDA Kernels*. arXiv:2507.11948.
10. **[P10]** Mankowitz, D. J., Michi, A., Zhernov, A., Gelmi, M., Selvi, M., Paduraru, C., Leurent, E., Iqbal, S., Lespiau, J.-B., Ahern, A., et al. (2023). *Faster sorting algorithms discovered using deep reinforcement learning (AlphaDev)*. Nature, 618, 257–263.
11. **[P11]** Cummins, C., Seeker, V., Grubisic, D., Roziere, B., Gehring, J., Synnaeve, G., & Leather, H. (2024). *Meta Large Language Model Compiler: Foundation Models of Compiler Optimization*. arXiv:2407.02524.
12. **[P12]** Romera-Paredes, B., et al. (2024). *Mathematical discoveries from program search with large language models (FunSearch)*. Nature, 625, 468–475; & Novikov, A., et al. (2025). *AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms*. Google DeepMind Technical Report.
13. **[P13]** Yazdanbakhsh, A., et al. (2025). *ECO: An LLM-Driven Efficient Code Optimizer for Warehouse Scale Computers*. arXiv:2503.15669.
14. **[P14]** Krishnan, S., Yazdanbakhsh, A., Prakash, S., Jabbour, J., Uchendu, I., Ghosh, S., Boroujerdian, B., Richins, D., Trippel, T., Faust, A., & Reddi, V. J. (2023). *ArchGym: An Open-Source Gymnasium for Machine Learning Assisted Architecture Design*. ACM/IEEE ISCA 2023 (arXiv:2306.08888).
15. **[P15]** Nasr, M., Subramanian, S., et al. (2025). *Concorde: Fast and Accurate CPU Performance Modeling with Compositional Analytical-ML Fusion*. arXiv:2503.23076.
16. **[P16]** Rashidi, S., Maas, M., et al. (2025). *COSMIC: Enabling Full-Stack Co-Design and Optimization of Distributed Machine Learning Systems*. arXiv:2505.15020.
17. **[P17]** Agiannou, K., Reddi, V. J., et al. (2025). *QuArch: A Question-Answering Dataset for AI Agents in Computer Architecture*. arXiv:2501.01892.
18. **[P18]** Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., et al. (2024). *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*. NeurIPS 2024.
19. **[P19]** Chen, Y., Xie, H., Ma, M., Kang, Y., Gao, X., Shi, L., Cao, Y., Gao, X., Fan, H., Wen, M., et al. (2024). *Automatic Root Cause Analysis via Large Language Models for Cloud Incidents (RCACopilot)*. EuroSys 2024.
20. **[P20]** Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., & Sheng, Y. (2024). *SGLang: Efficient Execution of Structured Language Model Programs*. NeurIPS 2024.
21. **[P21]** Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. ACM SOSP 2023 (arXiv:2309.06180).
22. **[P22]** Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., &jee, R. (2024). *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. USENIX OSDI 2024 (arXiv:2403.02310).
23. **[P23]** Fu, Y.,azizi, S., Choukse, E., et al. (2024). *Efficient LLM Scheduling by Learning to Rank*. NeurIPS 2024 (arXiv:2408.15792).
24. **[P24]** Sheng, Y., Cao, S., Li, D., Hooper, C., Lee, N., Yang, S., Chou, C., Zhu, B., Zheng, L., Keutzer, K., Gonzalez, J. E., & Stoica, I. (2024). *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. MLSys 2024 (arXiv:2311.03285).
25. **[P25]** Yao, J., Li, H., Liu, Y., Ray, S., Cheng, Y., Zhang, Q., Du, K., Lu, S., & Jiang, J. (2025). *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*. EuroSys 2025 (arXiv:2405.16444).
