# Course 08: Harvard CS249r — Architecture 2.0: Agentic AI for Computer Systems Design (Fall 2025)

**University**: Harvard University (School of Engineering and Applied Sciences)  
**Instructor**: Prof. Vijay Janapa Reddi  
**Official Course Website**: [https://harvard-edge.github.io/cs249r_fall2025/](https://harvard-edge.github.io/cs249r_fall2025/)  

---

## 1. Course Overview & Deliverable Structure

Harvard **CS249r (Fall 2025)** is a graduate research seminar exploring how autonomous AI agents are transforming full-stack computer systems design across three thematic phases:
1. **Phase 1 — AI for Software (Weeks 3–6)**: Code generation (`SWE-bench`), warehouse-scale code optimization (`ECO`), compiler autotuning (`LLM Compiler`, `Compiler-R1`), GPU kernel synthesis (`KernelBench`, `Kevin`, `AlphaEvolve`), and distributed systems co-design (`COSMIC`).
2. **Phase 2 — AI for Architecture (Weeks 7–10)**: Microarchitectural reasoning (`QuArch`), design space exploration gymnasiums (`ArchGym`), compositional analytical-ML CPU performance modeling (`Concorde`), accelerator dataflow co-design (`DOSA`, `AutoTVM`), learned memory hierarchies, and LLM/agent workload serving systems (`PagedAttention`, `RadixAttention`, `Sarathi-Serve`, `Learning-to-Rank Scheduling`, `S-LoRA`, `CacheBlend`).
3. **Phase 3 — AI for Chip Design (Weeks 11–13)**: Synthesizable RTL generation (`VerilogEval`, `RTLCoder`, `ChipNeMo`, `CVDP`, `MCTS-RTL`), physical layout & EDA automation (`ChatEDA`, `Circuit Training`, `DREAMPlace`), and formal SVA verification (`AssertLLM`, `Formal Verification of RTL`).

Per the [official CS249r Assignments specification](https://harvard-edge.github.io/cs249r_fall2025/course/assignments/), the semester deliverables consist of **Three Survey Paper Checkpoints (45%)**, **Final Paper Integration (15%)**, and **Weekly Reading Reflections & Paper Presentations (40%)**.

---

## 2. Repository File Index

| Deliverable / Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Checkpoint 1: Literature Survey & Weekly Synthesis** | [`checkpoint1_literature_survey.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/checkpoint1_literature_survey.md) | Curates 25 seminal papers across 4 systems domains, defines the quantitative extraction schema, synthesizes all 13 weekly readings & guest lectures, and includes a 20+20 min seminar presentation/discussion guide. |
| **Quantitative Dataset (JSON)** | [`systems_agent_dataset.json`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/systems_agent_dataset.json) | Structured dataset across all 25 papers (`pass@1`, `pass@10`, `verifier_latency_share_pct`, `token_amplification_factor`, `kv_cache_reuse_ratio`, `speedup_over_baseline`, `avg_turn_count`). |
| **Checkpoint 2: Data Analysis Engine** | [`checkpoint2_data_analysis.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/checkpoint2_data_analysis.py) | Runnable Python analysis script computing domain aggregates, Pearson correlations, Amdahl's Law verifier bottleneck curves, and KV-cache prefix-reuse effective amplification. |
| **Checkpoint 2: Data-Driven Report** | [`checkpoint2_report.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/checkpoint2_report.md) | Generated technical report with cross-stack statistical tables, Amdahl's Law analysis, hypothesis tests, and ASCII comparative visualizations. |
| **Checkpoint 3 & Final Integrated Survey Paper** | [`final_survey_paper.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/08-harvard-cs249r/final_survey_paper.md) | Complete publication-grade survey paper: *"Architecture 2.0: A Cross-Stack Quantitative Survey of Agentic AI for Computer Systems Design and Systems for Agentic AI"*. |

---

## 3. Quick Start: Reproducing the Checkpoint 2 Data Analysis

Run the quantitative analysis pipeline directly with Python 3 (zero external dependencies required):

```bash
python3 checkpoint2_data_analysis.py
```

### Expected Output
```text
[CS249r Checkpoint 2] Loaded 25 papers across 4 domains.
  - Hardware_EDA_RTL            : N=7, pass@1=51.9% -> pass@10=74.4% (+22.5%), VerifShare=74.6%, TokAmp=12.8x, GeomeanSpeedup=1.77x
  - Compiler_GPU_Kernel         : N=6, pass@1=36.7% -> pass@10=73.3% (+36.6%), VerifShare=72.9%, TokAmp=20.8x, GeomeanSpeedup=1.37x
  - Cloud_OS_Arch_DSE           : N=6, pass@1=53.1% -> pass@10=76.0% (+22.9%), VerifShare=51.0%, TokAmp=10.3x, GeomeanSpeedup=2.19x
  - Serving_Systems_for_Agents  : N=6, pass@1=90.5% -> pass@10=97.4% (+6.9%), VerifShare=8.9%, TokAmp=1.8x, GeomeanSpeedup=3.62x
[CS249r Checkpoint 2] Successfully wrote report to .../08-harvard-cs249r/checkpoint2_report.md
```
