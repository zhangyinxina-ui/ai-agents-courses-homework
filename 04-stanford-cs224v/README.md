# Course 4: Stanford CS224V — Agentic AI (Fall 2026, Prof. Monica Lam)
## Complete Homework Solutions & Reference Implementations (`OMIM:620450` / *TAF4* & SatIR/VERDICT)

This directory contains the complete, executed, and unit-tested deliverables for **Stanford CS224V: Agentic AI (Fall 2026)**:

---

## Directory Structure & Deliverables

### 1. Homework 1: Rare Disease Differential Diagnosis with RAG and GRILL (`hw1/`)
Assigned Disease: **`OMIM:620450`** (*Intellectual developmental disorder, autosomal dominant 73*, gene `TAF4`, target patient **`P10`** / `PMID_35904126_Proband_7`).
- [`hw1_rare_disease_diagnosis.ipynb`](./hw1_rare_disease_diagnosis.ipynb) & [`hw1/hw1_rare_disease_diagnosis.ipynb`](./hw1/hw1_rare_disease_diagnosis.ipynb) — Executed notebook with all 12 TODOs (`TODO 0`, `1.1`, `1.2`, `3.1`, `3.2`, `4.1`, `4.2`, `4.3`, `5.1`, `5.2`, `5.3`, `5.4`, `6.1`) and programmatic failure analysis output.
- [`hw1/todo_answers.json`](./hw1/todo_answers.json) — Complete structured JSON answers for all 12 TODOs.
- [`hw1/HW1_Rare_Disease_GRILL_Report.md`](./hw1/HW1_Rare_Disease_GRILL_Report.md) — In-depth diagnostic report tracing the 4-stage failure cascade (Table 1 snippet truncation in `e39`/`e57` $\rightarrow$ unfunded corrective directions `d48`/`d50`/`d35` $\rightarrow$ conflicted hypothesis `h9` leaking via `h5` into `answer.md` $\rightarrow$ downstream ranker penalizing `P10` and rewarding decoy `P07`).
- [`hw1/grill_belief_ledger_demo.py`](./hw1/grill_belief_ledger_demo.py) — Runnable Python reference implementation & `pytest` test suite showing how calibrating the GRILL belief ledger promotes ground-truth patient `P10` from Rank #4 (`0.03`) to Rank #1 (`1.000`).

### 2. Homework 2 Part 1: Building Simple ReAct Agents & AOS with SLIDERS (`hw2_part1/`)
- [`hw2_part1_building_simple_agents.ipynb`](./hw2_part1_building_simple_agents.ipynb) & [`hw2_part1/hw2_part1_building_simple_agents.ipynb`](./hw2_part1/hw2_part1_building_simple_agents.ipynb) — Executed notebook with all 16 TODOs (`TODO 0`, `1`, `1.2`, `2`, `3`, `4`, `5`, `5.1`, `6a`–`6e`, `7`, `8`, `9`, `9.1`, `9.2`, `10`) completed for `OMIM_620450`.
- [`hw2_part1/answers.json`](./hw2_part1/answers.json) — Complete structured JSON answers for all HW2 Part 1 TODOs.
- [`hw2_part1/results/claims.json`](./hw2_part1/results/claims.json) — Validated `ClaimBatch` JSON artifact linking `answer.md` lines `18`, `20`, `22` to hypotheses `h1`, `h7`, `h13` and their exact `ledger.json` evidence IDs (`e1, e2, e8, e9`, `e16, e28`, `e37, e50, e54`).
- [`hw2_part1/simple_agents_and_aos_solution.py`](./hw2_part1/simple_agents_and_aos_solution.py) — Standalone runnable ReAct agent, Pydantic schema validator, and AOS + SLIDERS workspace builder with `pytest` tests.
- [`hw2_part1/HW2_Part1_Simple_Agents_AOS_Report.md`](./hw2_part1/HW2_Part1_Simple_Agents_AOS_Report.md) — Detailed technical writeup for HW2 Part 1.

### 3. Homework 2 Part 2: SatIR Library Documentation Audit with VERDICT (`hw2_part2/`)
- [`hw2_part2_satir_verdict.ipynb`](./hw2_part2_satir_verdict.ipynb) & [`hw2_part2/hw2_part2_satir_verdict.ipynb`](./hw2_part2/hw2_part2_satir_verdict.ipynb) — Fully executed notebook completing all 6 TODOs (`P2-1(a)`, `P2-1(b)`, `P2-2(a)`, `P2-3(a)`, `P2-3(b)`, `P2-4`) auditing `verdict` (`v0.2.3`).
- [`hw2_part2/todo_answers.json`](./hw2_part2/todo_answers.json) — Structured answers for all 6 TODOs in HW2 Part 2.
- [`hw2_part2/HW2_Part2_SatIR_VERDICT_Report.md`](./hw2_part2/HW2_Part2_SatIR_VERDICT_Report.md) — Complete root-cause taxonomy and `findings.csv` vs. `consolidated_report.md` audit report.

---

## Verification & Test Execution

Run the full test suite across Homework 1 and Homework 2 Part 1 from this directory:

```bash
pytest -q hw1/grill_belief_ledger_demo.py hw2_part1/simple_agents_and_aos_solution.py
```
