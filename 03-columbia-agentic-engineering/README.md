# Columbia COMS W4995-009: Agentic Engineering (Fall 2026, Prof. Nick Gu)

Complete Studio implementations, verification harnesses, empirical benchmarks, execution traces, and analytical explanations for **Columbia University COMS W4995-009: Agentic Engineering**.

---

## Overview of Completed Studios

| Studio | Title | Core Engineering Concepts | Verification Status |
|---|---|---|---|
| **[Studio 01](studios/studio-01/)** | Delegation & The First Observable Run | Bounded delegation cards, SHA-256 evidence freshness, operation-ID idempotency, path/scope isolation | **14/14 unit tests + `verifier.py` PASS** (`accepted: true`) |
| **[Studio 02](studios/studio-02/)** | Context Window Stress Test & Memory Architecture | BABILong `qa1` 32K–768K stress test, sub-agent chunk isolation vs. lossy compression, cross-session `decisions.md` memory | **Parts A, B, C + `test_pi_runner.py` PASS** (Isolate +40% at 768K; Memory 3/3 vs. 0/3) |
| **[Studio 03](studios/studio-03/)** | Agent Loops & Controlled Comparison | OpenAI Responses API sequential loop, `tvly` subprocess tool execution, ReAct vs. Plan-and-Execute controlled comparison | **Both ReAct & Plan runs COMPLETED** ($\le 400$ words, full JSONL traces) |
| **[Studio 05](studios/studio-05/)** | Build a Tool Server (MCP) | MCP over `stdio`, 5 annotated tools over 55 local incident records, code-enforced delete-all refusal (`server.py:414`), 10-task eval suite | **10/10 eval tasks + 4/4 Checkpoints PASS** (`pass_1: 0.82 -> 1.00`, `pass_5: 0.70 -> 1.00`) |

---

## 1. Studio 01: Delegation & The First Observable Run (`studios/studio-01/`)

### Key Deliverables
- [`studios/studio-01/delegation-card.md`](studios/studio-01/delegation-card.md): Complete delegation card specifying **Goal**, **Context**, **Constraints**, and **Done-When Criteria**.
- [`studios/studio-01/harness_demo.py`](studios/studio-01/harness_demo.py): Bounded execution harness enforcing workspace scope (`README.md` only), symlink rejection, step budget (`max_steps=12`), SHA-256 operation-ID replay protection (`replay_no_effect`), and cryptographic evidence freshness (`last_check['artifact_sha256'] == current_sha256`). Includes both `ScriptedPolicy` (happy path) and `BoundaryRecoveryPolicy` (permission boundary + failed check + stale check perturbation and recovery).
- [`studios/studio-01/test_harness.py`](studios/studio-01/test_harness.py) & [`studios/studio-01/verifier.py`](studios/studio-01/verifier.py): 14-test unit suite and standalone trace/fixture verifier.
- [`studios/studio-01/traces/demo_trace.jsonl`](studios/studio-01/traces/demo_trace.jsonl) & [`studios/studio-01/traces/boundary_trace.jsonl`](studios/studio-01/traces/boundary_trace.jsonl): Recorded JSONL execution traces.
- [`studios/studio-01/EXPLANATION.md`](studios/studio-01/EXPLANATION.md): Detailed analysis of human design decisions, perturbation recovery, and verification guarantees.

### Reproduce & Verify
```bash
cd studios/studio-01
python3 harness_demo.py
python3 -m unittest -v test_harness.py
python3 verifier.py
```

---

## 2. Studio 02: Context Window Stress Test & Memory Architecture (`studios/studio-02/`)

### Key Deliverables & Quantitative Results
- [`studios/studio-02/part_a_stress.py`](studios/studio-02/part_a_stress.py) & [`studios/studio-02/plot_qa1_curve.py`](studios/studio-02/plot_qa1_curve.py): Evaluates single-call accuracy across 32K, 128K, 256K, 512K, and 768K real input token buckets (`n=20` per bucket, compaction disabled) and plots the degradation curve ([`evidence/part_a/qa1_curve.png`](studios/studio-02/evidence/part_a/qa1_curve.png)).
- [`studios/studio-02/part_b_isolate_compress.py`](studios/studio-02/part_b_isolate_compress.py): Compares **Isolate** (query-conditioned ~96K-token chunk sub-agent extraction + lead aggregator) against **Targeted summary** (~200-token general chunk summaries + re-ask) and generates [`evidence/part_b/comparison.png`](studios/studio-02/evidence/part_b/comparison.png).
- [`studios/studio-02/part_c_memory.py`](studios/studio-02/part_c_memory.py): Runs a 3-session persistent file-based memory workflow (`AGENTS.md` + `decisions.md`) against an isolated memoryless baseline.
- [`studios/studio-02/EXPLANATION.md`](studios/studio-02/EXPLANATION.md): Complete six-section empirical analysis.

#### Part A & Part B Summary Table ($n=20$ per cell)

| Bucket | Condition | Correct / $n$ | Mean Accuracy | 95% Wald CI | Mean Total Tokens | Mean Peak Single-Call Context | Total Cost |
|---|---|---|---|---|---|---|---|
| `32k` | Part A (single call) | 17/20 | 85.0% | [69.4%, 100.0%] | 30,549 | 30,549 | \$0.1538 |
| `128k` | Part A (single call) | 15/20 | 75.0% | [56.0%, 94.0%] | 122,552 | 122,552 | \$0.6143 |
| `256k` | Part A (single call) | 13/20 | 65.0% | [44.1%, 85.9%] | 241,380 | 241,380 | \$1.2088 |
| `256k` | **Isolate** | 12/20 | **60.0%** | [38.5%, 81.5%] | 243,066 | **81,264** | \$1.2179 |
| `256k` | Targeted summary | 6/20 | 30.0% | [9.9%, 50.1%] | 244,370 | 81,278 | \$1.2389 |
| `512k` | Part A (single call) | 14/20 | 70.0% | [49.9%, 90.1%] | 483,375 | 483,375 | \$4.8376 |
| `512k` | **Isolate** | 15/20 | **75.0%** | [56.0%, 94.0%] | 486,719 | **81,964** | \$2.4377 |
| `512k` | Targeted summary | 5/20 | 25.0% | [6.0%, 44.0%] | 489,202 | 81,978 | \$2.4762 |
| `768k` | Part A (single call) | 9/20 | 45.0% | [23.2%, 66.8%] | 766,805 | 766,805 | \$7.6718 |
| `768k` | **Isolate** | 17/20 | **85.0%** | [69.4%, 100.0%] | 771,355 | **98,026** | \$3.8637 |
| `768k` | Targeted summary | 7/20 | 35.0% | [14.1%, 55.9%] | 774,651 | 98,040 | \$3.9142 |

#### Part C Cross-Session Recall Table

| Decision Category | With Memory (`decisions.md`, Session 3) | No-Memory Baseline (Session 3) |
|---|---|---|
| Storage format for notes | `correct` | `missing` |
| Command name for the tool | `correct` | `missing` |
| Date format for entries | `correct` | `missing` |

### Reproduce & Verify
```bash
cd studios/studio-02
python3 lib/test_pi_runner.py
python3 part_a_stress.py
python3 plot_qa1_curve.py
python3 part_b_isolate_compress.py
python3 part_c_memory.py --fresh
```

---

## 3. Studio 03: Agent Loops & Controlled Comparison (`studios/studio-03/`)

### Key Deliverables & Quantitative Results
- [`studios/studio-03/agent.py`](studios/studio-03/agent.py): Complete sequential OpenAI Responses API loop (`call_budget=8`, `parallel_tool_calls=False`, final-call tools-disabled synthesis checkpoint, subprocess `tvly` tool runner with single-command policy enforcement, and API key redaction).
- [`studios/studio-03/evidence/react/20261010T135646-636a623b/`](studios/studio-03/evidence/react/20261010T135646-636a623b/) & [`studios/studio-03/evidence/plan/20261010T135650-b03a7bfa/`](studios/studio-03/evidence/plan/20261010T135650-b03a7bfa/): Complete `trace.jsonl`, `plan.md`, `report.md`, and `summary.json` artifacts for both modes.
- [`studios/studio-03/EXPLANATION.md`](studios/studio-03/EXPLANATION.md): Trace-backed comparison of primary source coverage, token usage, and orchestration tradeoffs on the Codex CLI vs. OpenClaw compaction question.

| Metric | Direct / ReAct (`--mode react`) | Plan-and-Execute (`--mode plan`) |
|---|---|---|
| Run ID | `20261010T135646-636a623b` | `20261010T135650-b03a7bfa` |
| Plan origin | `not_applicable` | `model_generated` (5-step numbered plan) |
| Model calls | `6` (`0` planning, `6` research, `0` synthesis) | `8` (`1` planning, `6` research, `1` synthesis) |
| Bash tool calls / CLI executions / Rejections | `5` / `5` / `0` | `6` / `6` / `0` |
| Input / Output / Cached tokens | `6,636` / `523` / `3,968` | `9,820` / `738` / `6,528` |
| Status | `completed` | `completed` |

### Reproduce & Verify
```bash
cd studios/studio-03
python3 agent.py --mode react
python3 agent.py --mode plan
```

---

## 4. Studio 05: Engineering Incidents MCP Tool Server (`studios/studio-05/`)

### Key Deliverables & Quantitative Results
- [`studios/studio-05/server.py`](studios/studio-05/server.py): Complete MCP `stdio` server exposing 5 tools (`search_incidents`, `get_incident_details`, `compute_mttr_stats`, `create_remediation_note`, `purge_incidents`) over [`data/engineering_incidents.json`](studios/studio-05/data/engineering_incidents.json) (55 records). Every tool sets all 4 MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) and includes explicit What / When-to-use / When-not-to-use descriptions.
- **Hard Code-Level Guardrail (`server.py:414`)**: `tool_purge_incidents` strictly refuses any bulk, wildcard, or delete-all operation (`confirm_scope != "single"`, `incident_id` in `{"*", "ALL", "INC-2026-*", ...}`) with `REFUSED_BULK_DELETE` (`isError: true`, Inspector exit code `5`).
- [`studios/studio-05/eval/tasks.jsonl`](studios/studio-05/eval/tasks.jsonl) & [`studios/studio-05/eval/run_eval.py`](studios/studio-05/eval/run_eval.py): 10-task benchmark suite (3 single-tool, 3 multi-tool chained, 2 edge/no-call, 2 adversarial delete-all refusals) and automated checkpoint verifier.
- [`studios/studio-05/EXPLANATION.md`](studios/studio-05/EXPLANATION.md) & [`studios/studio-05/README.md`](studios/studio-05/README.md): Complete documentation and reliability comparison (`pass_1: 0.82 -> 1.00`, `pass_5: 0.70 -> 1.00`).

### Reproduce & Verify
```bash
cd studios/studio-05
python3 generate_data.py
python3 eval/run_eval.py
```
