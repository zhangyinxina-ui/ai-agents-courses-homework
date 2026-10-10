# Course 1: CMU 11-768 — AI Agents (Fall 2026)

Complete homework solutions, unit-tested agent implementations, Harbor benchmark packages, analytical reports, and Chinese lecture guides for **CMU 11-768: AI Agents** (taught by Graham Neubig & Daniel Fried).

---

## Directory Structure & Completed Deliverables

### 1. [`assignment-1/`](./assignment-1/) — Building a General-Purpose Coding & Game-Playing Agent
- **Part 1 — Core Agent Loop, Skill Discovery & SWE-bench / Chess Server Debugging**:
  - [`src/assignment/agent/base.py`](./assignment-1/src/assignment/agent/base.py): Implemented `load_skills` (progressive disclosure of `.agents/skills/*/SKILL.md` YAML frontmatter), `build_prompt` (Jinja2 template rendering), and `Agent.run` (multi-turn tool-calling loop with token tracking and `StepLimitError` handling).
  - [`src/assignment/agent/code_agent.py`](./assignment-1/src/assignment/agent/code_agent.py): Configured `CodeAgent` system prompt (`<system_information>` + `<skills>`), task prompt, and `execute_tool_calls` for `bash` and `finish`.
  - [`chess_app/src/chess_app/game.py`](./assignment-1/chess_app/src/chess_app/game.py) & [`chess_app/tests/test_chess_server.py`](./assignment-1/chess_app/tests/test_chess_server.py): Fixed the terminal checkmating move crash where the server called Stockfish on an already game-over board, and added the regression test `test_checkmating_move_returns_final_state_without_engine_reply`.
  - [`artifacts/fix.patch`](./assignment-1/artifacts/fix.patch) & [`artifacts/django__django-15368.patch`](./assignment-1/artifacts/django__django-15368.patch): Patch artifacts for the `chess_app` fix and SWE-bench `django__django-15368` (`F()` expression in `bulk_update`).
- **Part 2 — Context Compaction & Token Usage Analysis**:
  - [`src/assignment/agent/base.py`](./assignment-1/src/assignment/agent/base.py): Implemented `COMPACTION_SYSTEM_PROMPT`, `compact_context`, and `maybe_compact_context`.
  - [`token-usage-analysis.md`](./assignment-1/token-usage-analysis.md) (and [`artifacts/token-usage-analysis.md`](./assignment-1/artifacts/token-usage-analysis.md)): Quantitative and structural analysis of cumulative token growth and sawtooth compaction resets.
- **Part 3 — Observation & Action Space Design (`ChessAgent`)**:
  - [`src/assignment/agent/tools.py`](./assignment-1/src/assignment/agent/tools.py), [`src/assignment/agent/chess_tools.py`](./assignment-1/src/assignment/agent/chess_tools.py), [`src/assignment/agent/chess_agent.py`](./assignment-1/src/assignment/agent/chess_agent.py): Implemented `PLAY_MOVE_TOOL`, `SIMULATE_MOVE_TOOL`, `RUN_PYTHON_TOOL`, `_play_move`, `_simulate_move`, `_run_python`, and `_invoke_skill`.
  - [`observation-experiment.md`](./assignment-1/observation-experiment.md) (and [`artifacts/observation-experiment.md`](./assignment-1/artifacts/observation-experiment.md)): Controlled ablation across observation formats (`ascii`, `fen`, `pgn`, `unicode`, combinations) and action spaces (`CodeAgent` bash vs. `ChessAgent` structured tools).

### 2. [`assignment-2/`](./assignment-2/) — Validating and Packaging Agent Visualization Tasks
- **Part 1 — Hybrid Validator**:
  - [`validator/solution.py`](./assignment-2/validator/solution.py): Implemented `judge_execution`, `judge_data_and_chart`, `judge_readability`, and `validate(run)` combining deterministic PNG/pixel/trajectory inspection with rubric-guided VLM judgment and context-overflow downscaling.
- **Part 2 — Adversarial Task Suite, Multi-Agent Runs & Ground-Truth Labels**:
  - [`tasks/task_01`](./assignment-2/tasks/task_01/) – [`tasks/task_05`](./assignment-2/tasks/task_05/): Five adversarial CSV aggregation & visualization tasks across two `task_class` families (`conditional_csv_aggregation` and `multi_series_derived_metric`).
  - [`runs/`](./assignment-2/runs/) & [`labels.json`](./assignment-2/labels.json): 15 agent runs across `glm` (`GLM-4.7-Flash`), `ministral` (`Ministral-3-14B-Instruct-2512`), and `qwen` (`Qwen2.5-Coder-3B-Instruct`), plus prediction files in [`output/`](./assignment-2/output/).
- **Part 3 — Harbor Benchmark Environments & Mutant Study**:
  - [`harbor/tasks/task_01`](./assignment-2/harbor/tasks/task_01/) – [`harbor/tasks/task_05`](./assignment-2/harbor/tasks/task_05/): Full Harbor task directories with `task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test_state.py`, `solution/solve.sh`, and two mutant solutions (`mutants/unfiltered_rows/solve.sh` scoring `0.0` and `mutants/unverified_styling/solve.sh` scoring `1.0`).
- **Reports**:
  - [`report.md`](./assignment-2/report.md), [`report.tex`](./assignment-2/report.tex), [`report.pdf`](./assignment-2/report.pdf), and [`AI_USAGE.md`](./assignment-2/AI_USAGE.md).

### 3. [`lectures_and_a3/`](./lectures_and_a3/) — Lectures 12–14 Chinese Study Guide & Assignment 3 Preview
- [`CMU_Lectures_12_13_14_zh.md`](./lectures_and_a3/CMU_Lectures_12_13_14_zh.md): In-depth Chinese translation and study notes for Lecture 12 (Long Context & Context Management), Lecture 13 (Agentic Evolution & Continual Learning), and Lecture 14 (GUI & Web Agents).
- [`assignment3_agent_training_reference.py`](./lectures_and_a3/assignment3_agent_training_reference.py): Self-contained reference implementation of Rejection Sampling SFT trajectory filtering (with assistant-only action-token masking) and multi-turn Group Relative Policy Optimization (GRPO).
