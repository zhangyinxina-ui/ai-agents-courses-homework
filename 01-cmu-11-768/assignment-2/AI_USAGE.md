# AI Usage Disclosure — Assignment 2

In accordance with the course policy for CMU 11-768 (AI Agents), we document the use of AI assistance during Assignment 2:

1. **Part 1 (`validator/solution.py`)**:
   - AI coding assistance was used to structure the two-stage hybrid validator combining deterministic artifact/trajectory checks (`judge_execution`, `_heuristic_data_and_chart_checks`, `_heuristic_readability_checks`) with rubric-guided VLM queries (`_ask_vlm`) and context-overflow downscaling.
2. **Part 2 (`tasks/`, `labels.json`, `runs/`)**:
   - AI assistance was used to help format task descriptors (`task.json`), verify ground-truth CSV aggregations across the 5 authored tasks (`task_01`–`task_05`), and structure the evaluation analysis.
3. **Part 3 (`harbor/tasks/`)**:
   - AI assistance was used to draft the Harbor `test_state.py` verifiers (including SHA-256 input integrity checks, sidecar schema validation, and input-perturbation re-execution probes), oracle `solve.sh` scripts, and the two mutant variants (`unfiltered_rows` and `unverified_styling`).
4. **Report (`report.tex` / `report.pdf`)**:
   - AI assistance was used for LaTeX formatting and table generation. All experimental claims, task designs, and verifier gap analyses were verified against the codebase and preflight checks.
