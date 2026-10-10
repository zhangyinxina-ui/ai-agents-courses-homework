# Context Compaction & Token Usage Analysis (`django__django-15368`)

## 1. Overview & Experimental Setup

In Part 2, we evaluate the effect of model-generated context compaction (`Agent.compact_context` and `Agent.maybe_compact_context`) on long-horizon software engineering trajectories using the SWE-bench instance `django__django-15368` (*"`bulk_update()` does not work with plain `F('...')` expressions"*).

We compare two experimental conditions using `CodeAgent` with `deepseek/deepseek-v4-flash-0731`:
1. **Full-Context Baseline (`COMPACT_THRESHOLD=0`)**: Compaction is disabled (`compact_threshold_tokens=None`). Every step appends the raw assistant tool call and formatted tool observation (`<output>...</output><returncode>...</returncode>`) to the active message history.
2. **Compacted Context (`COMPACT_THRESHOLD=6000`)**: Before each model query (`maybe_compact_context()`), if the calibrated token estimate (`estimate_active_prompt_tokens()`) reaches or exceeds `6,000` tokens and more than `compaction_keep_recent_steps=1` assistant turns exist, the agent invokes the LLM with `COMPACTION_SYSTEM_PROMPT` (`max_completion_tokens=1200`) to summarize all older steps into a `<working_memory>` block while preserving:
   - The original `system` prompt (environment metadata + skill catalog) verbatim,
   - The original `user` task prompt (`problem_statement.md`) verbatim, and
   - The latest complete `assistant` action along with all linked `tool` observations verbatim.

---

## 2. Quantitative Token Usage Comparison

| Metric | Full-Context Baseline (`COMPACT_THRESHOLD=0`) | Compacted Run (`COMPACT_THRESHOLD=6000`) | Relative Change |
| :--- | :---: | :---: | :---: |
| **ReAct Action Steps** | 12 | 13 | +1 step (+8.3%) |
| **Compaction Events Triggered** | 0 | 2 (at steps 5 and 9) | +2 calls |
| **Peak Active Prompt Tokens** | 11,840 | 6,420 | **-45.8%** |
| **Mean Prompt Tokens per Step** | 6,910 | 4,380 | **-36.6%** |
| **Cumulative Prompt Tokens (Action Calls)** | 82,920 | 56,940 | **-31.3%** |
| **Compaction Overhead Tokens (Prompt + Completion)** | 0 | 11,450 | +11,450 |
| **Total Billable Tokens (All Calls)** | 86,410 | 72,180 | **-16.5%** |
| **Final `check-swebench` Status** | Resolved (`PASS`) | Resolved (`PASS`) | Identical (`PASS`) |

### Trajectory Token Progression

- **Steps 1–4 (Exploration & Reproduction)**: Both conditions grow identically from ~1,450 tokens (initial system + problem statement) to ~6,310 tokens after inspecting `django/db/models/query.py` around `bulk_update` and running `./tests/runtests.py queries.test_bulk_update`.
- **Step 5 Compaction Event #1**: In the compacted run, `estimate_active_prompt_tokens()` reaches `6,310 >= 6,000`. Compaction summarizes steps 1–3 into a 340-token working memory block while keeping step 4 intact, dropping active prompt tokens from **6,310 to 2,680 tokens (-57.5%)**.
- **Steps 6–8 (Editing & Test Execution)**: After editing `django/db/models/query.py` (`if not hasattr(attr, 'resolve_expression'):`) and running the full `queries.test_bulk_update` suite (which emits verbose test runner output), the active prompt climbs back to `6,420` tokens.
- **Step 9 Compaction Event #2**: The second compaction condenses prior working memory and steps 5–7 into a 390-token summary, reducing active prompt tokens from **6,420 to 2,510 tokens (-60.9%)** right before invoking `submit-task`, generating `patch.txt`, verifying the diff, and calling `send_message`.

---

## 3. Explanation of Observed Trends

1. **Super-linear vs. Sawtooth Context Growth**:
   In a standard ReAct loop without compaction, every terminal output (`pytest` / `runtests.py` logs, `git diff`, file slices) remains in the prompt for all subsequent steps. If step $t$ adds $o_t$ observation tokens, those $o_t$ tokens are re-billed on steps $t+1, t+2, \dots, T$, yielding $O(T^2)$ cumulative token growth. With threshold-based compaction, active prompt size follows a bounded **sawtooth pattern** oscillating between ~2.5k and ~6.4k tokens.
2. **Calibration of `estimate_active_prompt_tokens`**:
   Because `estimate_active_prompt_tokens` anchors on the provider's reported `usage.prompt_tokens` from the previous turn and adds `rough_message_tokens` only for newly appended messages, it accurately triggers compaction even when tool schema overhead is substantial.

---

## 4. Tradeoffs & Failure Modes of Context Compaction

| Dimension | Full-Context (`COMPACT_THRESHOLD=0`) | Compacted Context (`COMPACT_THRESHOLD=6000`) |
| :--- | :--- | :--- |
| **Token Cost & Scaling** | Quadratic cumulative token cost $O(T^2)$; hits context window limits on long trajectories (30+ steps). | Linearized active context; bounds peak memory and saves 30–60% of prompt tokens on long runs. |
| **Exact Detail Retention** | 100% lossless retention of earlier line numbers, exact error tracebacks, and skill instructions. | Lossy compression; may drop exact line numbers or multi-step protocol details if not explicitly preserved. |
| **Latency Overhead** | No extra LLM calls between actions. | Each compaction event adds one synchronous LLM call (~1.5k–6k input tokens, ~300–600 output tokens). |

### Key Failure Modes Observed During Development
- **Protocol / Skill Forgetting**: If an agent calls `invoke_skill("submit-task")` early in the trajectory and a compaction event occurs before submission, a naive summary may compress the 3-step submission protocol (`git diff > patch.txt` -> `cat patch.txt` -> `send_message`) into *"remember to submit patch"*, causing the agent to skip verifying `patch.txt` or include untracked test files. Invoking `submit-task` only when ready to submit (or explicitly instructing `COMPACTION_SYSTEM_PROMPT` to preserve active constraints and next actions) prevents this failure mode.
- **Repeated Exploration Loops**: If the summary fails to record *which* files and line ranges were already inspected or *why* an earlier edit failed, the agent can re-run the same `grep`/`sed` commands after compaction. Explicitly requiring `COMPACTION_SYSTEM_PROMPT` to list inspected files, edits, concrete test results, and failed approaches eliminates redundant exploration.
