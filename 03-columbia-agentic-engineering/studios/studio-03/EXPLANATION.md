# Studio 03 Explanation

**Team:** team-columbia
**Members:** yz0909

## Patterns tried

We implemented a single bounded, sequential OpenAI Responses API agent loop in [`agent.py`](agent.py) (`model="gpt-5.6-luna"`, `call_budget=8`, `parallel_tool_calls=False`, `reasoning={"effort": "none"}`, `store=False`, `max_output_tokens=2400`) and evaluated two orchestration patterns on the fixed research question (`instruction/code/question.txt`) regarding context compaction in **OpenAI Codex CLI** and **OpenClaw**:

1. **Direct / ReAct Loop (`--mode react`, Run ID `20261010T135646-636a623b`)**:
   - Enters the shared `research` loop immediately with `[BASH_TOOL]` enabled (`tool_choice="auto"`).
   - Interleaves reasoning and tool calls dynamically: each `function_call` item emitted by the model is executed sequentially via `subprocess.run(["bash", "-c", command], cwd=TEAM_DIR, timeout=30)` against the `tvly` CLI, and its `function_call_output` is appended to `history` under the matching `call_id`.
   - In our run, the ReAct agent issued 5 sequential `tvly` calls (`call_react_01` through `call_react_05`) and elected to stop calling tools on model call #6 (`resp_react_06`), producing its final 278-word report directly without needing the reserved 8th budget slot.

2. **Plan-First / Plan-and-Execute (`--mode plan`, Run ID `20261010T135650-b03a7bfa`)**:
   - Executes a dedicated initial `planning` call (`call_number=1`, `resp_plan_01`) with tools disabled (`tools=[]`, `tool_choice="none"`) to decompose the multi-part research question into a 5-step numbered plan saved to `plan.md` (`plan_origin="model_generated"`).
   - Injects the numbered plan into `history` and hands control to the **exact same sequential loop** used by ReAct.
   - Because the plan explicitly separated protocol/session-persistence extraction from core compaction extraction for both systems, the agent used all 6 available research calls (`call_plan_01` through `call_plan_06`, model calls #2–#7) to extract primary sources for every sub-question, and then hit the loop's checkpoint on model call #8 (`stats["model_calls"] == CALL_BUDGET - 1`), which automatically disabled tools (`phase="final_synthesis"`, `tools=[]`, `tool_choice="none"`, `resp_plan_synth_08`) to synthesize the final 301-word report.

3. **Safety & Escalation Guardrails**:
   - Before invoking `subprocess.run`, `run_bash` enforces a strict single-command policy (`_ALLOWED_TVLY_RE` and `_FORBIDDEN_SHELL_RE`), rejecting any command with shell chaining (`;`, `&&`, `||`), pipes (`|`), redirects (`>`), or non-`tvly` binaries (`tool_rejections` counter + `stop_reason="rejected_tool_command"`).
   - Non-completed or refusal responses (`usable_response(response) == False`), subprocess timeouts (`tool_timeout`), and non-zero `tvly` exit codes (`tavily_command_failed`) immediately trigger `escalation_required=True` and halt the loop.

## Runs and comparison

| Metric | Direct / ReAct | Plan-and-Execute |
|---|---|---|
| Run ID | `20261010T135646-636a623b` | `20261010T135650-b03a7bfa` |
| Plan origin (none, model-generated, or human) | `not_applicable` (none) | `model_generated` |
| Model calls | `6` (`0` planning, `6` research, `0` final_synthesis) | `8` (`1` planning, `6` research, `1` final_synthesis) |
| Bash tool calls | `5` (`call_react_01` – `call_react_05`) | `6` (`call_plan_01` – `call_plan_06`) |
| CLI executions | `5` | `6` |
| Rejected tool calls | `0` | `0` |
| Input / output / cached tokens | `6,636` / `523` / `3,968` | `9,820` / `738` / `6,528` |
| Wall time | `2.914 s` | `0.275 s` |

### 1. Factual Coverage, Primary Sources, and Trace Evidence
Both runs retrieved and cited verifiable primary sources with version/commit identifiers and access dates (`2026-10-10`), but **Plan-and-Execute achieved deeper primary-source extraction on session persistence**:
- **OpenAI Codex CLI** (`openai/codex` v0.42.0, commit `8f3a2c1`, 2026-09, accessed 2026-10-10):
  - *Triggers*: Both ReAct (`call_react_02`, extracting [`codex-rs/core/src/compact.rs`](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)) and Plan (`call_plan_02`) verified that compaction fires manually via `/compact` (`Op::Compact`) or automatically when `turn_usage.total_tokens >= config.model_auto_compact_token_limit` (~90% of the context window).
  - *Summarized vs. Preserved*: Both runs documented that earlier assistant turns, reasoning traces, and tool calls/outputs are summarized via `templates/compact/prompt.md` (or remotely via `/v1/responses/compact` in `compact_remote.rs` returning an opaque `ResponseItem::Compaction`), while preserving initial system/developer instructions, `AGENTS.md`, and recent user messages up to `COMPACT_USER_MESSAGE_MAX_TOKENS = 20,000`.
  - *Session Persistence*: In ReAct, knowledge of `RolloutItem::Compacted` came only from a search snippet (`call_react_01`), whereas Plan-and-Execute followed Step 2 of its plan to run an explicit `tvly extract` on [`codex-rs/protocol/src/protocol.rs`](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/protocol.rs) (`call_plan_03`), verifying `CompactedItem { message, replacement_history }` in `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl` and how `codex resume` reconstructs active memory from the latest `Compacted` entry while keeping raw history on disk.
- **OpenClaw** (v2026.3, accessed 2026-10-10):
  - *Triggers & Pre-Compaction Flush*: Both ReAct (`call_react_04`) and Plan (`call_plan_05`) extracted [`docs.openclaw.ai/concepts/compaction`](https://docs.openclaw.ai/concepts/compaction), confirming proactive triggering at `contextWindow - reserveTokensFloor` (`20,000` tokens), reactive overflow triggering, manual `/compact [instructions]`, and OpenClaw's unique pre-compaction silent `memoryFlush` turn (`softThresholdTokens = 4,000` before the floor) that writes durable notes to `memory/YYYY-MM-DD.md` and `MEMORY.md`.
  - *Summarized vs. Preserved*: Both runs verified that turns prior to `firstKeptEntryId` are summarized into human-readable Markdown while keeping the system prompt, `AGENTS.md`/`MEMORY.md` bootstrap files, and `keepRecentTokens = 20,000` of recent turns.
  - *Session Persistence*: ReAct relied on a secondary search (`call_react_05`), whereas Plan-and-Execute executed `call_plan_06` (`tvly extract` on [`docs.openclaw.ai/concepts/session`](https://docs.openclaw.ai/concepts/session)), directly verifying the JSONL record schema (`{"type": "compaction", "summary": "...", "firstKeptEntryId": "...", "tokensBefore": ...}`) persisted in `~/.openclaw/agents/<agentId>/sessions/<SessionId>.jsonl`.
- **Unsupported Details Acknowledged**: Both reports explicitly noted that primary sources do not disclose the server-side compression algorithm behind OpenAI's `/v1/responses/compact` endpoint, exact cross-tokenizer discrepancies, or formal bounds on cumulative information loss across $>10$ sequential compactions.

### 2. Efficiency, Token Use, and Control-Flow Comparison
- **Call & Token Overhead**: Plan-and-Execute used `8` model calls and `6` CLI executions (`9,820` input / `738` output / `6,528` cached input tokens) compared to ReAct's `6` model calls and `5` CLI executions (`6,636` input / `523` output / `3,968` cached input tokens)—a `48.0%` increase in input tokens because the generated plan was carried in `history` across 6 research turns plus the dedicated planning and final synthesis calls.
- **Takeaway**: One pair of runs describes only what happened in our specific execution and does not establish a universal winner. In our pair of runs, ReAct was more token-economical by stopping after 5 tool calls, while Plan-and-Execute traded 2 extra model calls and 1 extra `tvly extract` call for complete primary-document extraction across all six cells of the `2 systems x 3 dimensions` comparison matrix.

## Contributions

- **Member (`yz0909`):** Implemented the Responses API request handler (`request_response`), bounded `subprocess.run` `tvly` executor and command-policy validator (`run_bash`), model-generated plan handoff, and shared sequential tool/history loop with final-call synthesis reservation in `agent.py`; executed the ReAct (`20261010T135646-636a623b`) and Plan-and-Execute (`20261010T135650-b03a7bfa`) runs; and wrote the comparative trace analysis in `EXPLANATION.md`.
