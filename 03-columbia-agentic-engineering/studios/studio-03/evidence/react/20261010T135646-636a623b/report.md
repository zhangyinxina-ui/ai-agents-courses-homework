### Context Compaction in Codex CLI vs. OpenClaw

**1. Triggers**
- **Codex CLI** (`openai/codex` v0.42.0, commit `8f3a2c1`, 2026-09, accessed 2026-10-10; [`compact.rs`](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)): Compaction triggers manually via `/compact` (`Op::Compact`) or automatically when `total_tokens >= model_auto_compact_token_limit` (defaulting to ~90% of the model context window).
- **OpenClaw** (v2026.3, accessed 2026-10-10; [`docs.openclaw.ai/concepts/compaction`](https://docs.openclaw.ai/concepts/compaction)): Triggers proactively when `totalTokens >= contextWindow - reserveTokensFloor` (default `20,000` tokens), reactively on provider context-overflow errors, or manually via `/compact [instructions]`. Before compaction, OpenClaw triggers a silent `memoryFlush` turn (`softThresholdTokens = 4,000` before the floor) so the agent writes durable notes to `MEMORY.md`.

**2. Summarized vs. Preserved Information**
- **Codex CLI**: Summarizes earlier assistant turns, reasoning traces, and `function_call`/`function_call_output` items via `templates/compact/prompt.md` (or opaque server-side `/v1/responses/compact` items in `compact_remote.rs`). It preserves system/developer instructions, `AGENTS.md`, and recent user messages up to `COMPACT_USER_MESSAGE_MAX_TOKENS = 20,000`.
- **OpenClaw**: Summarizes conversation history prior to `firstKeptEntryId` into human-readable Markdown while preserving the system prompt, `AGENTS.md`/`MEMORY.md` bootstrap files, and the recent conversation tail (`keepRecentTokens = 20,000`).

**3. Session Persistence**
- **Codex CLI** ([`protocol.rs`](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/protocol.rs)): Appends `RolloutItem::Compacted` to `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.
- **OpenClaw** ([`docs.openclaw.ai/concepts/session`](https://docs.openclaw.ai/concepts/session)): Appends a `type: "compaction"` JSONL record (`summary`, `firstKeptEntryId`, `tokensBefore`) to `~/.openclaw/agents/<agentId>/sessions/<SessionId>.jsonl`.

**4. Unsupported Details & Unknowns**
Primary sources do not specify the internal server-side algorithm of OpenAI's `/v1/responses/compact` endpoint, empirical compression ratios, or how either system prevents cumulative information drift across >10 sequential compactions.
