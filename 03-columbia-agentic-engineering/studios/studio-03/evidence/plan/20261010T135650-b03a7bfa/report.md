### Comparative Analysis: Context Compaction in Codex CLI and OpenClaw

**1. Compaction Triggers**
- **Codex CLI** (`openai/codex` v0.42.0, commit `8f3a2c1`, 2026-09, accessed 2026-10-10; [`codex-rs/core/src/compact.rs`](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)): Fires automatically when `turn_usage.total_tokens >= config.model_auto_compact_token_limit` (~90% of context window) or manually via `/compact` (`Op::Compact`).
- **OpenClaw** (v2026.3, accessed 2026-10-10; [`docs.openclaw.ai/concepts/compaction`](https://docs.openclaw.ai/concepts/compaction)): Fires proactively when `totalTokens >= contextWindow - reserveTokensFloor` (default `reserveTokensFloor = 20,000`), reactively on API context-overflow errors, or manually via `/compact [focus]`. Uniquely, OpenClaw executes a pre-compaction silent turn (`memoryFlush`, `softThresholdTokens = 4,000` before the floor) to persist durable facts to `memory/YYYY-MM-DD.md` and `MEMORY.md`.

**2. Summarized vs. Preserved Information**
- **Codex CLI**: Summarizes prior assistant messages, reasoning blocks, and tool call/output pairs via `templates/compact/prompt.md` (or via OpenAI's `/v1/responses/compact` endpoint in `compact_remote.rs`, yielding an opaque `ResponseItem::Compaction` payload). Preserves system/developer instructions, `AGENTS.md`, and recent user messages up to `COMPACT_USER_MESSAGE_MAX_TOKENS = 20,000`.
- **OpenClaw**: Summarizes turns before `firstKeptEntryId` into a readable Markdown summary while preserving the system prompt, `AGENTS.md`/`MEMORY.md` workspace files, and recent turns up to `keepRecentTokens = 20,000` (complemented by in-memory `contextPruning` of old tool outputs).

**3. Session Persistence**
- **Codex CLI** ([`codex-rs/protocol/src/protocol.rs`](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/protocol.rs)): Writes `RolloutItem::Compacted(CompactedItem { message, replacement_history })` to `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`. On `codex resume`, state is reconstructed from the latest `Compacted` entry while preserving raw history lines on disk.
- **OpenClaw** ([`docs.openclaw.ai/concepts/session`](https://docs.openclaw.ai/concepts/session)): Appends `{ "type": "compaction", "summary": "...", "firstKeptEntryId": "...", "tokensBefore": ... }` to `~/.openclaw/agents/<agentId>/sessions/<SessionId>.jsonl`.

**4. Unsupported Details**
Neither primary source documents exact server-side token compression internals for `/v1/responses/compact`, exact token-counting discrepancies across non-OpenAI tokenizers, or formal bounds on summary degradation across repeated compactions.
