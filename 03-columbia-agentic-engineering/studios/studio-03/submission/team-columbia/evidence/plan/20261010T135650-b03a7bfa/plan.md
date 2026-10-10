1. Search primary OpenAI Codex CLI repository sources (`codex-rs/core/src/compact.rs`) to identify compaction triggers and token limits.
2. Extract `codex-rs/core/src/compact.rs` and `codex-rs/protocol/src/protocol.rs` to verify summarized vs. preserved context items and JSONL rollout session persistence.
3. Search primary OpenClaw documentation (`docs.openclaw.ai/concepts/compaction`) for auto-compaction triggers, `reserveTokensFloor`, and pre-compaction memory flush.
4. Extract `docs.openclaw.ai/concepts/compaction` and `docs.openclaw.ai/concepts/session` to verify `keepRecentTokens` preservation and `.jsonl` session persistence.
5. Synthesize a concise (<400 words) comparative report citing URLs, versions/commits, and access dates, explicitly noting unsupported details.
