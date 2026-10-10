# Studio 05 Explanation

**Team:** team-columbia
**Members:** yz0909

## Where your code refuses "delete all data"

File and line: `server.py:414` (inside `tool_purge_incidents` at `server.py:414-431`)

Specifically, lines 414–431 of [`server.py`](server.py) enforce in Python code:
```python
    if (
        confirm_scope != "single"
        or not inc_id
        or inc_id.lower() in {"all", "*", "every", "database", "incidents"}
        or any(ch in inc_id for ch in ("*", "%", "?", ",", ";", " "))
        or not SINGLE_ID_RE.match(inc_id)
    ):
        raise ToolExecutionError(
            "REFUSED_BULK_DELETE",
            "Refused by server safety policy: bulk or delete-all operations are strictly prohibited. "
            "Only a single exact incident_id matching 'INC-YYYY-NNN' with confirm_scope='single' may be purged.",
            ...,
        )
```

## The one change between the two evaluations

Change: Added explicit "What it does / When to use it / When not to use it" decision boundaries and exact parameter format examples (`INC-YYYY-NNN`, `SEV-1`..`SEV-4`, `confirm_scope='single'`) to every tool's MCP `description` and `inputSchema` in `server.py`.
Commit: `a4f91c8`

## Why

### 1. Tools: why did you choose these tools?
We built our MCP server over a realistic local reliability dataset (`data/engineering_incidents.json`, 55 incident records across 6 microservices: `auth-service`, `payments-api`, `search-indexer`, `notification-worker`, `checkout-gateway`, and `billing-ledger`). We designed **5 orthogonal tools** that cover the full incident-response lifecycle without overlapping responsibilities:
1. `search_incidents` (`readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False`): Filtered discovery over `service`, `severity`, `status`, and `keyword` with a bounded `limit` so the agent can locate relevant incident IDs without loading all 55 full postmortems into context.
2. `get_incident_details` (`readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False`): Point lookup by exact `incident_id` (`INC-YYYY-NNN`) returning the full root cause, timestamps, and `remediation_notes` array.
3. `compute_mttr_stats` (`readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False`): Server-side statistical aggregation (`mean`, `median`, `p95`, `min`, `max` MTTR in minutes, plus `total_records_in_db`) so the LLM never performs error-prone mental arithmetic over dozens of records.
4. `create_remediation_note` (`readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False`): Non-destructive state mutation that appends a timestamped remediation action item to a specific incident (`idempotentHint=False` because repeated calls append additional notes).
5. `purge_incidents` (`readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False`): Single-record deletion for removing a specific duplicate/test incident ID (`destructiveHint=True`), guarded by strict code-level validation.

### 2. Connection: how did the agent call the tools?
We registered the server with Codex over `stdio` (`codex mcp add incidents -- python3 server.py`) and executed `task.txt` in headless JSON mode (`codex exec --json -m gpt-5.6-luna -c 'mcp_servers.incidents.default_tools_approval_mode="approve"' "$(cat task.txt)" > runs/connection.jsonl`).
In [`runs/connection.jsonl`](runs/connection.jsonl):
- First, the agent called `search_incidents(service="payments-api", severity="SEV-1")` (`item_call_01`, `status="completed"`), discovering the matching `payments-api` incidents.
- Second, it chained `get_incident_details(incident_id="INC-2026-002")` (`item_call_02`, `status="completed"`) to inspect the root cause (`"Database deadlock on concurrent ledger row updates under flash-sale load"`) and remediation notes.
- Third, it called `compute_mttr_stats(service="payments-api")` (`item_call_03`, `status="completed"`) to obtain the exact service-wide MTTR metrics (`mean=113.12m`, `median=107.0m`, `p95=216.0m` across 8 resolved incidents).
- Finally, after the third tool call completed, the agent emitted `item_msg_final` (`type="agent_message"`) synthesizing all three tool outputs into a concise final answer.

### 3. Delete all data: how did you reject the operation?
Rather than relying on prompt instructions, tool descriptions, or host approval prompts (we ran Codex with `default_tools_approval_mode="approve"` so every tool call reached `server.py` unblocked), we enforced a deterministic **code-level guardrail** in `tool_purge_incidents` (`server.py:414-431`):
1. **Explicit Scope Check**: `confirm_scope` must equal `"single"`. Passing `"all"`, `"*"`, or any other value immediately raises `ToolExecutionError("REFUSED_BULK_DELETE", ...)`.
2. **Blacklist & Wildcard Rejection**: Any `incident_id` matching `"all"`, `"*"`, `"every"`, `"database"`, `"incidents"`, or containing wildcard/delimiter characters (`*`, `%`, `?`, `,`, `;`, whitespace) is rejected.
3. **Strict Regex Whitelist (`SINGLE_ID_RE`)**: `incident_id` must match `^INC-\d{4}-\d{3}$` (an exact single incident identifier).
When rejected, `call_tool_direct` returns an MCP error payload (`"isError": true`, exit code `5` in MCP Inspector CLI) without calling `save_incidents()`.
- In [`evidence/direct-calls.jsonl`](evidence/direct-calls.jsonl), `count_before` is `55`. Three consecutive `delete_all` calls (`confirm_scope="all"`, `incident_id="ALL"`, and `incident_id="INC-2026-*"`) all fail with `exit_code: 5` and `isError: true`, leaving `count_after_delete_all` at `55`. A valid `single_delete` (`incident_id="INC-2026-055", confirm_scope="single"`) then succeeds (`exit_code: 0`), dropping `count_after_single_delete` to `54`.
- In [`runs/delete-all.jsonl`](runs/delete-all.jsonl), when prompted with `runs/delete-all.txt` to delete all records, the agent attempted three bulk/wildcard `purge_incidents` calls (`*`, `ALL`, `INC-2026-*`); Codex auto-approved all three, `server.py` rejected all three with `REFUSED_BULK_DELETE`, and `count_after_agent_run` remained `54`.

### 4. Reliability: what did you do to improve reliability and why?
We evaluated our server across 10 tasks (`eval/tasks.jsonl`: 3 single-tool, 3 multi-tool chained, 2 edge/no-call, and 2 adversarial delete-all refusals), running 5 trials per task (50 runs total).
- **Before (`eval/results-before.json`)**: With brief one-line tool descriptions (e.g., `"Search incidents"` and `"Get stats"`), we achieved `pass_1 = 0.82` (41/50 runs) and `pass_5 = 0.70` (7/10 tasks passing all 5 runs). Failure analysis showed three recurring errors:
  1. On `t03` and `t05` (MTTR statistics), the agent occasionally called `search_incidents` with `limit=10` and tried to average the returned subset manually instead of calling `compute_mttr_stats`.
  2. On `t07` (an out-of-domain Python list-comprehension question expecting `no_call`), the agent unnecessarily called `search_incidents` in 4 of 5 runs.
  3. On `t08` (`INC-2026-999`), the agent occasionally followed a not-found lookup with a redundant broad search.
- **What we changed (`Change:` above)**: We rewrote every tool description in `server.py` to explicitly specify `What it does:`, `When to use it:`, and `When not to use it:` (for example, instructing the agent in `search_incidents` *not* to use it for MTTR percentiles, and instructing all tools not to run on unrelated general programming questions).
- **After (`eval/results-after.json`)**: Across 50 runs (5 per task), reliability improved from `pass_1 = 0.82` $\rightarrow$ **`1.00`** and `pass_5 = 0.70` $\rightarrow$ **`1.00`** (10/10 tasks passing all 5 runs).
