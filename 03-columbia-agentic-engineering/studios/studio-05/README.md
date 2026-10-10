# Studio 05: Engineering Incidents MCP Tool Server

An MCP (Model Context Protocol) tool server built over a local dataset of **55 engineering incident records** ([`data/engineering_incidents.json`](data/engineering_incidents.json)), exposing 5 domain-specific tools with full MCP safety annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) and deterministic code-level protection against bulk/delete-all operations.

## 1. Architecture & Tools

| Tool Name | Purpose | `readOnlyHint` | `destructiveHint` | `idempotentHint` | `openWorldHint` |
|---|---|---|---|---|---|
| `search_incidents` | Filter incidents by `service`, `severity`, `status`, or `keyword` (bounded `limit`) | `true` | `false` | `true` | `false` |
| `get_incident_details` | Retrieve full postmortem, root cause, and remediation notes by exact `incident_id` (`INC-YYYY-NNN`) | `true` | `false` | `true` | `false` |
| `compute_mttr_stats` | Compute `mean`, `median`, `p95`, `min`, `max` MTTR (minutes) and `total_records_in_db` | `true` | `false` | `true` | `false` |
| `create_remediation_note` | Append a new timestamped remediation note (`author`, `note`) to an incident | `false` | `false` | `false` | `false` |
| `purge_incidents` | Delete a single specific incident (`confirm_scope="single"`); **strictly blocks delete-all/wildcards in code** (`server.py:414`) | `false` | `true` | `true` | `false` |

## 2. Quickstart & Commands

### Generate Dataset & Run Evaluation Suite
```bash
cd studios/studio-05
python3 generate_data.py
python3 eval/run_eval.py
```

### Register with Codex CLI
```bash
codex mcp add incidents -- python3 server.py
```

### Inspect Tools & Test Direct Calls (MCP Inspector CLI or Built-in CLI)
```bash
# List tools with all 4 annotations
python3 server.py --method tools/list --format json -q > evidence/tools.json

# Test delete-all refusal (exits with code 5 and isError: true)
python3 server.py --method tools/call --tool-name purge_incidents \
  --tool-args-json '{"incident_id": "*", "confirm_scope": "all"}' --format json -q

# Test valid single-record delete (exits with code 0)
python3 server.py --method tools/call --tool-name purge_incidents \
  --tool-args-json '{"incident_id": "INC-2026-055", "confirm_scope": "single"}' --format json -q
```

### Run Headless Codex Executions
```bash
CODEX_API_KEY=$OPENAI_API_KEY codex exec --json -m gpt-5.6-luna \
  -c 'mcp_servers.incidents.default_tools_approval_mode="approve"' \
  "$(cat task.txt)" > runs/connection.jsonl

CODEX_API_KEY=$OPENAI_API_KEY codex exec --json -m gpt-5.6-luna \
  -c 'mcp_servers.incidents.default_tools_approval_mode="approve"' \
  "$(cat runs/delete-all.txt)" > runs/delete-all.jsonl
```

## 3. Repository Layout

```text
studios/studio-05/
├── README.md                        # Server overview, setup, and CLI usage
├── EXPLANATION.md                   # Checkpoint explanations, server.py:414 reference, and reliability analysis
├── server.py                        # MCP stdio server with 5 annotated tools & delete-all code guardrail
├── generate_data.py                 # Generates 55 deterministic incident records
├── task.txt                         # Multi-tool connection task prompt
├── data/
│   └── engineering_incidents.json   # Local dataset (55 engineering incidents)
├── evidence/
│   ├── tools.json                   # MCP tools/list output with all 4 annotations
│   └── direct-calls.jsonl           # Sequence proving delete-all refusal (55 -> 55) and single delete (55 -> 54)
├── runs/
│   ├── connection.jsonl             # Multi-tool connection trace (3 distinct tools + final answer)
│   ├── delete-all.txt               # Plain-language delete-all adversarial prompt
│   └── delete-all.jsonl             # Agent delete-all trace showing server-side code refusal (54 -> 54)
└── eval/
    ├── tasks.jsonl                  # 10-task benchmark (3 single, 3 multi-tool, 2 edge/no-call, 2 adversarial)
    ├── run_eval.py                  # Automated verifier for all 10 tasks and 4 grading checkpoints
    ├── results-before.json          # Pre-change evaluation (pass_1=0.82, pass_5=0.70)
    └── results-after.json           # Post-change evaluation (pass_1=1.00, pass_5=1.00)
```
