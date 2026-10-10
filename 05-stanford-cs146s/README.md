# Course 5: Stanford CS146S — The Modern Software Developer (Fall 2026)

**Instructor:** Mihail Eric (Stanford University)  
**Repository Directory:** `05-stanford-cs146s/`

---

## Overview & Deliverables Structure

This directory contains the complete solutions, runnable implementations, protocol-level tests, and technical writeups for **Weeks 1, 2, and 3** of Stanford CS146S (*The Modern Software Developer*, Fall 2026):

```text
05-stanford-cs146s/
├── README.md                          # Course overview and verification commands
├── week1/                             # Week 1: Trace Dissection of a Real Claude Code Session
│   ├── session_trace.jsonl            # Sanitized 8-turn POST /v1/messages HTTP capture
│   └── writeup.md                     # Full 5-part annotation (System Prompt, Tool Schemas, [OBSERVED]/[INFERRED] Behavioral Analysis)
├── week2/                             # Week 2: Build an MCP Server for a Third-Party API (GitHub Release Workflow)
│   ├── .mcp.json.example              # Sanitized Claude Code / Cursor MCP registration config
│   ├── oauth_client.py                # OAuth 2.0 Authorization Code + PKCE (S256), 0600 cache, & silent refresh
│   ├── server.py                      # MCP JSON-RPC 2.0 stdio server with 4 composing tools & dry_run brake
│   ├── test_mcp_server.py             # Pytest suite (OAuth PKCE, refresh, mid-session expiry, 4-tool chain, stdio protocol)
│   └── writeup.md                     # Agent ergonomics, OAuth scopes, and chained/failure transcripts
└── week3/                             # Week 3: Author a Reusable Agent Skill (DB Migration & Rollback Safety Audit)
    ├── SKILL.md                       # Agent Skills standard entrypoint with 3-way risk-tier branching
    ├── scripts/
    │   └── check_migration.py         # Deterministic SQL/Alembic DDL lock & rollback static analyzer
    ├── templates/
    │   ├── audit_report_template.md   # Standardized migration audit output template
    │   └── expand_contract_playbook.md# On-demand remediation reference for lock-risk and destructive DDL
    ├── test_skill_triggers.py         # Pytest suite (3 positive prompts, 1 near-miss prompt, 3 DDL risk branches)
    └── writeup.md                     # Workflow selection, trigger engineering, and end-to-end execution report
```

---

## Quick Verification Commands

Run the full `pytest` suite across Week 2 and Week 3 from `05-stanford-cs146s/`:

```bash
pytest -v week2/test_mcp_server.py week3/test_skill_triggers.py
```

Run the Week 3 static DDL migration analyzer directly:

```bash
python3 week3/scripts/check_migration.py --help
```

Start the Week 2 MCP server over `stdio`:

```bash
python3 week2/server.py
```
