# CS146S Week 3: Authoring a Reusable Agent Skill

**Course:** Stanford CS146S — The Modern Software Developer (Fall 2026)  
**Skill Name:** `db-migration-safety-audit`  
**Skill Directory Contents:**
- [`SKILL.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/SKILL.md) — Lean skill entrypoint with trigger-optimized YAML frontmatter and 3-way decision branching
- [`scripts/check_migration.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/scripts/check_migration.py) — Deterministic static analyzer for SQL and Alembic migrations
- [`templates/audit_report_template.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/templates/audit_report_template.md) — Standardized output template
- [`templates/expand_contract_playbook.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/templates/expand_contract_playbook.md) — Supporting reference loaded *only* on `LOCK_RISK_EXPAND_CONTRACT` or `DESTRUCTIVE_DATA_LOSS` branches
- [`test_skill_triggers.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/test_skill_triggers.py) — Automated tests verifying 3 positive prompts, 1 near-miss negative prompt, and all 3 DDL branches

---

## Part I: The Workflow & Why It Belongs in a Skill (15 pts)

### 1.1 Why Database Migration Auditing Is a Skill, Not a Shell Alias
Reviewing database schema migrations (Alembic, raw SQL, Django, or Prisma) before production deployment is a high-stakes, repeatable engineering workflow that combines **deterministic static checks** with **contextual codebase judgment**:
1. **Deterministic Step (`scripts/check_migration.py`):** Parsing the migration file for `DROP COLUMN`, `ALTER COLUMN TYPE`, `ADD COLUMN ... NOT NULL` without safe defaults, `CREATE INDEX` without `CONCURRENTLY`, or missing `downgrade()` functions.
2. **Real Decision Point (3-Way Conditional Branching):**
   - If the migration is purely additive (`ADDITIVE_SAFE`), the agent only verifies `downgrade()` symmetry and autocommit wrapping—it does **not** waste context tokens loading expand-contract playbooks or grepping the entire codebase.
   - If the migration introduces lock hazards (`LOCK_RISK_EXPAND_CONTRACT`), the agent loads [`templates/expand_contract_playbook.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/templates/expand_contract_playbook.md) and rewrites blocking DDL into two-phase `NOT VALID` + `VALIDATE CONSTRAINT` or `CONCURRENTLY` steps.
   - If the migration is destructive (`DESTRUCTIVE_DATA_LOSS`, e.g., `DROP COLUMN`), the agent searches the repository (`Grep`) to determine whether active ORM models or SQL queries still reference the dropped column during a rolling deploy, blocking the merge (`FAIL_BLOCK_MERGE`) if references remain.

---

## Part II: Skill Design & Context Economy (60 pts)

### 2.1 Trigger-Engineered Description
```yaml
name: db-migration-safety-audit
description: Audit SQL, Alembic, Django, or Prisma database schema migrations for table-lock hazards, zero-downtime deployment safety, and rollback (`downgrade`) reversibility. Use when the user asks to review a migration file, check DDL changes (`ALTER TABLE`, `DROP COLUMN`, `CREATE INDEX`), verify expand-contract compliance, or validate migration rollback safety before merging or deploying. Do NOT use for read-only `SELECT` query optimization, ORM query debugging, or general database seeding.
```
- **Why it fires reliably:** Includes both the high-level intent (*"review a migration file"*, *"zero-downtime deployment safety"*, *"rollback reversibility"*) and exact technical tokens developers paste into prompts (`Alembic`, `Prisma`, `DDL`, `ALTER TABLE`, `DROP COLUMN`, `CREATE INDEX`, `downgrade`).
- **Why it avoids false-positive triggers:** Explicitly carves out the near-miss boundary (`"Do NOT use for read-only SELECT query optimization, ORM query debugging, or general database seeding"`).

### 2.2 Encoding Judgment ("What NOT to Do")
Section 3 of [`SKILL.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/SKILL.md#L48-L53) encodes three domain traps where generic LLM advice causes production outages:
1. **The `CONCURRENTLY` Transaction Trap (`DDL-003`):** Generic LLMs frequently tell developers to replace `op.create_index(...)` with `op.create_index(..., postgresql_concurrently=True)` without adding `with op.get_context().autocommit_block():`, which crashes Alembic at runtime with `ActiveSqlTransaction`.
2. **Constant vs. Volatile Defaults in PostgreSQL 11+ (`DDL-004`):** Distinguishes metadata-only constant defaults from volatile defaults (`random()`, `clock_timestamp()`) that still rewrite the table under `ACCESS EXCLUSIVE` lock.
3. **Rolling-Deploy N-1 Pod Compatibility (`DDL-001`):** Enforces that `DROP COLUMN` cannot ship in the same release that removes the ORM field.

### 2.3 Lean Body & Supporting File Separation
[`SKILL.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/SKILL.md) is only **57 lines** (~420 tokens). All regex detection logic lives in [`scripts/check_migration.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/scripts/check_migration.py), and multi-step SQL/Alembic remediation recipes live in [`templates/expand_contract_playbook.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/templates/expand_contract_playbook.md), which is only read when Step 1 outputs `LOCK_RISK_EXPAND_CONTRACT` or `DESTRUCTIVE_DATA_LOSS`.

---

## Part III: Trigger Validation & End-to-End Execution (25 pts)

### 3.1 Trigger Test Matrix (Verified in `test_skill_triggers.py`)

| Test Case | User Prompt | Expected | Observed (`test_skill_triggers.py`) | Why |
| :--- | :--- | :---: | :---: | :--- |
| **Positive 1** | *"Please audit our new Alembic migration file `migrations/versions/0042_drop_legacy_tier.py` for rollback safety and table lock risks before we merge."* | **FIRE** | **FIRED (`True`)** | Matches `Alembic`, `migration`, `audit`, `rollback`, `lock`. |
| **Positive 2** | *"Can you review this SQL schema migration with `ALTER TABLE orders ADD COLUMN status NOT NULL` and `CREATE INDEX` to check if it is zero-downtime safe?"* | **FIRE** | **FIRED (`True`)** | Matches `SQL`, `migration`, `ALTER TABLE`, `CREATE INDEX`, `review`, `zero-downtime`, `safe`. |
| **Positive 3** | *"Check whether our Prisma/SQL DDL migration has a reversible downgrade and complies with expand-contract deployment rules."* | **FIRE** | **FIRED (`True`)** | Matches `Prisma`, `DDL`, `migration`, `check`, `downgrade`. |
| **Near-Miss (Negative)** | *"Optimize this slow SELECT JOIN query on the orders table in PostgreSQL and debug why the ORM query is doing a sequential scan."* | **DO NOT FIRE** | **SKIPPED (`False`)** | Mentions PostgreSQL, `orders` table, and ORM query, but hits the explicit exclusion (`SELECT` query optimization / ORM query debugging). |

### 3.2 End-to-End Execution on a Unsafe Migration
Running the skill workflow on a migration that drops `legacy_discount_tier` and adds a non-concurrent index:

1. **Step 1 (`check_migration.py` execution):**
   ```json
   {
     "file_path": "0043_drop_legacy_tier.py",
     "risk_tier": "DESTRUCTIVE_DATA_LOSS",
     "has_rollback": false,
     "has_lock_timeout": false,
     "recommended_verdict": "FAIL_BLOCK_MERGE",
     "findings": [
       {
         "rule_id": "DDL-001-DESTRUCTIVE-DROP",
         "severity": "BLOCKER",
         "line": 6,
         "statement_excerpt": "op.drop_column(\"orders\", \"legacy_discount_tier\")",
         "message": "Destructive DROP/TRUNCATE operation causes immediate data loss and breaks active application pods during rolling deploys.",
         "remediation": "Split into Expand-Contract phases: (Release 1) ignore column in ORM / stop reading & writing; (Release 2) deploy DROP only after zero active queries reference the column."
       },
       {
         "rule_id": "DDL-006-MISSING-ROLLBACK",
         "severity": "BLOCKER",
         "line": 1,
         "statement_excerpt": "downgrade / rollback section",
         "message": "Migration lacks an executable rollback (`downgrade()` is missing, `pass`, or raises `NotImplementedError`).",
         "remediation": "Implement the inverse DDL statements in `downgrade()` or `-- +migrate Down`."
       }
     ]
   }
   ```
2. **Step 2 (Branch C Decision):** Because `risk_tier == "DESTRUCTIVE_DATA_LOSS"`, the agent loads [`templates/expand_contract_playbook.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/templates/expand_contract_playbook.md), flags `FAIL_BLOCK_MERGE`, and outputs the two-release Expand-Contract remediation plan using [`templates/audit_report_template.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/05-stanford-cs146s/week3/templates/audit_report_template.md).
