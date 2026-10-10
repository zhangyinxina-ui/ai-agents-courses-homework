---
name: db-migration-safety-audit
description: Audit SQL, Alembic, Django, or Prisma database schema migrations for table-lock hazards, zero-downtime deployment safety, and rollback (`downgrade`) reversibility. Use when the user asks to review a migration file, check DDL changes (`ALTER TABLE`, `DROP COLUMN`, `CREATE INDEX`), verify expand-contract compliance, or validate migration rollback safety before merging or deploying. Do NOT use for read-only `SELECT` query optimization, ORM query debugging, or general database seeding.
---

# Database Schema Migration & Rollback Safety Audit

Follow this procedure whenever auditing a database schema migration (`.sql` or Alembic/Django `.py` migration file).

## Step 1: Run Deterministic Static DDL Classification First

Before reading the surrounding application code, run the bundled analyzer on the target migration file(s):

```bash
python3 week3/scripts/check_migration.py <path-to-migration-file> --format json
```

The analyzer parses DDL operations and classifies the migration into one of three risk tiers:
- **`ADDITIVE_SAFE`** (e.g., nullable `ADD COLUMN`, `CREATE TABLE`, `CREATE INDEX CONCURRENTLY` outside a transaction block).
- **`LOCK_RISK_EXPAND_CONTRACT`** (e.g., `ADD COLUMN ... NOT NULL` with volatile default, non-concurrent `CREATE INDEX`, `ALTER COLUMN TYPE`, adding a foreign key without `NOT VALID`).
- **`DESTRUCTIVE_DATA_LOSS`** (e.g., `DROP TABLE`, `DROP COLUMN`, `TRUNCATE`, or irreversible `DELETE`/`UPDATE` inside a schema migration).

## Step 2: Decision Point — Branch by Classification Tier

Do **not** apply a one-size-fits-all checklist. Branch based on `risk_tier` from Step 1:

### Branch A: `ADDITIVE_SAFE`
1. Verify that the `downgrade()` / rollback block drops the exact objects created in `upgrade()` in reverse dependency order.
2. Confirm that `CREATE INDEX CONCURRENTLY` (if present) executes inside an autocommit block (`op.get_context().autocommit_block()`), because PostgreSQL rejects `CONCURRENTLY` inside a transaction.
3. Produce the final audit using [`templates/audit_report_template.md`](templates/audit_report_template.md) without loading the expand-contract playbook.

### Branch B: `LOCK_RISK_EXPAND_CONTRACT`
1. Read [`templates/expand_contract_playbook.md`](templates/expand_contract_playbook.md) to select the zero-downtime rewrite pattern.
2. Inspect the codebase (`Grep` for the table and column names) to check whether running application instances read or write the affected table on the hot path.
3. Require an explicit `SET LOCAL lock_timeout = '2s';` guard and rewrite blocking operations into two-phase steps (e.g., `ADD CONSTRAINT ... NOT VALID` followed by `VALIDATE CONSTRAINT`).

### Branch C: `DESTRUCTIVE_DATA_LOSS`
1. **Hard Block (`BLOCK_MERGE`):** Dropping a column or table while old application pods are still running during a rolling deploy causes immediate `UndefinedColumn` / `UndefinedTable` 500 errors, and permanently destroys data on rollback.
2. Read [`templates/expand_contract_playbook.md`](templates/expand_contract_playbook.md) (Section 2: Destructive DDL) and verify whether a prior release already removed all ORM/SQL references to the target column/table.
3. If application code in the current diff still references the dropped column/table—or if `downgrade()` cannot restore dropped data—mark the audit **`FAIL (BLOCK_MERGE)`** and provide the exact 3-release expand-contract split.

## Step 3: What NOT to Do (Judgment Rules)

- **Never approve a empty `downgrade(): pass`** on an `ADDITIVE_SAFE` or `LOCK_RISK` migration; every additive change has a deterministic inverse.
- **Never suggest `CREATE INDEX CONCURRENTLY` without checking transaction wrapping:** In Alembic/PostgreSQL, `CONCURRENTLY` inside the default transaction raises `ActiveSqlTransaction: CREATE INDEX CONCURRENTLY cannot run inside a transaction block`.
- **Never trust `ADD COLUMN ... NOT NULL DEFAULT ...` blindly:** While PostgreSQL 11+ metadata-optimizes *constant* defaults, *volatile* defaults (e.g., `clock_timestamp()`, `random()`) still rewrite the entire table under an `ACCESS EXCLUSIVE` lock.

## Step 4: Format Output

Render the final review following [`templates/audit_report_template.md`](templates/audit_report_template.md), citing exact file paths and line numbers for every finding.
