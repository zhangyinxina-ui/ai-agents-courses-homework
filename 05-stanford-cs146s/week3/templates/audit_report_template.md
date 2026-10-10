# Database Migration Safety Audit Report

- **Migration File:** `<file_path>`
- **Risk Tier (`check_migration.py`):** `ADDITIVE_SAFE` | `LOCK_RISK_EXPAND_CONTRACT` | `DESTRUCTIVE_DATA_LOSS`
- **Rollback (`downgrade`) Present:** `Yes` | `No`
- **Lock Timeout Guard Present:** `Yes` | `No`
- **Final Audit Verdict:** `PASS` | `WARN_REQUIRES_GUARD` | `FAIL_BLOCK_MERGE`

## 1. Static DDL Findings

| Rule ID | Severity | File:Line | Statement Excerpt | Hazard & Remediation |
| :--- | :--- | :--- | :--- | :--- |
| `<rule_id>` | `BLOCKER` / `WARNING` | `<file>:<line>` | `<sql_or_alembic>` | `<explanation>` |

## 2. Application Codebase Impact Check

- **ORM / Query References Found:** `<list of files/lines referencing modified table/columns>`
- **Rolling-Deploy Compatibility:** `<whether N-1 application pods can run against N schema>`

## 3. Required Remediation Diff (If Verdict != `PASS`)

```sql
-- Safe replacement DDL or Alembic upgrade()/downgrade() implementation
```
