# Zero-Downtime Expand-Contract Playbook (Supporting Reference)

Load this reference **only** when `check_migration.py` classifies a migration as `LOCK_RISK_EXPAND_CONTRACT` or `DESTRUCTIVE_DATA_LOSS`.

---

## 1. Fixing `LOCK_RISK_EXPAND_CONTRACT` Hazards

### 1.1 Adding a `NOT NULL` Column Without Table Rewrite
**Unsafe (`ACCESS EXCLUSIVE` table scan/failure on existing rows):**
```sql
ALTER TABLE orders ADD COLUMN payment_intent_id VARCHAR(64) NOT NULL;
```

**Safe Expand-Contract Pattern:**
```sql
-- Migration 1 (Expand): Add nullable column + NOT VALID check constraint
SET LOCAL lock_timeout = '2s';
ALTER TABLE orders ADD COLUMN payment_intent_id VARCHAR(64);
ALTER TABLE orders ADD CONSTRAINT ck_orders_payment_intent_not_null
    CHECK (payment_intent_id IS NOT NULL) NOT VALID;

-- Migration 2 (Backfill + Validate under SHARE UPDATE EXCLUSIVE lock):
ALTER TABLE orders VALIDATE CONSTRAINT ck_orders_payment_intent_not_null;
```

### 1.2 Creating an Index Without Blocking Writes in Alembic
**Unsafe:**
```python
def upgrade() -> None:
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"])
```

**Safe (Autocommit + `postgresql_concurrently=True`):**
```python
def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_orders_customer_id",
            "orders",
            ["customer_id"],
            postgresql_concurrently=True,
        )

def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_orders_customer_id",
            table_name="orders",
            postgresql_concurrently=True,
        )
```

---

## 2. Fixing `DESTRUCTIVE_DATA_LOSS` (`DROP COLUMN` / `DROP TABLE`)

Never execute `DROP COLUMN` in the same release that removes the column from application code:
1. **Release $N$ (Code Change Only):** Remove the column from SQLAlchemy/Django/Prisma models (or mark `__deferred__` / ignored) so running pods never `SELECT` or `INSERT` it. If the column has a `NOT NULL` constraint in DB, drop only the `NOT NULL` constraint (`ALTER TABLE ... ALTER COLUMN ... DROP NOT NULL`).
2. **Release $N+1$ (Contract Migration):** After confirming zero active queries reference the column, deploy the `ALTER TABLE ... DROP COLUMN ...` migration with a pre-drop backup snapshot or archive table.
