"""Trigger evaluation and static analyzer tests for CS146S Week 3 Skill (`db-migration-safety-audit`)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week3.scripts.check_migration import analyze_migration_text

SKILL_MD_PATH = Path(__file__).resolve().parent / "SKILL.md"


def parse_skill_frontmatter(skill_md_path: Path = SKILL_MD_PATH) -> dict[str, str]:
    """Extract YAML frontmatter keys (`name`, `description`) from SKILL.md."""
    text = skill_md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    assert match is not None, "SKILL.md must begin with valid YAML frontmatter delimited by ---"
    frontmatter: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            frontmatter[k.strip()] = v.strip()
    return frontmatter


def should_trigger_skill(user_prompt: str, description: str) -> tuple[bool, str]:
    """Deterministic skill-router matcher evaluating positive intent vs explicit negative exclusions in description."""
    prompt_lower = user_prompt.lower()
    desc_lower = description.lower()

    # 1. Check explicit negative exclusions encoded in SKILL.md description ("Do NOT use for ...")
    negative_patterns = [
        r"\bselect\b.*\bquery\s+optimization\b",
        r"\bslow\s+select\b",
        r"\boptimize\s+this\s+slow\s+select\b",
        r"\borm\s+query\s+debugging\b",
        r"\bdatabase\s+seeding\b",
    ]
    for pat in negative_patterns:
        if re.search(pat, prompt_lower):
            return False, f"Blocked by negative exclusion pattern: {pat}"

    # 2. Check positive migration / DDL / rollback trigger concepts present in description
    positive_concepts = [
        ("migration", ("migration", "alembic", "prisma", "django", "ddl", "alter table", "drop column")),
        ("safety_or_audit", ("audit", "review", "check", "lock", "rollback", "downgrade", "zero-downtime", "safe")),
    ]
    matched_groups = []
    for group_name, keywords in positive_concepts:
        if any(kw in prompt_lower and kw in desc_lower for kw in keywords):
            matched_groups.append(group_name)

    if len(matched_groups) == len(positive_concepts):
        return True, f"Matched positive trigger groups: {matched_groups}"
    return False, f"Insufficient trigger overlap: {matched_groups}"


def test_skill_frontmatter_and_three_positive_plus_one_near_miss_prompts() -> None:
    """Verify SKILL.md frontmatter triggers on 3 positive prompts and rejects 1 near-miss prompt."""
    fm = parse_skill_frontmatter()
    assert fm["name"] == "db-migration-safety-audit"
    description = fm["description"]

    positive_prompts = [
        "Please audit our new Alembic migration file migrations/versions/0042_drop_legacy_tier.py for rollback safety and table lock risks before we merge.",
        "Can you review this SQL schema migration with ALTER TABLE orders ADD COLUMN status NOT NULL and CREATE INDEX to check if it is zero-downtime safe?",
        "Check whether our Prisma/SQL DDL migration has a reversible downgrade and complies with expand-contract deployment rules.",
    ]
    for prompt in positive_prompts:
        fired, reason = should_trigger_skill(prompt, description)
        assert fired is True, f"Expected skill to trigger on positive prompt: {prompt!r} ({reason})"

    near_miss_prompt = (
        "Optimize this slow SELECT JOIN query on the orders table in PostgreSQL and debug why the ORM query is doing a sequential scan."
    )
    fired, reason = should_trigger_skill(near_miss_prompt, description)
    assert fired is False, f"Expected near-miss prompt NOT to trigger skill: {near_miss_prompt!r} ({reason})"


def test_check_migration_branch_a_additive_safe() -> None:
    """Verify Branch A (`ADDITIVE_SAFE`) passes when nullable column + concurrent index in autocommit + downgrade exist."""
    alembic_safe = '''
from alembic import op
import sqlalchemy as sa

def upgrade() -> None:
    op.add_column("orders", sa.Column("coupon_code", sa.String(length=32), nullable=True))
    with op.get_context().autocommit_block():
        op.create_index("ix_orders_coupon_code", "orders", ["coupon_code"], postgresql_concurrently=True)

def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index("ix_orders_coupon_code", table_name="orders", postgresql_concurrently=True)
    op.drop_column("orders", "coupon_code")
'''
    res = analyze_migration_text(alembic_safe, file_path="0041_add_coupon_code.py")
    assert res.risk_tier == "ADDITIVE_SAFE"
    assert res.has_rollback is True
    assert res.recommended_verdict == "PASS"
    assert res.findings == []


def test_check_migration_branch_b_lock_risk() -> None:
    """Verify Branch B (`LOCK_RISK_EXPAND_CONTRACT`) flags non-concurrent index and unsafe NOT NULL column."""
    sql_lock_risk = """
-- +migrate Up
ALTER TABLE orders ADD COLUMN payment_intent_id VARCHAR(64) NOT NULL;
CREATE INDEX idx_orders_payment_intent ON orders (payment_intent_id);

-- +migrate Down
DROP INDEX idx_orders_payment_intent;
ALTER TABLE orders DROP COLUMN payment_intent_id;
"""
    res = analyze_migration_text(sql_lock_risk, file_path="0042_payment_intent.sql")
    assert res.risk_tier == "LOCK_RISK_EXPAND_CONTRACT"
    rule_ids = {f.rule_id for f in res.findings}
    assert "DDL-004-UNSAFE-NOT-NULL-ADD" in rule_ids
    assert "DDL-002-NONCONCURRENT-INDEX" in rule_ids


def test_check_migration_branch_c_destructive_data_loss() -> None:
    """Verify Branch C (`DESTRUCTIVE_DATA_LOSS`) blocks merge on DROP COLUMN and empty downgrade."""
    alembic_destructive = '''
from alembic import op
import sqlalchemy as sa

def upgrade() -> None:
    op.drop_column("orders", "legacy_discount_tier")

def downgrade() -> None:
    pass
'''
    res = analyze_migration_text(alembic_destructive, file_path="0043_drop_legacy_tier.py")
    assert res.risk_tier == "DESTRUCTIVE_DATA_LOSS"
    assert res.has_rollback is False
    assert res.recommended_verdict == "FAIL_BLOCK_MERGE"
    rule_ids = {f.rule_id for f in res.findings}
    assert "DDL-001-DESTRUCTIVE-DROP" in rule_ids
    assert "DDL-006-MISSING-ROLLBACK" in rule_ids
