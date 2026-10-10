"""Deterministic DDL & Alembic Migration Safety Analyzer for CS146S Week 3 Skill."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

RiskTier = Literal["ADDITIVE_SAFE", "LOCK_RISK_EXPAND_CONTRACT", "DESTRUCTIVE_DATA_LOSS"]


@dataclass
class Finding:
    rule_id: str
    severity: Literal["BLOCKER", "WARNING", "INFO"]
    line: int
    statement_excerpt: str
    message: str
    remediation: str


@dataclass
class MigrationAuditResult:
    file_path: str
    risk_tier: RiskTier
    has_rollback: bool
    has_lock_timeout: bool
    recommended_verdict: Literal["PASS", "WARN_REQUIRES_GUARD", "FAIL_BLOCK_MERGE"]
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_migration_text(content: str, file_path: str = "<memory>") -> MigrationAuditResult:
    """Analyze SQL or Alembic Python migration source and classify its DDL safety tier."""
    lines = content.splitlines()
    findings: list[Finding] = []
    has_destructive = False
    has_lock_risk = False
    has_lock_timeout = bool(re.search(r"lock_timeout", content, re.IGNORECASE))

    # Check rollback presence (either SQL "-- +migrate Down" / "-- rollback" or Python "def downgrade()")
    has_rollback = False
    if re.search(r"--\s*(?:\+migrate\s+Down|rollback\b)", content, re.IGNORECASE):
        has_rollback = True
    elif re.search(r"def\s+downgrade\s*\(", content):
        # Check if downgrade is just `pass` or `raise NotImplementedError`
        m = re.search(r"def\s+downgrade\s*\([^)]*\)\s*(?:->\s*None)?\s*:\s*\n((?:[ \t]+.*\n?)+)", content)
        if m:
            body_lines = [
                ln.strip()
                for ln in m.group(1).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            if body_lines and body_lines != ["pass"] and not any("NotImplementedError" in b for b in body_lines):
                has_rollback = True

    has_autocommit_block = bool(re.search(r"autocommit_block\s*\(", content))
    in_rollback_section = False

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if re.match(r"^--\s*(?:\+migrate\s+Down|rollback\b)", line, re.IGNORECASE) or re.match(
            r"^def\s+downgrade\s*\(", line
        ):
            in_rollback_section = True
            continue
        if in_rollback_section or not line or line.startswith("#") or line.startswith("--"):
            continue

        # 1. Destructive DDL detection (DROP TABLE, DROP COLUMN, TRUNCATE, op.drop_column, op.drop_table)
        if re.search(
            r"\b(DROP\s+TABLE|DROP\s+COLUMN|TRUNCATE\s+TABLE)\b|\bop\.(drop_column|drop_table)\s*\(",
            line,
            re.IGNORECASE,
        ):
            has_destructive = True
            findings.append(
                Finding(
                    rule_id="DDL-001-DESTRUCTIVE-DROP",
                    severity="BLOCKER",
                    line=idx,
                    statement_excerpt=line[:120],
                    message="Destructive DROP/TRUNCATE operation causes immediate data loss and breaks active application pods during rolling deploys.",
                    remediation=(
                        "Split into Expand-Contract phases: (Release 1) ignore column in ORM / stop reading & writing; "
                        "(Release 2) deploy DROP only after zero active queries reference the column."
                    ),
                )
            )

        # 2. Non-concurrent index creation on existing tables
        if re.search(r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\b", line, re.IGNORECASE) and not re.search(
            r"\bCONCURRENTLY\b", line, re.IGNORECASE
        ):
            has_lock_risk = True
            findings.append(
                Finding(
                    rule_id="DDL-002-NONCONCURRENT-INDEX",
                    severity="WARNING",
                    line=idx,
                    statement_excerpt=line[:120],
                    message="Standard CREATE INDEX acquires a SHARE lock that blocks all writes (INSERT/UPDATE/DELETE) for the duration of index build.",
                    remediation="Use `CREATE INDEX CONCURRENTLY` inside an autocommit block.",
                )
            )
        elif re.search(r"\bop\.create_index\s*\(", line) and "postgresql_concurrently=True" not in content:
            has_lock_risk = True
            findings.append(
                Finding(
                    rule_id="DDL-002-ALEMBIC-NONCONCURRENT-INDEX",
                    severity="WARNING",
                    line=idx,
                    statement_excerpt=line[:120],
                    message="Alembic `op.create_index` without `postgresql_concurrently=True` blocks concurrent table writes.",
                    remediation="Pass `postgresql_concurrently=True` inside `with op.get_context().autocommit_block():`.",
                )
            )

        # 3. CREATE INDEX CONCURRENTLY without autocommit_block in Alembic file
        if (
            file_path.endswith(".py")
            and ("postgresql_concurrently=True" in line or re.search(r"CONCURRENTLY", line, re.IGNORECASE))
            and not has_autocommit_block
        ):
            has_lock_risk = True
            findings.append(
                Finding(
                    rule_id="DDL-003-CONCURRENT-INDEX-IN-TX",
                    severity="BLOCKER",
                    line=idx,
                    statement_excerpt=line[:120],
                    message="CREATE INDEX CONCURRENTLY executed inside a transaction block will fail at runtime with `ActiveSqlTransaction`.",
                    remediation="Wrap the index operation in `with op.get_context().autocommit_block():`.",
                )
            )

        # 4. ADD COLUMN ... NOT NULL without safe default or volatile default
        if re.search(r"\bADD\s+(?:COLUMN\s+)?\w+\s+[^;]*\bNOT\s+NULL\b", line, re.IGNORECASE):
            if not re.search(r"\bDEFAULT\b", line, re.IGNORECASE) or re.search(
                r"\b(random|clock_timestamp|uuid_generate_v4)\s*\(", line, re.IGNORECASE
            ):
                has_lock_risk = True
                findings.append(
                    Finding(
                        rule_id="DDL-004-UNSAFE-NOT-NULL-ADD",
                        severity="BLOCKER",
                        line=idx,
                        statement_excerpt=line[:120],
                        message="Adding a NOT NULL column without a constant default fails on populated tables or forces a full-table rewrite under ACCESS EXCLUSIVE lock.",
                        remediation="Add the column as nullable first, backfill in batches, then add a CHECK constraint with `NOT VALID` and `VALIDATE CONSTRAINT`.",
                    )
                )

        # 5. ALTER COLUMN TYPE (full table rewrite under ACCESS EXCLUSIVE lock)
        if re.search(r"\bALTER\s+COLUMN\s+\w+\s+(?:SET\s+DATA\s+)?TYPE\b|\bop\.alter_column\([^)]*type_=", line, re.IGNORECASE):
            has_lock_risk = True
            findings.append(
                Finding(
                    rule_id="DDL-005-ALTER-COLUMN-TYPE",
                    severity="WARNING",
                    line=idx,
                    statement_excerpt=line[:120],
                    message="Altering a column type typically rewrites the entire table under an ACCESS EXCLUSIVE lock.",
                    remediation="Use an expand-contract dual-write column migration or ensure binary-compatible cast with `lock_timeout`.",
                )
            )

    if not has_rollback:
        findings.append(
            Finding(
                rule_id="DDL-006-MISSING-ROLLBACK",
                severity="WARNING" if not has_destructive else "BLOCKER",
                line=1,
                statement_excerpt="downgrade / rollback section",
                message="Migration lacks an executable rollback (`downgrade()` is missing, `pass`, or raises `NotImplementedError`).",
                remediation="Implement the inverse DDL statements in `downgrade()` or `-- +migrate Down`.",
            )
        )

    if has_destructive:
        risk_tier: RiskTier = "DESTRUCTIVE_DATA_LOSS"
        verdict: Literal["PASS", "WARN_REQUIRES_GUARD", "FAIL_BLOCK_MERGE"] = "FAIL_BLOCK_MERGE"
    elif has_lock_risk or any(f.severity == "BLOCKER" for f in findings):
        risk_tier = "LOCK_RISK_EXPAND_CONTRACT"
        verdict = "FAIL_BLOCK_MERGE" if any(f.severity == "BLOCKER" for f in findings) else "WARN_REQUIRES_GUARD"
    else:
        risk_tier = "ADDITIVE_SAFE"
        verdict = "PASS" if has_rollback else "WARN_REQUIRES_GUARD"

    return MigrationAuditResult(
        file_path=file_path,
        risk_tier=risk_tier,
        has_rollback=has_rollback,
        has_lock_timeout=has_lock_timeout,
        recommended_verdict=verdict,
        findings=findings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SQL/Alembic migration for DDL lock & rollback safety.")
    parser.add_argument("migration_file", help="Path to .sql or .py migration file")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    args = parser.parse_args()

    path = Path(args.migration_file)
    if not path.is_file():
        sys.stderr.write(f"Error: file not found: {path}\n")
        sys.exit(2)

    result = analyze_migration_text(path.read_text(encoding="utf-8"), file_path=str(path))
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"File: {result.file_path}")
        print(f"Risk Tier: {result.risk_tier} | Verdict: {result.recommended_verdict}")
        for f in result.findings:
            print(f"  [{f.severity}] {f.rule_id} (L{f.line}): {f.message}")


if __name__ == "__main__":
    main()
