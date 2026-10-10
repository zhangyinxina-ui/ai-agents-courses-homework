#!/usr/bin/env python3
"""Generate realistic local dataset data/engineering_incidents.json (55 records)
for Studio 05 MCP Tool Server.
"""
from __future__ import annotations

import json
from pathlib import Path

SERVICES = [
    "auth-service",
    "payments-api",
    "search-indexer",
    "notification-worker",
    "checkout-gateway",
    "billing-ledger",
]

SEVERITIES = ["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
STATUSES = ["resolved", "resolved", "resolved", "mitigated", "investigating"]
ENVIRONMENTS = ["prod-us-east", "prod-eu-west", "prod-ap-south", "staging"]

ROOT_CAUSES = {
    "auth-service": [
        "Redis cluster connection pool exhaustion during OAuth token refresh burst",
        "Expired intermediate TLS certificate on internal mTLS identity proxy",
        "JWKS public key cache stampede after automated key rotation",
    ],
    "payments-api": [
        "Upstream payment gateway webhook retry storm due to missing idempotency key lock",
        "Database deadlock on concurrent ledger row updates under flash-sale load",
        "gRPC deadline exceeded between payments-api and fraud-scoring microservice",
    ],
    "search-indexer": [
        "Elasticsearch bulk indexing queue backpressure caused by unthrottled catalog backfill",
        "Out-of-memory (OOMKilled) crash in vector embedding worker pod on oversized document",
        "Stale shard routing table after availability zone failover",
    ],
    "notification-worker": [
        "Kafka consumer group rebalance loop triggered by slow SMTP relay timeouts",
        "Rate-limit 429 throttling from external push notification provider",
        "Poison-pill malformed JSON payload in high-priority alert topic",
    ],
    "checkout-gateway": [
        "Nginx ingress controller worker connection limit reached during peak traffic",
        "Cart state deserialization error after schema migration v42",
        "DNS resolution latency spike on external tax calculation endpoint",
    ],
    "billing-ledger": [
        "PostgreSQL autovacuum lock contention on monthly invoice partition table",
        "Floating-point rounding drift check triggered safety halt in currency conversion job",
        "S3 export bucket IAM policy regression blocked nightly reconciliation snapshot",
    ],
}


def build_incidents(n: int = 55) -> list[dict]:
    records = []
    for idx in range(1, n + 1):
        inc_id = f"INC-2026-{idx:03d}"
        service = SERVICES[(idx - 1) % len(SERVICES)]
        # Ensure every service cycles across all 4 severities evenly
        severity = SEVERITIES[((idx - 1) + ((idx - 1) // len(SERVICES))) % len(SEVERITIES)]
        status = "resolved" if idx <= 48 else STATUSES[idx % len(STATUSES)]
        env = ENVIRONMENTS[(idx - 1) % len(ENVIRONMENTS)]
        causes = ROOT_CAUSES[service]
        root_cause = causes[(idx - 1) % len(causes)]

        base_mttr = {"SEV-1": 42, "SEV-2": 75, "SEV-3": 130, "SEV-4": 210}[severity]
        mttr_minutes = base_mttr + ((idx * 7) % 35) - 12
        day = ((idx - 1) % 28) + 1
        month = 8 if idx <= 28 else 9
        hour = (idx * 3) % 24

        records.append({
            "incident_id": inc_id,
            "title": f"{service} {severity} incident #{idx}: {root_cause[:58]}",
            "service": service,
            "severity": severity,
            "status": status,
            "environment": env,
            "detected_at": f"2026-{month:02d}-{day:02d}T{hour:02d}:15:00Z",
            "resolved_at": f"2026-{month:02d}-{day:02d}T{(hour + 1) % 24:02d}:{(15 + mttr_minutes) % 60:02d}:00Z" if status == "resolved" else None,
            "mttr_minutes": mttr_minutes if status == "resolved" else None,
            "oncall_engineer": f"eng-oncall-{(idx % 6) + 1}",
            "root_cause": root_cause,
            "summary": (
                f"Incident {inc_id} affected {service} in {env} ({severity}). "
                f"Root cause: {root_cause}. Status: {status}."
            ),
            "remediation_notes": [
                {
                    "note_id": f"NOTE-{idx:03d}-1",
                    "author": f"eng-oncall-{(idx % 6) + 1}",
                    "created_at": f"2026-{month:02d}-{day:02d}T{(hour + 2) % 24:02d}:00:00Z",
                    "note": f"Mitigated {inc_id} on {service} and added alerting threshold check.",
                }
            ],
        })
    return records


def main() -> None:
    here = Path(__file__).resolve().parent
    data_path = here / "data" / "engineering_incidents.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    incidents = build_incidents(55)
    data_path.write_text(json.dumps(incidents, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(incidents)} incident records to {data_path}")


if __name__ == "__main__":
    main()
