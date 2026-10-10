#!/usr/bin/env python3
"""Studio 05 MCP Tool Server: Engineering Incidents Reliability & Postmortem Server.

Exposes 5 tools over local data (`data/engineering_incidents.json`, >= 50 records):
  1. search_incidents        (readOnlyHint=True,  destructiveHint=False, idempotentHint=True,  openWorldHint=False)
  2. get_incident_details    (readOnlyHint=True,  destructiveHint=False, idempotentHint=True,  openWorldHint=False)
  3. compute_mttr_stats      (readOnlyHint=True,  destructiveHint=False, idempotentHint=True,  openWorldHint=False)
  4. create_remediation_note (readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
  5. purge_incidents         (readOnlyHint=False, destructiveHint=True,  idempotentHint=True,  openWorldHint=False)

Enforces in code (lines 212-230) that any attempt to delete all data or perform a
wildcard/bulk delete (e.g., confirm_scope="all", incident_id="*", "ALL", "INC-*")
is strictly refused with a structured safety error (isError=True).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = HERE / "data" / "engineering_incidents.json"
SINGLE_ID_RE = re.compile(r"^INC-\d{4}-\d{3}$")


class ToolExecutionError(RuntimeError):
    """Structured error returned with isError=True in MCP tool responses."""

    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            **self.details,
        }


def load_incidents(data_path: Path = DEFAULT_DATA_PATH) -> list[dict[str, Any]]:
    if not data_path.exists():
        from generate_data import build_incidents

        data_path.parent.mkdir(parents=True, exist_ok=True)
        records = build_incidents(55)
        data_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        return records
    return json.loads(data_path.read_text(encoding="utf-8"))


def save_incidents(records: list[dict[str, Any]], data_path: Path = DEFAULT_DATA_PATH) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool Metadata & Schemas (with all 4 required MCP annotations on every tool)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_incidents",
        "description": (
            "What it does: Searches and filters local engineering incident records by service name, "
            "severity (SEV-1 to SEV-4), status (resolved, mitigated, investigating), or text keyword, "
            "returning matching incident summaries and total database record count. "
            "When to use it: Use this tool when you need to discover incidents matching specific filter "
            "criteria or check how many total incident records exist in the database. "
            "When not to use it: Do not use this tool when you already have an exact incident_id and need "
            "its full remediation notes (use get_incident_details instead), or when computing aggregate "
            "MTTR percentiles (use compute_mttr_stats instead)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Optional service filter (e.g., 'auth-service', 'payments-api', 'search-indexer', 'notification-worker', 'checkout-gateway', 'billing-ledger').",
                },
                "severity": {
                    "type": "string",
                    "description": "Optional severity filter ('SEV-1', 'SEV-2', 'SEV-3', or 'SEV-4').",
                },
                "status": {
                    "type": "string",
                    "description": "Optional status filter ('resolved', 'mitigated', or 'investigating').",
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional case-insensitive substring to match against title, summary, or root_cause.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching incident summaries to return (1 to 55, default 10).",
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "get_incident_details",
        "description": (
            "What it does: Retrieves the complete record for a single engineering incident by its exact "
            "incident_id (e.g., 'INC-2026-001'), including timestamps, root cause, on-call engineer, and "
            "all attached remediation notes. "
            "When to use it: Use this tool when you know a specific incident_id and need its full root-cause "
            "analysis, resolution timeline, or remediation note history. "
            "When not to use it: Do not use this tool to browse or filter multiple incidents by service or "
            "severity (use search_incidents instead) or to mutate/delete records."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "incident_id": {
                    "type": "string",
                    "description": "Exact incident identifier in format 'INC-YYYY-NNN' (e.g., 'INC-2026-001').",
                },
            },
            "required": ["incident_id"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "compute_mttr_stats",
        "description": (
            "What it does: Computes aggregate Mean Time To Resolution (MTTR) metrics in minutes—count, "
            "mean, median, p95, min, max, and total_records_in_db—across resolved engineering incidents, "
            "optionally filtered by service and/or severity. "
            "When to use it: Use this tool when asked for quantitative reliability metrics, average/median/p95 "
            "resolution times, or total database record counts. "
            "When not to use it: Do not use this tool to fetch individual incident narratives or remediation "
            "notes (use get_incident_details or search_incidents instead)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Optional service name to scope MTTR calculation (e.g., 'payments-api').",
                },
                "severity": {
                    "type": "string",
                    "description": "Optional severity level to scope MTTR calculation (e.g., 'SEV-1').",
                },
            },
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "create_remediation_note",
        "description": (
            "What it does: Appends a new timestamped postmortem remediation note (with author and note text) "
            "to an existing incident record in the local JSON dataset. "
            "When to use it: Use this tool only when explicitly instructed to log, append, or record a new "
            "remediation action item or postmortem note on a specific incident_id. "
            "When not to use it: Do not use this tool for read-only queries, and do not use it to delete or "
            "purge incidents (use purge_incidents for single-record deletion)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "incident_id": {
                    "type": "string",
                    "description": "Exact incident ID to attach the remediation note to (e.g., 'INC-2026-005').",
                },
                "author": {
                    "type": "string",
                    "description": "Identifier or name of the engineer recording the note (e.g., 'sre-lead').",
                },
                "note": {
                    "type": "string",
                    "description": "Non-empty remediation action item or postmortem observation to append.",
                },
            },
            "required": ["incident_id", "author", "note"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "purge_incidents",
        "description": (
            "What it does: Permanently deletes a single engineering incident record from the local dataset "
            "by its exact incident_id when confirm_scope is set to 'single'. "
            "When to use it: Use this tool only when explicitly asked to remove a single specific duplicate "
            "or test incident record identified by an exact 'INC-YYYY-NNN' ID. "
            "When not to use it: Never use this tool to delete all records, wipe the database, or delete "
            "multiple incidents via wildcards ('*', 'ALL', 'INC-*', confirm_scope='all'); bulk deletion is "
            "strictly refused by server code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "incident_id": {
                    "type": "string",
                    "description": "Exact single incident ID to purge ('INC-YYYY-NNN'). Wildcards or 'ALL' are rejected.",
                },
                "confirm_scope": {
                    "type": "string",
                    "description": "Must be 'single' to confirm single-record deletion. Any bulk/all scope is rejected.",
                    "default": "single",
                },
            },
            "required": ["incident_id", "confirm_scope"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def tool_search_incidents(
    arguments: dict[str, Any],
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    records = load_incidents(data_path)
    service = (arguments.get("service") or "").strip().lower()
    severity = (arguments.get("severity") or "").strip().upper()
    status = (arguments.get("status") or "").strip().lower()
    keyword = (arguments.get("keyword") or "").strip().lower()
    limit = int(arguments.get("limit", 10) or 10)
    limit = max(1, min(limit, 55))

    matched = []
    for r in records:
        if service and r["service"].lower() != service:
            continue
        if severity and r["severity"].upper() != severity:
            continue
        if status and r["status"].lower() != status:
            continue
        if keyword:
            haystack = f"{r['incident_id']} {r['title']} {r['summary']} {r['root_cause']}".lower()
            if keyword not in haystack:
                continue
        matched.append({
            "incident_id": r["incident_id"],
            "service": r["service"],
            "severity": r["severity"],
            "status": r["status"],
            "environment": r["environment"],
            "mttr_minutes": r["mttr_minutes"],
            "title": r["title"],
        })

    return {
        "total_records_in_db": len(records),
        "matched_count": len(matched),
        "returned_count": len(matched[:limit]),
        "incidents": matched[:limit],
    }


def tool_get_incident_details(
    arguments: dict[str, Any],
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    records = load_incidents(data_path)
    inc_id = (arguments.get("incident_id") or "").strip()
    if not inc_id:
        raise ToolExecutionError("MISSING_INCIDENT_ID", "incident_id is required.", {"total_records_in_db": len(records)})
    for r in records:
        if r["incident_id"].upper() == inc_id.upper():
            return {
                "found": True,
                "total_records_in_db": len(records),
                "incident": r,
            }
    raise ToolExecutionError(
        "INCIDENT_NOT_FOUND",
        f"No incident found with incident_id={inc_id!r}.",
        {"incident_id": inc_id, "total_records_in_db": len(records)},
    )


def tool_compute_mttr_stats(
    arguments: dict[str, Any],
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    records = load_incidents(data_path)
    service = (arguments.get("service") or "").strip().lower()
    severity = (arguments.get("severity") or "").strip().upper()

    filtered = []
    for r in records:
        if service and r["service"].lower() != service:
            continue
        if severity and r["severity"].upper() != severity:
            continue
        if r.get("status") == "resolved" and isinstance(r.get("mttr_minutes"), (int, float)):
            filtered.append(float(r["mttr_minutes"]))

    if not filtered:
        return {
            "total_records_in_db": len(records),
            "resolved_sample_count": 0,
            "service_filter": service or None,
            "severity_filter": severity or None,
            "mean_mttr_minutes": None,
            "median_mttr_minutes": None,
            "p95_mttr_minutes": None,
            "min_mttr_minutes": None,
            "max_mttr_minutes": None,
        }

    sorted_vals = sorted(filtered)
    p95_idx = min(len(sorted_vals) - 1, max(0, math.ceil(0.95 * len(sorted_vals)) - 1))
    return {
        "total_records_in_db": len(records),
        "resolved_sample_count": len(sorted_vals),
        "service_filter": service or None,
        "severity_filter": severity or None,
        "mean_mttr_minutes": round(statistics.mean(sorted_vals), 2),
        "median_mttr_minutes": round(statistics.median(sorted_vals), 2),
        "p95_mttr_minutes": round(sorted_vals[p95_idx], 2),
        "min_mttr_minutes": round(sorted_vals[0], 2),
        "max_mttr_minutes": round(sorted_vals[-1], 2),
    }


def tool_create_remediation_note(
    arguments: dict[str, Any],
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    records = load_incidents(data_path)
    inc_id = (arguments.get("incident_id") or "").strip()
    author = (arguments.get("author") or "").strip()
    note_text = (arguments.get("note") or "").strip()

    if not inc_id or not author or not note_text:
        raise ToolExecutionError(
            "INVALID_NOTE_ARGUMENTS",
            "incident_id, author, and non-empty note are all required.",
            {"total_records_in_db": len(records)},
        )

    for r in records:
        if r["incident_id"].upper() == inc_id.upper():
            notes = r.setdefault("remediation_notes", [])
            new_note = {
                "note_id": f"NOTE-{r['incident_id'].split('-')[-1]}-{len(notes) + 1}",
                "author": author,
                "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "note": note_text,
            }
            notes.append(new_note)
            save_incidents(records, data_path)
            return {
                "updated": True,
                "incident_id": r["incident_id"],
                "added_note": new_note,
                "remediation_note_count": len(notes),
                "total_records_in_db": len(records),
            }

    raise ToolExecutionError(
        "INCIDENT_NOT_FOUND",
        f"Cannot attach remediation note: incident_id={inc_id!r} not found.",
        {"incident_id": inc_id, "total_records_in_db": len(records)},
    )


def tool_purge_incidents(
    arguments: dict[str, Any],
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    records = load_incidents(data_path)
    inc_id = str(arguments.get("incident_id") or "").strip()
    confirm_scope = str(arguments.get("confirm_scope") or "").strip().lower()

    # HARD SAFETY GUARDRAIL (Checkpoint 3): Block any delete-all, wildcard, or bulk deletion in code.
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
            {
                "requested_incident_id": inc_id,
                "requested_confirm_scope": confirm_scope,
                "records_deleted": 0,
                "total_records_in_db": len(records),
            },
        )

    remaining = [r for r in records if r["incident_id"] != inc_id]
    if len(remaining) == len(records):
        raise ToolExecutionError(
            "INCIDENT_NOT_FOUND",
            f"No incident found with incident_id={inc_id!r}; 0 records deleted.",
            {"incident_id": inc_id, "records_deleted": 0, "total_records_in_db": len(records)},
        )

    save_incidents(remaining, data_path)
    return {
        "purged": True,
        "purged_incident_id": inc_id,
        "records_deleted": len(records) - len(remaining),
        "total_records_before": len(records),
        "total_records_in_db": len(remaining),
    }


DISPATCH_TABLE = {
    "search_incidents": tool_search_incidents,
    "get_incident_details": tool_get_incident_details,
    "compute_mttr_stats": tool_compute_mttr_stats,
    "create_remediation_note": tool_create_remediation_note,
    "purge_incidents": tool_purge_incidents,
}


def list_tool_definitions() -> dict[str, Any]:
    return {"tools": TOOL_DEFINITIONS}


def call_tool_direct(
    name: str,
    arguments: Optional[dict[str, Any]] = None,
    data_path: Path = DEFAULT_DATA_PATH,
) -> tuple[int, dict[str, Any]]:
    """Execute a tool call and return (inspector_exit_code, mcp_result_envelope).
    Returns exit_code=0 on success, exit_code=5 when the tool returns an error (isError=True).
    """
    args = arguments or {}
    if name not in DISPATCH_TABLE:
        err_payload = {
            "error": "UNKNOWN_TOOL",
            "message": f"Unknown tool {name!r}.",
        }
        return 5, {
            "result": {
                "content": [{"type": "text", "text": json.dumps(err_payload)}],
                "isError": True,
            }
        }
    try:
        output = DISPATCH_TABLE[name](args, data_path=data_path)
        return 0, {
            "result": {
                "content": [{"type": "text", "text": json.dumps(output)}],
                "isError": False,
            }
        }
    except ToolExecutionError as exc:
        return 5, {
            "result": {
                "content": [{"type": "text", "text": json.dumps(exc.to_payload())}],
                "isError": True,
            }
        }
    except Exception as exc:
        err_payload = {"error": type(exc).__name__, "message": str(exc)}
        return 5, {
            "result": {
                "content": [{"type": "text", "text": json.dumps(err_payload)}],
                "isError": True,
            }
        }


# ---------------------------------------------------------------------------
# Standard MCP JSON-RPC 2.0 over stdio transport
# ---------------------------------------------------------------------------

def serve_stdio(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Serve MCP JSON-RPC 2.0 protocol over stdin/stdout (newline-delimited JSON)."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        # Notifications (no id) do not receive a response
        if req_id is None:
            continue

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "engineering-incidents-mcp",
                        "version": "1.0.0",
                    },
                },
            }
        elif method == "ping":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": list_tool_definitions()}
        elif method == "tools/call":
            t_name = params.get("name", "")
            t_args = params.get("arguments") or {}
            _code, env = call_tool_direct(t_name, t_args, data_path=data_path)
            resp = {"jsonrpc": "2.0", "id": req_id, "result": env["result"]}
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tools/list", "tools/call"), default=None)
    parser.add_argument("--tool-name", default=None)
    parser.add_argument("--tool-args-json", default="{}")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--format", default="json")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.method == "tools/list":
        print(json.dumps(list_tool_definitions(), indent=2))
        return 0
    if args.method == "tools/call":
        parsed_args = json.loads(args.tool_args_json) if args.tool_args_json else {}
        exit_code, envelope = call_tool_direct(args.tool_name or "", parsed_args, data_path=args.data_path)
        print(json.dumps(envelope, indent=2))
        return exit_code

    serve_stdio(data_path=args.data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
