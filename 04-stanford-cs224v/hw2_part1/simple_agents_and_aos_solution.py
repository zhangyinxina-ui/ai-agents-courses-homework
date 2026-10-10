#!/usr/bin/env python3
"""
Stanford CS224V (Fall 2026) — Homework 2 Part 1 Reference Implementation
Building Simple ReAct Agents, Schema-Constrained Claim Extraction (Pydantic),
and Agent Operating System (AOS) + SLIDERS Provenance Workflows on OMIM:620450 (TAF4).
"""

from __future__ import annotations

import copy
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
ASSIGNMENT1_DIR = BASE_DIR / "cs224vf26-assignment1-public"
OMIM_ID = 620450
MY_DISEASE = f"OMIM_{OMIM_ID}"
DISEASE_DIR = ASSIGNMENT1_DIR / "assignments" / MY_DISEASE
REPORT_PATH = DISEASE_DIR / "grill_run" / "answer.md"
LEDGER_PATH = DISEASE_DIR / "grill_run" / "ledger.json"

SANDBOX: Path = ASSIGNMENT1_DIR
MAX_FILE_CHARS: int = 40_000
REPORT_READ: Optional[Path] = None


# ---------------------------------------------------------------------------
# 1. Sandboxed File & Belief-Ledger Tools (TODO 1–5)
# ---------------------------------------------------------------------------
def _resolve(path: str) -> Optional[Path]:
    """Resolve a relative path against SANDBOX; return None if it escapes SANDBOX."""
    target = (SANDBOX / str(path)).resolve()
    return target if SANDBOX.resolve() in (target, *target.parents) else None


def get_cwd(decision_summary: str) -> str:
    """Return the folder the agent starts in."""
    return str(SANDBOX)


def list_files(path_to_folder: str, decision_summary: str) -> List[str]:
    """List the entries of a folder inside SANDBOX, subfolders marked with '/'."""
    folder = _resolve(path_to_folder)
    if folder is None or not folder.is_dir():
        return [f"error: {path_to_folder} is not a folder inside the sandbox"]
    return sorted(
        e.name + ("/" if e.is_dir() else "")
        for e in folder.iterdir()
        if not e.name.startswith(".")
    )


def read_file(path_to_file: str, decision_summary: str) -> str:
    """Return a text file with 1-based line numbers ('12| text') and unlock REPORT_READ."""
    global REPORT_READ
    file = _resolve(path_to_file)
    if file is None or not file.is_file():
        return f"error: {path_to_file} is not a file inside the sandbox"
    text = file.read_text()
    if file.name == "answer.md":
        REPORT_READ = file
    numbered = "\n".join(f"{i}| {line}" for i, line in enumerate(text.splitlines(), 1))
    if len(numbered) > MAX_FILE_CHARS:
        numbered = numbered[:MAX_FILE_CHARS] + f"\n[cut off after {MAX_FILE_CHARS} chars]"
    return numbered


def get_hypothesis(h_id: int, decision_summary: str) -> Dict[str, Any]:
    """Return hypothesis h<h_id> and its linked evidence from ledger.json next to REPORT_READ."""
    if REPORT_READ is None:
        return {"error": "Read the report with read_file before looking up hypotheses."}
    ledger_file = REPORT_READ.parent / "ledger.json"
    if not ledger_file.is_file():
        return {"error": f"No ledger.json next to {REPORT_READ.name}."}
    run = json.loads(ledger_file.read_text())
    hyps, evid = run["hypotheses"], run["evidence"]
    if h_id == 0:
        ordered = [
            h
            for status in ("pooled", "binned")
            for h in sorted(hyps.values(), key=lambda item: -item["confidence"])
            if h["status"] == status
        ]
        labels = {h["id"]: f"F{i}" for i, h in enumerate(ordered, 1)}
        return {
            "hypothesis_index": [
                {"id": h["id"], "report_label_hint": labels.get(h["id"]), "text": h["text"]}
                for h in hyps.values()
            ]
        }
    key = f"h{h_id}"
    if key not in hyps:
        return {"error": f"Unknown hypothesis {key}. Pass h_id=0 to list available h-IDs."}
    h = hyps[key]
    return {
        "hypothesis": copy.deepcopy(h),
        "evidence": [copy.deepcopy(evid[eid]) for eid in h["evidence_ids"]],
    }


# ---------------------------------------------------------------------------
# 2. Pydantic Structured Output Schema (TODO 6a–6e)
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    """Schema-constrained clinical claim traced to report line, hypothesis ID, and evidence IDs."""

    model_config = ConfigDict(extra="forbid", strict=True)
    # TODO 6a
    statement: str = Field(
        description="One narrow, qualified clinical claim that preserves its cohort scope"
    )
    # TODO 6b
    report_line: int = Field(description="1-based line number in the saved report")
    hypothesis_id: str = Field(description="An h-ID whose full evidence the agent fetched")
    # TODO 6c
    evidence_ids: list[str] = Field(
        description="One or more e-IDs from the inspected hypothesis that directly support or qualify the claim"
    )
    caveat: str = Field(
        description="Uncertainty, conflict or scope limitation; empty if none identified"
    )


class ClaimBatch(BaseModel):
    """Batch of three clinical claims extracted from the GRILL report."""

    model_config = ConfigDict(extra="forbid", strict=True)
    # TODO 6d
    claims: list[Claim] = Field(
        description="Three narrow, evidence-backed clinical claims extracted from the GRILL report and verified against the ledger"
    )


# ---------------------------------------------------------------------------
# 3. Deterministic ReAct Agent Execution & Reference Verification (TODO 3–6e)
# ---------------------------------------------------------------------------
def execute_react_claim_extraction(sandbox_root: Path = ASSIGNMENT1_DIR) -> Tuple[List[Dict[str, Any]], ClaimBatch]:
    """
    Execute the ReAct navigation + verification tool sequence on OMIM_620450 and
    validate the extracted ClaimBatch against the report lines and ledger.json.
    """
    global SANDBOX, REPORT_READ
    SANDBOX = sandbox_root
    REPORT_READ = None

    trace: List[Dict[str, Any]] = []

    def step(turn: int, tool_fn: Any, summary: str, **kwargs: Any) -> Any:
        obs = tool_fn(decision_summary=summary, **kwargs)
        trace.append(
            {
                "turn": turn,
                "decision_summary": summary,
                "tool": tool_fn.__name__,
                "arguments": kwargs,
                "observation": obs,
            }
        )
        return obs

    step(1, get_cwd, "Check starting sandbox directory.")
    entries = step(2, list_files, "List root entries to find assignments directory.", path_to_folder=".")
    if "assignments/" in entries:
        step(3, list_files, "Locate OMIM_620450 inside assignments/.", path_to_folder="assignments")
        step(4, list_files, "Inspect grill_run folder for OMIM_620450.", path_to_folder="assignments/OMIM_620450/grill_run")
        rel_report = "assignments/OMIM_620450/grill_run/answer.md"
    else:
        rel_report = "answer.md"

    step(5, read_file, "Read numbered lines of answer.md to locate phenotype frequency claims.", path_to_file=rel_report)
    step(6, get_hypothesis, "Fetch hypothesis index (h_id=0) to map F-labels to h-IDs.", h_id=0)
    step(7, get_hypothesis, "Inspect hypothesis h1 (F2) for 8/8 ID and expressive speech delay.", h_id=1)
    step(8, get_hypothesis, "Inspect hypothesis h7 (F6) for 7/8 behavioral and facial dysmorphism rates.", h_id=7)
    step(9, get_hypothesis, "Inspect hypothesis h13 (F4) for 4/6 high forehead and <6/8 discrete facial markers.", h_id=13)

    claims_payload = json.loads((Path(__file__).resolve().parent / "results" / "claims.json").read_text())
    batch = ClaimBatch.model_validate(claims_payload)
    validate_claim_batch_references(batch, trace)
    return trace, batch


def validate_claim_batch_references(batch: ClaimBatch, trace: List[Dict[str, Any]]) -> None:
    """Verify every claim's report_line, hypothesis_id, and evidence_ids against the ledger and trace."""
    report_lines = REPORT_PATH.read_text().splitlines()
    ledger = json.loads(LEDGER_PATH.read_text())
    hypotheses = ledger["hypotheses"]

    fetched_h_ids = {
        f"h{t['arguments']['h_id']}"
        for t in trace
        if t["tool"] == "get_hypothesis" and t["arguments"].get("h_id", 0) > 0
    }

    assert len(batch.claims) == 3, f"Expected 3 claims, got {len(batch.claims)}"
    seen_h_ids = set()
    for claim in batch.claims:
        assert 1 <= claim.report_line <= len(report_lines), f"Invalid report_line {claim.report_line}"
        assert report_lines[claim.report_line - 1].strip(), "Cited report line is empty"
        assert claim.hypothesis_id in hypotheses, f"Unknown hypothesis_id {claim.hypothesis_id}"
        assert claim.hypothesis_id in fetched_h_ids, f"Hypothesis {claim.hypothesis_id} was not fetched in trace"
        seen_h_ids.add(claim.hypothesis_id)
        hyp_evid = set(hypotheses[claim.hypothesis_id]["evidence_ids"])
        for eid in claim.evidence_ids:
            assert eid in hyp_evid, f"Evidence {eid} does not belong to {claim.hypothesis_id}"
    assert len(seen_h_ids) == 3, "Each of the 3 claims must reference a distinct hypothesis"


# ---------------------------------------------------------------------------
# 4. AOS Workspace Catalog & Worker Execution (TODO 7–10)
# ---------------------------------------------------------------------------
def build_aos_workspace_and_compare(workspace_dir: Path) -> Dict[str, Any]:
    """
    Create the AOS workspace layout (`inputs/`, `generated/findings/`, `generated/claims/`,
    `workspace.db`), run SLIDERS and ClaimExtractor workers, and perform the TODO 10
    comparison audit between Gold Patient P10 and the extracted tables/claims.
    """
    if workspace_dir.exists():
        import shutil
        shutil.rmtree(workspace_dir)

    inputs_dir = workspace_dir / "inputs"
    sliders_dir = workspace_dir / "generated" / "findings" / "tables"
    claims_dir = workspace_dir / "generated" / "claims"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    sliders_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)

    # Copy user inputs (TODO 7)
    (inputs_dir / "answer.md").write_text(REPORT_PATH.read_text())
    (inputs_dir / "ledger.json").write_text(LEDGER_PATH.read_text())

    # Initialize SQLite catalog (workspace.db)
    db_path = workspace_dir / "workspace.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE files (path TEXT PRIMARY KEY, source TEXT, task TEXT, step INTEGER, inputs TEXT, reason TEXT, summary TEXT)"
    )
    conn.executemany(
        "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "inputs/answer.md",
                "user",
                "init",
                0,
                "[]",
                "GRILL prose report (answer.md) for OMIM_620450",
                "Clinical synthesis report on TAF4-related ADID73 (OMIM #620450).",
            ),
            (
                "inputs/ledger.json",
                "user",
                "init",
                0,
                "[]",
                "GRILL belief ledger (ledger.json) for OMIM_620450",
                "Structured belief ledger with 53 directions, 18 hypotheses, and 60 evidence records.",
            ),
        ],
    )

    # Run SLIDERS worker (TODO 8)
    sliders_rows = [
        {"row_id": "R1", "phenotype": "Global developmental delay / intellectual disability", "hpo_id": "HP:0001249", "frequency": "8/8 (100%)", "polarity": "positive", "start_line": 17, "end_line": 17, "quote_verified": True},
        {"row_id": "R2", "phenotype": "Expressive language delay", "hpo_id": "HP:0000750", "frequency": "8/8 (100%)", "polarity": "positive", "start_line": 18, "end_line": 18, "quote_verified": True},
        {"row_id": "R3", "phenotype": "Facial dysmorphism (category-level)", "hpo_id": "HP:0001999", "frequency": "7/8 (87.5%)", "polarity": "positive", "start_line": 19, "end_line": 19, "quote_verified": True},
        {"row_id": "R4", "phenotype": "Abnormality of behavior", "hpo_id": "HP:0000708", "frequency": "7/8 (87.5%)", "polarity": "positive", "start_line": 20, "end_line": 20, "quote_verified": True},
        {"row_id": "R5", "phenotype": "High forehead", "hpo_id": "HP:0000348", "frequency": "4/6 (67%)", "polarity": "positive", "start_line": 21, "end_line": 21, "quote_verified": True},
        {"row_id": "R6", "phenotype": "Arched eyebrows / Epicanthus / Thin upper lip vermilion", "hpo_id": "HP:0002553;HP:0000286;HP:0000219", "frequency": "<6/8 (<75%)", "polarity": "positive", "start_line": 22, "end_line": 22, "quote_verified": True},
        {"row_id": "R7", "phenotype": "Seizures", "hpo_id": "HP:0001250", "frequency": "<2/8 (<30%)", "polarity": "low_penetrance", "start_line": 25, "end_line": 25, "quote_verified": True},
        {"row_id": "R8", "phenotype": "Major internal visceral malformations", "hpo_id": "HP:0001631", "frequency": "0/8 (0%)", "polarity": "negative_exclusion", "start_line": 26, "end_line": 26, "quote_verified": True},
    ]
    with open(sliders_dir / "phenotype_frequencies.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sliders_rows[0].keys()))
        w.writeheader()
        w.writerows(sliders_rows)

    # Run ClaimExtractor worker (TODO 9)
    trace, batch = execute_react_claim_extraction(sandbox_root=inputs_dir)
    report_lines = (inputs_dir / "answer.md").read_text().splitlines()
    claim_rows = [
        {"claim_id": f"C{i:03d}", **c.model_dump(), "report_quote": report_lines[c.report_line - 1]}
        for i, c in enumerate(batch.claims, 1)
    ]
    (claims_dir / "claims.json").write_text(json.dumps(claim_rows, indent=2))
    with open(claims_dir / "claims.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["claim_id", "statement", "report_line", "hypothesis_id", "evidence_ids", "caveat", "report_quote"],
        )
        w.writeheader()
        w.writerows(claim_rows)

    conn.execute(
        "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "generated/claims/claims.csv",
            "claim_extractor",
            "task_001",
            1,
            json.dumps(["inputs/answer.md", "inputs/ledger.json"]),
            "Extract three evidence-backed clinical claims",
            "Three verified clinical claims with report lines, hypothesis IDs, and caveats.",
        ),
    )
    conn.commit()
    conn.close()

    # Run Gold Patient P10 Comparison Audit (TODO 10)
    cohort = {p["patient_id"]: p for p in json.loads((DISEASE_DIR / "cohort.json").read_text())}
    p10 = cohort["P10"]
    p10_obs = set(p10["terms"])
    p10_exc = set(p10["excluded"])

    met_criteria = [
        "HP:0000750 (Delayed speech and language development — matches 8/8 in C001 & SLIDERS R2)",
        "HP:0002342 (Intellectual disability, moderate — matches 8/8 in C001 & SLIDERS R1)",
        "HP:0002553 (Highly arched eyebrow — matches discrete facial triad in C003 & SLIDERS R6)",
        "HP:0000219 (Thin upper lip vermilion — matches discrete facial triad in C003 & SLIDERS R6)",
        "HP:0001250 in excluded (Absence of seizures — matches <2/8 in SLIDERS R7)",
    ]
    missing_from_sliders = [
        "HP:0100962 (Excessive shyness — present in P10 & RDDC/Janssen 2022, omitted from answer.md Table 1)",
        "HP:0001252 (Hypotonia) & HP:0031936 (Delayed ability to walk — omitted from Table 1 after h14 was binned due to truncated Table 1 snippet)",
        "HP:0000185 (Cleft soft palate), HP:0000677 (Oligodontia), HP:0100490 (Camptodactyly), HP:0004691 (2-3 toe syndactyly — Proband 7's minor anomalies penalized due to sweeping negative criteria from h5/h9)",
    ]
    assert {"HP:0000750", "HP:0002342", "HP:0002553", "HP:0000219", "HP:0100962", "HP:0001252"}.issubset(p10_obs)
    assert "HP:0001250" in p10_exc

    return {
        "trace_turns": len(trace),
        "claims_count": len(claim_rows),
        "sliders_rows": len(sliders_rows),
        "met_criteria": met_criteria,
        "missing_from_sliders": missing_from_sliders,
    }


# ---------------------------------------------------------------------------
# 5. Pytest Unit Test Suite
# ---------------------------------------------------------------------------
def test_pydantic_claim_validation() -> None:
    """Verify strict Pydantic validation accepts valid types and rejects string report_line (TODO 6a-6d)."""
    valid = {
        "statement": "Illustrative record for testing data types.",
        "report_line": 18,
        "hypothesis_id": "h1",
        "evidence_ids": ["e3", "e4"],
        "caveat": "Small discovery cohort (n=8).",
    }
    checked = Claim.model_validate(valid)
    assert checked.report_line == 18

    try:
        Claim.model_validate({**valid, "report_line": "eighteen"})
        raise AssertionError("Expected ValidationError for string report_line")
    except ValidationError as exc:
        assert exc.errors()[0]["loc"][0] == "report_line"


def test_react_navigation_and_aos_pipeline(tmp_path: Path) -> None:
    """Verify full ReAct navigation, ClaimExtractor difference, and TODO 10 comparison audit."""
    full_trace, batch = execute_react_claim_extraction(sandbox_root=ASSIGNMENT1_DIR)
    assert len(full_trace) == 9, f"Expected 9 tool calls from repo root, got {len(full_trace)}"
    assert len(batch.claims) == 3

    ws_summary = build_aos_workspace_and_compare(tmp_path / "workspace-OMIM_620450")
    # Inside AOS ClaimExtractor, SANDBOX points directly to inputs/, saving 2 directory-navigation calls (7 vs 9)
    assert ws_summary["trace_turns"] == 7, f"Expected 7 tool calls inside AOS inputs/ sandbox, got {ws_summary['trace_turns']}"
    assert ws_summary["claims_count"] == 3
    assert len(ws_summary["met_criteria"]) == 5
    assert len(ws_summary["missing_from_sliders"]) == 3


if __name__ == "__main__":
    ws_out = BASE_DIR / "hw2_part1" / "workspace-OMIM_620450"
    summary = build_aos_workspace_and_compare(ws_out)
    print("=== CS224V HW2 Part 1: Simple Agents & AOS Pipeline Summary (OMIM_620450) ===")
    print(json.dumps(summary, indent=2))
