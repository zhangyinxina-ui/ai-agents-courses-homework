#!/usr/bin/env python3
"""Part C: file-based memory across three sessions vs. a memoryless baseline."""
import argparse
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.pi_runner import (  # noqa: E402
    DEFAULT_MODEL,
    md_table,
    run_pi,
    write_json,
    write_text,
)

CODE_DIR = Path(__file__).resolve().parent
STUDIO_ROOT = CODE_DIR if (CODE_DIR / "submission").exists() else CODE_DIR.parent.parent
SUBMISSION_ROOT = STUDIO_ROOT / "submission"
TEAM_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def resolve_team_and_out(team, out):
    if not team and not out:
        team = "team-columbia"
    if team is not None and (team == "_template" or not TEAM_NAME_RE.match(team)):
        print(f"error: --team must match {TEAM_NAME_RE.pattern!r} and must not be '_template'", file=sys.stderr)
        sys.exit(2)
    if out:
        out_dir = Path(out)
    elif team:
        out_dir = SUBMISSION_ROOT / team / "evidence"
    else:
        print("error: pass --team <name> or --out <dir>", file=sys.stderr)
        sys.exit(2)
    sessions_dir = (CODE_DIR / "work" / "sessions" / (team or "default")).resolve()
    return out_dir, sessions_dir


FALLBACK_AGENTS_MD = """# Project memory

Before starting any task, read `decisions.md` in this directory if it exists, and follow any
decisions already recorded there.

Whenever you make a decision while working, append a new dated entry to `decisions.md` with the
decision and the reason for it. Use this format:

## YYYY-MM-DD: <short title>
Decision: <what you decided>
Reason: <why>

Never delete or rewrite earlier entries in `decisions.md`; only append.
"""

STARTER_AGENTS_MD = Path(__file__).resolve().parent / "part_c" / "AGENTS.md"

SESSION1_PROMPT = (
    "Plan a small command-line note-taking tool. You must make and record ALL THREE of the "
    "following decisions, each with a one-sentence reason. Do not skip any of them and do not "
    "stop until all three are recorded:\n"
    "1. The storage format for notes.\n"
    "2. The command name for the tool.\n"
    "3. The date format used when recording each note.\n"
    "Follow the project instructions for where and how to record decisions. Before you finish, "
    "re-read decisions.md and confirm it contains all three decisions above; if one is missing, "
    "add it. Do not write any other code or files for this task."
)

SESSION2_PROMPT = (
    "Write fib.py that prints the first 20 Fibonacci numbers. While doing this, make at least "
    "one small implementation decision (for example, whether to start the sequence at 0 or 1, "
    "or iterative vs. recursive), and record it following the project instructions."
)

SESSION3_PROMPT = "What did we decide in session 1 about the note-taking tool, and why?"

SCOPE_INSTRUCTION = (
    "Only use information already available in your current working directory to answer. "
    "Do not read, open, or guess the contents of any file in a parent directory, a sibling "
    "directory, or anywhere else outside your current working directory."
)

DATE_ENTRY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

CATEGORIES = {
    "storage_format": {
        "anchor": re.compile(r"stor(?:e|age|ing)|file per note|plain[- ]text|markdown|\bjson\b|\bformat\b", re.IGNORECASE),
        "label": "storage format for notes",
    },
    "command_name": {
        "anchor": re.compile(r"command|name the tool|tool name|call(?:ed|ing)? (?:it|the tool)|`[a-z]+`|\bcli\b", re.IGNORECASE),
        "label": "command name for the tool",
    },
    "date_format": {
        "anchor": re.compile(r"date|iso[- ]?8601|yyyy[- ]mm[- ]dd|timestamp", re.IGNORECASE),
        "label": "date format for entries",
    },
}

STOPWORDS = set(
    "the a an of for to in on with and or is are was were be been being this that these those "
    "it its as at by from will would should could note notes tool decision decided record "
    "recording use used using format name command storage date reason because".split()
)


def significant_words(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def extract_ground_truth(decisions_md_text: str) -> dict:
    blocks = re.split(r"\n\s*\n", decisions_md_text)
    ground_truth = {}
    for key, spec in CATEGORIES.items():
        found = ""
        for block in blocks:
            if spec["anchor"].search(block):
                found = block.strip()
                break
        ground_truth[key] = found
    return ground_truth


def classify_recall(answer_text: str, ground_truth: dict) -> dict:
    result = {}
    for key, spec in CATEGORIES.items():
        truth_block = ground_truth.get(key, "")
        mentions_category = bool(spec["anchor"].search(answer_text))
        if not mentions_category:
            result[key] = "missing"
            continue
        if not truth_block:
            result[key] = "invented"
            continue
        truth_words = significant_words(truth_block)
        truth_words -= significant_words(spec["label"])
        answer_words = significant_words(answer_text)
        overlap = truth_words & answer_words
        result[key] = "correct" if overlap else "invented"
    return result


def count_dated_entries(text: str) -> int:
    return len(DATE_ENTRY_RE.findall(text))


def get_agents_md_text() -> str:
    if STARTER_AGENTS_MD.exists():
        return STARTER_AGENTS_MD.read_text(encoding="utf-8")
    return FALLBACK_AGENTS_MD


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--team", default="team-columbia", help="team name; writes to studio/studio-02/submission/<name>/evidence/")
    ap.add_argument("--out", default=None, help="output directory (overrides --team; default: derived from --team)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--fresh", action="store_true", help="wipe work/part_c/ and start the three sessions over")
    args = ap.parse_args()

    out_dir, sessions_dir = resolve_team_and_out(args.team, args.out)
    part_c_dir = out_dir / "part_c"
    part_c_dir.mkdir(parents=True, exist_ok=True)

    with_memory_dir = CODE_DIR / "work" / "part_c" / "with-memory"
    no_memory_dir = CODE_DIR / "work" / "part_c" / "no-memory"

    if args.fresh:
        shutil.rmtree(CODE_DIR / "work" / "part_c", ignore_errors=True)

    with_memory_dir.mkdir(parents=True, exist_ok=True)
    no_memory_dir.mkdir(parents=True, exist_ok=True)

    agents_md_text = get_agents_md_text()
    if STARTER_AGENTS_MD.exists():
        try:
            agents_md_source = str(STARTER_AGENTS_MD.relative_to(CODE_DIR))
        except ValueError:
            agents_md_source = str(STARTER_AGENTS_MD)
    else:
        agents_md_source = "(built-in fallback: part_c/AGENTS.md was not found)"
    (with_memory_dir / "AGENTS.md").write_text(agents_md_text, encoding="utf-8")
    print(f"AGENTS.md source: {agents_md_source}")

    tool_list = "read,write,edit,ls"
    raw_dir = part_c_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    todays_date = _dt.date.today().isoformat()
    with_memory_extra_args = [
        "--append-system-prompt",
        agents_md_text + f"\n\nToday's date is {todays_date}.",
    ]

    print("=== Part C, session 1 (with memory): plan note-taking tool ===", flush=True)
    s1 = run_pi(
        prompt=SESSION1_PROMPT,
        cwd=str(with_memory_dir),
        model=args.model,
        tools=tool_list,
        session_dir=str(sessions_dir),
        raw_events_path=str(raw_dir / "session1.raw.jsonl"),
        timeout=args.timeout,
        approve=True,
        no_context_files=True,
        extra_args=with_memory_extra_args,
    )
    decisions_path = with_memory_dir / "decisions.md"
    decisions_after_s1 = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
    entries_after_s1 = count_dated_entries(decisions_after_s1)
    missing_categories = [spec["label"] for key, spec in CATEGORIES.items() if not extract_ground_truth(decisions_after_s1).get(key)]
    print(f"  session 1 ok={s1.ok} cost=${s1.total_cost_usd:.4f} decisions.md exists={decisions_path.exists()} dated_entries~={entries_after_s1} missing={missing_categories}", flush=True)

    s1_retry_cost = 0.0
    if missing_categories:
        retry_prompt = (
            "decisions.md is still missing a decision for: " + "; ".join(missing_categories) + ". "
            "Add the missing one(s) now, each with a one-sentence reason, following the project "
            "instructions for how and where to record decisions. Do not change or remove any "
            "existing entries."
        )
        s1_retry = run_pi(
            prompt=retry_prompt,
            cwd=str(with_memory_dir),
            model=args.model,
            tools=tool_list,
            session_dir=str(sessions_dir),
            raw_events_path=str(raw_dir / "session1-retry.raw.jsonl"),
            timeout=args.timeout,
            approve=True,
            no_context_files=True,
            extra_args=with_memory_extra_args,
        )
        s1_retry_cost = s1_retry.total_cost_usd
        decisions_after_s1 = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
        entries_after_s1 = count_dated_entries(decisions_after_s1)
        missing_categories = [spec["label"] for key, spec in CATEGORIES.items() if not extract_ground_truth(decisions_after_s1).get(key)]

    print("=== Part C, session 2 (with memory): write fib.py ===", flush=True)
    s2 = run_pi(
        prompt=SESSION2_PROMPT,
        cwd=str(with_memory_dir),
        model=args.model,
        tools=tool_list,
        session_dir=str(sessions_dir),
        raw_events_path=str(raw_dir / "session2.raw.jsonl"),
        timeout=args.timeout,
        approve=True,
        no_context_files=True,
        extra_args=with_memory_extra_args,
    )
    decisions_after_s2 = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
    entries_after_s2 = count_dated_entries(decisions_after_s2)
    fib_path = with_memory_dir / "fib.py"
    print(f"  session 2 ok={s2.ok} cost=${s2.total_cost_usd:.4f} fib.py exists={fib_path.exists()} dated_entries~={entries_after_s2}", flush=True)

    ground_truth = extract_ground_truth(decisions_after_s2 or decisions_after_s1)

    print("=== Part C, session 3 (with memory): recall query ===", flush=True)
    s3_mem = run_pi(
        prompt=SESSION3_PROMPT,
        cwd=str(with_memory_dir),
        model=args.model,
        tools="read",
        session_dir=str(sessions_dir),
        raw_events_path=str(raw_dir / "session3-with-memory.raw.jsonl"),
        timeout=args.timeout,
        approve=True,
        no_context_files=True,
        extra_args=["--append-system-prompt", with_memory_extra_args[1] + "\n\n" + SCOPE_INSTRUCTION],
    )
    recall_with_memory = classify_recall(s3_mem.answer_text, ground_truth)
    print(f"  session 3 (with memory) ok={s3_mem.ok} cost=${s3_mem.total_cost_usd:.4f} recall={recall_with_memory}", flush=True)

    print("=== Part C, baseline (no memory): recall query ===", flush=True)
    s3_baseline = run_pi(
        prompt=SESSION3_PROMPT,
        cwd=str(no_memory_dir),
        model=args.model,
        tools="read",
        session_dir=str(sessions_dir),
        raw_events_path=str(raw_dir / "session3-no-memory.raw.jsonl"),
        timeout=args.timeout,
        approve=True,
        no_context_files=True,
        extra_args=["--append-system-prompt", SCOPE_INSTRUCTION],
    )
    recall_no_memory = classify_recall(s3_baseline.answer_text, ground_truth)
    print(f"  baseline (no memory) ok={s3_baseline.ok} cost=${s3_baseline.total_cost_usd:.4f} recall={recall_no_memory}", flush=True)

    write_text(str(part_c_dir / "decisions.md"), decisions_after_s2 or decisions_after_s1 or "(decisions.md was never created)")
    write_text(str(part_c_dir / "fib.py"), fib_path.read_text(encoding="utf-8") if fib_path.exists() else "(fib.py was never created)")
    write_text(str(part_c_dir / "session3-with-memory-answer.md"), s3_mem.answer_text)
    write_text(str(part_c_dir / "session3-no-memory-answer.md"), s3_baseline.answer_text)

    record = {
        "model": args.model,
        "agents_md_source": agents_md_source,
        "session1": {
            "ok": s1.ok, "error": s1.error, "usage": s1.usage,
            "cost_usd": round(s1.total_cost_usd + s1_retry_cost, 6),
            "retried": s1_retry_cost > 0,
            "decisions_md_created": decisions_path.exists(), "dated_entries": entries_after_s1,
            "missing_after_retry": missing_categories,
        },
        "session2": {
            "ok": s2.ok, "error": s2.error, "usage": s2.usage, "cost_usd": round(s2.total_cost_usd, 6),
            "fib_py_created": fib_path.exists(), "dated_entries_after": entries_after_s2,
        },
        "session3_with_memory": {
            "ok": s3_mem.ok, "error": s3_mem.error, "usage": s3_mem.usage, "cost_usd": round(s3_mem.total_cost_usd, 6),
            "answer_text": s3_mem.answer_text, "recall": recall_with_memory,
        },
        "session3_no_memory_baseline": {
            "ok": s3_baseline.ok, "error": s3_baseline.error, "usage": s3_baseline.usage,
            "cost_usd": round(s3_baseline.total_cost_usd, 6),
            "answer_text": s3_baseline.answer_text, "recall": recall_no_memory,
        },
        "ground_truth_blocks": ground_truth,
        "total_cost_usd": round(
            s1.total_cost_usd + s1_retry_cost + s2.total_cost_usd + s3_mem.total_cost_usd + s3_baseline.total_cost_usd, 6
        ),
    }
    write_json(str(part_c_dir / "part_c.json"), record)
    write_json(str(part_c_dir / "results.json"), record)

    rows = []
    for key, spec in CATEGORIES.items():
        rows.append([spec["label"], recall_with_memory[key], recall_no_memory[key]])
    table_md = md_table(["decision", "with memory (session 3)", "no memory baseline (session 3)"], rows)

    lines = [
        "# Part C: File-based Memory vs. Memoryless Baseline -- summary",
        "",
        f"Model: `{args.model}`",
        f"AGENTS.md source: {agents_md_source}",
        "",
        f"Session 1 wrote decisions.md with ~{entries_after_s1} dated entries.",
        f"Session 2 (unrelated fib.py task) left decisions.md with ~{entries_after_s2} dated entries.",
        "",
        "Recall of the three session-1 decisions, asked fresh in session 3:",
        "",
        table_md,
        "",
        "'correct' = the answer's content for that decision overlaps with what decisions.md actually "
        "recorded. 'invented' = the answer addresses that decision but does not match what was "
        "recorded (or nothing was ever recorded for it). 'missing' = the answer does not address "
        "that decision at all. This is an automated keyword-overlap heuristic; the actual answer "
        "text is saved alongside this file for a human to check.",
        "",
        f"With-memory session 3 answer: see session3-with-memory-answer.md",
        f"No-memory baseline session 3 answer: see session3-no-memory-answer.md",
        "",
    ]
    write_text(str(part_c_dir / "summary.md"), "\n".join(lines))
    print("\n" + table_md)
    print(f"\nWrote {part_c_dir / 'part_c.json'}, results.json, decisions.md, fib.py, and summary.md")


if __name__ == "__main__":
    main()
