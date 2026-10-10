#!/usr/bin/env python3
"""Part B: fix Part A's BABILong qa1 failure with Isolate and Targeted
summary.

Operates on the SAME items Part A scored (benchmarks/subset_qa1_topend.json,
lowest --n ids per bucket -- run part_a_stress.py first so there is a
baseline to compare against).
"""
import argparse
import csv
import json
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.pi_runner import DEFAULT_MODEL, md_table, real_slice_by_tokens, real_token_count, run_pi, write_json, write_text  # noqa: E402
from part_a_stress import BASE, load_items, resolve_team_and_out, score_babilong  # noqa: E402

THINKING = "low"
CHUNK_TARGET_TOKENS = 96_000

FACT_VERBS = r"(?:moved|went|travell?ed|journeyed|walked|ran|goes|go)"

COLOR_PART_A = "#2a78d6"
COLOR_ISOLATE = "#eb6834"
COLOR_SUMMARY = "#1baf7a"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    INPUT_RATE_LOW = 0.25
    INPUT_RATE_HIGH = 0.50
    INPUT_RATE_THRESHOLD = 272_000
    OUTPUT_RATE = 1.8

    def __init__(self, max_usd=None):
        self.max_usd = max_usd
        self.spent = 0.0
        self.log = []
        self._lock = threading.Lock()
        self._in_flight = 0.0

    @classmethod
    def estimate_call_cost_usd(cls, input_tokens: int, output_tokens_guess: int = 200) -> float:
        rate = cls.INPUT_RATE_HIGH if input_tokens > cls.INPUT_RATE_THRESHOLD else cls.INPUT_RATE_LOW
        return input_tokens / 1e6 * rate + output_tokens_guess / 1e6 * cls.OUTPUT_RATE

    def check_before(self, label: str, input_tokens: int, output_tokens_guess: int = 200) -> float:
        est = self.estimate_call_cost_usd(input_tokens, output_tokens_guess)
        with self._lock:
            if self.max_usd is not None:
                projected = self.spent + self._in_flight + est
                if projected > self.max_usd:
                    raise BudgetExceeded(
                        f"refusing to submit '{label}' (~{input_tokens:,} est. input tokens, est ${est:.4f}): "
                        f"completed=${self.spent:.4f} + in_flight=${self._in_flight:.4f} + this=${est:.4f} "
                        f"= ${projected:.4f} > --max-usd ${self.max_usd:.2f}."
                    )
            self._in_flight += est
        return est

    def record(self, label: str, reserved_est: float, actual_cost_usd: float) -> None:
        with self._lock:
            self._in_flight = max(0.0, self._in_flight - reserved_est)
            self.spent += actual_cost_usd or 0.0
            self.log.append({"label": label, "cost_usd": round(actual_cost_usd or 0.0, 6),
                              "running_total_usd": round(self.spent, 6)})

    def release(self, reserved_est: float) -> None:
        with self._lock:
            self._in_flight = max(0.0, self._in_flight - reserved_est)

    def snapshot(self) -> dict:
        with self._lock:
            return {"max_usd": self.max_usd, "completed_spend_usd": round(self.spent, 6), "n_calls": len(self.log)}


def call_pi(budget: Budget, label: str, prompt: str, model: str, output_tokens_guess: int, **kwargs):
    est = budget.check_before(label, real_token_count(prompt), output_tokens_guess)
    try:
        result = run_pi(prompt=prompt, model=model, thinking=THINKING, **kwargs)
    except Exception:
        budget.release(est)
        raise
    budget.record(label, est, result.total_cost_usd)
    return result


def split_into_n_chunks(text: str, n: int) -> list:
    total = real_token_count(text)
    target = max(1, total // n)
    remaining = text
    chunks = []
    for _ in range(n - 1):
        if not remaining:
            break
        piece = real_slice_by_tokens(remaining, target)
        if not piece:
            break
        chunks.append(piece)
        remaining = remaining[len(piece):]
    if remaining:
        chunks.append(remaining)
    return chunks


def chunk_count_for(haystack_or_item) -> int:
    if isinstance(haystack_or_item, dict):
        if haystack_or_item.get("n_chunks"):
            return int(haystack_or_item["n_chunks"])
        toks = haystack_or_item.get("real_input_tokens") or real_token_count(haystack_or_item.get("haystack", ""))
        return max(1, round(toks / CHUNK_TARGET_TOKENS))
    return max(1, round(real_token_count(haystack_or_item) / CHUNK_TARGET_TOKENS))


def _movement_matches(person: str, text: str):
    pat = re.compile(re.escape(person) + r" " + FACT_VERBS + r" (?:back )?to the (\w+)", re.I)
    return [(m.start(), m.group(1)) for m in pat.finditer(text)]


def run_data_check(items: list) -> dict:
    zero = single = multi = mismatches = 0
    dist = {}
    for it in items:
        person = it.get("person")
        if not person:
            continue
        matches = _movement_matches(person, it["haystack"])
        k = len(matches)
        dist[k] = dist.get(k, 0) + 1
        if k == 0:
            zero += 1
            continue
        single += 1 if k == 1 else 0
        multi += 1 if k > 1 else 0
        if matches[-1][1].lower() != it["target"].lower():
            mismatches += 1

    n = len(items)
    checked = n - zero
    return {
        "n_items_checked": n,
        "n_single_mention": single,
        "n_multi_mention": multi,
        "n_zero_regex_hits": zero,
        "mention_count_distribution": {str(k): v for k, v in sorted(dist.items())},
        "n_cases_where_last_mention_disagrees_with_target": mismatches,
        "finding": (
            f"{multi}/{n} items have the target person moving MORE THAN ONCE (regex hits for "
            f"'PERSON moved/went/travelled/... (back) to the ROOM'); {single} have exactly one "
            f"mention, {zero} had no literal regex hit. Across the {checked} items with at least "
            f"one hit, the LAST mention's room matches the gold target in {checked - mismatches}/{checked} "
            f"cases -- when a person moves more than once, the most recent move is usually the right "
            f"answer. Every demonstration below is therefore told explicitly to take the LAST "
            f"reported location, never the first."
        ),
    }


ISOLATE_CHUNK_INSTRUCTIONS = (
    "Does the text above state where {person} is -- for example, that they are in, or moved, "
    "went, or travelled to, a specific room or place? If yes, quote the exact sentence that "
    "states it, and say nothing else. If no, answer NOT FOUND and say nothing else."
)

ISOLATE_LEAD_TEMPLATE = (
    "Below are short reports from {n} consecutive chunks of a long story, given in document "
    "order (chunk 1 came first in the story, chunk {n} came last):\n\n{reports}\n\n"
    "Question: Where is {person}?\n\n"
    "People in this story move between rooms over time, and {person} may be reported in more "
    "than one chunk. If more than one chunk states a location for {person}, the correct answer "
    "is the LAST one in document order (the highest-numbered chunk that reports a location) -- "
    "ignore earlier ones, they are out of date. Answer with ONLY the room name, exactly as "
    "stated: no article (a/an/the), no punctuation, no explanation, nothing else. If no chunk "
    "reports a location for {person}, answer NOT FOUND."
)

SUMMARY_CHUNK_INSTRUCTIONS = (
    "Summarize the text above in about 200 tokens. If any sentence states that a person is in, "
    "or moved, went, or travelled to, a specific room or place, you MUST preserve that statement "
    "clearly in your summary -- do not drop it, generalize it away, or omit it even if it seems "
    "minor compared to the rest of the text."
)

SUMMARY_REASK_TEMPLATE = (
    "Below are summaries of {n} consecutive chunks of a long story, given in document order "
    "(chunk 1 came first in the story, chunk {n} came last):\n\n{summaries}\n\n"
    "Question: Where is {person}?\n\n"
    "People in this story move between rooms over time. If more than one summary states a "
    "location for {person}, the correct answer is the LAST one in document order -- ignore "
    "earlier ones, they are out of date. Answer with ONLY the room name: no article (a/an/the), "
    "no punctuation, no explanation, nothing else. If no summary states a location for "
    "{person}, answer NOT FOUND."
)


def prompt_tokens_from_usage(usage: dict) -> int:
    return (usage.get("input") or 0) + (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0)


def make_chunk_task(budget, model, condition, item, chunk_idx, chunk_text, work_root, session_dir, raw_dir, timeout):
    person = item["person"]
    if condition == "isolate":
        prompt = f"{chunk_text}\n\n{ISOLATE_CHUNK_INSTRUCTIONS.format(person=person)}\n"
        out_guess = 60
    else:
        prompt = f"{chunk_text}\n\n{SUMMARY_CHUNK_INSTRUCTIONS}\n"
        out_guess = 250
    label = f"{condition}/{item['bucket']}-item-{item['id']}/chunk-{chunk_idx:02d}"
    cwd = work_root / f"{item['bucket']}-item-{item['id']}" / f"chunk-{chunk_idx:02d}"
    raw_path = raw_dir / f"{condition}-{item['bucket']}-item-{item['id']}-chunk-{chunk_idx:02d}.raw.jsonl"

    def task():
        result = call_pi(
            budget, label, prompt, model, out_guess,
            cwd=str(cwd), no_tools=True, session_dir=str(session_dir),
            raw_events_path=str(raw_path), timeout=timeout, approve=True, no_context_files=True,
        )
        text = result.answer_text.strip() if result.ok else f"[ERROR: {result.error}]"
        return {
            "chunk": chunk_idx, "ok": result.ok, "error": result.error,
            "report" if condition == "isolate" else "summary": text,
            "input_tokens": prompt_tokens_from_usage(result.usage), "output_tokens": result.usage.get("output"),
            "cost_usd": round(result.total_cost_usd, 6),
        }
    return task


def make_lead_task(budget, model, condition, item, chunk_records, work_root, session_dir, raw_dir, timeout):
    person = item["person"]
    n = len(chunk_records)
    if condition == "isolate":
        reports = "\n\n".join(f"Chunk {c['chunk']}: {c['report']}" for c in chunk_records)
        prompt = ISOLATE_LEAD_TEMPLATE.format(n=n, reports=reports, person=person)
        out_guess, suffix = 30, "lead"
    else:
        summaries = "\n\n".join(f"Chunk {c['chunk']} summary: {c['summary']}" for c in chunk_records)
        prompt = SUMMARY_REASK_TEMPLATE.format(n=n, summaries=summaries, person=person)
        out_guess, suffix = 30, "reask"
    label = f"{condition}/{item['bucket']}-item-{item['id']}/{suffix}"
    cwd = work_root / f"{item['bucket']}-item-{item['id']}" / suffix
    raw_path = raw_dir / f"{condition}-{item['bucket']}-item-{item['id']}-{suffix}.raw.jsonl"

    def task():
        result = call_pi(
            budget, label, prompt, model, out_guess,
            cwd=str(cwd), no_tools=True, session_dir=str(session_dir),
            raw_events_path=str(raw_path), timeout=timeout, approve=True, no_context_files=True,
        )
        scoring = score_babilong(result.answer_text, item["target"]) if result.ok else {
            "raw_answer": None, "correct": False}
        return {
            "ok": result.ok, "error": result.error, "answer_text": result.answer_text, **scoring,
            "input_tokens": prompt_tokens_from_usage(result.usage), "output_tokens": result.usage.get("output"),
            "cost_usd": round(result.total_cost_usd, 6),
        }
    return task


def finalize_item(item, condition, chunk_records, lead_rec):
    chunk_in = [c["input_tokens"] or 0 for c in chunk_records]
    chunk_out = [c["output_tokens"] or 0 for c in chunk_records]
    lead_in = lead_rec["input_tokens"] or 0
    lead_out = lead_rec["output_tokens"] or 0
    total_tokens = sum(chunk_in) + sum(chunk_out) + lead_in + lead_out
    peak_context = max(chunk_in + [lead_in]) if chunk_in else lead_in
    cost = sum(c["cost_usd"] for c in chunk_records) + lead_rec["cost_usd"]
    key = "lead" if condition == "isolate" else "reask"
    return {
        "bucket": item["bucket"], "id": item["id"], "condition": condition, "person": item["person"],
        "target": item["target"], "n_chunks": len(chunk_records), "chunks": chunk_records,
        key: lead_rec, "correct": lead_rec["correct"], "raw_answer": lead_rec.get("raw_answer"),
        "total_tokens": total_tokens, "peak_context_tokens": peak_context, "cost_usd": round(cost, 6),
    }


def wald_ci(p: float, n: int):
    if n == 0:
        return 0.0, 0.0
    se = math.sqrt(p * (1 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def stats_for(rows, correct_key="correct"):
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get(correct_key))
    p = n_correct / n if n else 0.0
    lo, hi = wald_ci(p, n)
    return {"n": n, "n_correct": n_correct, "mean_pct": p * 100, "ci_lo_pct": lo * 100, "ci_hi_pct": hi * 100}


def write_report(part_b_dir, buckets, model, thinking, data_check, part_a_by_key, final_items):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [("Part A (single call)", None), ("Isolate", "isolate"), ("Targeted summary", "summary")]
    cells = {}
    for b in buckets:
        pa_rows = [r for (bkt, idn), r in part_a_by_key.items() if bkt == b]
        s = stats_for(pa_rows, "correct")
        s["mean_total_tokens"] = round(sum(r["real_input_tokens"] for r in pa_rows) / len(pa_rows)) if pa_rows else 0
        s["mean_peak_context"] = s["mean_total_tokens"]
        s["cost_usd"] = sum(r["cost_usd"] for r in pa_rows)
        cells[(b, "Part A (single call)")] = s

        for label, cond in (("Isolate", "isolate"), ("Targeted summary", "summary")):
            rows = [r for r in final_items if r["bucket"] == b and r["condition"] == cond]
            s = stats_for(rows, "correct")
            s["mean_total_tokens"] = round(sum(r["total_tokens"] for r in rows) / len(rows)) if rows else 0
            s["mean_peak_context"] = round(sum(r["peak_context_tokens"] for r in rows) / len(rows)) if rows else 0
            s["cost_usd"] = sum(r["cost_usd"] for r in rows)
            cells[(b, label)] = s

    lines = [
        "# Part B: Isolate and Targeted summary -- summary", "",
        f"Model `{model}`, thinking `{thinking}`. Chunking targets ~{CHUNK_TARGET_TOKENS:,} tokens per chunk; "
        "the actual chunk count per item follows from its real token size.", "",
        "## Data check: how many times is the target person mentioned?", "",
        data_check["finding"], "",
        "## Bucket x condition table", "",
    ]
    rows = []
    for b in buckets:
        for label, _ in conditions:
            s = cells.get((b, label))
            if not s:
                continue
            rows.append([
                b, label, f"{s['n_correct']}/{s['n']}", f"{s['mean_pct']:.1f}%",
                f"[{s['ci_lo_pct']:.1f}%, {s['ci_hi_pct']:.1f}%]",
                f"{s['mean_total_tokens']:,}", f"{s['mean_peak_context']:,}", f"${s['cost_usd']:.4f}",
            ])
    table_str = md_table(
        ["bucket", "condition", "correct/n", "mean", "95% CI", "mean total tokens",
         "mean peak single-call context", "total cost"], rows)
    lines.append(table_str)
    lines.append("")

    lines += ["## The lesson: peak context, not total tokens", "", (
        "Part A's single call for an item has to hold the ENTIRE haystack in context at once: peak "
        "context = total input tokens. Isolate's and Targeted summary's peak single-call context is "
        "bounded near one chunk's size instead, while their TOTAL tokens barely move relative to Part "
        "A -- the same haystack still gets read in full, just split across more, smaller calls. "
        "Isolation and targeted summarization do not save tokens; they cap what any single call has to "
        "hold in context at once."
    ), ""]

    write_text(str(part_b_dir / "summary.md"), "\n".join(lines))
    print("\n" + table_str)
    print(f"wrote {part_b_dir / 'summary.md'}")

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x_positions = {b: i for i, b in enumerate(buckets)}
    series = [("Part A (single call)", COLOR_PART_A, -0.12), ("Isolate", COLOR_ISOLATE, 0.0),
              ("Targeted summary", COLOR_SUMMARY, 0.12)]
    for label, color, offset in series:
        xs, ys, lo, hi, ns = [], [], [], [], []
        for b in buckets:
            s = cells.get((b, label))
            if not s:
                continue
            xs.append(x_positions[b] + offset)
            ys.append(s["mean_pct"])
            lo.append(s["mean_pct"] - s["ci_lo_pct"])
            hi.append(s["ci_hi_pct"] - s["mean_pct"])
            ns.append(s["n"])
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linestyle="-", marker="o", markersize=8,
                    linewidth=2.2, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.0, capsize=5,
                    elinewidth=1.5, ecolor=color, label=f"{label} (95% CI, n={ns[0]})")

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([b.upper() for b in buckets])
    ax.set_xlim(-0.5, len(buckets) - 0.5)
    ax.set_ylim(-5, 118)
    ax.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.set_xlabel("Bucket", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Exact-match accuracy (0-100%)", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_title("BABILong qa1: Part A baseline vs. Isolate vs. Targeted summary, 95% CI",
                 color=INK_PRIMARY, fontsize=12.5, loc="left", pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    caption = "Error bars are 95% Wald binomial CIs -- wide at small n; see summary.md for the full table."
    fig.text(0.02, 0.005, caption, fontsize=7.5, color=INK_MUTED, wrap=True, va="bottom")
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(part_b_dir / "comparison.png", facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {part_b_dir / 'comparison.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"provider/model-id (default: {DEFAULT_MODEL})")
    ap.add_argument("--buckets", default="256k,512k,768k", help="comma-separated buckets to run")
    ap.add_argument("--n", type=int, default=20, help="items per bucket (default: 20; same ids Part A used)")
    ap.add_argument("--team", default="team-columbia", help="team name; writes to studio/studio-02/submission/<name>/evidence/")
    ap.add_argument("--out", default=None, help="output directory (overrides --team; default: derived from --team)")
    ap.add_argument("--concurrency", type=int, default=4, help="max concurrent pi calls (default: 4)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-usd", type=float, default=None,
                     help="stop submitting new calls once the running total would exceed this (default: no limit)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate; make no pi calls")
    args = ap.parse_args()

    out_dir, session_dir = resolve_team_and_out(args.team, args.out)

    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    items_by_bucket = load_items(buckets, args.n)
    all_items = [it for b in buckets for it in items_by_bucket.get(b, [])]
    if not all_items:
        print("No items loaded -- check benchmarks/subset_qa1_topend.json and --buckets/--n.", file=sys.stderr)
        sys.exit(1)

    print("=== data check (mention counts) ===")
    data_check = run_data_check(all_items)
    print(data_check["finding"])
    print(f"\nLoaded {len(all_items)} items across {len(buckets)} bucket(s).")

    part_b_dir = out_dir / "part_b"

    if args.dry_run:
        total_est = 0.0
        for item in all_items:
            n_chunks = chunk_count_for(item)
            per_chunk_tok = (item.get("real_input_tokens") or real_token_count(item["haystack"])) // n_chunks
            for condition, out_g in (("isolate", 60), ("summary", 250)):
                est = n_chunks * Budget.estimate_call_cost_usd(per_chunk_tok, out_g) + Budget.estimate_call_cost_usd(1500, 30)
                total_est += est
        print(f"\nplanned output directory: {part_b_dir}")
        print(f"--dry-run: {len(all_items)} items x 2 conditions, estimated total ~${total_est:.4f} "
              f"(--max-usd default: no limit)")
        return

    raw_dir = part_b_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    work_root = BASE / "work" / "part_b"

    budget = Budget(max_usd=args.max_usd)

    work = [(it["bucket"], it["id"], condition, it) for it in all_items for condition in ("isolate", "summary")]
    print(f"{len(work)} (item, condition) pairs to run.")

    errors = []
    chunk_tasks = {}
    for bucket, idn, condition, item in work:
        n_chunks = chunk_count_for(item)
        chunks = split_into_n_chunks(item["haystack"], n_chunks)
        while len(chunks) < n_chunks:
            chunks.append(chunks[-1] if chunks else "")
        tasks = [
            make_chunk_task(budget, args.model, condition, item, i, chunk_text, work_root, session_dir, raw_dir, args.timeout)
            for i, chunk_text in enumerate(chunks[:n_chunks], start=1)
        ]
        chunk_tasks[(bucket, idn, condition)] = tasks

    print(f"\n=== Stage A: {sum(len(v) for v in chunk_tasks.values())} chunk calls, "
          f"max {args.concurrency} concurrent ===")
    chunk_results = {k: [None] * len(v) for k, v in chunk_tasks.items()}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(t): (key, i) for key, tasks in chunk_tasks.items() for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            key, i = futures[fut]
            try:
                chunk_results[key][i] = fut.result()
            except (BudgetExceeded, Exception) as e:
                errors.append((key, str(e)))
                print(f"  *** ERROR for {key} chunk {i + 1}: {e} ***", file=sys.stderr, flush=True)
            done += 1
            if done % 100 == 0 or done == len(futures):
                print(f"  ...{done}/{len(futures)} chunk calls done "
                      f"(spend so far ${budget.snapshot()['completed_spend_usd']:.4f})", flush=True)

    lead_tasks = {}
    skipped = []
    for key, results_list in chunk_results.items():
        bucket, idn, condition = key
        if any(r is None for r in results_list):
            skipped.append(key)
            continue
        item = next(it for it in all_items if it["bucket"] == bucket and it["id"] == idn)
        lead_tasks[key] = make_lead_task(budget, args.model, condition, item, results_list, work_root, session_dir,
                                          raw_dir, args.timeout)
    if skipped:
        print(f"\nSkipping lead/reask for {len(skipped)} items with a failed chunk: {skipped}", file=sys.stderr)

    print(f"\n=== Stage B: {len(lead_tasks)} lead/reask calls, max {args.concurrency} concurrent ===")
    lead_results = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(t): key for key, t in lead_tasks.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                lead_results[key] = fut.result()
                bucket, idn, condition = key
                v = "correct" if lead_results[key]["correct"] else "WRONG"
                print(f"  {condition} {bucket} id={idn}: {v} answer={lead_results[key].get('raw_answer')!r}", flush=True)
            except (BudgetExceeded, Exception) as e:
                errors.append((key, str(e)))
                print(f"  *** ERROR for {key}: {e} ***", file=sys.stderr, flush=True)

    final_items = []
    for key, lead_rec in lead_results.items():
        bucket, idn, condition = key
        item = next(it for it in all_items if it["bucket"] == bucket and it["id"] == idn)
        final_items.append(finalize_item(item, condition, chunk_results[key], lead_rec))

    part_a_path = out_dir / "part_a" / "results.json"
    part_a_by_key = {}
    if part_a_path.exists():
        part_a_results = json.loads(part_a_path.read_text(encoding="utf-8"))
        part_a_by_key = {(r["bucket"], r["id"]): r for r in part_a_results}
        for rec in final_items:
            pa = part_a_by_key.get((rec["bucket"], rec["id"]))
            rec["part_a_correct"] = pa.get("correct") if pa else None
            rec["part_a_real_input_tokens"] = pa.get("real_input_tokens") if pa else None
            rec["part_a_cost_usd"] = pa.get("cost_usd") if pa else None
    else:
        print(f"NOTE: {part_a_path} not found -- results.json will not carry a Part A comparison; "
              "run part_a_stress.py first for a full summary.md.", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} (item,condition) pairs failed: {errors}", file=sys.stderr)

    results = {
        "model": args.model, "thinking": THINKING, "buckets": buckets,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "data_check": data_check,
        "items": sorted(final_items, key=lambda r: (r["bucket"], r["id"], r["condition"])),
        "n_errors": len(errors), "errors": [{"key": list(k), "error": e} for k, e in errors],
        "budget": budget.snapshot(),
    }
    write_json(str(part_b_dir / "results.json"), results)
    print(f"\nWrote {part_b_dir / 'results.json'} with {len(results['items'])} item-condition rows.")

    csv_path = part_b_dir / "results.csv"
    fieldnames = ["bucket", "id", "condition", "n_chunks", "correct", "raw_answer", "target", "total_tokens",
                  "peak_context_tokens", "cost_usd", "part_a_correct", "part_a_real_input_tokens", "part_a_cost_usd"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results["items"]:
            writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fieldnames})
    print(f"Wrote {csv_path}")

    if part_a_by_key:
        write_report(part_b_dir, buckets, args.model, THINKING, data_check, part_a_by_key, final_items)

    snap = budget.snapshot()
    print(f"\nTotal spend: ${snap['completed_spend_usd']:.4f} across {snap['n_calls']} calls "
          f"(--max-usd: {snap['max_usd'] if snap['max_usd'] is not None else 'no limit'})")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
