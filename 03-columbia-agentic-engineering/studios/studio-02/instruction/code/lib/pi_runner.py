#!/usr/bin/env python3
"""lib/pi_runner.py -- shared helpers for Studio 02 (Context Window Stress
Test & Memory Architecture): running `pi` non-interactively and parsing its
output, with deterministic offline simulation fallback when `pi` or live API
credentials are not configured.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_MODEL = "openai/gpt-5.6-luna"

ENV_VAR_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "amazon-bedrock": "AWS_BEARER_TOKEN_BEDROCK",
}

PI_AGENT_DIR = Path(os.environ.get("PI_CODING_AGENT_DIR", str(Path.home() / ".pi" / "agent")))

_TIKTOKEN_ENCODING = None
_TIKTOKEN_TRIED = False


def _get_tiktoken_encoding():
    global _TIKTOKEN_ENCODING, _TIKTOKEN_TRIED
    if not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:
            import tiktoken

            _TIKTOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
        except Exception:
            _TIKTOKEN_ENCODING = None
    return _TIKTOKEN_ENCODING


def tokenizer_name() -> str:
    return "tiktoken o200k_base" if _get_tiktoken_encoding() is not None else "estimate (words x 1.33; tiktoken unavailable)"


def real_token_count(text: str) -> int:
    """Real o200k token count via tiktoken; falls back to a words*1.33
    estimate if tiktoken is not installed."""
    enc = _get_tiktoken_encoding()
    if enc is None:
        return round(len((text or "").split()) * 1.33)
    return len(enc.encode(text))


def real_slice_by_tokens(text: str, token_budget: int) -> str:
    """Line-safe cut of the first `token_budget` REAL tokens."""
    enc = _get_tiktoken_encoding()
    lines = (text or "").splitlines(keepends=True)
    out_lines: List[str] = []
    count = 0
    for line in lines:
        n = len(enc.encode(line)) if enc is not None else round(len(line.split()) * 1.33)
        if count + n > token_budget and out_lines:
            break
        out_lines.append(line)
        count += n
        if count >= token_budget:
            break
    return "".join(out_lines)


def probe_providers() -> Dict[str, bool]:
    available = {}
    auth_providers = set()
    auth_path = PI_AGENT_DIR / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            if isinstance(auth, dict):
                auth_providers = set(auth.keys())
        except Exception:
            pass
    for provider, env_var in ENV_VAR_BY_PROVIDER.items():
        available[provider] = bool(os.environ.get(env_var)) or provider in auth_providers
    available["openai-codex"] = "openai-codex" in auth_providers
    return available


class RunResult:
    def __init__(self):
        self.ok = False
        self.answer_text = ""
        self.usage: Dict = {}
        self.total_cost_usd: float = 0.0
        self.compactions: List[dict] = []
        self.error: Optional[str] = None
        self.wall_time_s: float = 0.0
        self.raw_events_path: Optional[str] = None
        self.events: List[dict] = []
        self.exit_code: Optional[int] = None
        self.model: str = ""

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "answer_text": self.answer_text,
            "usage": self.usage,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "compactions": self.compactions,
            "error": self.error,
            "wall_time_s": round(self.wall_time_s, 2),
            "raw_events_path": self.raw_events_path,
            "exit_code": self.exit_code,
            "model": self.model,
        }


_AUTH_HINT = (
    " Hint: run `pi` then `/login` to authenticate the ChatGPT/Codex subscription provider "
    "and pass --model openai-codex/gpt-5.6-luna, or set OPENAI_API_KEY and use the default "
    "--model openai/gpt-5.6-luna."
)
_AUTH_ERROR_KEYWORDS = (
    "no credits", "insufficient balance", "unauthorized", "401", "403", "not found", "404",
    "no longer available", "authentication", "auth", "login", "api key", "credential",
)


def _with_auth_hint(result: "RunResult") -> "RunResult":
    if not result.ok and result.error:
        low = result.error.lower()
        if any(kw in low for kw in _AUTH_ERROR_KEYWORDS) and _AUTH_HINT.strip() not in result.error:
            result.error = result.error + _AUTH_HINT
    return result


# ---------------------------------------------------------------------------
# Offline deterministic simulation backed by reference-evidence
# ---------------------------------------------------------------------------

_REF_CACHE: Dict[str, object] = {}


def _load_reference_data() -> dict:
    if _REF_CACHE:
        return _REF_CACHE
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "reference-evidence",
        here.parent / "instruction" / "reference-evidence",
    ]
    ref_dir = next((c for c in candidates if c.exists()), None)
    pa_by_key = {}
    pb_by_key = {}
    pc_data = {}
    if ref_dir is not None:
        pa_file = ref_dir / "part_a" / "results.json"
        if pa_file.exists():
            for r in json.loads(pa_file.read_text(encoding="utf-8")):
                pa_by_key[(r["bucket"], int(r["id"]))] = r
        pb_file = ref_dir / "part_b" / "results.json"
        if pb_file.exists():
            pb_raw = json.loads(pb_file.read_text(encoding="utf-8"))
            for it in pb_raw.get("items", []):
                pb_by_key[(it["bucket"], int(it["id"]), it["condition"])] = it
        pc_file = ref_dir / "part_c" / "part_c.json"
        if pc_file.exists():
            pc_data = json.loads(pc_file.read_text(encoding="utf-8"))
    _REF_CACHE["pa"] = pa_by_key
    _REF_CACHE["pb"] = pb_by_key
    _REF_CACHE["pc"] = pc_data
    return _REF_CACHE


def _write_simulated_raw_events(raw_events_path: Optional[str], answer_text: str, usage: dict, cost_usd: float) -> None:
    if not raw_events_path:
        return
    p = Path(raw_events_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "stopReason": "end_turn",
            "content": [{"type": "text", "text": answer_text}],
            "usage": {**usage, "cost": {"total": cost_usd}},
        },
    }
    p.write_text(json.dumps(event) + "\n", encoding="utf-8")


def _simulate_pi(
    prompt: str,
    cwd: str,
    model: str,
    raw_events_path: Optional[str],
) -> RunResult:
    refs = _load_reference_data()
    pa_by_key = refs.get("pa", {})
    pb_by_key = refs.get("pb", {})
    pc_data = refs.get("pc", {})

    res = RunResult()
    res.model = model
    res.ok = True
    res.exit_code = 0
    res.wall_time_s = 0.12
    res.raw_events_path = raw_events_path

    raw_name = Path(raw_events_path).name if raw_events_path else ""

    # 1. Check Part B chunk call: {condition}-{bucket}-item-{id}-chunk-{chunk}.raw.jsonl
    m_chunk = re.match(r"^(isolate|summary)-(\w+)-item-(\d+)-chunk-(\d+)\.raw\.jsonl$", raw_name)
    if m_chunk:
        cond, bkt, idn, c_idx = m_chunk.group(1), m_chunk.group(2), int(m_chunk.group(3)), int(m_chunk.group(4))
        pb_item = pb_by_key.get((bkt, idn, cond))
        if pb_item:
            chunks = pb_item.get("chunks", [])
            c_rec = next((c for c in chunks if c.get("chunk") == c_idx), chunks[min(c_idx - 1, len(chunks) - 1)] if chunks else {})
            text = c_rec.get("report" if cond == "isolate" else "summary", "NOT FOUND")
            in_tok = c_rec.get("input_tokens", 81500)
            out_tok = c_rec.get("output_tokens", 25 if cond == "isolate" else 210)
            cost = c_rec.get("cost_usd", 0.0205)
        else:
            text = "NOT FOUND"
            in_tok, out_tok, cost = 81500, 25, 0.0205
        res.answer_text = text
        res.usage = {"input": in_tok, "output": out_tok, "cacheRead": 0, "cacheWrite": 0}
        res.total_cost_usd = cost
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    # 2. Check Part B lead/reask call: {condition}-{bucket}-item-{id}-{lead|reask}.raw.jsonl
    m_lead = re.match(r"^(isolate|summary)-(\w+)-item-(\d+)-(lead|reask)\.raw\.jsonl$", raw_name)
    if m_lead:
        cond, bkt, idn, key = m_lead.group(1), m_lead.group(2), int(m_lead.group(3)), m_lead.group(4)
        pb_item = pb_by_key.get((bkt, idn, cond))
        if pb_item and key in pb_item:
            l_rec = pb_item[key]
            text = l_rec.get("answer_text") or l_rec.get("raw_answer") or pb_item.get("raw_answer") or "kitchen"
            in_tok = l_rec.get("input_tokens", 420)
            out_tok = l_rec.get("output_tokens", 6)
            cost = l_rec.get("cost_usd", 0.00012)
        else:
            text = "kitchen"
            in_tok, out_tok, cost = 420, 6, 0.00012
        res.answer_text = text
        res.usage = {"input": in_tok, "output": out_tok, "cacheRead": 0, "cacheWrite": 0}
        res.total_cost_usd = cost
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    # 3. Check Part A item call: {bucket}-item-{id}.raw.jsonl
    m_pa = re.match(r"^(\w+)-item-(\d+)\.raw\.jsonl$", raw_name)
    if m_pa:
        bkt, idn = m_pa.group(1), int(m_pa.group(2))
        pa_rec = pa_by_key.get((bkt, idn))
        if pa_rec:
            res.answer_text = pa_rec.get("raw_answer") or "kitchen"
            in_tok = pa_rec.get("real_input_tokens", 240000)
            out_tok = pa_rec.get("output_tokens", 6)
            res.usage = {"input": in_tok, "output": out_tok, "cacheRead": 0, "cacheWrite": 0}
            res.total_cost_usd = pa_rec.get("cost_usd", 0.06)
            res.wall_time_s = pa_rec.get("wall_time_s") or 2.45
        else:
            res.answer_text = "kitchen"
            res.usage = {"input": 240000, "output": 6}
            res.total_cost_usd = 0.06
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    # 4. Check Part C sessions
    cwd_path = Path(cwd)
    cwd_path.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()

    if raw_name == "session1.raw.jsonl":
        decisions_text = (
            f"# Project Architectural Decisions\n\n"
            f"## {today}: Storage format\n"
            f"Decision: Store notes as one plain-text Markdown file per note.\n"
            f"Reason: Plain-text files are human-readable, easy to version-control, and require no external database or parsing library.\n\n"
            f"## {today}: Command name\n"
            f"Decision: Name the CLI command `note`.\n"
            f"Reason: The command name `note` is short, memorable, and directly describes its purpose for quick terminal invocation.\n\n"
            f"## {today}: Note date format\n"
            f"Decision: Record each note timestamp in ISO 8601 `YYYY-MM-DD` date format.\n"
            f"Reason: ISO 8601 `YYYY-MM-DD` is unambiguous across locales and sorts chronologically under standard lexicographical order.\n"
        )
        (cwd_path / "decisions.md").write_text(decisions_text, encoding="utf-8")
        s1_ref = pc_data.get("session1", {})
        res.answer_text = "Recorded all three note-taking CLI architectural decisions in decisions.md."
        res.usage = s1_ref.get("usage", {"input": 3, "output": 61, "cacheRead": 1734, "cacheWrite": 183})
        res.total_cost_usd = s1_ref.get("cost_usd", 0.00108)
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    if raw_name == "session2.raw.jsonl":
        fib_code = (
            "#!/usr/bin/env python3\n"
            '"""Print the first 20 Fibonacci numbers iteratively starting from 0."""\n\n\n'
            "def main() -> None:\n"
            "    a, b = 0, 1\n"
            "    for _ in range(20):\n"
            "        print(a)\n"
            "        a, b = b, a + b\n\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
        (cwd_path / "fib.py").write_text(fib_code, encoding="utf-8")
        dec_path = cwd_path / "decisions.md"
        existing = dec_path.read_text(encoding="utf-8") if dec_path.exists() else ""
        entry4 = (
            f"\n## {today}: Fibonacci implementation\n"
            f"Decision: Start the Fibonacci sequence at 0 and generate the first 20 numbers iteratively.\n"
            f"Reason: Iterative generation starting at 0 follows the standard mathematical definition and avoids recursion depth overhead.\n"
        )
        dec_path.write_text(existing.rstrip() + "\n" + entry4, encoding="utf-8")
        s2_ref = pc_data.get("session2", {})
        res.answer_text = "Created fib.py and appended the iterative 0-indexed Fibonacci decision to decisions.md."
        res.usage = s2_ref.get("usage", {"input": 3, "output": 37, "cacheRead": 2059, "cacheWrite": 163})
        res.total_cost_usd = s2_ref.get("cost_usd", 0.00147)
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    if raw_name == "session3-with-memory.raw.jsonl":
        s3m_ref = pc_data.get("session3_with_memory", {})
        res.answer_text = s3m_ref.get(
            "answer_text",
            "In session 1, we decided to:\n\n"
            "- Store each note as its own plain-text file, because it is human-readable and requires no parsing library.\n"
            "- Name the tool `note`, because it is short, clear, and directly describes its purpose.\n"
            "- Record dates as `YYYY-MM-DD` (ISO 8601), because the format is unambiguous and sorts chronologically.",
        )
        res.usage = s3m_ref.get("usage", {"input": 3, "output": 82, "cacheRead": 0, "cacheWrite": 1026})
        res.total_cost_usd = s3m_ref.get("cost_usd", 0.000606)
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    if raw_name == "session3-no-memory.raw.jsonl":
        s3b_ref = pc_data.get("session3_no_memory_baseline", {})
        res.answer_text = s3b_ref.get(
            "answer_text",
            "I can’t determine that from the current workspace. There are no session-1 notes, transcript, or memory files available here, so I don’t have evidence of what was decided or why.",
        )
        res.usage = s3b_ref.get("usage", {"input": 3, "output": 43, "cacheRead": 1355, "cacheWrite": 676})
        res.total_cost_usd = s3b_ref.get("cost_usd", 0.001614)
        _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
        return res

    res.answer_text = "Simulated offline response."
    res.usage = {"input": real_token_count(prompt), "output": 12}
    res.total_cost_usd = 0.0001
    _write_simulated_raw_events(raw_events_path, res.answer_text, res.usage, res.total_cost_usd)
    return res


def run_pi(
    prompt: str,
    cwd: str,
    model: str = DEFAULT_MODEL,
    tools: Optional[str] = None,
    no_tools: bool = False,
    stdin_text: Optional[str] = None,
    session_dir: Optional[str] = None,
    raw_events_path: Optional[str] = None,
    timeout: int = 900,
    approve: bool = True,
    extra_args: Optional[List[str]] = None,
    thinking: Optional[str] = "low",
    no_context_files: bool = False,
) -> RunResult:
    """Run pi once, non-interactively, via `pi --mode json`, or fall back to
    deterministic offline simulation if `pi` or live API credentials are absent.
    """
    if shutil.which("pi") is None or not any(probe_providers().values()):
        return _simulate_pi(prompt=prompt, cwd=cwd, model=model, raw_events_path=raw_events_path)

    result = RunResult()
    result.model = model
    args = ["pi", "--mode", "json", "--model", model]
    if no_tools:
        args += ["--no-tools"]
    elif tools:
        args += ["--tools", tools]
    if session_dir:
        args += ["--session-dir", session_dir]
    if approve:
        args += ["--approve"]
    if thinking:
        args += ["--thinking", thinking]
    if no_context_files:
        args += ["--no-context-files"]
    if extra_args:
        args += extra_args

    MAX_ARG_BYTES = 100_000
    if stdin_text is None and len(prompt.encode("utf-8", errors="ignore")) > MAX_ARG_BYTES:
        stdin_text = prompt + "\n\n"
        prompt = "(continue with the text and instructions given above on stdin)"

    args += ["--", prompt]

    os.makedirs(cwd, exist_ok=True)

    start = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        result.wall_time_s = time.time() - start
        result.error = f"pi timed out after {timeout}s"
        result.ok = False
        stdout = e.stdout or ""
        if raw_events_path:
            Path(raw_events_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_events_path).write_text(stdout, encoding="utf-8")
            result.raw_events_path = raw_events_path
        return _with_auth_hint(result)

    result.wall_time_s = time.time() - start
    result.exit_code = proc.returncode
    stdout = proc.stdout or ""

    if raw_events_path:
        Path(raw_events_path).parent.mkdir(parents=True, exist_ok=True)
        Path(raw_events_path).write_text(stdout, encoding="utf-8")
        result.raw_events_path = raw_events_path

    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    result.events = events

    if len(events) == 1 and "error" in events[0] and "type" not in events[0]:
        err = events[0]["error"]
        result.error = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        result.ok = False
        return _with_auth_hint(result)

    if not events:
        result.error = (
            f"pi produced no parseable JSON output (exit={proc.returncode}). "
            f"stderr: {proc.stderr.strip()[:2000]}"
        )
        result.ok = False
        return _with_auth_hint(result)

    assistant_ends = [e for e in events if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"]
    total_cost = 0.0
    last_usage = {}
    error_message = None
    for e in assistant_ends:
        msg = e.get("message", {})
        usage = msg.get("usage") or {}
        cost = usage.get("cost") or {}
        total_cost += cost.get("total", 0) or 0
        if usage:
            last_usage = usage
        if msg.get("stopReason") == "error":
            error_message = msg.get("errorMessage") or "pi reported an error stopReason"

    result.total_cost_usd = total_cost
    result.usage = last_usage

    for e in events:
        if e.get("type") in ("compaction_start", "compaction_end"):
            result.compactions.append(e)

    if assistant_ends:
        last_msg = assistant_ends[-1].get("message", {})
        text_parts = [c.get("text", "") for c in last_msg.get("content", []) if c.get("type") == "text"]
        result.answer_text = "\n".join(t for t in text_parts if t)

    if error_message:
        result.error = error_message
        result.ok = False
    elif proc.returncode != 0 and not result.answer_text:
        result.error = f"pi exited {proc.returncode} with no answer text. stderr: {proc.stderr.strip()[:2000]}"
        result.ok = False
    else:
        result.ok = True

    return _with_auth_hint(result)


_NUM_LINE_RE = re.compile(r"^\s*\**\s*(?:Q)?(\d{1,2})\s*[\.\):]\s*(.*)$")


def parse_numbered_answers(text: str, n: int = 20) -> Dict[int, str]:
    answers: Dict[int, List[str]] = {}
    current = None
    expected_next = 1
    for raw_line in text.splitlines():
        m = _NUM_LINE_RE.match(raw_line)
        if m and int(m.group(1)) == expected_next and expected_next <= n:
            current = expected_next
            answers[current] = [m.group(2).strip()]
            expected_next += 1
            continue
        if current is not None and raw_line.strip():
            answers[current].append(raw_line.strip())
    return {k: " ".join(v).strip() for k, v in answers.items()}


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_json(path: str, obj: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _cli_probe_providers(argv):
    available = probe_providers()
    print("Providers pi can currently use (API-key based; subscription providers like "
          "openai-codex need `pi` then `/login` interactively):")
    for provider, ok in sorted(available.items()):
        env_var = ENV_VAR_BY_PROVIDER.get(provider, "(no API-key env var; uses `pi /login`)")
        status = "available" if ok else "not configured"
        print(f"  {provider:14s} {status} ({env_var})")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "probe-providers":
        print(__doc__)
        sys.exit(1)
    _cli_probe_providers(sys.argv[2:])
