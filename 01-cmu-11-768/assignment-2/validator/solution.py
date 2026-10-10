"""The validator you write. This is the only file you need to modify.

Each function evaluates a single run object and returns the errors it finds.
An empty list means that no errors were found. `validator/runner.py` defines
what a run object contains, and `ASSIGNMENT.md` defines the four error families you may report.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from openai import BadRequestError
from PIL import Image, UnidentifiedImageError

from validator.model import complete, detokenize, tokenize
from validator.prediction import Error, ErrorFamily
from validator.runner import Run

CONTEXT_FRACTION = 0.5
MAX_OUTPUT_TOKENS = 400
OMISSION = "\n[... omitted to fit model context ...]\n"
COMPACTION_NOTE = "\nSome evidence may be omitted. Do not treat omissions as agent errors.\n"
IMAGE_DOWNSCALE_FACTOR = 0.7
MIN_IMAGE_LONG_SIDE = 64


def _has_vlm_endpoint() -> bool:
    """Return True when a live VLM judge endpoint is configured in the environment."""
    return bool(os.getenv("VALIDATOR_BASE_URL") and os.getenv("VALIDATOR_API_KEY"))


def _extract_commands_and_outputs(messages: list[dict[str, Any]]) -> list[tuple[str, int | None, str]]:
    """Extract (command, returncode, output) tuples in chronological order from run.messages."""
    steps: list[tuple[str, int | None, str]] = []
    pending_cmds: dict[str, str] = {}
    last_cmd: str | None = None

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}
                cmd = args.get("command", "") if isinstance(args, dict) else ""
                tc_id = tc.get("id") or ""
                if cmd:
                    pending_cmds[tc_id] = cmd
                    last_cmd = cmd
            extra = msg.get("extra") or {}
            for act in extra.get("actions") or []:
                if isinstance(act, dict) and act.get("command"):
                    tc_id = act.get("tool_call_id") or ""
                    pending_cmds[tc_id] = act["command"]
                    last_cmd = act["command"]
        elif role == "tool":
            tc_id = msg.get("tool_call_id") or ""
            cmd = pending_cmds.get(tc_id, last_cmd or "")
            extra = msg.get("extra") or {}
            rc = extra.get("returncode")
            out = extra.get("raw_output")
            if out is None:
                content = msg.get("content") or ""
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        rc = parsed.get("returncode", rc)
                        out = parsed.get("output", content)
                    else:
                        out = content
                except Exception:
                    out = content
            steps.append((cmd, rc, str(out or "")))
    return steps


def _extract_final_python_code(steps: list[tuple[str, int | None, str]]) -> str:
    """Extract the Python plotting code from the last successful plotting command(s)."""
    code_snippets: list[str] = []
    for cmd, rc, _out in steps:
        if rc not in (0, None):
            continue
        if any(tok in cmd for tok in ("matplotlib", "plt.", "savefig", ".py", "pandas", "seaborn", "PIL")):
            code_snippets.append(cmd)
    if not code_snippets:
        return "\n".join(cmd for cmd, _, _ in steps)
    return "\n\n".join(code_snippets[-3:])


def _excerpt(text: str, tokens: list[int], limit: int) -> str:
    if len(tokens) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return (
        (detokenize(tokens[:head]) if head else "")
        + OMISSION
        + (detokenize(tokens[-tail:]) if tail else "")
    )


def _compact(trajectory: str, inputs: str, fixed_prompt: str) -> tuple[str, str]:
    trajectory_info = tokenize(trajectory)
    context = trajectory_info.get("max_model_len")
    if type(context) is not int or context <= 0:
        raise RuntimeError("Validator server did not report a valid max_model_len")
    trajectory_tokens = trajectory_info["tokens"]
    input_tokens = tokenize(inputs)["tokens"]
    fixed_tokens = tokenize(fixed_prompt + COMPACTION_NOTE + OMISSION * 2)["tokens"]
    budget = int(context * CONTEXT_FRACTION) - len(fixed_tokens) - MAX_OUTPUT_TOKENS
    if budget < 0:
        raise RuntimeError("Task instructions exceed the validator's context budget")
    total = len(trajectory_tokens) + len(input_tokens)
    trajectory_budget = budget * len(trajectory_tokens) // max(total, 1)
    return (
        _excerpt(trajectory, trajectory_tokens, trajectory_budget),
        _excerpt(inputs, input_tokens, budget - trajectory_budget),
    )


def _downscale(data: bytes) -> bytes | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    longest = max(width, height)
    if longest <= MIN_IMAGE_LONG_SIDE:
        return None
    scale = max(MIN_IMAGE_LONG_SIDE / longest, IMAGE_DOWNSCALE_FACTOR)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size == (width, height):
        return None
    with Image.open(io.BytesIO(data)) as image:
        if mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        resized = image.resize(new_size, Image.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()


def _context_overflow(error: BadRequestError) -> bool:
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "context_length_exceeded",
            "maximum context length",
            "maximum model length",
            "maximum input length",
        )
    )


def _ask_vlm(run: Run, family: ErrorFamily, rubric_question: str) -> list[Error]:
    """Query the VLM judge with context compaction and image downscaling fallback."""
    if not _has_vlm_endpoint():
        return []

    inputs = "\n".join(
        f"{name}:\n{path.read_text(errors='replace')[:4000]}"
        for name, path in run.inputs.items()
    )
    steps = _extract_commands_and_outputs(run.messages)
    compact_steps = "\n".join(
        f"[Step {i+1} | rc={rc}]\nCMD: {cmd}\nOUT: {out[:600]}"
        for i, (cmd, rc, out) in enumerate(steps)
    )
    image_bytes = run.figure.read_bytes() if (run.figure and run.figure.is_file()) else None

    def make_prompt(shown_trajectory: str, shown_inputs: str) -> str:
        return (
            f"{rubric_question}\n"
            "First verify every explicit requirement in the Task against the final script in Trajectory, "
            "the Input files, and the rendered Figure.\n"
            "Answer YES if and only if there is a clear violation of this error family; otherwise answer NO.\n"
            "Start your response with YES or NO on the first line, followed by concise, specific evidence.\n\n"
            f"Task:\n{run.instructions}\n\n"
            f"Trajectory:\n{shown_trajectory}\n\n"
            f"Input files:\n{shown_inputs or '(none)'}\n"
        )

    def ask(prompt: str, image: bytes | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            encoded = base64.b64encode(image).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        return complete([{"role": "user", "content": content}], max_tokens=MAX_OUTPUT_TOKENS)

    try:
        response = ask(make_prompt(compact_steps, inputs), image_bytes)
    except BadRequestError as error:
        if not _context_overflow(error):
            raise
        warnings.warn(
            f"Run {run.run_id} ({family.value}): model context exceeded; compacting text.",
            stacklevel=2,
        )
        shown_trajectory, shown_inputs = _compact(compact_steps, inputs, make_prompt("", ""))
        prompt = make_prompt(shown_trajectory, shown_inputs) + COMPACTION_NOTE
        image = image_bytes
        while True:
            try:
                response = ask(prompt, image)
                break
            except BadRequestError as retry_error:
                if not _context_overflow(retry_error):
                    raise
                smaller = _downscale(image) if image is not None else None
                if smaller is None:
                    raise
                image = smaller

    answer = response["choices"][0]["message"]["content"].strip()
    if not answer.upper().startswith("YES"):
        return []
    evidence = answer[3:].lstrip(" :.-\n") or answer
    return [Error(family=family, evidence=evidence)]


def judge_execution(run: Run) -> list[Error]:
    """Whether the run produced a figure to judge at all."""
    # 1. Deterministic check: figure path missing or not a file
    if run.figure is None or not run.figure.is_file():
        steps = _extract_commands_and_outputs(run.messages)
        saved_elsewhere = []
        for cmd, _rc, _out in steps:
            for match in re.findall(r"savefig\(\s*['\"]([^'\"]+)['\"]", cmd):
                if match != "figure.png" and not match.endswith("/figure.png"):
                    saved_elsewhere.append(match)
        if saved_elsewhere:
            return [
                Error(
                    family=ErrorFamily.EXECUTION_FAILURE,
                    evidence=(
                        f"figure.png was not created in the workspace root; the agent saved to "
                        f"{saved_elsewhere[-1]!r} instead of 'figure.png'."
                    ),
                )
            ]
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence="No figure.png artifact was produced by the run.",
            )
        ]

    # 2. Deterministic check: file size and valid PNG decoding
    try:
        size = run.figure.stat().st_size
    except OSError as exc:
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"Could not stat figure.png: {exc}",
            )
        ]
    if size < 500:
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"figure.png is truncated or empty ({size} bytes).",
            )
        ]

    try:
        with Image.open(run.figure) as img:
            img.verify()
        with Image.open(run.figure) as img:
            fmt = img.format
            width, height = img.size
            thumb = img.convert("RGB")
            thumb.thumbnail((128, 128))
            rgb = np.asarray(thumb)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"figure.png is not a valid readable PNG image: {exc}",
            )
        ]

    if fmt != "PNG":
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"figure.png has format {fmt!r} instead of 'PNG'.",
            )
        ]
    if min(width, height) < 64:
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"figure.png dimensions ({width}x{height}) are too small to contain a valid chart.",
            )
        ]

    # 3. Check for completely blank canvas (solid single-color or near-empty canvas)
    flat = rgb.reshape(-1, 3)
    unique_colors = len(np.unique(flat, axis=0))
    if unique_colors <= 2:
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"figure.png is a blank canvas with only {unique_colors} distinct color(s).",
            )
        ]
    # Fraction of non-background pixels (assuming most common color is background)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    bg_fraction = float(counts.max()) / float(len(flat))
    if bg_fraction > 0.998:
        return [
            Error(
                family=ErrorFamily.EXECUTION_FAILURE,
                evidence=f"figure.png is effectively blank ({bg_fraction:.2%} uniform background pixels).",
            )
        ]

    return []


def _heuristic_data_and_chart_checks(run: Run) -> list[Error]:
    """Programmatic trajectory + instruction + input analysis for WRONG_DATA and WRONG_CHART."""
    errors: list[Error] = []
    steps = _extract_commands_and_outputs(run.messages)
    code = _extract_final_python_code(steps)
    inst = run.instructions
    inst_lower = inst.lower()

    # --- WRONG_DATA checks ---
    wrong_data_reasons: list[str] = []

    # 1. Check if input files were provided but never referenced in the agent's plotting code
    if run.inputs and code:
        referenced_any = any(name in code for name in run.inputs)
        if not referenced_any and not any(
            tok in code for tok in ("glob", "listdir", "read_csv", "json.load", "open(")
        ):
            wrong_data_reasons.append(
                f"Task provides input file(s) {sorted(run.inputs)}, but the plotting script "
                "never reads them and hardcodes numbers instead."
            )

    # 2. Check explicit positive filtering requirements in instructions vs code
    if ("only" in inst_lower or "filter" in inst_lower) and code:
        for quoted in re.findall(r"`([^`]+)`", inst):
            if quoted in ("figure.png", "plot.py", "plotted_values.json") or quoted.endswith(".csv"):
                continue
            q_low = quoted.lower()
            if any(
                phrase in inst_lower
                for phrase in (
                    f"only `{q_low}`",
                    f"`status` is `{q_low}`",
                    f"`calibration` is `{q_low}`",
                    f"`tier` is `{q_low}`",
                    f"`audit_state` is `{q_low}`",
                    f"`grid_sync` is `{q_low}`",
                    f"inspection is `{q_low}`",
                )
            ):
                if quoted not in code and q_low not in code.lower():
                    wrong_data_reasons.append(
                        f"Instructions require filtering on `{quoted}`, but `{quoted}` never appears in the plotting code."
                    )

    # 3. Check explicit aggregation metric mismatches
    if "weighted" in inst_lower and ".mean(" in code and "sum(" not in code and "weights" not in code and "average(" not in code:
        wrong_data_reasons.append(
            "Instructions require a weighted aggregation, but the script computes an unweighted mean."
        )
    if "median" in inst_lower and ".mean(" in code and "median" not in code and "boxplot" not in code:
        wrong_data_reasons.append(
            "Instructions request median values, but the script computes `.mean()` instead of median."
        )
    if ("cumulative" in inst_lower or "running total" in inst_lower) and "cumsum" not in code and "accumulate" not in code:
        # Check if manual loop accumulation wasn't used either
        if "+=" not in code:
            wrong_data_reasons.append(
                "Instructions require a cumulative / running total calculation, which is missing from the script."
            )

    if wrong_data_reasons:
        errors.append(Error(family=ErrorFamily.WRONG_DATA, evidence=" ".join(wrong_data_reasons)))

    # --- WRONG_CHART checks ---
    wrong_chart_reasons: list[str] = []

    if code:
        # Check chart type requirements
        if "box plot" in inst_lower or "boxplot" in inst_lower:
            if "boxplot" not in code and "bxp" not in code:
                wrong_chart_reasons.append("Task requests a box plot, but `boxplot` was not used.")
        if "horizontal bar" in inst_lower:
            if "barh" not in code and "orientation='horizontal'" not in code and 'orientation="horizontal"' not in code:
                wrong_chart_reasons.append("Task requests a horizontal bar chart (`barh`), but vertical bars were drawn.")
        if "vertical bar" in inst_lower or "one vertical bar" in inst_lower:
            if "barh(" in code:
                wrong_chart_reasons.append("Task requests vertical bars, but horizontal bars (`barh`) were drawn.")
        if "scatter" in inst_lower and "bar chart" not in inst_lower and "line" not in inst_lower:
            if "scatter" not in code and "plot(" not in code:
                wrong_chart_reasons.append("Task requests a scatter plot, which is missing in the script.")
        if "heatmap" in inst_lower:
            if not any(tok in code for tok in ("imshow", "pcolormesh", "heatmap", "matshow")):
                wrong_chart_reasons.append("Task requests a heatmap, but no heatmap plotting function was called.")

        # Check log scale requirement
        if "log scale" in inst_lower or "logarithmic" in inst_lower or "log-scale" in inst_lower:
            if not any(tok in code for tok in ("set_xscale('log')", 'set_xscale("log")', "set_yscale('log')", 'set_yscale("log")', "semilogx", "semilogy", "loglog", "LogNorm")):
                wrong_chart_reasons.append("Task requires a logarithmic scale, but linear scaling was used.")

        # Check title / axis label requirements
        if "title" in inst_lower:
            if not any(tok in code for tok in ("set_title", "plt.title", "suptitle", "title=")):
                wrong_chart_reasons.append("Task requires a chart title, but none was set in the plotting code.")
        if "label both" in inst_lower or ("label the x" in inst_lower and "y axis" in inst_lower):
            has_xlabel = any(tok in code for tok in ("set_xlabel", "plt.xlabel", "xlabel="))
            has_ylabel = any(tok in code for tok in ("set_ylabel", "plt.ylabel", "ylabel="))
            if not (has_xlabel and has_ylabel):
                wrong_chart_reasons.append("Task requires both x-axis and y-axis labels, but at least one axis label is missing.")

        # Check errorbar requirement
        if ("error bar" in inst_lower or "errorbar" in inst_lower or "confidence interval" in inst_lower) and not any(
            tok in code for tok in ("yerr", "xerr", "errorbar", "fill_between")
        ):
            wrong_chart_reasons.append("Task requires error bars or confidence intervals, which were omitted.")

        # Check reference / threshold line requirement
        if ("dashed" in inst_lower and ("threshold" in inst_lower or "target" in inst_lower or "reference line" in inst_lower)):
            if not any(tok in code for tok in ("axhline", "axvline", "hlines", "vlines", "--", "dashed")):
                wrong_chart_reasons.append("Task requires a dashed reference/threshold line, which is missing.")

    if wrong_chart_reasons:
        errors.append(Error(family=ErrorFamily.WRONG_CHART, evidence=" ".join(wrong_chart_reasons)))

    return errors


def _heuristic_readability_checks(run: Run) -> list[Error]:
    """Pixel-level and code-level readability inspection for HARD_TO_READ."""
    if run.figure is None or not run.figure.is_file():
        return []
    try:
        with Image.open(run.figure) as img:
            width, height = img.size
            rgb = np.asarray(img.convert("RGB"))
    except Exception:
        return []

    reasons: list[str] = []

    # 1. Check severe aspect ratio / tiny resolution
    if width < 320 or height < 240:
        reasons.append(f"Figure resolution ({width}x{height}) is too small for legible text and tick labels.")
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > 4.5:
        reasons.append(f"Extreme aspect ratio ({width}x{height}, {aspect:.1f}:1) compresses the plot area and makes labels hard to read.")

    # 2. Check border clipping: non-background ink heavily touching the outer 2px edges of the canvas
    # Estimate background color from the four corners
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dtype=int)
    bg = np.median(corners, axis=0)
    top_edge_ink = float((np.abs(rgb[:2, :, :].astype(int) - bg).sum(axis=2) > 30).mean())
    bottom_edge_ink = float((np.abs(rgb[-2:, :, :].astype(int) - bg).sum(axis=2) > 30).mean())
    left_edge_ink = float((np.abs(rgb[:, :2, :].astype(int) - bg).sum(axis=2) > 30).mean())
    right_edge_ink = float((np.abs(rgb[:, -2:, :].astype(int) - bg).sum(axis=2) > 30).mean())

    clipped_edges = []
    if top_edge_ink > 0.08:
        clipped_edges.append("top")
    if bottom_edge_ink > 0.08:
        clipped_edges.append("bottom")
    if left_edge_ink > 0.08:
        clipped_edges.append("left")
    if right_edge_ink > 0.08:
        clipped_edges.append("right")
    if clipped_edges:
        reasons.append(
            f"Text or chart elements are clipped against the {', '.join(clipped_edges)} image border(s)."
        )

    # 3. Check trajectory code for known overcrowding / collision patterns
    steps = _extract_commands_and_outputs(run.messages)
    code = _extract_final_python_code(steps)
    if code:
        # Multi-subplot chart without tight_layout, constrained_layout, or subplots_adjust
        if any(tok in code for tok in ("subplots(2", "subplots(3", "subplots(nrows=2", "subplots(nrows=3", "GridSpec")):
            if not any(tok in code for tok in ("tight_layout", "constrained_layout", "subplots_adjust", "bbox_inches='tight'", 'bbox_inches="tight"')):
                reasons.append(
                    "Multi-panel subplot layout was rendered without `tight_layout()`, `constrained_layout`, or `subplots_adjust`, causing subplot titles and axis labels to collide."
                )

    if reasons:
        return [Error(family=ErrorFamily.HARD_TO_READ, evidence=" ".join(reasons))]
    return []


def judge_data_and_chart(run: Run) -> list[Error]:
    """Whether the figure plots the requested data, built the requested way."""
    heuristic_errors = _heuristic_data_and_chart_checks(run)
    if not _has_vlm_endpoint():
        return heuristic_errors

    vlm_wrong_data = _ask_vlm(
        run,
        ErrorFamily.WRONG_DATA,
        (
            "Does the plotted data in the figure or final script differ from what the task requested? "
            "Check for: (1) missing or incorrect row filtering, (2) wrong aggregation or mathematical formula "
            "(e.g. unweighted vs weighted mean, sum vs mean, wrong unit conversion), (3) hardcoded numbers "
            "that ignore the input file, or (4) mismatched category-to-value pairings."
        ),
    )
    vlm_wrong_chart = _ask_vlm(
        run,
        ErrorFamily.WRONG_CHART,
        (
            "Does the figure violate any structural or chart-design requirement stated in the task? "
            "Check for: (1) wrong chart type, (2) wrong category/series ordering along axes, "
            "(3) missing title, axis labels, legend, colorbar, or required annotations/reference lines, "
            "(4) wrong axis scale (linear vs log) or violated baseline/axis range constraints."
        ),
    )
    return _merge_errors(heuristic_errors + vlm_wrong_data + vlm_wrong_chart)


def judge_readability(run: Run) -> list[Error]:
    """Whether the figure can be read."""
    heuristic_errors = _heuristic_readability_checks(run)
    if not _has_vlm_endpoint():
        return heuristic_errors

    vlm_readability = _ask_vlm(
        run,
        ErrorFamily.HARD_TO_READ,
        (
            "Inspect the rendered PNG image visually: is the chart difficult or impossible to read due to "
            "visual defects such as overlapping tick labels, colliding text annotations, clipped titles/labels "
            "at the image border, legend occluding data marks, or indistinguishable colors?"
        ),
    )
    return _merge_errors(heuristic_errors + vlm_readability)


def _merge_errors(errors: list[Error]) -> list[Error]:
    """Deduplicate errors by ErrorFamily, combining distinct evidence statements."""
    by_family: dict[ErrorFamily, list[str]] = {}
    for err in errors:
        ev = err.evidence.strip()
        if ev and ev not in by_family.setdefault(err.family, []):
            by_family[err.family].append(ev)
    return [
        Error(family=fam, evidence=" | ".join(evidences))
        for fam, evidences in by_family.items()
    ]


def validate(run: Run) -> list[Error]:
    """Everything wrong with one run."""
    execution = judge_execution(run)
    if execution:
        return execution
    return _merge_errors(judge_data_and_chart(run) + judge_readability(run))
