"""Verifier for task_01: Completed GPU-hours by cluster queue."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

APP = Path("/app")
FIGURE = APP / "figure.png"
SIDECAR = APP / "plotted_values.json"
CSV = APP / 'cluster_jobs.csv'
SCRIPT = APP / "plot.py"

CATEGORIES = ['alpha', 'beta', 'delta', 'gamma']
EXPECTED = [24.0, 34.0, 22.0, 30.0]
PERTURBED = [30.0, 34.0, 22.0, 30.0]
EXTRA_ROW = '113,alpha,completed,2,3.0\n'
CSV_SHA256 = '2b92744a514f3d82246390b7d62d6405230946187c59a095ee0e4ee7c5b59bbc'


def test_figure_exists():
    assert FIGURE.exists(), f"no figure at {FIGURE}; the task requires one"


def test_figure_is_a_valid_nonblank_png():
    size = FIGURE.stat().st_size
    assert size > 1000, f"{FIGURE} is only {size} bytes, so it is not a real chart"
    with Image.open(FIGURE) as image:
        image.verify()
    with Image.open(FIGURE) as image:
        assert image.format == "PNG", f"{FIGURE} is a {image.format}, not a PNG"
        assert min(image.size) > 100, f"{FIGURE} is {image.size}, too small to read"
        pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
    distinct = len(np.unique(pixels, axis=0))
    assert distinct > 4, f"{FIGURE} has {distinct} distinct colours: it is a blank canvas"


def test_input_csv_was_not_modified():
    assert CSV.exists(), f"input file {CSV} is missing"
    digest = hashlib.sha256(CSV.read_bytes()).hexdigest()
    assert digest == CSV_SHA256, f"{CSV} was modified (sha256 {digest} != {CSV_SHA256})"


def test_sidecar_parses_and_matches_expected_values():
    assert SIDECAR.exists(), f"no {SIDECAR}; instruction.md requires it"
    declared = json.loads(SIDECAR.read_text())
    assert isinstance(declared, dict), f"{SIDECAR} holds {type(declared).__name__}, expected dict"
    assert sorted(declared) == CATEGORIES, f"keys {sorted(declared)} != {CATEGORIES}"
    for key, value in declared.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{SIDECAR}[{key!r}] is not a JSON number"
        )
    actual = [float(declared[k]) for k in CATEGORIES]
    assert actual == pytest.approx(EXPECTED, rel=1e-5), (
        f"plotted_values.json values {actual} do not match expected {EXPECTED}"
    )


def test_script_reruns_and_tracks_perturbed_input():
    assert SCRIPT.exists() and SCRIPT.stat().st_size > 0, "missing /app/plot.py"
    with tempfile.TemporaryDirectory(prefix="verify_reexec_") as tmp:
        workdir = Path(tmp)
        shutil.copy2(SCRIPT, workdir / "plot.py")
        text = CSV.read_text()
        if not text.endswith("\n"):
            text += "\n"
        (workdir / CSV.name).write_text(text + EXTRA_ROW)
        res = subprocess.run(
            [sys.executable, "plot.py"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert res.returncode == 0, f"plot.py failed on re-execution: {res.stderr}"
        assert (workdir / "figure.png").is_file(), "re-execution did not write figure.png"
        assert (workdir / "plotted_values.json").is_file(), "re-execution did not write plotted_values.json"
        perturbed_declared = json.loads((workdir / "plotted_values.json").read_text())
        perturbed_actual = [float(perturbed_declared[k]) for k in CATEGORIES]
        assert perturbed_actual == pytest.approx(PERTURBED, rel=1e-5), (
            f"perturbed values {perturbed_actual} != {PERTURBED}; script may be hardcoding values"
        )
