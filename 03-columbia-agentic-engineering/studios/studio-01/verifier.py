"""Studio 01 Verifier: Verifies both the fixture README installation line and
the execution traces in traces/ (demo_trace.jsonl and boundary_trace.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_demo import BoundaryRecoveryPolicy, ScriptedPolicy, resolve_fixture_dir, run_and_save_trace


def verify_traces_and_harness(base_dir: Path) -> dict:
    traces_dir = base_dir / 'traces'
    traces_dir.mkdir(parents=True, exist_ok=True)
    demo_trace = traces_dir / 'demo_trace.jsonl'
    boundary_trace = traces_dir / 'boundary_trace.jsonl'

    fixture_dir = resolve_fixture_dir()
    if not demo_trace.exists():
        run_and_save_trace(ScriptedPolicy(), demo_trace, fixture_dir)
    if not boundary_trace.exists():
        run_and_save_trace(BoundaryRecoveryPolicy(), boundary_trace, fixture_dir)

    demo_events = [json.loads(line) for line in demo_trace.read_text(encoding='utf-8').splitlines() if line.strip()]
    boundary_events = [json.loads(line) for line in boundary_trace.read_text(encoding='utf-8').splitlines() if line.strip()]

    # Verify demo trace
    demo_check = next((e for e in reversed(demo_events) if e['action'] == 'check' and e['status'] == 'ok'), None)
    demo_finish = demo_events[-1] if demo_events else {}
    demo_passed = (
        bool(demo_check)
        and demo_check['result'].get('passed') is True
        and demo_finish.get('action') == 'finish'
        and demo_finish.get('status') == 'ok'
        and demo_finish.get('result', {}).get('accepted') is True
        and demo_finish.get('result', {}).get('artifact_sha256') == demo_check['result'].get('artifact_sha256')
    )

    # Verify boundary trace (must show rejected boundary/stale checks followed by recovery & accepted finish)
    rejected_count = sum(1 for e in boundary_events if e['status'] == 'rejected')
    b_check = next((e for e in reversed(boundary_events) if e['action'] == 'check' and e['status'] == 'ok'), None)
    b_finish = boundary_events[-1] if boundary_events else {}
    boundary_passed = (
        rejected_count >= 3
        and bool(b_check)
        and b_check['result'].get('passed') is True
        and b_finish.get('action') == 'finish'
        and b_finish.get('status') == 'ok'
        and b_finish.get('result', {}).get('accepted') is True
        and b_finish.get('result', {}).get('artifact_sha256') == b_check['result'].get('artifact_sha256')
    )

    expected_line = (fixture_dir / 'approved-installation.txt').read_text(encoding='utf-8').strip()
    passed = bool(demo_passed and boundary_passed and expected_line == 'pip install -e .')

    return {
        'passed': passed,
        'verifier': 'approved-line-v1',
        'expected_line': expected_line,
        'artifact_sha256': demo_finish.get('result', {}).get('artifact_sha256', hashlib.sha256(expected_line.encode()).hexdigest()),
        'demo_trace_events': len(demo_events),
        'boundary_trace_events': len(boundary_events),
        'boundary_rejections_caught': rejected_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=None, help='Optional workspace directory with README.md to check directly')
    args = parser.parse_args()

    if args.workspace is not None:
        root = args.workspace.resolve()
        data = (root / 'README.md').read_bytes()
        target = (root / 'approved-installation.txt').read_text().strip()
        passed = target in data.decode().splitlines()
        print(json.dumps({
            'passed': passed,
            'artifact_sha256': hashlib.sha256(data).hexdigest(),
            'verifier': 'approved-line-v1',
        }))
        raise SystemExit(0 if passed else 1)

    base_dir = Path(__file__).resolve().parent
    report = verify_traces_and_harness(base_dir)
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
