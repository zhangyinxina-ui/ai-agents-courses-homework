"""Offline & LLM-compatible teaching harness for Studio 01.

Supports deterministic offline execution (ScriptedPolicy and BoundaryRecoveryPolicy)
when OPENAI_API_KEY is not set, as well as optional OpenAI LLM policy execution.
Generates verifiable JSONL execution traces in traces/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Rejected(RuntimeError):
    """An action violated the teaching contract."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Action:
    operation_id: str
    name: str
    arguments: dict[str, Any]


class Harness:
    def __init__(self, workspace: Path, *, max_steps: int = 12) -> None:
        self.root = workspace.resolve(strict=True)
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.last_check: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.cache: dict[str, tuple[str, dict[str, Any]]] = {}
        # Instructor-owned target; the tool API cannot modify it.
        self.expected = (self.root / 'approved-installation.txt').read_text().strip()
        if not self.expected:
            raise ValueError('Empty instructor-owned target')

    def _file(self, path: Any) -> Path:
        if path != 'README.md':
            raise Rejected('Only README.md is in scope')
        raw = self.root / 'README.md'
        if raw.is_symlink():
            raise Rejected('Symlinks are not permitted in the teaching fixture')
        resolved = raw.resolve(strict=True)
        if resolved.parent != self.root or not resolved.is_file():
            raise Rejected('Resource outside approved workspace')
        return resolved

    def _event(self, action: Action, status: str, result: Any) -> None:
        self.events.append({
            'event_id': f'e{len(self.events)+1:03}',
            'operation_id': action.operation_id,
            'action': action.name,
            'arguments': action.arguments,
            'status': status,
            'result': result,
        })

    def execute(self, action: Action) -> dict[str, Any]:
        if not isinstance(action.arguments, dict) or not isinstance(action.operation_id, str) or not action.operation_id:
            raise Rejected('Invalid action envelope')
        try:
            signature = digest(json.dumps([action.name, action.arguments], sort_keys=True).encode())
        except (TypeError, ValueError) as exc:
            raise Rejected('Arguments must be JSON-compatible') from exc
        if action.operation_id in self.cache:
            old_sig, old_result = self.cache[action.operation_id]
            if signature != old_sig:
                self._event(action, 'rejected', 'Operation ID reused with different parameters')
                raise Rejected('Operation ID reused with different parameters')
            self._event(action, 'replay_no_effect', old_result)
            return dict(old_result)
        try:
            if self.done:
                raise Rejected('Task already terminated')
            if self.steps >= self.max_steps:
                raise Rejected('Budget exhausted')
            self.steps += 1
            allowed_keys = {
                'read': {'path'},
                'replace': {'path', 'text'},
                'check': {'path'},
                'finish': set(),
            }
            if action.name not in allowed_keys:
                raise Rejected('Unknown or prohibited action')
            if set(action.arguments) != allowed_keys[action.name]:
                raise Rejected('Unexpected or missing argument keys')
            if action.name == 'finish':
                current = digest(self._file('README.md').read_bytes())
                if not self.last_check or not self.last_check['passed']:
                    raise Rejected('No passing check')
                if self.last_check['artifact_sha256'] != current:
                    raise Rejected('Check is stale for the current artifact')
                self.done = True
                result = {'accepted': True, 'artifact_sha256': current}
            else:
                path = self._file(action.arguments['path'])
                if action.name == 'read':
                    result = {'text': path.read_text(), 'artifact_sha256': digest(path.read_bytes())}
                elif action.name == 'replace':
                    text = action.arguments['text']
                    if not isinstance(text, str) or len(text.encode()) > 10000:
                        raise Rejected('Replacement must be a string of at most 10KB')
                    path.write_text(text)
                    result = {'written': 'README.md', 'artifact_sha256': digest(path.read_bytes())}
                else:
                    data = path.read_bytes()
                    lines = data.decode().splitlines()
                    result = {
                        'passed': self.expected in lines,
                        'artifact_sha256': digest(data),
                        'verifier': 'approved-line-v1',
                    }
                    self.last_check = dict(result)
            self.cache[action.operation_id] = (signature, dict(result))
            self._event(action, 'ok', result)
            return result
        except Rejected as exc:
            self._event(action, 'rejected', str(exc))
            raise


class ScriptedPolicy:
    """Deterministic control-flow policy for the happy-path observable run."""

    def run(self, harness: Harness) -> None:
        harness.execute(Action('read-1', 'read', {'path': 'README.md'}))
        harness.execute(Action('write-1', 'replace', {
            'path': 'README.md',
            'text': '# Course fixture\n\nInstallation\n\n' + harness.expected + '\n',
        }))
        check = harness.execute(Action('check-1', 'check', {'path': 'README.md'}))
        if not check['passed']:
            raise Rejected('Fixture check failed; do not claim success')
        harness.execute(Action('finish-1', 'finish', {}))


class BoundaryRecoveryPolicy:
    """Section 3 (Change & Explain): Perturbs execution with a permission boundary
    violation, an unverified finish attempt, and a stale-check mutation, then makes
    an evidence-led correction to achieve a clean verified finish."""

    def run(self, harness: Harness) -> None:
        # 1. Read initial fixture
        harness.execute(Action('read-1', 'read', {'path': 'README.md'}))

        # 2. Change condition #1: Permission boundary check (attempting to read outside scope)
        try:
            harness.execute(Action('boundary-1', 'read', {'path': 'approved-installation.txt'}))
        except Rejected:
            pass  # Observed and recorded in trace as 'rejected': 'Only README.md is in scope'

        # 3. Change condition #2: Initial check before fix fails verification
        initial_check = harness.execute(Action('check-initial', 'check', {'path': 'README.md'}))
        assert not initial_check['passed']
        try:
            harness.execute(Action('finish-premature', 'finish', {}))
        except Rejected:
            pass  # Observed and recorded in trace as 'rejected': 'No passing check'

        # 4. Write approved installation line and run check
        harness.execute(Action('write-1', 'replace', {
            'path': 'README.md',
            'text': '# Course fixture\n\nInstallation\n\n' + harness.expected + '\n',
        }))
        harness.execute(Action('check-1', 'check', {'path': 'README.md'}))

        # 5. Change condition #3: Modify README after check (causes stale check rejection)
        harness.execute(Action('write-2', 'replace', {
            'path': 'README.md',
            'text': '# Course fixture\n\nInstallation\n\n' + harness.expected + '\n\nVerified for Studio 01.\n',
        }))
        try:
            harness.execute(Action('finish-stale', 'finish', {}))
        except Rejected:
            pass  # Observed and recorded in trace as 'rejected': 'Check is stale for the current artifact'

        # 6. Evidence-led correction: Re-run check on current artifact so sha256 matches, then finish
        final_check = harness.execute(Action('check-2', 'check', {'path': 'README.md'}))
        if not final_check['passed']:
            raise Rejected('Fixture check failed after recovery')
        harness.execute(Action('finish-1', 'finish', {}))


def resolve_fixture_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / 'studio_fixture').exists():
        return here / 'studio_fixture'
    alt = here / 'instruction' / 'code' / 'studio_fixture'
    if alt.exists():
        return alt
    raise FileNotFoundError('Could not locate studio_fixture directory')


def run_and_save_trace(policy: Any, output_path: Path, fixture_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='agent-course-') as tmp:
        root = Path(tmp) / 'fixture'
        shutil.copytree(fixture_dir, root)
        h = Harness(root)
        policy.run(h)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in h.events), encoding='utf-8')
        return {
            'accepted': h.done,
            'events': len(h.events),
            'steps': h.steps,
            'policy': policy.__class__.__name__,
            'trace': str(output_path),
            'final_artifact_sha256': h.last_check['artifact_sha256'] if h.last_check else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_trace = Path(__file__).resolve().parent / 'traces' / 'demo_trace.jsonl'
    parser.add_argument('--output', type=Path, default=default_trace)
    parser.add_argument('--mode', choices=('normal', 'boundary', 'both'), default='both')
    args = parser.parse_args()

    fixture_dir = resolve_fixture_dir()

    if args.mode in ('normal', 'both'):
        summary = run_and_save_trace(ScriptedPolicy(), args.output, fixture_dir)
        print(json.dumps({
            'accepted': summary['accepted'],
            'events': summary['events'],
            'policy': 'scripted fixture, NOT an LLM' if not os.environ.get('OPENAI_API_KEY') else summary['policy'],
            'trace': summary['trace'],
        }, ensure_ascii=False))

    if args.mode in ('boundary', 'both'):
        boundary_out = args.output.parent / 'boundary_trace.jsonl' if args.mode == 'both' else args.output
        b_summary = run_and_save_trace(BoundaryRecoveryPolicy(), boundary_out, fixture_dir)
        if args.mode == 'boundary':
            print(json.dumps(b_summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
