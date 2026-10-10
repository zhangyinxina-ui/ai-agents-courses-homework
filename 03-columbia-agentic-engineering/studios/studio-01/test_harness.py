import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_demo import (
    Action,
    BoundaryRecoveryPolicy,
    Harness,
    Rejected,
    ScriptedPolicy,
    resolve_fixture_dir,
    run_and_save_trace,
)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'fixture'
        shutil.copytree(resolve_fixture_dir(), self.root)
        self.h = Harness(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def call(self, name, args=None, op=None):
        return self.h.execute(Action(op or f'id-{len(self.h.events)}', name, args or {}))

    def write_correct(self):
        return self.call('replace', {'path': 'README.md', 'text': self.h.expected + '\n'})

    def test_happy_path(self):
        ScriptedPolicy().run(self.h)
        self.assertTrue(self.h.done)

    def test_boundary_recovery_path(self):
        BoundaryRecoveryPolicy().run(self.h)
        self.assertTrue(self.h.done)
        statuses = [e['status'] for e in self.h.events]
        self.assertIn('rejected', statuses)
        self.assertEqual(self.h.events[-1]['action'], 'finish')
        self.assertEqual(self.h.events[-1]['status'], 'ok')

    def test_no_evidence_no_finish(self):
        with self.assertRaisesRegex(Rejected, 'No passing'):
            self.call('finish')

    def test_stale_check_rejected(self):
        self.write_correct()
        self.call('check', {'path': 'README.md'})
        self.call('replace', {'path': 'README.md', 'text': self.h.expected + '\nChanged\n'})
        with self.assertRaisesRegex(Rejected, 'stale'):
            self.call('finish')

    def test_prohibited_tool(self):
        with self.assertRaisesRegex(Rejected, 'prohibited'):
            self.call('shell', {'cmd': 'anything'})

    def test_path_traversal(self):
        with self.assertRaisesRegex(Rejected, 'scope'):
            self.call('read', {'path': '../README.md'})

    def test_extra_argument(self):
        with self.assertRaisesRegex(Rejected, 'argument'):
            self.call('read', {'path': 'README.md', 'admin': True})

    def test_id_reuse_with_different_arguments(self):
        self.call('read', {'path': 'README.md'}, 'same')
        with self.assertRaisesRegex(Rejected, 'different'):
            self.call('finish', {}, 'same')

    def test_replay_does_not_repeat_side_effect(self):
        a = Action('write', 'replace', {'path': 'README.md', 'text': self.h.expected + '\n'})
        self.h.execute(a)
        (self.root / 'README.md').write_text('External change\n')
        before = self.h.steps
        self.h.execute(a)
        self.assertEqual((self.root / 'README.md').read_text(), 'External change\n')
        self.assertEqual(self.h.steps, before)
        self.assertEqual(self.h.events[-1]['status'], 'replay_no_effect')

    def test_budget(self):
        self.h = Harness(self.root, max_steps=1)
        self.call('read', {'path': 'README.md'})
        with self.assertRaisesRegex(Rejected, 'Budget'):
            self.call('read', {'path': 'README.md'})

    def test_failed_check_not_accepted(self):
        self.assertFalse(self.call('check', {'path': 'README.md'})['passed'])
        with self.assertRaisesRegex(Rejected, 'No passing'):
            self.call('finish')

    def test_symlink_rejected(self):
        target = Path(self.temp.name) / 'outside'
        target.write_text('outside')
        (self.root / 'README.md').unlink()
        (self.root / 'README.md').symlink_to(target)
        with self.assertRaisesRegex(Rejected, 'Symlinks'):
            self.call('read', {'path': 'README.md'})

    def test_new_action_after_finish(self):
        ScriptedPolicy().run(self.h)
        with self.assertRaisesRegex(Rejected, 'terminated'):
            self.call('read', {'path': 'README.md'})

    def test_trace_generation_jsonl_valid(self):
        trace_file = Path(self.temp.name) / 'traces' / 'test_trace.jsonl'
        summary = run_and_save_trace(ScriptedPolicy(), trace_file, resolve_fixture_dir())
        self.assertTrue(summary['accepted'])
        self.assertTrue(trace_file.exists())
        lines = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[-1]['result']['accepted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
