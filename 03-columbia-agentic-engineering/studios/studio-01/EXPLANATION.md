# Studio 01 Explanation: Delegation & The First Observable Run

## 1. Summary of What the Agent & Harness Did

In Studio 01, we framed a bounded recurring delegation task in [`delegation-card.md`](delegation-card.md) and executed it inside a deterministic, observable verification harness ([`harness_demo.py`](harness_demo.py)). Rather than trusting an agent's self-reported completion claim, the harness enforces:
1. **Strict Scope & Path Isolation**: Only `README.md` inside the temporary workspace root is accessible. Symlinks, `../` traversal, and modifications to the instructor-owned `approved-installation.txt` (`pip install -e .`) are blocked by `Harness._file()`.
2. **Action Whitelisting & Step Budget**: Only `read`, `replace`, `check`, and `finish` are permitted within a hard ceiling of `max_steps = 12`.
3. **Idempotency via Operation IDs**: Every `Action` carries an `operation_id` whose `[name, arguments]` payload is hashed with SHA-256. Re-submitting the same `operation_id` with identical parameters returns `replay_no_effect` without re-executing side effects or consuming step budget; reusing an `operation_id` with altered parameters raises `Rejected("Operation ID reused with different parameters")`.
4. **Evidence Freshness on Completion**: `finish()` compares the current SHA-256 digest of `README.md` against `self.last_check["artifact_sha256"]`. If `check` was never run, failed, or became stale because `README.md` was modified after `check`, termination is rejected.

We generated two complete JSONL execution traces in [`traces/`](traces/):
- [`traces/demo_trace.jsonl`](traces/demo_trace.jsonl): The 4-step happy-path run (`read-1` $\rightarrow$ `write-1` $\rightarrow$ `check-1` $\rightarrow$ `finish-1`), transitioning `README.md` from SHA-256 `67716a99...` (containing the unapproved `pip install wrong-course-package`) to SHA-256 `78b182dd...` (containing `pip install -e .`).
- [`traces/boundary_trace.jsonl`](traces/boundary_trace.jsonl): The 10-event boundary perturbation and recovery run (`BoundaryRecoveryPolicy`).

## 2. Human Decisions and Why They Were Made

1. **Separating Instructor-Owned Specification from Agent-Writable Scope**: We kept `approved-installation.txt` outside the agent's read/write scope in `_file()`. If an agent could edit the ground-truth file, it could trivially pass verification by rewriting the test target to match its own broken output (reward hacking).
2. **Enforcing Cryptographic Evidence Freshness (`artifact_sha256`)**: A common failure mode in agent loops is running a test suite, making one "small cleanup edit" afterward, and calling `finish()` without re-verifying. By binding `self.last_check` to the exact SHA-256 digest of the file bytes at verification time, any subsequent `replace` invalidates the check until `check` is executed again.
3. **Offline Deterministic Mode**: To guarantee 100% reproducibility in CI and local grading without leaking API keys, `ScriptedPolicy` and `BoundaryRecoveryPolicy` exercise the exact control-flow invariants deterministically while producing structured JSONL event logs.

## 3. Perturbation Experiment (Change & Explain) and Verification

In `BoundaryRecoveryPolicy` (recorded in [`traces/boundary_trace.jsonl`](traces/boundary_trace.jsonl)), we deliberately changed three operating conditions to observe how the harness responds and how an evidence-led correction recovers:

| Event ID | Operation ID | Perturbation / Action | Observed Status & Result | Evidence-Led Correction |
|---|---|---|---|---|
| `e002` | `boundary-1` | **Permission Boundary**: Called `read(path="approved-installation.txt")` | `rejected`: `"Only README.md is in scope"` | Respect workspace file whitelist; only operate on `README.md`. |
| `e003`–`e004` | `check-initial`, `finish-premature` | **Failed Check**: Ran `check` on unmodified fixture (`passed: false`) and called `finish()` | `rejected`: `"No passing check"` | Apply `replace` with approved installation line `pip install -e .` (`write-1`, `e005`) and run `check-1` (`e006`, `passed: true`, SHA-256 `78b182dd...`). |
| `e007`–`e008` | `write-2`, `finish-stale` | **Stale Evidence**: Appended a footer to `README.md` (changing SHA-256 to `040cb275...`) and immediately called `finish()` | `rejected`: `"Check is stale for the current artifact"` | Re-run `check-2` (`e009`) on the updated artifact (`040cb275...`) before calling `finish-1` (`e010`, `accepted: true`). |

All 13 unit tests in [`test_harness.py`](test_harness.py) and the trace verifier [`verifier.py`](verifier.py) pass with exit code `0`.

## 4. Remaining Uncertainties

- **Semantic vs. Syntactic Verification**: `approved-line-v1` checks exact line membership (`self.expected in lines`). An agent could still insert misleading prose elsewhere in `README.md` while keeping `pip install -e .` on its own line. Stronger verifiers would parse Markdown AST sections or execute isolated container smoke tests.
- **Filesystem TOCTOU**: In a multi-process environment, a file could theoretically change between `check` and `finish` or during `_file()` resolution, though hash-comparing file bytes inside `finish()` detects any content mutation prior to acceptance.
