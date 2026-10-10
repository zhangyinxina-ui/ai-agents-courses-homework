# Delegation Card: Weekly AI Engineering Research Digest & Course Fixture Verification

## Goal (Task)
Produce two bounded, verifiable artifacts per run:
1. **Verified Course Fixture (`README.md`)**: Inspect the target repository `README.md`, replace any unapproved installation command (`pip install wrong-course-package`) with the instructor-approved line from `approved-installation.txt` (`pip install -e .`), execute the deterministic verification check (`approved-line-v1`), and record an immutable SHA-256 execution trace in `traces/`.
2. **Weekly AI Engineering Research Brief (`outputs/research_digest.md`)**: Compile a structured 3-item technical digest covering recent agentic engineering updates from Anthropic Engineering, OpenAI News, and the COMS W4995-009 lecture materials on bounded delegation, tool harnesses, and evidence freshness.

## Context
- **Target Audience**: Columbia COMS W4995-009 (*Agentic Engineering*) students and teaching staff reviewing bounded agent execution and verification harnesses.
- **Inputs & Fixtures**:
  - `studio_fixture/README.md` (initial target artifact containing a deliberately unapproved installation command).
  - `studio_fixture/approved-installation.txt` (instructor-owned ground-truth specification: `pip install -e .`).
  - `studio_fixture/verifier.py` and `harness_demo.py` (scoped execution harness with SHA-256 artifact digest tracking and idempotency cache).
- **Available Tools**:
  - `read(path)`: Read-only inspection of `README.md` within the workspace root.
  - `replace(path, text)`: Scoped write to `README.md` (capped at 10 KB, symlinks and path traversal strictly rejected).
  - `check(path)`: Runs `approved-line-v1` verifier and binds `passed` status to the current `artifact_sha256`.
  - `finish()`: Terminal action that succeeds only if `last_check["passed"] is True` and `last_check["artifact_sha256"] == current_sha256`.

## Constraints (Restrictions)
- **Scope & Path Isolation**: Only `README.md` inside the approved workspace may be read or modified. Any attempt to access `../`, absolute paths outside the root, `approved-installation.txt`, or symlinks must be immediately rejected (`Rejected` exception).
- **Tool Whitelist & No Arbitrary Shell**: Only `{read, replace, check, finish}` are permitted. Arbitrary shell execution (`shell`, `bash`, `rm`) is prohibited.
- **Step Budget**: Maximum of `max_steps = 12` operations per run. Exhausting the budget raises `Rejected("Budget exhausted")`.
- **Idempotency Contract**: Reusing an `operation_id` with identical arguments must replay the cached result without side effects (`replay_no_effect`); reusing an `operation_id` with altered arguments must be rejected.
- **No Credentials in Repo**: Never store API keys or tokens in source files, traces, or git history. Operate deterministically offline when `OPENAI_API_KEY` is absent.

## Done-When Criteria (Success Criteria)
1. **Fresh Passing Verification**: `check("README.md")` returns `{"passed": true, "verifier": "approved-line-v1"}` and its `artifact_sha256` exactly matches the SHA-256 digest of `README.md` at the moment `finish()` is called (no post-check mutations).
2. **Trace Completeness**: `traces/demo_trace.jsonl` and `traces/boundary_trace.jsonl` contain valid JSONL events with sequential `event_id`s, operation IDs, SHA-256 digests, and terminal status `{"accepted": true}`.
3. **Test & Verifier Suite**: `pytest` / `unittest` (`test_harness.py`) and `python3 verifier.py` pass 100% of checks with exit code `0`.
