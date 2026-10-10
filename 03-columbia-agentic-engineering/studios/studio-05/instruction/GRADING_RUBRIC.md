# Studio 05 Grading Rubric

**Total:** 10 points: 2 for each of the four checkpoints and 2 for the explanation. Lines 1.1 to 4.2 are worth 1 point each. Every line is either met or not met.

| Line | Points | Condition for credit | Read from |
|---|---|---|---|
| 1.1 | 1 | The server has 3 to 8 tools over your own local data, at least one is marked `destructiveHint: true`, and every tool sets all four annotations: `readOnlyHint`, `destructiveHint`, `idempotentHint` and `openWorldHint`. | `evidence/tools.json` |
| 1.2 | 1 | Every description says what the tool does, when to use it, and when not to use it. | `evidence/tools.json` |
| 2.1 | 1 | One Codex run of the task in `task.txt` completes calls to at least two different tools of your server. | `runs/connection.jsonl`, `task.txt` |
| 2.2 | 1 | That run ends with a final answer after the last call. | `runs/connection.jsonl` |
| 3.1 | 1 | Every delete-all call is refused (exit code 5), and `count_after_delete_all` equals `count_before`, which is greater than 1. The single delete succeeds (exit code 0), and `count_after_single_delete` is one less than `count_after_delete_all`. | `evidence/direct-calls.jsonl` |
| 3.2 | 1 | In a Codex run told in plain words to delete all data with your tools, Codex blocks none of its calls for approval, and `count_after_agent_run` equals `count_after_single_delete`. `EXPLANATION.md` names the file and line where your code refuses, and that line is code, not a prompt, a tool description or a host setting. | `runs/delete-all.jsonl`, `runs/delete-all.txt`, `evidence/direct-calls.jsonl`, `EXPLANATION.md`, your server's code |
| 4.1 | 1 | At least 10 tasks, each with an expected outcome and a check, and at least 2 expect no tool call or a refusal. | `eval/tasks.jsonl` |
| 4.2 | 1 | Both results files record 5 runs of every task, made with Codex and gpt-5.6-luna, and their `pass_1` and `pass_5` match those runs. A `Change:` line names the one thing you changed, and a `Commit:` line names that commit in your repository. | `eval/results-before.json`, `eval/results-after.json`, `EXPLANATION.md` |
| 5 | 2 | `EXPLANATION.md` answers the four questions. | `EXPLANATION.md` |

All files are in `studio/studio-05/submission/<team>/` in your team's private repository, at the commit you submit.

- A grading agent reads your repository at that commit and checks each line. It does not run your code. A line the files cannot settle goes to a TA.
- A late or missing file scores 0 for its lines.
- How high your agent scores is not graded.
- Every member must be able to explain the code behind every line (merge defense).
