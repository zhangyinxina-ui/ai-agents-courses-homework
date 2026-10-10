Team: team-columbia  Members: yz0909

## What we ran
We executed the three-part Context Window Stress Test and Memory Architecture benchmark suite using model `openai/gpt-5.6-luna` (`--thinking low`, `--no-context-files`, compaction disabled for Part A/B):
```bash
python3 part_a_stress.py --model openai/gpt-5.6-luna --buckets 32k,128k,256k,512k,768k --n 20 --team team-columbia
python3 plot_qa1_curve.py --team team-columbia
python3 part_b_isolate_compress.py --model openai/gpt-5.6-luna --buckets 256k,512k,768k --n 20 --concurrency 4 --team team-columbia
python3 part_c_memory.py --model openai/gpt-5.6-luna --team team-columbia --fresh
```
In Part A, each BABILong `qa1` haystack was delivered in a single message (`--no-tools`, `.pi/settings.json` setting `{"compaction": {"enabled": false}}`) and scored via normalized exact-match against the ground-truth room name. In Part B, the `256k`, `512k`, and `768k` buckets were split into ~96K-token chunks (`3`, `6`, and `8` chunks per item respectively) and evaluated under two mitigation architectures: **Isolate** (per-chunk sub-agent extraction of verbatim location statements or `NOT FOUND`, followed by a lead synthesis call taking the last reported location) and **Targeted summary** (per-chunk ~200-token summary instructed to preserve room movements, followed by a re-ask call). In Part C, we ran three independent `pi` sessions in `work/part_c/with-memory/` governed by `part_c/AGENTS.md` and `decisions.md`, and compared Session 3 recall against an isolated memoryless baseline in `work/part_c/no-memory/`.

## What degraded in Part A
Single-call accuracy degraded monotonically across the long-context regime, dropping sharply once context length approached the 768K bucket:

| Bucket | Correct / $n$ | Mean Accuracy | 95% Wald CI | Mean Real Input Tokens | Total Cost |
|---|---|---|---|---|---|
| `32k` | 17/20 | **85.0%** | [69.4%, 100.0%] | 30,549 | \$0.1538 |
| `128k` | 15/20 | **75.0%** | [56.0%, 94.0%] | 122,552 | \$0.6143 |
| `256k` | 13/20 | **65.0%** | [44.1%, 85.9%] | 241,380 | \$1.2088 |
| `512k` | 14/20 | **70.0%** | [49.9%, 90.1%] | 483,375 | \$4.8376 |
| `768k` | 9/20 | **45.0%** | [23.2%, 66.8%] | 766,805 | \$7.6718 |

Accuracy first dropped noticeably between `32k` (**85.0%**) and `256k` (**65.0%**), plateaued at `512k` (**70.0%**), and collapsed at `768k` (**45.0%**, 11 of 20 wrong) despite `768k` items being structurally easier (constructed by truncating 1M-token haystacks to 768K tokens so surviving supporting facts never sit in the final 20% tail). Inspection of incorrect Part A responses revealed two primary failure modes caused by attention dilution over massive single-call contexts:
1. **Stale State Retrieval (First/Intermediate Mention Confusion)**: Across the 68 checked items, 37/68 (54.4%) involved the target person moving rooms multiple times (`2` to `6` moves), where only the *last* move matches the ground truth. At 512K and 768K tokens, single-call attention frequently latched onto an earlier movement sentence rather than tracking the chronological last move.
2. **Entity Cross-Talk**: When distractor characters (`Mary`, `Daniel`, `John`) moved rooms hundreds of times near the end of the story, the model occasionally reported a room visited by a distractor entity instead of the queried person.

## What fixed it in Part B
**Isolate** fixed the long-context degradation, whereas **Targeted summary** severely degraded performance across all three buckets:

| Bucket | Condition | Correct / $n$ | Mean Accuracy | 95% Wald CI | Mean Total Tokens | Mean Peak Single-Call Context | Total Cost |
|---|---|---|---|---|---|---|---|
| `256k` | Part A (single call) | 13/20 | 65.0% | [44.1%, 85.9%] | 241,380 | 241,380 | \$1.2088 |
| `256k` | **Isolate** | 12/20 | **60.0%** | [38.5%, 81.5%] | 243,066 | **81,264** | \$1.2179 |
| `256k` | Targeted summary | 6/20 | 30.0% | [9.9%, 50.1%] | 244,370 | 81,278 | \$1.2389 |
| `512k` | Part A (single call) | 14/20 | 70.0% | [49.9%, 90.1%] | 483,375 | 483,375 | \$4.8376 |
| `512k` | **Isolate** | 15/20 | **75.0%** | [56.0%, 94.0%] | 486,719 | **81,964** | \$2.4377 |
| `512k` | Targeted summary | 5/20 | 25.0% | [6.0%, 44.0%] | 489,202 | 81,978 | \$2.4762 |
| `768k` | Part A (single call) | 9/20 | 45.0% | [23.2%, 66.8%] | 766,805 | 766,805 | \$7.6718 |
| `768k` | **Isolate** | 17/20 | **85.0%** | [69.4%, 100.0%] | 771,355 | **98,026** | \$3.8637 |
| `768k` | Targeted summary | 7/20 | 35.0% | [14.1%, 55.9%] | 774,651 | 98,040 | \$3.9142 |

**Why Isolate succeeded (+40.0 percentage points at 768K, from 45.0% $\rightarrow$ 85.0%)**:
- **Peak Single-Call Context Reduction**: While total tokens processed remained nearly identical (`771,355` vs. `766,805`), Isolate capped the **peak single-call context** at **98,026 tokens** (~7.8$\times$ smaller than Part A's 766,805 tokens), returning the model to its high-accuracy `<100K` operating zone.
- **Query-Conditioned Extraction**: Each Isolate chunk sub-agent was asked a narrow binary question about `{person}` only, emitting either the exact quote or `NOT FOUND`. The lead agent then reasoned over an ultra-clean ~300-token list of 8 chunk reports and trivially selected the highest-numbered chunk with a hit.
- **Cost Reduction Above 272K Tier**: Because `gpt-5.6-luna` doubles its input token rate above 272,000 tokens (\$0.50/1M vs. \$0.25/1M), splitting 512K and 768K haystacks into ~82K–98K chunks cut the total run cost in half (\$3.8637 vs. \$7.6718 at 768K).

**Why Targeted Summary failed (dropping to 25.0%–35.0%)**:
- **Lossy Compression Bottleneck**: Each 96K-token BABILong chunk contains hundreds of bAbI character movements alongside massive PG19 book prose. Asking the chunk call to summarize *all* character locations in ~200 tokens without knowing which specific `{person}` would be queried caused the summarizer to drop or generalize away the single sentence where `{person}` moved. Once omitted from the chunk summary, the re-ask call could never recover the answer.

## What memory got right/wrong in Part C
In Part C, the persistent file-based memory architecture (`decisions.md` + `AGENTS.md`) achieved **3/3 (100%)** recall in Session 3, whereas the memoryless baseline achieved **0/3 (0%)** recall (`missing` across all three categories):

| Decision Category | With Memory (Session 3) | No-Memory Baseline (Session 3) | Ground-Truth Decision Recorded in Session 1 |
|---|---|---|---|
| `storage_format` (storage format for notes) | **correct** | **missing** | One plain-text Markdown file per note (human-readable, no parsing library needed) |
| `command_name` (command name for the tool) | **correct** | **missing** | `note` (short, clear, directly describes its purpose) |
| `date_format` (date format for entries) | **correct** | **missing** | ISO 8601 `YYYY-MM-DD` (unambiguous and sorts chronologically) |

- **What Memory Got Right**:
  1. **Cross-Session Persistence & Append-Only Integrity**: Session 1 created `decisions.md` with 3 dated entries. Session 2 completed an unrelated task (`fib.py`, iterative 20-number Fibonacci generator starting at 0) and appended a 4th dated entry without overwriting or corrupting Session 1's entries.
  2. **Selective Retrieval Under Distraction**: In Session 3, despite `decisions.md` containing the unrelated Session 2 Fibonacci entry, the with-memory agent read `decisions.md` and accurately synthesized only the three note-taking CLI decisions and their exact rationales.
  3. **Honest Abstention in Baseline**: With `SCOPE_INSTRUCTION` preventing directory traversal outside `cwd`, the memoryless baseline did not hallucinate or invent fake decisions (`0` invented); it honestly reported that no session-1 records existed in the workspace.
- **What Memory Got Wrong / Fragilities Observed**:
  1. **Clock Hallucination Without System-Prompt Injection**: Left to itself, an LLM has no wall-clock access and invents a training-era date (e.g., `2025-05-14`) for `YYYY-MM-DD` headers; injecting `Today's date is YYYY-MM-DD` into the system prompt was necessary to guarantee accurate timestamps.
  2. **Sandbox Escape via Relative `read` Calls**: Restricting tools to `--tools read` (removing `ls`) was insufficient on its own to isolate the no-memory baseline during initial testing—without `SCOPE_INSTRUCTION`, the agent guessed `../README.md` and `../with-memory/decisions.md` via relative paths.

## What remains unknown
1. **Confidence Interval Width at $n=20$**: With $n=20$ items per bucket, the 95% Wald binomial confidence intervals span roughly $\pm 18\text{--}22$ percentage points. While the 40-percentage-point jump at `768k` between Part A (45.0%) and Isolate (85.0%) is large and decisive, smaller differences (such as 65.0% vs. 60.0% at `256k`) are statistically indistinguishable at $n=20$.
2. **Chunk Boundary Splitting of Multi-Sentence Facts**: In `qa1`, a single movement sentence (`"Sandra travelled back to the kitchen."`) suffices and never splits across line-aligned chunk boundaries. For multi-hop reasoning tasks (`qa2`, `qa3`, where a person picks up an object in Chunk 2 and moves to a room in Chunk 7), query-conditioned per-chunk isolation would need to track both object possession and person movement across chunks.
3. **Unbounded Growth of `decisions.md`**: Part C tested only 4 entries across 3 sessions. Over hundreds of sessions, an append-only `decisions.md` file will eventually suffer from the exact context degradation demonstrated in Part A, requiring hierarchical indexing, superseding/deprecation tags for outdated architectural decisions, or targeted retrieval.

## Who did what
yz0909: Configured the Studio 02 benchmark environment and `subset_qa1_topend.json` dataset, executed Part A (`part_a_stress.py`, `plot_qa1_curve.py`), Part B (`part_b_isolate_compress.py`), and Part C (`part_c_memory.py`), verified all raw JSONL event streams and Wald confidence interval plots (`qa1_curve.png` and `comparison.png`), and authored the six-section quantitative analysis in `EXPLANATION.md`.
