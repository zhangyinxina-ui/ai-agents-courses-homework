# Stanford CS224V (Fall 2026) — Homework 2 Part 1 Report
## Building Simple ReAct Agents, Schema-Constrained Claim Extraction, and Agent Operating System (AOS) + SLIDERS Provenance Workflows (`OMIM:620450`)

- **Assigned Disease (`TODO 0`):** `OMIM_620450` — *Intellectual developmental disorder, autosomal dominant 73* (`TAF4`)
- **Ground-Truth Patient:** **`P10`** (`PMID_35904126_Proband_7`, Janssen et al., 2022)
- **Deliverables in this Folder:**
  - [`hw2_part1_building_simple_agents.ipynb`](./hw2_part1_building_simple_agents.ipynb) — Executed notebook with all 16 TODOs completed (`TODO 0`–`TODO 10`)
  - [`answers.json`](./answers.json) — Complete structured answers for all notebook TODOs
  - [`results/claims.json`](./results/claims.json) — Schema-validated `ClaimBatch` output from `TODO 6e`
  - [`simple_agents_and_aos_solution.py`](./simple_agents_and_aos_solution.py) — Standalone runnable ReAct + Pydantic + AOS/SLIDERS implementation & `pytest` test suite
  - [`workspace-OMIM_620450/`](./workspace-OMIM_620450/) — Generated AOS workspace with `inputs/`, `generated/findings/tables/`, `generated/claims/claims.csv`, and SQLite `workspace.db` catalog

---

## 1. Part 1: ReAct Tool Loop & Schema-Constrained Claim Extraction (`TODO 0` – `TODO 6e`)

### 1.1 Simple LLM Invocation & Deep Research Tool Design (`TODO 0`, `TODO 1`, `TODO 1.2`)
- **`TODO 0` & `TODO 1`:** Set `OMIM_ID = 620450` (`MY_DISEASE = "OMIM_620450"`) and invoked `response = llm.invoke(prompt)` on the report title `# Clinical Identification of Intellectual Developmental Disorder, Autosomal Dominant 73 (OMIM #620450)`.
- **`TODO 1.2` (Designing Tools for a STORM-Class Deep Researcher):** Beyond local sandbox file navigation (`get_cwd`, `list_files`, `read_file`, `get_hypothesis`), a STORM-style deep researcher (Shao et al., NAACL 2024) requires four specialized tool families:
  1. **Biomedical Literature & Ontology Search Tools:** `search_pubmed_pmc`, `search_semantic_scholar`, `query_omim_api`, and `lookup_hpo_ontology` (preventing the HW1 RAG failure where generic web search returned only database landing pages).
  2. **Full-Text & Structured Table Extraction Tools:** `read_paper_section` and `extract_structured_table` (preventing the HW1 GRILL failure where Janssen et al., 2022 truncated right after the header of Table 1).
  3. **Perspective-Guided Interview Tools:** `generate_expert_personas` and `conduct_perspective_interview` (spawning dysmorphologist, neurogeneticist, speech-language, and statistical-skeptic personas).
  4. **Belief-Ledger & Span-Verification Tools:** `propose_hypothesis`, `link_evidence`, and `verify_claim_grounding`.

### 1.2 Case-Specific Prompt & LangChain ReAct Loop (`TODO 2` – `TODO 5.1`)
- **`TODO 2` (`AGENT_TASK`):** Engineered a 5-sentence prompt instructing the agent to locate `answer.md` for `Intellectual developmental disorder, autosomal dominant 73 (OMIM 620450)` without a pre-supplied path, read its numbered lines via `read_file`, map `F`-labels via `get_hypothesis(0)`, inspect full hypothesis/evidence records via `get_hypothesis(h_id)`, preserve exact cohort denominators (`n/N`) and percentages without transferring verdicts onto negated/conflicted claims, and output three claims with `statement`, `report_line`, `hypothesis_id`, `evidence_ids`, and `caveat`.
- **`TODO 3`–`5` (`run_agent`):**
  - `agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT, middleware=limits)`
  - `result = agent.invoke({"messages": [{"role": "user", "content": task}]})`
  - `messages = result["messages"]`
- **`TODO 5.1` (Tool Call Trace Analysis):** Starting from `cs224vf26-assignment1-public/`, the ReAct agent executed **9 tool calls**:
  1. `get_cwd()` $\rightarrow$ repository root
  2. `list_files(".")` $\rightarrow$ `['assignments/', 'data/', 'vendor/']`
  3. `list_files("assignments")` $\rightarrow$ located `OMIM_620450/`
  4. `list_files("assignments/OMIM_620450/grill_run")` $\rightarrow$ located `answer.md` and `ledger.json`
  5. `read_file("assignments/OMIM_620450/grill_run/answer.md")` $\rightarrow$ read 147 numbered lines and unlocked `REPORT_READ`
  6. `get_hypothesis(0)` $\rightarrow$ retrieved `F1`–`F18` to `h1`–`h18` mapping index
  7. `get_hypothesis(1)` $\rightarrow$ verified `h1` (`F2`, `conf=0.88`, `evidence_ids=['e1', 'e2', 'e8', 'e9']`)
  8. `get_hypothesis(7)` $\rightarrow$ verified `h7` (`F6`, `conf=0.85`, `evidence_ids=['e16', 'e28']`)
  9. `get_hypothesis(13)` $\rightarrow$ verified `h13` (`F3`, `conf=0.88`, `evidence_ids=['e37', 'e50', 'e54']`)

### 1.3 Pydantic Schema-Constrained Output (`TODO 6a` – `TODO 6e`)

| Claim ID | `report_line` | `hypothesis_id` | `evidence_ids` | Statement Summary | Attached Caveat |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **`C001`** | `18` | `h1` (`F2`) | `["e1", "e2", "e8", "e9"]` | In Janssen et al. (2022), global developmental delay / ID (`HP:0001263` / `HP:0001249`) and disproportionate expressive language delay (`HP:0000750`) both occurred in `8/8 (100%)` de novo pLoF probands. | Small discovery cohort ($n=8$ pLoF probands); missense carriers (`Kennis et al., 2026`) can present with isolated DLD and non-verbal $\text{IQ} \ge 70$. |
| **`C002`** | `20` | `h7` (`F6`) | `["e16", "e28"]` | In Janssen et al. (2022) ($n=8$ de novo pLoF probands), behavioral abnormalities (`HP:0000708`) and category-level facial dysmorphisms (`HP:0001999`) each occurred in `7/8 (87.5%)` patients. | Category-level dysmorphism is penetrant (`87.5%`), but individual morphological features are variable and non-pathognomonic. |
| **`C003`** | `22` | `h13` (`F3`) | `["e37", "e50", "e54"]` | In Janssen et al. (2022), discrete facial anomalies—arched eyebrows (`HP:0002553`), epicanthus (`HP:0000286`), and thin upper lip vermilion (`HP:0000219`)—each occurred in `<6/8 (<75%)` patients. | No single discrete facial feature exceeds `75%` penetrance, preventing diagnosis from physical exam alone. |

---

## 2. Part 2: Agent Operating System (AOS), SLIDERS, and Gold-Patient Audit (`TODO 7` – `TODO 10`)

- **`TODO 7` (Workspace Initialization & Cataloging):** Created `workspaces/OMIM_620450` via `AOS.create(...)` and added 3 files (`inputs/answer.md`, `inputs/ledger.json`, and `inputs/results/claims.json`). Each file received a `source='user'` row in `workspace.db` with an accurate one-sentence model summary and git commit provenance.
- **`TODO 8` (SLIDERS Table Extraction & Quote Verification):** Executed `aos.step("sliders", question, inputs=["inputs/answer.md"], output_dir="generated/findings", ...)`. SLIDERS extracted structured phenotype frequency, differential diagnosis, and negative exclusion tables backed by `evidence.csv` line locators (`start_line`–`end_line`), achieving **100% quote verification** against `answer.md`.
- **`TODO 9`, `9.1`, `9.2` (`ClaimExtractor` Worker & Planner Autonomy):**
  - Registered `ClaimExtractor` and invoked `aos.run(GOAL, max_steps=3)` without naming the worker.
  - **`TODO 9.1`:** The AOS planner selected `claim_extractor` on Step 1 (`status='ok'`), matching its description (`inputs = answer.md and ledger.json; writes output_dir/claims.json, claims.csv and tool_trace.json`) to the goal and recording `source='claim_extractor'` with derivation `['inputs/answer.md', 'inputs/ledger.json']` in `workspace.db`.
  - **`TODO 9.2`:** Because `ClaimExtractor.run` re-pointed `SANDBOX` directly to `report.parent` (`inputs/` inside the workspace) rather than the assignment-1 repository root, the ReAct agent needed only **7 tool calls instead of 9** (saving 2 directory-navigation calls under `assignments/OMIM_620450/grill_run/`).
- **`TODO 10` (Gold Patient `P10` vs. SLIDERS Table & Claims Audit):**
  - **Criteria Met by `P10`:** `P10` (`PMID_35904126_Proband_7`) meets all positive core and craniofacial criteria (`HP:0002342` Moderate ID, `HP:0000750` Delayed speech, `HP:0002553` Highly arched eyebrow, `HP:0000219` Thin upper lip vermilion, and `HP:0001250` Seizures excluded).
  - **What Was Missing from SLIDERS / `answer.md`:** Due to snippet truncation of Table 1 in Janssen et al. (2022) (`e39`, `e57` in `ledger.json`), SLIDERS and `answer.md` missed `HP:0100962 (Excessive shyness)`, omitted `HP:0001252 (Hypotonia)` and `HP:0031936 (Delayed ability to walk)` from Table 1 (`h14` binned as conflicted), and over-generalized `0/8` visceral malformations (`h5`/`h9`) into a broad anti-malformation rule that penalized `P10`'s minor oral/dental/digital findings (`HP:0000185` Cleft soft palate, `HP:0000677` Oligodontia, `HP:0100490` Camptodactyly, `HP:0004691` 2-3 toe syndactyly).
