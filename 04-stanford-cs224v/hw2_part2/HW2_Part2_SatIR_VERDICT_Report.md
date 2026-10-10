# CS224V Homework 2 (Part 2): Turning Clinical Text into Constraints with SatIR and VERDICT

**Course:** Stanford CS224V — Agentic AI (Fall 2026)  
**Deliverables:** Completed `hw2_part2_satir_verdict.ipynb`, `todo_answers.json`, and Technical Audit Report  

---

## Executive Summary

Clinical trial matching traditionally pits two prose documents—a patient's longitudinal chart and a trial's eligibility criteria—against a language model in a single free-text call. While convenient, free-text LLM matchers suffer from three structural flaws:
1. **Throughput & Cost Scaling:** Screening $N$ patients against $M$ trials (e.g., 59 patients $\times$ 3,621 trials in the SIGIR benchmark = 213,639 pairs) requires $O(N \times M)$ LLM calls or lossy vector embeddings that collapse negation, numerical thresholds, and logical structure.
2. **Silent Assumptions on Missing Data:** Clinical charts are systematically incomplete. When a trial requires "no history of giant coronary aneurysm" or "negative *Aspergillus* precipitin test" and the chart is silent, a free-text LLM silently guesses without recording what it assumed.
3. **Unfaithful Explanations:** Free-text rationales frequently fail counterfactual self-faithfulness tests—continuing to output the same verdict even when the stated disqualifying reason is removed from the chart.

**SatIR** ([arXiv:2604.08849](https://arxiv.org/abs/2604.08849)) and **VERDICT** ([arXiv:2609.03366](https://arxiv.org/abs/2609.03366)) decompose clinical trial matching into a neuro-symbolic pipeline where **the language model owns translation** (Steps 1–2: declaring canonical typed variables, compiling SMT-LIB assertions, and extracting observed chart facts) and **deterministic code + the Z3 SMT solver own reasoning, retrieval, policy application, and auditing**.

| Pipeline Stage | Language Model | Deterministic Code / Z3 Solver | Key Artifact in Notebook |
| :--- | :---: | :---: | :--- |
| Read trial criteria & declare canonical variables | ✅ | | `(declare-const ...)` in `pair['smt']` |
| Compile criteria into named SMT-LIB assertions | ✅ | | `(assert (! ... :named REQ...))` in `pair['smt']` |
| Extract patient chart facts against declared variables | ✅ | | `pair['observed']` |
| Apply silence policy per predicate category | | ✅ | `POLICY` (29 rules, 5 evidence states) |
| Solve satisfiability (`SAT` / `UNSAT`) | | ✅ | `verdict(pair)` |
| Project SMT via quantifier elimination into SQL gate | | ✅ | `PROJECTION[trial_id][side]['gate']` |
| Screen 3,621 trials in 2.95s/patient | | ✅ | `RETRIEVAL` |
| Audit assumptions, pivotal conditions, & self-faithfulness | | ✅ | `evidence_states(p)`, `pivotal(p)`, `retract_pivotal(p)` |

---

## Notebook Data Summary (SIGIR Benchmark Subset)

Across the 12 pre-solved patient–trial pairs in `PAIRS` (drawn from the 59-patient, 3,621-trial SIGIR benchmark):
- **5 pairs are `ELIGIBLE`** (`inclusion_status = sat`, `exclusion_status = sat`):
  - `sigir-20144__NCT00000520` (*Prevention of Coronary Aneurysms in Kawasaki Syndrome*): 4/4 inclusion variables explicitly observed in the chart (`patient_is_child = True`, `patient_has_diagnosis_of_acute_febrile_mucocutaneous_lymph_node_syndrome_now = True`, CDC criteria refinement `= True`, `presented_on_or_before_tenth_day_of_illness = True`).
  - `sigir-20151__NCT02150941a`: eligible (`SAT`/`SAT`).
  - `sigir-201522__NCT01259336c` (*Efficacy Of Itraconazole In Chronic Cavitary Pulmonary Aspergillosis*): 6 declared inclusion variables; 3 observed `True` in chart (sputum culture positive, sputum hyphae detected), 3 unrecorded and set to `UNKNOWN_DEFER` by policy (precipitin test, BAL hyphae, BAL culture).
  - `sigir-201527__NCT01991314a`: eligible (`SAT`/`SAT`).
  - `sigir-20158__NCT00393913`: eligible (`SAT`/`SAT`).
- **7 pairs are `INELIGIBLE`** (`inclusion_status = unsat`, `exclusion_status = sat`), and **all 7 flip to `ELIGIBLE`** when their pivotal conditions are retracted (`retract_pivotal(p)`):

| Pair ID | Trial Title (Shortened) | Raw `unsat_core` Size | Pivotal (`violated`) Size | Violated Requirements |
| :--- | :--- | :---: | :---: | :--- |
| `sigir-20144__NCT02114099` | Atorvastatin on Persistent Coronary Aneurysm in Children | 3 | 2 | `REQ0_COMPONENT1` (`giant_aneurysm_now`), `REQ1_COMPONENT0` (`age > 10`) |
| `sigir-201524__NCT00929721` | Global Active Surveillance for Community Acquired Pneumonia | 3 | 1 | `REQ0_COMPONENT0` (`age >= 50.0`) |
| `sigir-20156__NCT00796913` | Remission Induction and Sustenance in Graves' Disease 2 | 2 | 2 | `REQ0_COMPONENT0` (Graves' in remission), `REQ0_COMPONENT1` (hyperthyroidism in remission) |
| `sigir-201418__NCT00774137a` | New Diagnostic Tests for AKI After Heart Surgery | 3 | 1 | `REQ_INCLUSION_AT_LEAST_ONE` (CABG/valve surgery + risk factor) |
| `sigir-20158__NCT00233194a` | Trial `NCT00233194a` | 2 | 1 | `REQ2_COMPONENT0_PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE` |
| `sigir-201516__NCT00851487` | Trial `NCT00851487` | 6 | 1 | `REQ1_COMPONENT0_OTHER_REQUIREMENTS` |
| `sigir-20141__NCT01084239` | Rule Out MI by Cardiac Computed Tomography | 8 | 2 | `REQ1_COMPONENT0` (`cardiac_risk_factor_count >= 2`), `REQ5_COMPONENT0` (`sinus_rhythm_now`) |

---

## Detailed Answers to Assignment TODOs

### TODO P2-1(a): Which Step Is the Fragile One? (6 marks)

The most fragile step is **Step 1 (the language model naming/canonicalizing variables and extracting chart values against them)**, followed closely by **Step 2 (the language model writing SMT assertions)**, whereas **Step 3 (the Z3 SMT solver)** is deterministic, sound, and mathematically exact given its inputs.

#### Why Step 1 (and Step 2) Are Fragile While the Solver Is Not
1. **Vocabulary & Canonicalization Drift (Step 1):** Both the trial compiler (`trial_compiler/`) and the patient compiler (`patient_compiler/`) must map free-text clinical prose onto identical canonical predicate names (e.g., mapping "Kawasaki disease" and "acute febrile mucocutaneous lymph node syndrome" both to `patient_has_diagnosis_of_acute_febrile_mucocutaneous_lymph_node_syndrome_now`, or preserving exact units and temporal scopes like `patient_age_value_recorded_now_in_years`). If the model fails to canonicalize two synonymous phrasings to the exact same identifier—or mis-extracts a negated/hedged finding from the patient note—the solver treats them as independent variables.
2. **Logical Connective & Polarity Miscompilation (Step 2):** When translating complex prose criteria into SMT-LIB (Cell 13), the model must decide whether commas and clinical lists represent conjunctions (`and`) or disjunctions (`or`), and whether a qualifier (`@@...`) is mandatory or an alternative.

#### What a Failure Looks Like in the Notebook Output
- **Step 1 (Variable Naming / Chart Reading) Failure:** In `evidence_states(pair)` (Cell 48) or `basis(p)` (Cell 22), a finding that is clearly documented in the prose patient note would fail to match the trial's declared variable name. Consequently, `name in pair['observed']` would evaluate to `False` (`chart is silent` in Cell 15), and the variable would silently fall back to `policy_value(name)`. For a `1b` ("named specific disease") or `Symptom/Sign / Exam Finding (Current Visit)` predicate whose policy rule is `assumed_false`—such as `patient_has_finding_of_sinus_rhythm_now` in `sigir-20141__NCT01084239` (where the patient note does not explicitly write "sinus rhythm" even if an ECG is normal) or `patient_has_diagnosis_of_giant_aneurysm_now` in `sigir-20144__NCT02114099`—a missed chart extraction turns a chart fact into an `[ASSUMED]` `assumed_false`, causing `verdict(p)` to output a false `UNSAT -> INELIGIBLE` and listing that requirement in `pivotal(p)`.
- **Step 2 (Assertion Writing) Failure:** If the model translates an "at least one of" clinical criterion as `(and ...)` instead of `(or ...)`—for instance, in `sigir-20156__NCT00796913` (*Remission Induction and Sustenance in Graves' Disease 2*), where `REQ0_COMPONENT0` (`patient_has_finding_of_graves_disease_in_remission_after_antithyroid_drug_therapy_now`) and `REQ0_COMPONENT1` (`patient_has_finding_of_hyperthyroidism_in_remission_after_antithyroid_drug_therapy_now`) are asserted as separate conjuncts rather than alternatives—the solver faithfully proves `UNSAT` whenever either phrasing fails, inflating `violated` in `pivotal(p)`.

---

### TODO P2-1(b): Checking What the Policy Filled In (8 marks)

In Cell 22 (`basis(BY_ID['sigir-201522__NCT01259336c'])`, *Efficacy Of Itraconazole In Chronic Cavitary Pulmonary Aspergillosis*), the trial asks about 6 inclusion variables: 3 are answered `True` by the chart (`patient_has_finding_of_sputum_culture_positive_for_aspergillus_inthehistory`, `patient_has_finding_of_hyphae_of_aspergillus_detected_inthehistory@@specimen_is_sputum`, and the stem `patient_has_finding_of_hyphae_of_aspergillus_detected_inthehistory`), while 3 are silent in the chart and assigned `UNKNOWN_DEFER` by the policy:
1. `patient_has_undergone_aspergillus_fumigatus_1_precipitin_level_inthehistory_outcome_is_positive` (positive *Aspergillus fumigatus* precipitin blood/serology test; `Procedure / Past or Ongoing Diagnostic Procedure`)
2. `patient_has_finding_of_hyphae_of_aspergillus_detected_inthehistory@@specimen_is_bronchoalveolar_lavage_fluid` (*Aspergillus* hyphae detected in bronchoalveolar lavage [BAL] fluid; `Symptom/Sign / Other Symptom/Sign`)
3. `patient_has_undergone_aspergillus_species_culture_inthehistory_outcome_is_positive` (positive BAL fluid culture for *Aspergillus* species; `Procedure / Past or Ongoing Diagnostic Procedure`)

#### Which One to Check First and Why
If a coordinator had to verify one of the three deferred items first, they should check **`patient_has_undergone_aspergillus_fumigatus_1_precipitin_level_inthehistory_outcome_is_positive`** (the serum *Aspergillus* IgG/precipitin test) first. Clinically and operationally, checking or ordering a peripheral blood serology test (precipitins) is fast, inexpensive, and non-invasive, whereas the other two deferred items (`...@@specimen_is_bronchoalveolar_lavage_fluid` and BAL culture) both require an invasive bronchoscopy with bronchoalveolar lavage (BAL)—which is rarely performed if sputum and serology already establish the diagnosis, and carries procedural risk in a 65-year-old patient with hemoptysis (*"productive cough with tinges of blood"*) and a post-tuberculosis cavitary mass.

#### Pushing on the Premise — What Has to Be True to Skip the Check Entirely and Trust the Verdict
Inspecting the compiled SMT assertion for `sigir-201522__NCT01259336c` (`REQ0_COMPONENT0_OTHER_REQUIREMENTS`), the trial's microbiological criterion (*"Positive results in the aspergillus precipitin test, demonstration of aspergillus hyphae in sputum or BAL fluid or cultures of BAL/sputum growing aspergillus species"*) is compiled as a single 5-way **disjunction (`or`)**:

```smt
(assert (! (or
  patient_has_undergone_aspergillus_fumigatus_1_precipitin_level_inthehistory_outcome_is_positive
  patient_has_finding_of_hyphae_of_aspergillus_detected_inthehistory@@specimen_is_sputum
  patient_has_finding_of_hyphae_of_aspergillus_detected_inthehistory@@specimen_is_bronchoalveolar_lavage_fluid
  patient_has_undergone_aspergillus_species_culture_inthehistory_outcome_is_positive
  patient_has_finding_of_sputum_culture_positive_for_aspergillus_inthehistory)
  :named REQ0_COMPONENT0_OTHER_REQUIREMENTS))
```

Because two disjuncts in that `or` clause are **already `explicit_true` from the chart** (`sputum_culture_positive = True` and `hyphae_of_aspergillus_detected...@@specimen_is_sputum = True`), the disjunction `REQ0_COMPONENT0_OTHER_REQUIREMENTS` evaluates to `True` **even if all three `UNKNOWN_DEFER` variables are `False`!**
Therefore, to skip checking the three deferred items entirely, two things must be true:
1. **Logical Sufficiency (Disjunctive Structure):** The trial criterion genuinely requires only *one* of the five microbiological/immunological tests (`or`), as compiled in `REQ0_COMPONENT0_OTHER_REQUIREMENTS`, rather than requiring BAL or precipitins in addition to sputum.
2. **Chart Extraction Fidelity:** We trust the patient compiler's inference that the patient note's morphological description (*"Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls"*) is a confirmed *Aspergillus* sputum culture/hyphae finding (rather than an unspeciated fungus that still requires *A. fumigatus* precipitin confirmation).

---

### TODO P2-2(a): Limits of the Gate (8 marks)

Because the SatIR retrieval gate (Part 2, Cells 26–33) projects a trial's SMT program into a set of **independent clause rows** and checks each clause in isolation (marking a clause supported as soon as at least one of its atoms is not contradicted), it cannot enforce cross-clause mutual exclusion or functional dependencies across variables.

#### Concrete Pairs of Requirements the Gate Lets Through Together That No Real Patient Could Satisfy
1. **Cross-Clause Numeric / Temporal Contradiction (e.g., Disjoint Age Bounds Across Clauses):**
   Suppose Clause 1 in a trial gate requires `(or (< patient_age_value_recorded_now_in_years 5) patient_has_diagnosis_of_acute_febrile_mucocutaneous_lymph_node_syndrome_now)` and Clause 2 requires `(> patient_age_value_recorded_now_in_years 10)` (as in `NCT02114099`'s `REQ1_COMPONENT0_OTHER_REQUIREMENTS`), or Clause 1 requires `(< patient_age_value_recorded_now_in_years 18)` while Clause 2 requires `(>= patient_age_value_recorded_now_in_years 50.0)` (as in `NCT00929721`). If a patient's chart is silent on age—or if Clause 1 is supported via another atom—the gate checks Clause 1 in isolation and Clause 2 in isolation. Because the SQL row join does not enforce a single simultaneous valuation of `patient_age_value_recorded_now_in_years` across rows, the gate lets the trial through even though $\text{age} < 18 \wedge \text{age} \ge 50$ is impossible for any real human.
2. **Simultaneous Presence and Absence of the Same Clinical State (or Mutually Exclusive Biological States):**
   Consider a trial where one clause requires `patient_has_finding_of_hyperthyroidism_now = True` (or `patient_is_pregnant_now = True`) while another clause (or the exclusion gate, as in `NCT02114099` Cell 29: `(assert (or (not patient_sex_is_female_now) (not patient_is_pregnant_now)))`) requires `(not patient_is_pregnant_now)` or `(not patient_sex_is_female_now)`. If a chart is silent on pregnancy and sex, each clause row is independently satisfiable (`pregnant = True` in row 1; `female = False` or `pregnant = False` in row 2), even though a biological patient cannot simultaneously be pregnant and not pregnant (or pregnant and male). Similarly, in `NCT02114099` (Cell 32), the inclusion program (5 variables, 3 assertions) projects to **0 gate clauses** (`NCT02114099 inclusion -> 0`), so `sigir-20144` (age 2) surfaces on the Kawasaki disease atom (Cell 42) and passes the gate completely, leaving Z3 in Step 3 to catch the age $>10$ and giant aneurysm violations.

---

### TODO P2-3(a): Coordinator Versus Regulator (8 marks)

The **clinical trial coordinator** phoning the patient and the **regulator** auditing the service a year later have fundamentally different operational goals—one is making a *case-level, actionable screening decision right now*, while the other is evaluating *system-level governance, calibration, and explanation integrity across a population*.

1. **What the Coordinator Most Wants: Assumptions + Pivotal Conditions**
   - **When the verdict is `ELIGIBLE`:** The coordinator most wants **Assumptions (and Deferred items)** (`evidence_states(p)` / `basis(p)`, Cells 22 & 48). A bare `ELIGIBLE` label hides whether the patient matched on documented chart facts (like `sigir-20144__NCT00000520`, where 4/4 variables come from `CHART`) or on unrecorded `UNKNOWN_DEFER` / `ASSUMED_FALSE` defaults (like `sigir-201522__NCT01259336c`, where 3/6 variables are `POLICY: UNKNOWN_DEFER`, or `sigir-20141__NCT01084239`, where 31/48 variables rest on policy rules). Before phoning the patient, the coordinator needs the exact checklist of unrecorded items to verify on the call.
   - **When the verdict is `INELIGIBLE`:** The coordinator most wants **Pivotal Conditions** (`pivotal(p)`, Cell 51)—not the raw Z3 `unsat_core`. As shown in `sigir-20144__NCT02114099` (Cells 19 vs. 51) and `sigir-201524__NCT00929721` (Cell 57), the raw `unsat_core` includes requirements the patient actually *meets* (e.g., `REQ0_COMPONENT0` for Kawasaki disease in `NCT02114099`, or 3 requirements in `NCT00929721` when only `REQ0_COMPONENT0` `(>= age 50.0)` is violated). Pivotal conditions tell the coordinator the exact minimal set of violated requirements—and whether a violation rests on a hard chart fact (`age = 2.0` vs. `> 10`, unfixable) or a policy assumption (`giant_aneurysm = assumed_false`, potentially overturned by an outside echocardiogram).
2. **What the Regulator Most Wants: Policy + Self-Faithfulness**
   - **The Written Policy (`POLICY`, Cell 46):** A regulator auditing the service a year later needs proof that chart silence was governed by an explicit, versioned, clinically justified rule table (29 rules across categories like `1a common chronic disease -> unknown_defer` vs. `1b named specific disease -> assumed_false`) rather than arbitrary, non-reproducible LLM guesses that drift across patients or demographics.
   - **Self-Faithfulness (`retract_pivotal(p)`, Cells 52–55):** Regulators must verify that the explanations recorded in audit logs are **causally faithful** to the decision mechanism, not post-hoc rationalizations. While deterministic constraint solvers flip 7/7 ineligible demo pairs upon retracting pivotal conditions (Cell 53), free-text LLM matchers frequently fail counterfactual chart-editing tests.

---

### TODO P2-3(b): Fixed Policy Versus Model Judgment (8 marks)

#### 1. Fixed Written Policy (VERDICT's Approach in Cells 45–48)
- **Strengths (What is good):**
  - *Determinism, Reproducibility, and Auditability:* Every silent chart predicate in a given clinical category (e.g., `Uncommon Chronic Disease` $\rightarrow$ `assumed_false`; `Past or Ongoing Diagnostic Procedure` $\rightarrow$ `unknown_defer`) is resolved identically across all 3,621 trials and 59 patients. Every assumption is explicitly logged in `evidence_states(p)` with its category and rationale.
  - *Zero Marginal Inference Cost & Calibration Control:* Applying the 29-rule lookup table in `policy_value(var)` requires zero LLM calls and allows clinical governance boards to tune sensitivity vs. specificity globally.
- **Weaknesses (What is bad):**
  - *Context-Blindness to Note Type and Clinical Implication:* A fixed rule table only sees the predicate's category (`CATEGORY[var]['level2']`), not the clinical context of the note. For example, in `sigir-20141__NCT01084239` (*Multicenter Study to Rule Out Myocardial Infarction by Cardiac CT*), `patient_has_finding_of_sinus_rhythm_now` is classified as `Symptom/Sign / Exam Finding (Current Visit)` where silence maps to `ASSUMED_FALSE`. Because the brief ED vignette did not explicitly write "sinus rhythm", the fixed policy assumes the patient is *not* in sinus rhythm (`assumed_false`), making `REQ5_COMPONENT0_OTHER_REQUIREMENTS` one of the two pivotal violations (Cell 51)! A clinician reading an uncomplicated vitals/ECG note without arrhythmia would infer sinus rhythm is likely present or defer (`unknown_defer`), whereas the rigid policy rule falsely disqualifies the patient.

#### 2. Model Judgment on Each Chart's Merits (LLM-as-Judge / Free-Text Matcher)
- **Strengths (What is good):**
  - *Contextual Nuance and Pragmatic Clinical Inference:* A language model reading the full note can condition on note length, care setting (brief ED triage vignette vs. comprehensive longitudinal EHR), and correlated findings (e.g., recognizing that an uncomplicated vitals line shouldn't be treated as proof of arrhythmia/non-sinus rhythm).
- **Weaknesses (What is bad):**
  - *Silent, Unaudited Assumptions & Run-to-Run Variance:* When an LLM judges silence on the fly, it embeds unstated assumptions inside prose rationales, may treat the exact same silent predicate as `False` for Patient A and `True` for Patient B, and often fails self-faithfulness (Cells 54–55).

#### Recommendation: Which to Use and When
Use a **Fixed Written Policy as the default backbone** for high-throughput clinical trial screening, regulatory auditing, and formal SMT solving (`SatIR`/`VERDICT`), because silent assumptions must be visible and reproducible. Pair it with **targeted Model Judgment as an exception/escalation layer** strictly for predicates marked `unknown_defer` or `assumed_false` that end up in `pivotal(p)` (like `sinus_rhythm_now` in `sigir-20141__NCT01084239`): when a policy assumption alone causes an `INELIGIBLE` verdict, a contextual model check (or coordinator flag) can estimate the posterior probability from the note context without sacrificing the solver's formal audit trail.

---

### TODO P2-4: Combining GRILL with SatIR/VERDICT for Rare-Disease Diagnosis (15 marks)

#### 1. Architecture of the Combined System (`GRILL-VERDICT`)
In Homework 1, **GRILL** discovers candidate disease criteria from literature via an append-only belief ledger (`directions`, `hypotheses`, `evidence`), and **SLIDERS** (HW2 Part 1) distills the resulting report into structured phenotype tables—yet the final cohort ranking still relies on an uncalibrated LLM prompt (`report_match.json`) that weighs `for` and `against` arguments in free text. We combine **GRILL** with **SatIR/VERDICT** in a four-stage closed-loop neuro-symbolic architecture:

1. **Literature-to-SMT Disease Compiler (GRILL $\rightarrow$ Step 1/2 Compiler):**
   Each `pooled` (supported) hypothesis in GRILL's `ledger.json` is compiled into formal SMT declarations and named assertions over canonical **HPO terms**, inheritance modes, age-of-onset intervals, and genetic loci—annotated with literature penetrance weights (obligate $100\%$, very frequent $80\text{--}99\%$, frequent $30\text{--}79\%$, occasional $5\text{--}29\%$).
2. **Five-State Phenopacket Evidence Policy (VERDICT Policy + Assumptions):**
   Each patient in `cohort.json` has `terms` (observed HPO codes) and `excluded` (explicitly absent HPO codes). We map these directly to VERDICT's five evidence states (Cell 46):
   - $t \in \text{patient.terms}$ (or HPO ancestor via ontology subsumption) $\rightarrow$ **`explicit_true`**
   - $t \in \text{patient.excluded}$ $\rightarrow$ **`explicit_false`**
   - $t \notin \text{terms} \cup \text{excluded}$ (chart silence) $\rightarrow$ governed by an **HPO-aware silence policy**: highly salient/obligate structural malformations default to `assumed_false`, whereas subtle, age-dependent, or lab-specific phenotypes default to **`unknown_defer`**.
3. **MaxSMT / Weighted Solver & VERDICT Audit (`pivotal` + `retract_pivotal`):**
   A Z3 MaxSMT solver checks each of the 10 cohort patients against the compiled disease constraints, producing formal **pivotal conditions** (which obligate/high-frequency disease features a patient explicitly violates) and **self-faithfulness checks** (whether retracting a single contested literature hypothesis flips the top-ranked patient).
4. **Closed-Loop Corrective Directions (`UNSAT` $\rightarrow$ GRILL Frontier):**
   Because the disease constraints are themselves *hypotheses*, if all 10 cohort patients return hard `UNSAT` (or if the top two candidates tie on soft weights), the solver's pivotal conditions automatically spawn a **corrective direction** in GRILL's ledger to query the literature specifically on the discriminating or violated phenotype.

#### 2. Two Concrete Examples Where `GRILL-VERDICT` Outperforms `GRILL` Alone
1. **Disentangling `explicit_false` (Excluded HPO Terms) from Chart Silence (`unknown_defer`) — The Hypotonia Trap:**
   In Homework 1, a GRILL report may state that hypotonia (`HP:0001252`) or microcephaly (`HP:0000252`) is a hallmark feature present in $>85\%$ of patients. Free-text LLM matchers frequently rank a distractor patient #1 based on overlapping generic features (e.g., global developmental delay) even when that distractor's phenopacket explicitly records `HP:0001252` in `excluded` (`explicit_false`), or conversely penalize the true patient when an occasional feature is simply unmentioned (`unknown_defer`). In `GRILL-VERDICT`, `explicit_false` on a hallmark requirement triggers a named violation in `pivotal(p)`, while `unknown_defer` is surfaced in `evidence_states(p)` as an unrecorded assumption rather than a false negative.
2. **Eliminating Run-to-Run Rank Instability Between Rank 1 and Rank 2 via Pivotal Conditions & Self-Faithfulness:**
   In Homework 1 (TODO 3.1 & 5.1), running the LLM matcher 3 times on the exact same GRILL report yields shifting probabilities and rank inversions among the top candidates because the LLM inconsistently trades off feature counts. In `GRILL-VERDICT`, `pivotal(p)` deterministically outputs the exact set of violated requirements separating Rank 1 from Rank 2, and `retract_pivotal(p)` measures whether the top rank depends on a single low-confidence (`binned`/`screened`) hypothesis.

#### 3. Two Specific Technical Challenges and Concrete Workarounds
1. **Challenge 1 — Incomplete Penetrance and Variable Expressivity (`UNSAT` Under Strict Boolean Logic):**
   - *What goes wrong:* Unlike clinical trial inclusion criteria (which are hard legal gates like `age > 10`), Mendelian rare diseases exhibit incomplete penetrance: a true patient with ReNU syndrome (`OMIM:620851`) or CHDFIDD (`OMIM:620450`) may legitimately lack—and have `explicit_false` recorded for—a symptom present in $70\%$ of published cases (e.g., seizures `HP:0001250`). A pure hard-constraint SMT program would declare the true patient `UNSAT` (`INELIGIBLE`).
   - *Workaround:* Compile only pathognomonic/obligate ($100\%$) criteria as hard `(assert ...)` statements, and compile variable-penetrance HPO phenotypes as **weighted soft assertions (`assert-soft` in Z3 MaxSMT)** with log-odds weights derived from GRILL's extracted cohort frequencies ($w_i = \log \frac{p_i}{1 - p_i}$). When a soft constraint is violated by `explicit_false`, it incurs a verifiable penalty in the audit table rather than a hard `UNSAT`.
2. **Challenge 2 — HPO Ontology Granularity Mismatch Between Literature and Phenopackets:**
   - *What goes wrong:* Just as SatIR drops non-canonical `@@` refinements during quantifier elimination (Cell 30), GRILL's literature compiler might declare a specific term like `Generalized tonic-clonic seizure` (`HP:0002069`) while the patient's phenopacket records the parent term `Seizure` (`HP:0001250`) or a sibling term, causing an exact variable-name join to treat the chart as silent.
   - *Workaround:* Automatically inject **HPO DAG subsumption axioms** as `AUTO_AUXILIARY` implications into the SMT program—analogous to `AUTO_AUXILIARY_QUAL_IMPLIES_STEM_0` in `sigir-201522__NCT01259336c` (Cell 22)—so `(assert (=> child_HPO parent_HPO))` propagates `explicit_true` upward to ancestor terms and `explicit_false` downward to descendant terms.
