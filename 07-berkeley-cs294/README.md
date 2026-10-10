# UC Berkeley CS294-196 / CS194-196: Advanced LLM Agents — Complete Coursework & Labs

**Course:** UC Berkeley CS294-196 — *Advanced LLM Agents* (Instructor: Prof. Dawn Song, `https://rdi.berkeley.edu/adv-llm-agents/`)  
**Verification Status:** 5/5 comprehensive lab test suites passing (`pytest labs/ -v`)

---

## Repository Structure

| Component | File Path | Description & Topics Covered |
| :--- | :--- | :--- |
| **12 Weekly Lecture Quizzes** | [`quizzes/Quiz_01_to_12_Solutions.md`](./quizzes/Quiz_01_to_12_Solutions.md) | Complete question-and-answer solutions across all 12 lectures: POMDP/CoALA Foundations, ToT/LATS/ADaPT Search, Constrained Tool Calling, MemGPT & Voyager Memory, SWE-bench & ACI, Lean 4 Formal Verification, OSWorld/WebArena SoM GUI Agents, Multi-Agent Debate, DPO/PRM Post-Training, Indirect Prompt Injection & Taint Tracking, $\tau$-bench $\text{pass}^k$, and Compound AI Systems (DSPy/TextGrad). |
| **Lean 4 Formal Verification Lab** | [`labs/lean4_formal_verification_lab.py`](./labs/lean4_formal_verification_lab.py) | Natural-language-to-Lean-4 `Autoformalizer`, anti-reward-hacking guardrails (`sorry`/`admit` & header mutation rejection), structured compiler diagnostic & `⊢` goal-state parser, and closed-loop `Lean4ProofRepairAgent`. |
| **Foundation Agent Labs 1–3** | [`labs/foundation_labs_1_to_3.py`](./labs/foundation_labs_1_to_3.py) | **Lab 1:** `ReActRestaurantAgent` (3-tool quantitative sentiment & Bayesian shrinkage pipeline).<br>**Lab 2:** `TieredMemoryAgent` (MemGPT-style Core Memory, FIFO Queue with eviction interrupts, and Archival RAG).<br>**Lab 3:** `MultiAgentReflexionSystem` (Actor-Critic-Judge collaboration with episodic Reflexion memory). |
| **Automated Pytest Suite** | [`labs/test_berkeley_labs.py`](./labs/test_berkeley_labs.py) | End-to-end offline unit tests verifying all Lean 4 proof repair trajectories and Foundation Labs 1–3. |
| **Written Technical Article** | [`article/Verifiable_LLM_Agents_Article.md`](./article/Verifiable_LLM_Agents_Article.md) | Technical synthesis article (>800 words): *"From Heuristic Generation to Formal Guarantees: Verifiable LLM Agents and Neurosymbolic Proof Search"*. |

---

## Quickstart: Running Labs & Unit Tests

```bash
# Run the complete pytest suite for Berkeley CS294-196
python3 -m pytest labs/ -v

# Execute individual lab benchmarks
python3 labs/lean4_formal_verification_lab.py
python3 labs/foundation_labs_1_to_3.py
```
