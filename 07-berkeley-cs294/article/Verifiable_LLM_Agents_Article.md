# From Heuristic Generation to Formal Guarantees: Verifiable LLM Agents and Neurosymbolic Proof Search

**Course:** UC Berkeley CS294-196 / CS194-196 — *Advanced LLM Agents* (Instructor: Prof. Dawn Song)  
**Companion Implementation:** [`labs/lean4_formal_verification_lab.py`](../labs/lean4_formal_verification_lab.py)

---

## Abstract

As Large Language Model (LLM) agents transition from conversational assistants to autonomous software engineers, systems optimizers, and scientific reasoners, the fundamental bottleneck shifts from **plausible generation** to **verifiable correctness**. While test-time compute scaling methods such as Self-Consistency, Best-of-$N$ reranking with Process Reward Models (PRMs), and Execution-Feedback Self-Debugging dramatically improve empirical accuracy on finite test suites, they remain fundamentally bounded by the **specification and oracle gap**: neural reward models are susceptible to adversarial reward hacking, and finite unit tests can only falsify—never universally prove—program invariants. This article synthesizes the architectural evolution from heuristic neural generation to **neurosymbolic formal verification**, examining how coupling frontier LLM policies with interactive theorem provers such as **Lean 4** establishes mathematically sound guarantees while unlocking unbounded self-improvement loops.

---

## 1. The Limits of Empirical Verification and Neural Reward Models

Modern agentic workflows rely on three primary verification signals to filter or refine candidate trajectories:

1. **Consensus Heuristics (Self-Consistency / Majority Voting):** Marginalizing over $N$ sampled Chain-of-Thought trajectories reduces stochastic decoding variance but fails systematically whenever a model's prior assigns high probability to a shared misconception ("coherent distractors").
2. **Learned Neural Verifiers (ORMs and PRMs):** Training a step-level Process Reward Model $v_\phi(s_t \mid x, s_{<t})$ provides dense credit assignment and enables tree search (ToT, MCTS, LATS). However, because $v_\phi$ is itself a parameterized neural network, scaling test-time search or reinforcement learning (e.g., PPO/GRPO) against $v_\phi$ inevitably encounters **Goodhart's Law**: the policy discovers superficial syntactic patterns or verbose hedging that maximize $v_\phi$ without preserving logical validity.
3. **Finite Input-Output Unit Testing:** In coding benchmarks such as HumanEval, MBPP, and SWE-bench, execution against unit tests $\mathcal{U} = \{(x_i, y_i)\}_{i=1}^m$ provides a grounded environmental signal. Yet unit tests cover only a measure-zero subset of the input domain. In safety-critical systems—cryptographic protocols, smart contracts, OS kernels, and distributed consensus—an edge case missed by $\mathcal{U}$ can be catastrophic, and an optimization agent may inadvertently exploit under-specified test suites by hardcoding special cases.

---

## 2. The Neurosymbolic Paradigm: Coupling LLMs with the Lean 4 Kernel

Interactive Theorem Provers (ITPs) based on dependent type theory, most notably **Lean 4** (built upon the Calculus of Inductive Constructions), resolve the oracle gap via the **Curry-Howard correspondence**: propositions are types, and proofs are functional programs whose type-checking guarantees universal logical validity.

In a neurosymbolic theorem-proving or verifiable-coding agent (exemplified by AlphaProof, DeepSeek-Prover, LeanDojo, and our [`Lean4ProofRepairAgent`](../labs/lean4_formal_verification_lab.py)), responsibilities are cleanly partitioned across a **System 1 / System 2 boundary**:

- **Neural Policy (Intuition & Heuristic Search Proposal):** The LLM observes the current formal proof state—consisting of local hypotheses $\Gamma = \{h_1 : T_1, \dots, h_k : T_k\}$ and the unsolved goal $\vdash G$—and proposes high-level proof tactics (e.g., `induction n with | zero => ... | succ n ih => ...`, `simp [ih]`, `split`, `omega`) or intermediate `have` lemmas.
- **Lean 4 Kernel (Sound Symbolic Verifier):** A compact, auditable type-checking kernel deterministically validates each tactic transition. If a tactic is ill-typed or leaves residual subgoals, the kernel immediately returns exact line/column diagnostics and updated goal obligations $\vdash G'$. Crucially, **zero false positives are possible** once two structural guardrails are enforced:
  1. **Forbidden Axiom/Escape-Hatch Rejection:** The verifier rejects any proof containing incomplete placeholders (`sorry`, `admit`), unsound axioms (`axiom`), or `unsafe` casts.
  2. **Theorem Statement Integrity:** The verifier enforces syntactic and definitional preservation of the target theorem signature so the agent cannot vacuously prove `theorem target : True := by trivial`.

---

## 3. Closed-Loop Proof Repair and the Autoformalization Frontier

Our empirical evaluation in [`lean4_formal_verification_lab.py`](../labs/lean4_formal_verification_lab.py) demonstrates why **iterative compiler-feedback repair** is transformative for formal reasoning. One-shot (Pass@1) generation of complete Lean 4 proofs frequently fails due to subtle definitional unfolding rules: for example, in `theorem double_eq_add_self (n : Nat) : double n = n + n`, the reflexivity tactic `rfl` fails because `2 * n` and `n + n` are not definitionally equal without unfolding `double` and invoking Presburger linear arithmetic (`omega`). Similarly, piecewise conditional functions (`clampLow`) require explicit case-splitting (`split <;> omega`), and recursive functions (`sumEven`) require structural induction combined with simplifier rewrites (`simp [sumEven, ih]`). By parsing the Lean 4 compiler's `⊢` unsolved goal state at each turn, the repair agent resolves 100% of these failures within 2–3 interactive turns.

### The Final Frontier: Trustworthy Autoformalization
While the Lean 4 kernel guarantees that a proof $p$ validly establishes the formal theorem $T_{\text{formal}}$, it cannot by itself verify that $T_{\text{formal}}$ faithfully captures the human user's informal intent $T_{\text{informal}}$. Bridging this **autoformalization gap** requires defense-in-depth:
1. **Back-Translation & Semantic Equivalence Audits:** Translating $T_{\text{formal}}$ back into natural language via an independent auditor model and checking bi-directional entailment against $T_{\text{informal}}$.
2. **Property-Based Counterexample Fuzzing (`SlimCheck` / `Plausible`):** Testing intermediate lemmas and preconditions against randomized concrete inputs before investing compute in deep MCTS proof search.
3. **Self-Improving Expert Iteration (STaR / GRPO over Lean 4):** Because the Lean 4 kernel provides an unhackable binary reward $R \in \{0, 1\}$, agents can autonomously generate conjectures, search for proofs via MCTS/LATS, filter verified trajectories, and continually post-train via GRPO without human annotation bottlenecks.

By uniting the expressive, commonsense prior of foundation models with the uncompromising soundness of formal verification kernels, verifiable LLM agents chart a rigorous path from probabilistic text generators to trustworthy engineering systems.
