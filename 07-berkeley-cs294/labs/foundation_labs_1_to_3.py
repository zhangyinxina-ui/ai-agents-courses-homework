"""
UC Berkeley CS294-196: Advanced LLM Agents (Prof. Dawn Song)
Foundation Agent Labs 1, 2, and 3:
  - Lab 1: ReAct Restaurant Evaluation & Quantitative Data Analysis Agent
  - Lab 2: Conversational Memory & Tiered Agentic RAG Agent (MemGPT Architecture)
  - Lab 3: Reflexion & Multi-Agent Actor-Critic-Judge Collaboration
"""

from __future__ import annotations

import collections
import dataclasses
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# LAB 1: ReAct Restaurant Evaluation & Quantitative Data Analysis Agent
# ============================================================================

RESTAURANT_REVIEWS_DB: Dict[str, List[str]] = {
    "tartine bakery": [
        "The morning buns and sourdough croissants are phenomenal and unforgettable, though the line is quite long.",
        "Incredible pastries and excellent espresso, service is brisk and efficient.",
        "Croissants were delicious and flaky, coffee was good, seating was slightly cramped.",
        "Phenomenal bread quality and great atmosphere in the Mission.",
    ],
    "chez panisse": [
        "Phenomenal seasonal tasting menu with unforgettable organic ingredients and impeccable service.",
        "Delicious farm-to-table dishes and excellent wine pairing, truly phenomenal dining experience.",
        "Great ambiance and delicious food, though reservation availability is tough.",
    ],
    "golden boy pizza": [
        "Good clam and garlic focaccia slice, cheap and tasty late night spot, service is average.",
        "Delicious square slices and good sauce, cramped standing room only.",
        "Good value pizza in North Beach, nothing fancy but solid.",
    ],
    "mediocre diner": [
        "The pancakes were bland and lukewarm, service was slow and disappointing.",
        "Overpriced coffee and mediocre eggs, noisy dining room.",
    ],
}

SENTIMENT_LEXICON: Dict[str, int] = {
    "phenomenal": 5,
    "unforgettable": 5,
    "impeccable": 5,
    "incredible": 5,
    "excellent": 5,
    "delicious": 4,
    "flaky": 4,
    "great": 4,
    "good": 4,
    "tasty": 4,
    "solid": 4,
    "brisk": 4,
    "average": 3,
    "cramped": 2,
    "tough": 2,
    "noisy": 2,
    "mediocre": 2,
    "bland": 1,
    "lukewarm": 1,
    "slow": 1,
    "disappointing": 1,
    "overpriced": 1,
}


@dataclasses.dataclass
class ReActStep:
    step_num: int
    thought: str
    action_name: str
    action_arg: Any
    observation: Any


@dataclasses.dataclass
class RestaurantAnalysisReport:
    restaurant_name: str
    num_reviews: int
    review_scores: List[float]
    raw_mean_score: float
    bayesian_weighted_score: float
    trajectory: List[ReActStep]


class ReActRestaurantAgent:
    """
    Lab 1: Multi-step ReAct Data Analysis Agent that orchestrates 3 sequential tools:
      1. `fetch_reviews(restaurant_name)` -> retrieves unstructured review texts.
      2. `score_review_sentiments(reviews)` -> maps review adjectives to quantitative [1.0, 5.0] scores.
      3. `compute_bayesian_rating(scores, prior_mean=3.5, prior_weight=2.0)` -> computes shrinkage-adjusted
         overall rating and confidence interval.
    """

    def __init__(self, db: Optional[Dict[str, List[str]]] = None):
        self.db = db or RESTAURANT_REVIEWS_DB

    def tool_fetch_reviews(self, restaurant_name: str) -> List[str]:
        key = restaurant_name.strip().lower()
        for k, reviews in self.db.items():
            if k in key or key in k:
                return list(reviews)
        return []

    @staticmethod
    def tool_score_review_sentiments(reviews: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for rev in reviews:
            tokens = re.findall(r"[a-z]+", rev.lower())
            matched = [SENTIMENT_LEXICON[t] for t in tokens if t in SENTIMENT_LEXICON]
            if matched:
                scores.append(round(sum(matched) / len(matched), 3))
            else:
                scores.append(3.0)
        return scores

    @staticmethod
    def tool_compute_bayesian_rating(
        scores: Sequence[float],
        prior_mean: float = 3.5,
        prior_weight: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Computes (raw_mean, bayesian_shrinkage_score):
            S_Bayes = (N * mean(scores) + m * prior_mean) / (N + m)
        """
        if not scores:
            return 0.0, prior_mean
        n = len(scores)
        raw_mean = sum(scores) / n
        bayes = (n * raw_mean + prior_weight * prior_mean) / (n + prior_weight)
        return round(raw_mean, 3), round(bayes, 3)

    def evaluate_restaurant(self, query: str) -> RestaurantAnalysisReport:
        # Extract target restaurant name
        clean_name = query.strip()
        for k in self.db:
            if k in query.lower():
                clean_name = k.title()
                break

        trajectory: List[ReActStep] = []

        # Step 1: Fetch reviews
        t1 = f"I need to retrieve customer reviews for '{clean_name}'."
        reviews = self.tool_fetch_reviews(clean_name)
        trajectory.append(ReActStep(1, t1, "fetch_reviews", clean_name, reviews))

        # Step 2: Quantify sentiment scores per review
        t2 = f"Retrieved {len(reviews)} reviews. Now I will extract quantitative [1, 5] sentiment scores."
        scores = self.tool_score_review_sentiments(reviews)
        trajectory.append(ReActStep(2, t2, "score_review_sentiments", reviews, scores))

        # Step 3: Aggregate via Bayesian rating formula
        t3 = "Now I will compute the raw mean and Bayesian shrinkage-adjusted rating."
        raw_mean, bayes_score = self.tool_compute_bayesian_rating(scores)
        trajectory.append(
            ReActStep(
                3,
                t3,
                "compute_bayesian_rating",
                scores,
                {"raw_mean": raw_mean, "bayesian_score": bayes_score},
            )
        )

        return RestaurantAnalysisReport(
            restaurant_name=clean_name,
            num_reviews=len(reviews),
            review_scores=scores,
            raw_mean_score=raw_mean,
            bayesian_weighted_score=bayes_score,
            trajectory=trajectory,
        )


# ============================================================================
# LAB 2: Conversational Memory & Tiered Agentic RAG Agent (MemGPT Style)
# ============================================================================

@dataclasses.dataclass
class ArchivalPassage:
    passage_id: str
    text: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class TieredMemoryAgent:
    """
    Lab 2: OS-Inspired Tiered Memory & Agentic RAG Agent (MemGPT, Packer et al., 2023).
    Maintains three distinct memory tiers:
      1. `core_memory`: High-priority working memory dictionary (user persona, constraints, preferences)
         explicitly editable via `core_memory_upsert(key, value)`.
      2. `fifo_queue`: Sliding window of recent conversation messages (`max_fifo_turns`).
         When `len(fifo_queue) > max_fifo_turns`, triggers a **Memory Pressure Interrupt**
         that summarizes and flushes evicted turns into `recall_archive` and extracts
         persistent entities into `core_memory`.
      3. `archival_storage`: Long-term document & episodic store searchable via lexical/BM25 similarity.
    """

    def __init__(self, max_fifo_turns: int = 4):
        self.max_fifo_turns = max_fifo_turns
        self.core_memory: Dict[str, str] = {}
        self.fifo_queue: collections.deque = collections.deque()
        self.recall_summaries: List[str] = []
        self.archival_storage: List[ArchivalPassage] = []
        self.page_fault_count: int = 0

    def core_memory_upsert(self, key: str, value: str) -> None:
        self.core_memory[key.strip().lower()] = value.strip()

    def archival_insert(self, passage_id: str, text: str, **metadata: Any) -> None:
        self.archival_storage.append(
            ArchivalPassage(passage_id=passage_id, text=text, metadata=metadata)
        )

    def archival_search(self, query: str, top_k: int = 2) -> List[ArchivalPassage]:
        q_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        scored: List[Tuple[float, int]] = []
        for idx, p in enumerate(self.archival_storage):
            p_terms = set(re.findall(r"[a-z0-9_]+", p.text.lower()))
            overlap = len(q_terms & p_terms) / max(1.0, math.sqrt(len(p_terms)))
            if overlap > 0:
                scored.append((overlap, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.archival_storage[i] for _, i in scored[:top_k]]

    def _extract_facts_to_core_memory(self, text: str) -> None:
        # Automatically extract structured user preferences/constraints
        m_diet = re.search(r"allergic to ([a-z]+)|vegetarian|vegan", text, flags=re.I)
        if m_diet:
            self.core_memory_upsert("dietary_restriction", m_diet.group(0).lower())
        m_lang = re.search(r"prefer ([a-z0-9\+\#]+) for (coding|verification|backend)", text, flags=re.I)
        if m_lang:
            self.core_memory_upsert("preferred_stack", m_lang.group(1))
        m_budget = re.search(r"budget is \$?(\d+)", text, flags=re.I)
        if m_budget:
            self.core_memory_upsert("max_budget_usd", m_budget.group(1))

    def _handle_memory_pressure_interrupt(self) -> None:
        """Evicts oldest turns when FIFO queue exceeds capacity, flushing summary to archival/recall."""
        while len(self.fifo_queue) > self.max_fifo_turns:
            evicted_role, evicted_msg = self.fifo_queue.popleft()
            self._extract_facts_to_core_memory(evicted_msg)
            summary = f"[Evicted {evicted_role}]: {evicted_msg}"
            self.recall_summaries.append(summary)
            self.archival_insert(
                passage_id=f"evicted_{len(self.recall_summaries)}",
                text=summary,
                source="fifo_eviction",
            )
            self.page_fault_count += 1

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Processes a user turn, triggers memory interrupts if needed, and synthesizes a grounded reply."""
        self._extract_facts_to_core_memory(user_message)
        self.fifo_queue.append(("user", user_message))
        self._handle_memory_pressure_interrupt()

        retrieved_docs = self.archival_search(user_message, top_k=2)
        context_notes = [d.text for d in retrieved_docs]

        response_parts = [
            f"CoreMemory={dict(self.core_memory)}",
            f"ArchivalHits={len(retrieved_docs)}",
        ]
        if context_notes:
            response_parts.append(f"Retrieved: {' | '.join(context_notes)}")

        reply = " ; ".join(response_parts)
        self.fifo_queue.append(("assistant", reply))
        self._handle_memory_pressure_interrupt()

        return {
            "reply": reply,
            "core_memory": dict(self.core_memory),
            "fifo_size": len(self.fifo_queue),
            "retrieved_passages": retrieved_docs,
            "page_fault_count": self.page_fault_count,
        }


# ============================================================================
# LAB 3: Reflexion & Multi-Agent Actor-Critic Collaboration
# ============================================================================

@dataclasses.dataclass
class CriticReview:
    approved: bool
    defects: List[str]
    score: float


@dataclasses.dataclass
class MultiAgentRound:
    round_num: int
    actor_draft: Dict[str, Any]
    critic_review: CriticReview
    episodic_reflection: str


class MultiAgentReflexionSystem:
    """
    Lab 3: Multi-Agent Actor-Critic-Judge Collaboration with Episodic Reflexion (Shinn et al., 2023).
    Task: Synthesize a verifiable distributed systems deployment plan & SQL migration script
    that satisfies 4 strict safety & consistency invariants:
      1. No unguarded `DROP TABLE` or `TRUNCATE` statements.
      2. Must wrap schema changes in an explicit `BEGIN ... COMMIT` transaction.
      3. Must include a rollback/down-migration block (`-- ROLLBACK:`).
      4. Must enforce a rate-limit / canary rollout percentage `canary_percent <= 10`.
    """

    def __init__(self, max_rounds: int = 4):
        self.max_rounds = max_rounds

    @staticmethod
    def critic_evaluate(draft: Dict[str, Any]) -> CriticReview:
        defects: List[str] = []
        sql = draft.get("sql_migration", "")
        canary = draft.get("canary_percent", 100)

        if re.search(r"\b(DROP\s+TABLE|TRUNCATE)\b", sql, flags=re.I):
            defects.append("Safety violation: destructive DROP TABLE / TRUNCATE detected.")
        if "BEGIN;" not in sql or "COMMIT;" not in sql:
            defects.append("Atomicity violation: SQL migration must be wrapped in `BEGIN; ... COMMIT;`.")
        if "-- ROLLBACK:" not in sql:
            defects.append("Reversibility violation: missing `-- ROLLBACK:` down-migration specification.")
        if canary > 10:
            defects.append(f"Rollout violation: canary_percent={canary}% exceeds 10% safety threshold.")

        score = max(0.0, 1.0 - 0.25 * len(defects))
        return CriticReview(approved=(len(defects) == 0), defects=defects, score=score)

    @staticmethod
    def actor_propose(
        task_spec: str,
        previous_draft: Optional[Dict[str, Any]],
        reflections: Sequence[str],
    ) -> Dict[str, Any]:
        if previous_draft is None:
            # Round 1 initial draft has typical zero-shot flaws: destructive SQL, no transaction, 50% canary
            return {
                "sql_migration": (
                    "DROP TABLE IF EXISTS legacy_sessions;\n"
                    "ALTER TABLE users ADD COLUMN last_login_ts TIMESTAMP;"
                ),
                "canary_percent": 50,
            }

        sql = previous_draft["sql_migration"]
        canary = previous_draft["canary_percent"]
        combined_refl = " ".join(reflections)

        if "destructive DROP TABLE" in combined_refl:
            sql = sql.replace("DROP TABLE IF EXISTS legacy_sessions;\n", "")
        if "BEGIN; ... COMMIT;" in combined_refl and "BEGIN;" not in sql:
            sql = "BEGIN;\n" + sql.strip() + "\nCOMMIT;"
        if "missing `-- ROLLBACK:`" in combined_refl and "-- ROLLBACK:" not in sql:
            sql = sql + "\n-- ROLLBACK: ALTER TABLE users DROP COLUMN last_login_ts;"
        if "exceeds 10% safety threshold" in combined_refl:
            canary = 5

        return {
            "sql_migration": sql,
            "canary_percent": canary,
        }

    def run_collaboration(self, task_spec: str) -> List[MultiAgentRound]:
        history: List[MultiAgentRound] = []
        reflections: List[str] = []
        draft: Optional[Dict[str, Any]] = None

        for r in range(1, self.max_rounds + 1):
            draft = self.actor_propose(task_spec, draft, reflections)
            review = self.critic_evaluate(draft)
            if review.approved:
                refl = "All 4 safety, transaction, rollback, and canary invariants verified."
            else:
                refl = "Reflexion Memory Update: Fix -> " + " | ".join(review.defects)
                reflections.append(refl)

            history.append(
                MultiAgentRound(
                    round_num=r,
                    actor_draft=dict(draft),
                    critic_review=review,
                    episodic_reflection=refl,
                )
            )
            if review.approved:
                break

        return history


if __name__ == "__main__":
    print("=== Berkeley CS294-196 Foundation Lab 1: ReAct Restaurant Agent ===")
    r_agent = ReActRestaurantAgent()
    for name in ("Tartine Bakery", "Chez Panisse", "Mediocre Diner"):
        rep = r_agent.evaluate_restaurant(f"How good is {name}?")
        print(
            f"{rep.restaurant_name:<18} | Reviews: {rep.num_reviews} | "
            f"Raw Mean: {rep.raw_mean_score:.2f} | Bayesian Score: {rep.bayesian_weighted_score:.2f}"
        )

    print("\n=== Berkeley CS294-196 Foundation Lab 2: Tiered Memory (MemGPT) Agent ===")
    mem_agent = TieredMemoryAgent(max_fifo_turns=3)
    mem_agent.archival_insert("doc_lean", "Lean 4 uses the Calculus of Inductive Constructions and `omega` for linear integer arithmetic.")
    for msg in [
        "Hi! I am allergic to peanuts and I prefer Lean4 for verification.",
        "My cloud compute budget is $250.",
        "We are building a neurosymbolic prover.",
        "What tactic does Lean 4 use for linear integer arithmetic, and what are my constraints?",
    ]:
        out = mem_agent.chat(msg)
    print(f"Final Core Memory: {out['core_memory']}")
    print(f"FIFO Queue Size: {out['fifo_size']} | Memory Pressure Page Faults: {out['page_fault_count']}")

    print("\n=== Berkeley CS294-196 Foundation Lab 3: Multi-Agent Actor-Critic Reflexion ===")
    ma_sys = MultiAgentReflexionSystem(max_rounds=4)
    rounds = ma_sys.run_collaboration("Deploy `users.last_login_ts` migration safely")
    for rd in rounds:
        print(f"Round {rd.round_num}: approved={rd.critic_review.approved} | score={rd.critic_review.score:.2f} | {rd.episodic_reflection}")
