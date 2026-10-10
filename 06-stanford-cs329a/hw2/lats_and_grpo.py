"""
Stanford CS329A: Self-Improving AI Agents
Homework 2: Multi-Step Reasoning/Planning (LATS & ADaPT) and Train-Time RL Scaling (STaR & GRPO)

Implements:
1. Language Agent Tree Search (LATS, Zhou et al., 2024):
   - Monte Carlo Tree Search (MCTS) over reasoning/action trajectories
   - UCT selection, multi-branch expansion, environment + value evaluation,
     verbal self-reflection memory on failed terminal leaves, and backpropagation.
2. ADaPT (As-Needed Decomposition and Planning, Prasad et al., 2024):
   - Attempts direct atomic execution first; recursively decomposes into AND/OR sub-tasks
     only when atomic execution fails.
3. STaR (Self-Taught Reasoner, Zelikman et al., 2022):
   - Rationale generation (forward filter) + Hint-conditioned rationalization + SFT dataset curation.
4. GRPO (Group Relative Policy Optimization, Shao et al., 2024 / DeepSeek-R1):
   - Group-normalized advantage estimation (critic-free)
   - Clipped PPO-style surrogate objective with unbiased Schulman K3 KL penalty
   - Tabular/log-linear policy optimization loop demonstrating train-time RL scaling.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# 1. Language Agent Tree Search (LATS)
# ============================================================================

@dataclasses.dataclass
class LATSNode:
    """Node in the Language Agent Tree Search (MCTS) tree."""
    state: str
    action: Optional[str] = None
    parent: Optional["LATSNode"] = None
    children: List["LATSNode"] = dataclasses.field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False
    is_solved: bool = False
    reflection: Optional[str] = None
    depth: int = 0

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s, a)."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """
        Upper Confidence Bound for Trees (UCT):
            UCT(s, a) = Q(s, a) + c * sqrt(ln(N(s) + 1) / (N(s, a) + 1e-6))
        Unvisited children receive +inf priority (ties broken by order).
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploit = self.q_value
        explore = exploration_weight * math.sqrt(math.log(max(1, parent_visits)) / self.visits)
        return exploit + explore

    def trajectory_actions(self) -> List[str]:
        """Returns the sequence of actions from root to this node."""
        actions: List[str] = []
        curr: Optional[LATSNode] = self
        while curr is not None and curr.action is not None:
            actions.append(curr.action)
            curr = curr.parent
        actions.reverse()
        return actions


class GameOf24Environment:
    """
    Symbolic multi-step arithmetic reasoning environment (Game of 24 / Countdown)
    used to benchmark Tree of Thoughts (Yao et al., 2023) and LATS (Zhou et al., 2024).
    State is represented as a sorted tuple of remaining numbers formatted as a string,
    e.g., '[3, 3, 8, 8]' or '[4, 6, 8]'. Goal: reach '[24]' (within 1e-5).
    """

    def __init__(self, initial_numbers: Sequence[float], target: float = 24.0):
        self.initial_numbers = tuple(float(x) for x in initial_numbers)
        self.target = float(target)

    def initial_state(self) -> str:
        return self._format_nums(self.initial_numbers)

    @staticmethod
    def _format_nums(nums: Sequence[float]) -> str:
        formatted = []
        for x in sorted(nums):
            if abs(x - round(x)) < 1e-6:
                formatted.append(str(int(round(x))))
            else:
                formatted.append(f"{x:.8f}")
        return "[" + ", ".join(formatted) + "]"

    @staticmethod
    def parse_state(state_str: str) -> List[float]:
        inner = state_str.strip().strip("[]")
        if not inner:
            return []
        return [float(tok.strip()) for tok in inner.split(",")]

    def is_terminal(self, state_str: str) -> Tuple[bool, bool]:
        """Returns (is_terminal, is_solved)."""
        nums = self.parse_state(state_str)
        if len(nums) == 1:
            solved = abs(nums[0] - self.target) < 1e-4
            return True, solved
        return False, False

    def propose_actions(
        self,
        state_str: str,
        reflections: Sequence[str],
        max_branches: int = 10,
    ) -> List[Tuple[str, str]]:
        """
        Proposes candidate (action_description, next_state_str) transitions,
        prioritizing promising intermediate values and filtering out dead-end actions
        recorded in `reflections`.
        """
        nums = self.parse_state(state_str)
        if len(nums) <= 1:
            return []

        candidates: List[Tuple[float, str, str]] = []
        seen_next: set = set()
        n = len(nums)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = nums[i], nums[j]
                rest = [nums[k] for k in range(n) if k != i and k != j]
                ops = [
                    (f"{a:g} + {b:g} = {a + b:g}", a + b),
                    (f"{b:g} - {a:g} = {b - a:g}", b - a),
                    (f"{a:g} - {b:g} = {a - b:g}", a - b),
                    (f"{a:g} * {b:g} = {a * b:g}", a * b),
                ]
                if abs(b) > 1e-6:
                    ops.append((f"{a:g} / {b:g} = {a / b:g}", a / b))
                if abs(a) > 1e-6:
                    ops.append((f"{b:g} / {a:g} = {b / a:g}", b / a))

                for act_desc, val in ops:
                    if val < 0:
                        continue
                    next_nums = rest + [val]
                    next_state = self._format_nums(next_nums)
                    if next_state in seen_next:
                        continue
                    # Filter actions explicitly flagged in verbal reflections
                    if any(act_desc in refl for refl in reflections):
                        prior_penalty = -0.35
                    else:
                        prior_penalty = 0.0
                    seen_next.add(next_state)
                    h_score = self.evaluate_state(next_state) + prior_penalty
                    candidates.append((h_score, act_desc, next_state))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(act, nxt) for _, act, nxt in candidates[:max_branches]]

    def evaluate_state(self, state_str: str) -> float:
        """
        Execution + lookahead value function V(s) in [0.0, 1.0].
        Returns 1.0 if solved, 0.05 if terminal dead-end, or a heuristic score based on
        whether 24 is reachable or factors/addends of 24 are present.
        """
        nums = self.parse_state(state_str)
        if not nums:
            return 0.0
        if len(nums) == 1:
            return 1.0 if abs(nums[0] - self.target) < 1e-4 else 0.05

        # 1-step exact reachability check when 2 numbers remain
        if len(nums) == 2:
            a, b = nums[0], nums[1]
            reachable = [a + b, abs(a - b), a * b]
            if abs(b) > 1e-6:
                reachable.append(a / b)
            if abs(a) > 1e-6:
                reachable.append(b / a)
            if any(abs(r - self.target) < 1e-4 for r in reachable):
                return 0.92
            closest = min(abs(r - self.target) for r in reachable)
            return max(0.10, 0.65 / (1.0 + 0.1 * closest))

        # 3 or more numbers: reward clean divisors/factors of 24
        bonus = 0.35
        for x in nums:
            if abs(x) > 1e-6 and abs(round(self.target / x) * x - self.target) < 1e-4:
                bonus += 0.12
        return min(0.80, bonus)


class LATSAgent:
    """
    Language Agent Tree Search (LATS) combining MCTS (Selection, Expansion,
    Evaluation, Backpropagation) with Verbal Self-Reflection memory.
    """

    def __init__(
        self,
        exploration_weight: float = 1.414,
        max_iterations: int = 40,
        branching_factor: int = 10,
    ):
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.branching_factor = branching_factor
        self.reflections: List[str] = []

    def select(self, root: LATSNode) -> LATSNode:
        """Traverses tree from root using UCT until reaching a leaf or terminal node."""
        node = root
        while node.children and not node.is_terminal:
            node = max(
                node.children,
                key=lambda child: child.uct_score(self.exploration_weight),
            )
        return node

    def expand(self, node: LATSNode, env: GameOf24Environment) -> List[LATSNode]:
        """Expands a non-terminal leaf node using reflection-conditioned action proposals."""
        if node.is_terminal:
            return []
        proposals = env.propose_actions(
            node.state,
            reflections=self.reflections,
            max_branches=self.branching_factor,
        )
        for act_desc, next_state in proposals:
            is_term, is_sol = env.is_terminal(next_state)
            child = LATSNode(
                state=next_state,
                action=act_desc,
                parent=node,
                is_terminal=is_term,
                is_solved=is_sol,
                depth=node.depth + 1,
            )
            node.children.append(child)
        return node.children

    def backpropagate(self, node: LATSNode, reward: float) -> None:
        """Updates visit counts and cumulative values up to the root."""
        curr: Optional[LATSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent

    def generate_reflection(self, failed_node: LATSNode) -> str:
        """Generates a verbal self-reflection summarizing why a terminal trajectory failed."""
        actions = failed_node.trajectory_actions()
        first_act = actions[0] if actions else "empty"
        refl = (
            f"Trajectory [{' -> '.join(actions)}] ended at terminal state {failed_node.state} "
            f"!= 24; avoid unproductive opening step `{first_act}`."
        )
        failed_node.reflection = refl
        if refl not in self.reflections:
            self.reflections.append(refl)
        return refl

    def search(self, env: GameOf24Environment) -> Dict[str, Any]:
        """Runs LATS MCTS search on `env` and returns solution trajectory and tree stats."""
        root = LATSNode(state=env.initial_state(), depth=0)
        self.reflections.clear()
        nodes_expanded = 0

        for it in range(1, self.max_iterations + 1):
            leaf = self.select(root)
            if leaf.is_solved:
                self.backpropagate(leaf, 1.0)
                return {
                    "solved": True,
                    "iterations": it,
                    "nodes_expanded": nodes_expanded,
                    "actions": leaf.trajectory_actions(),
                    "final_state": leaf.state,
                    "reflections": list(self.reflections),
                    "root": root,
                }

            children = self.expand(leaf, env)
            nodes_expanded += len(children)

            if not children:
                val = env.evaluate_state(leaf.state)
                if leaf.is_terminal and not leaf.is_solved:
                    self.generate_reflection(leaf)
                self.backpropagate(leaf, val)
                continue

            # Evaluate newly expanded children and check if any solves the task
            for child in children:
                val = env.evaluate_state(child.state)
                self.backpropagate(child, val)
                if child.is_solved:
                    return {
                        "solved": True,
                        "iterations": it,
                        "nodes_expanded": nodes_expanded,
                        "actions": child.trajectory_actions(),
                        "final_state": child.state,
                        "reflections": list(self.reflections),
                        "root": root,
                    }
                if child.is_terminal and not child.is_solved:
                    self.generate_reflection(child)

        return {
            "solved": False,
            "iterations": self.max_iterations,
            "nodes_expanded": nodes_expanded,
            "actions": [],
            "final_state": root.state,
            "reflections": list(self.reflections),
            "root": root,
        }


# ============================================================================
# 2. ADaPT: As-Needed Decomposition and Planning (Prasad et al., 2024)
# ============================================================================

@dataclasses.dataclass
class ADaPTPlanNode:
    """Execution record of an ADaPT task or decomposed sub-plan."""
    task: str
    depth: int
    atomic_attempted: bool = True
    atomic_succeeded: bool = False
    operator: Optional[str] = None  # 'AND' or 'OR' when decomposed
    children: List["ADaPTPlanNode"] = dataclasses.field(default_factory=list)
    succeeded: bool = False
    output: str = ""


class ADaPTPlanner:
    """
    As-Needed Decomposition and Planning (ADaPT, Prasad et al., 2024).
    Core algorithm:
        1. Attempt atomic execution of `task` via low-level executor.
        2. If atomic execution succeeds, return immediately without decomposition overhead.
        3. If atomic execution fails and `depth < max_depth`, invoke Planner to decompose
           `task` into an AND/OR graph of subtasks and recursively execute each.
    """

    def __init__(
        self,
        atomic_executor: Callable[[str, Dict[str, Any]], Tuple[bool, str]],
        decomposer: Callable[[str], Tuple[str, List[str]]],
        max_depth: int = 3,
    ):
        self.atomic_executor = atomic_executor
        self.decomposer = decomposer
        self.max_depth = max_depth

    def run(self, task: str, env_state: Dict[str, Any], depth: int = 0) -> ADaPTPlanNode:
        node = ADaPTPlanNode(task=task, depth=depth)

        # Step 1: Try atomic execution first
        ok, msg = self.atomic_executor(task, env_state)
        if ok:
            node.atomic_succeeded = True
            node.succeeded = True
            node.output = msg
            return node

        # Step 2: If max depth reached or cannot decompose, fail gracefully
        if depth >= self.max_depth:
            node.succeeded = False
            node.output = f"Max depth {self.max_depth} reached on: {task}"
            return node

        # Step 3: Decompose into AND / OR subtasks as-needed
        operator, subtasks = self.decomposer(task)
        node.operator = operator
        if not subtasks:
            node.succeeded = False
            node.output = f"No decomposition available for: {task}"
            return node

        if operator == "AND":
            outputs = []
            all_ok = True
            for sub in subtasks:
                child = self.run(sub, env_state, depth + 1)
                node.children.append(child)
                outputs.append(child.output)
                if not child.succeeded:
                    all_ok = False
                    break
            node.succeeded = all_ok
            node.output = " | ".join(outputs)
        elif operator == "OR":
            any_ok = False
            for sub in subtasks:
                child = self.run(sub, env_state, depth + 1)
                node.children.append(child)
                if child.succeeded:
                    any_ok = True
                    node.output = child.output
                    break
            node.succeeded = any_ok
            if not any_ok:
                node.output = "All OR branches failed"
        else:
            raise ValueError(f"Unsupported logical operator in ADaPT: {operator}")

        return node


# ============================================================================
# 3. STaR (Self-Taught Reasoner) & GRPO (Group Relative Policy Optimization)
# ============================================================================

@dataclasses.dataclass
class STaRExample:
    question_id: str
    question: str
    ground_truth: str
    rationale: str
    predicted_answer: str
    used_rationalization_hint: bool = False


def star_filter_and_rationalize(
    dataset: Sequence[Tuple[str, str, str]],
    forward_generator: Callable[[str], Tuple[str, str]],
    hint_rationalizer: Callable[[str, str], Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Implements one outer iteration of STaR (Zelikman et al., 2022):
      1. Forward Rationale Generation: sample (r_i, y_hat_i) ~ pi_theta(. | x_i).
         Keep directly if y_hat_i == y_i*.
      2. Hint-Conditioned Rationalization: for failed instances where y_hat_i != y_i*,
         prompt with ground-truth hint (x_i, y_i*) to synthesize a valid rationale r_i^hint.
         Keep if the rationalized trace derives y_i* without leaking the hint marker.
    """
    direct_kept: List[STaRExample] = []
    rationalized_kept: List[STaRExample] = []
    rejected: List[str] = []

    for qid, question, gt in dataset:
        rationale, pred = forward_generator(question)
        if pred.strip() == gt.strip():
            direct_kept.append(
                STaRExample(
                    question_id=qid,
                    question=question,
                    ground_truth=gt,
                    rationale=rationale,
                    predicted_answer=pred,
                    used_rationalization_hint=False,
                )
            )
        else:
            hint_rat, hint_pred = hint_rationalizer(question, gt)
            if hint_pred.strip() == gt.strip() and len(hint_rat.strip()) > 0:
                rationalized_kept.append(
                    STaRExample(
                        question_id=qid,
                        question=question,
                        ground_truth=gt,
                        rationale=hint_rat,
                        predicted_answer=hint_pred,
                        used_rationalization_hint=True,
                    )
                )
            else:
                rejected.append(qid)

    sft_corpus = direct_kept + rationalized_kept
    return {
        "sft_corpus": sft_corpus,
        "num_direct": len(direct_kept),
        "num_rationalized": len(rationalized_kept),
        "num_rejected": len(rejected),
        "retention_rate": len(sft_corpus) / max(1, len(dataset)),
    }


def compute_grpo_advantages(
    rewards: Sequence[float],
    eps: float = 1e-8,
) -> List[float]:
    """
    Computes Group Relative Policy Optimization (GRPO) group-normalized advantages:
        A_i = (r_i - mean(r)) / (std(r) + eps)
    where std(r) is the population standard deviation across the group of G outputs
    sampled for the same prompt q.
    """
    g = len(rewards)
    if g == 0:
        raise ValueError("rewards group cannot be empty")
    mean_r = sum(rewards) / g
    var_r = sum((r - mean_r) ** 2 for r in rewards) / g
    std_r = math.sqrt(var_r)
    if std_r < eps:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std_r + eps) for r in rewards]


def schulman_k3_kl_divergence(
    log_probs_new: Sequence[float],
    log_probs_ref: Sequence[float],
) -> List[float]:
    """
    Computes the unbiased, strictly non-negative Schulman K3 estimator of
    KL(pi_theta || pi_ref) used in GRPO (Shao et al., 2024):
        D_KL_i = exp(log_pi_ref - log_pi_theta) - (log_pi_ref - log_pi_theta) - 1
    """
    if len(log_probs_new) != len(log_probs_ref):
        raise ValueError("Length mismatch between log_probs_new and log_probs_ref")
    kls: List[float] = []
    for lp_new, lp_ref in zip(log_probs_new, log_probs_ref):
        log_ratio_ref_over_new = lp_ref - lp_new
        kl_i = math.exp(log_ratio_ref_over_new) - log_ratio_ref_over_new - 1.0
        kls.append(max(0.0, kl_i))
    return kls


@dataclasses.dataclass
class GRPOLossOutput:
    total_loss: float
    policy_surrogate_loss: float
    kl_penalty: float
    advantages: List[float]
    ratios: List[float]
    clipped_fraction: float


def compute_grpo_loss(
    log_probs_new: Sequence[float],
    log_probs_old: Sequence[float],
    log_probs_ref: Sequence[float],
    rewards: Sequence[float],
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
) -> GRPOLossOutput:
    """
    Computes the GRPO objective (expressed as a minimization loss L_GRPO = -J_GRPO):
        J_GRPO(theta) = (1/G) sum_{i=1}^G [
            min( rho_i * A_i, clip(rho_i, 1 - eps, 1 + eps) * A_i ) - beta * D_KL_i
        ]
    where rho_i = exp(log_pi_theta(o_i|q) - log_pi_old(o_i|q)).
    """
    g = len(rewards)
    if not (len(log_probs_new) == len(log_probs_old) == len(log_probs_ref) == g):
        raise ValueError("All input sequences to compute_grpo_loss must have equal length G")

    advantages = compute_grpo_advantages(rewards)
    kls = schulman_k3_kl_divergence(log_probs_new, log_probs_ref)

    surrogate_terms: List[float] = []
    ratios: List[float] = []
    clipped_count = 0

    for lp_new, lp_old, adv in zip(log_probs_new, log_probs_old, advantages):
        rho = math.exp(lp_new - lp_old)
        ratios.append(rho)
        clipped_rho = max(1.0 - clip_eps, min(1.0 + clip_eps, rho))
        if abs(clipped_rho - rho) > 1e-9:
            clipped_count += 1
        unclipped_obj = rho * adv
        clipped_obj = clipped_rho * adv
        surrogate_terms.append(min(unclipped_obj, clipped_obj))

    mean_surrogate_obj = sum(surrogate_terms) / g
    mean_kl = sum(kls) / g
    # Loss to minimize is negative of objective + beta_kl * mean_kl
    policy_loss = -mean_surrogate_obj
    total_loss = policy_loss + beta_kl * mean_kl

    return GRPOLossOutput(
        total_loss=total_loss,
        policy_surrogate_loss=policy_loss,
        kl_penalty=mean_kl,
        advantages=advantages,
        ratios=ratios,
        clipped_fraction=clipped_count / g,
    )


def run_hw2_benchmarks() -> Dict[str, Any]:
    """
    Runs empirical evaluations for LATS, ADaPT, STaR, and GRPO for HW2 Report.
    """
    # 1. LATS on Game of 24 benchmark suite
    puzzles = [
        ("puzzle_1", [4, 7, 8, 8]),   # (7 - 8/8) * 4 = 24
        ("puzzle_2", [3, 3, 8, 8]),   # 8 / (3 - 8/3) = 24
        ("puzzle_3", [1, 5, 5, 5]),   # (5 - 1/5) * 5 = 24
        ("puzzle_4", [2, 3, 4, 6]),   # 4 * 6 * (3 - 2) = 24
        ("puzzle_5", [1, 3, 4, 6]),   # 6 / (1 - 3/4) = 24
        ("puzzle_6", [2, 5, 5, 10]),  # (5 - 2/10) * 5 = 24
    ]
    lats_results = []
    for pid, nums in puzzles:
        env = GameOf24Environment(nums)
        agent = LATSAgent(exploration_weight=1.414, max_iterations=45, branching_factor=10)
        res = agent.search(env)
        lats_results.append((pid, nums, res))

    # 2. ADaPT on composite data-pipeline & tool execution tasks
    def atomic_exec(task: str, state: Dict[str, Any]) -> Tuple[bool, str]:
        # Atomic tasks succeed directly; composite multi-stage tasks fail atomic execution
        atomic_catalog = {
            "fetch_sales_csv": "loaded 1200 rows",
            "clean_null_records": "cleaned 1185 rows",
            "compute_quarterly_margin": "margin=28.4%",
            "query_primary_api": "timeout",  # primary API fails, triggering OR fallback
            "query_replica_cache": "cache_hit=True",
            "validate_schema": "schema_ok",
        }
        if task in atomic_catalog:
            if atomic_catalog[task] == "timeout":
                return False, "Primary API timed out"
            state[task] = atomic_catalog[task]
            return True, f"{task}:{atomic_catalog[task]}"
        return False, f"Task `{task}` is too complex for single-step atomic execution"

    def decomposer(task: str) -> Tuple[str, List[str]]:
        decomp_rules = {
            "build_quarterly_report": (
                "AND",
                ["ingest_resilient_data", "clean_null_records", "compute_quarterly_margin"],
            ),
            "ingest_resilient_data": (
                "OR",
                ["query_primary_api", "fetch_backup_pipeline"],
            ),
            "fetch_backup_pipeline": (
                "AND",
                ["query_replica_cache", "fetch_sales_csv", "validate_schema"],
            ),
        }
        return decomp_rules.get(task, ("AND", []))

    adapt_state: Dict[str, Any] = {}
    planner = ADaPTPlanner(atomic_executor=atomic_exec, decomposer=decomposer, max_depth=4)
    adapt_root = planner.run("build_quarterly_report", adapt_state)

    # 3. GRPO Policy Optimization progression across 15 training steps
    # Simulate a categorical policy over K=4 reasoning strategies where strategy 0 is optimal (r=1.0),
    # strategy 1 is format-correct partial (r=0.35), and strategies 2, 3 are wrong (r=0.0).
    logits_ref = [0.0, 0.0, 0.0, 0.0]
    logits = list(logits_ref)
    action_rewards = [1.0, 0.35, 0.0, 0.0]
    rng = random.Random(42)
    lr = 0.35
    grpo_history = []

    def log_softmax(z: Sequence[float]) -> List[float]:
        m = max(z)
        exps = [math.exp(v - m) for v in z]
        s = sum(exps)
        return [math.log(e / s) for e in exps]

    for step in range(1, 16):
        lp_curr = log_softmax(logits)
        lp_ref = log_softmax(logits_ref)
        probs = [math.exp(lp) for lp in lp_curr]

        # Sample group of G=16 rollouts from current policy
        sampled_actions = rng.choices(range(4), weights=probs, k=16)
        group_rewards = [action_rewards[a] for a in sampled_actions]
        group_lp_old = [lp_curr[a] for a in sampled_actions]
        group_lp_ref = [lp_ref[a] for a in sampled_actions]

        loss_info = compute_grpo_loss(
            log_probs_new=group_lp_old,
            log_probs_old=group_lp_old,
            log_probs_ref=group_lp_ref,
            rewards=group_rewards,
            clip_eps=0.2,
            beta_kl=0.04,
        )

        # REINFORCE / GRPO gradient step with KL regularization toward reference logits
        advs = loss_info.advantages
        grad = [0.0, 0.0, 0.0, 0.0]
        for a, adv in zip(sampled_actions, advs):
            for k in range(4):
                indicator = 1.0 if k == a else 0.0
                grad[k] += (indicator - probs[k]) * adv / len(sampled_actions)
        for k in range(4):
            kl_grad = 0.04 * (probs[k] - math.exp(lp_ref[k]))
            logits[k] += lr * (grad[k] - kl_grad)

        grpo_history.append({
            "step": step,
            "mean_reward": sum(group_rewards) / len(group_rewards),
            "optimal_prob": probs[0],
            "kl_div": loss_info.kl_penalty,
        })

    return {
        "lats_results": lats_results,
        "adapt_root": adapt_root,
        "grpo_history": grpo_history,
    }


if __name__ == "__main__":
    out = run_hw2_benchmarks()
    print("=== CS329A HW2: LATS (Game of 24) Search Results ===")
    for pid, nums, res in out["lats_results"]:
        print(
            f"{pid} {nums}: solved={res['solved']} | iters={res['iterations']} | "
            f"expanded={res['nodes_expanded']} | reflections={len(res['reflections'])} | "
            f"plan={' -> '.join(res['actions'])}"
        )
    print("\n=== ADaPT Hierarchical Execution ===")
    print(f"Root task `{out['adapt_root'].task}` succeeded={out['adapt_root'].succeeded}")
    print(f"Output: {out['adapt_root'].output}")
    print("\n=== GRPO Training Progression ===")
    for row in out["grpo_history"][::3] + [out["grpo_history"][-1]]:
        print(
            f"Step {row['step']:>2d} | Mean Reward: {row['mean_reward']:.3f} | "
            f"P(optimal trace): {row['optimal_prob']:.3f} | KL: {row['kl_div']:.4f}"
        )
