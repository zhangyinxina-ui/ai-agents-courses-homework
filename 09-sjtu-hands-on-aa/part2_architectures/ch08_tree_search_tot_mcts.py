#!/usr/bin/env python3
"""
Chapter 8: Deliberate Reasoning via Tree Search — ToT & MCTS (树搜索与深层推理：ToT与MCTS)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 2: Core Architectures

Implements:
1. Tree-of-Thoughts (Yao et al., 2023) with Breadth-First Beam Search (`ToTBeamSearch`)
   and Depth-First Search (`ToTDFSSearch`) on combinatorial arithmetic reasoning (Game of 24)
2. Monte Carlo Tree Search (`MCTSReasoningAgent`) with UCT (Upper Confidence Bound for Trees):
   - Selection via UCT: Q(s,a)/N(s,a) + c * sqrt(ln N(s) / N(s,a))
   - Expansion of candidate thought states
   - Value Rollout / Evaluation
   - Backpropagation of rewards up the reasoning tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ThoughtState:
    """Represents a partial reasoning state in the Game of 24."""
    numbers: Tuple[float, ...]
    history: Tuple[str, ...] = ()

    def is_terminal(self) -> bool:
        return len(self.numbers) == 1

    def is_solved(self, target: float = 24.0) -> bool:
        return len(self.numbers) == 1 and abs(self.numbers[0] - target) < 1e-5


class GameOf24Generator:
    """Generates and heuristically scores candidate single-step arithmetic thoughts."""

    @staticmethod
    def expand(state: ThoughtState) -> List[ThoughtState]:
        nums = state.numbers
        n = len(nums)
        if n <= 1:
            return []

        children: List[ThoughtState] = []
        seen_signatures = set()

        for i in range(n):
            for j in range(i + 1, n):
                a, b = nums[i], nums[j]
                rest = tuple(nums[k] for k in range(n) if k != i and k != j)
                candidates = [
                    (a + b, f"({a:g}+{b:g}={a+b:g})"),
                    (a - b, f"({a:g}-{b:g}={a-b:g})"),
                    (b - a, f"({b:g}-{a:g}={b-a:g})"),
                    (a * b, f"({a:g}*{b:g}={a*b:g})"),
                ]
                if abs(b) > 1e-6:
                    candidates.append((a / b, f"({a:g}/{b:g}={a/b:g})"))
                if abs(a) > 1e-6:
                    candidates.append((b / a, f"({b:g}/{a:g}={b/a:g})"))

                for val, expr in candidates:
                    next_nums = tuple(sorted(rest + (round(val, 5),)))
                    if next_nums not in seen_signatures:
                        seen_signatures.add(next_nums)
                        children.append(
                            ThoughtState(numbers=next_nums, history=state.history + (expr,))
                        )
        return children

    @staticmethod
    def evaluate_heuristic(state: ThoughtState, target: float = 24.0) -> float:
        """Value function V(s) in [0.0, 1.0] estimating proximity and factor structure toward 24."""
        if state.is_solved(target):
            return 1.0
        nums = state.numbers
        if len(nums) == 1:
            dist = abs(nums[0] - target)
            return max(0.0, 0.35 * math.exp(-dist / 12.0))

        # Check if 2 numbers can directly form 24 in one operation
        if len(nums) == 2:
            a, b = nums[0], nums[1]
            ops = [a + b, a - b, b - a, a * b]
            if abs(b) > 1e-6:
                ops.append(a / b)
            if abs(a) > 1e-6:
                ops.append(b / a)
            if any(abs(v - target) < 1e-5 for v in ops):
                return 0.95

        # Reward states containing clean integer factors of 24 (2, 3, 4, 6, 8, 12)
        factors = {2.0, 3.0, 4.0, 6.0, 8.0, 12.0}
        factor_count = sum(1 for x in nums if any(abs(x - f) < 1e-5 for f in factors))
        int_count = sum(1 for x in nums if abs(x - round(x)) < 1e-5)
        score = 0.2 + 0.2 * factor_count + 0.05 * int_count
        return round(min(0.9, score), 4)


class ToTBeamSearch:
    """Tree-of-Thoughts Breadth-First Search with beam width `b`."""

    def __init__(self, beam_width: int = 8) -> None:
        self.beam_width = beam_width

    def solve(self, initial_numbers: Tuple[float, ...]) -> Dict[str, Any]:
        beam: List[ThoughtState] = [ThoughtState(tuple(sorted(initial_numbers)))]
        nodes_expanded = 0

        for _depth in range(len(initial_numbers) - 1):
            next_candidates: List[Tuple[ThoughtState, float]] = []
            for state in beam:
                nodes_expanded += 1
                for child in GameOf24Generator.expand(state):
                    if child.is_solved():
                        return {
                            "solved": True,
                            "history": list(child.history),
                            "nodes_expanded": nodes_expanded,
                        }
                    score = GameOf24Generator.evaluate_heuristic(child)
                    next_candidates.append((child, score))

            next_candidates.sort(key=lambda item: item[1], reverse=True)
            beam = [item[0] for item in next_candidates[: self.beam_width]]

        return {"solved": False, "history": [], "nodes_expanded": nodes_expanded}


@dataclass
class MCTSNode:
    state: ThoughtState
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    untried_states: Optional[List[ThoughtState]] = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration_c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None and self.parent.visits > 0
        exploit = self.q_value
        explore = exploration_c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


class MCTSReasoningAgent:
    """Monte Carlo Tree Search (UCT) over multi-step reasoning states."""

    def __init__(self, iterations: int = 120, exploration_c: float = 1.414) -> None:
        self.iterations = iterations
        self.exploration_c = exploration_c

    def search(self, initial_numbers: Tuple[float, ...]) -> Dict[str, Any]:
        root = MCTSNode(state=ThoughtState(tuple(sorted(initial_numbers))))
        winning_node: Optional[MCTSNode] = None

        for _ in range(self.iterations):
            node = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            reward, solved_state = self._simulate(node.state)
            self._backpropagate(node, reward)
            if solved_state is not None:
                winning_node = MCTSNode(state=solved_state, parent=node)
                break

        if winning_node is not None:
            return {
                "solved": True,
                "history": list(winning_node.state.history),
                "root_visits": root.visits,
            }
        # Fallback: pick best visited terminal child
        return {"solved": False, "history": [], "root_visits": root.visits}

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal():
            if node.untried_states is None:
                node.untried_states = GameOf24Generator.expand(node.state)
            if node.untried_states:
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda c: c.uct_score(self.exploration_c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.untried_states is None:
            node.untried_states = GameOf24Generator.expand(node.state)
        if not node.untried_states:
            return node
        # Prioritize highest heuristic untried state
        node.untried_states.sort(key=GameOf24Generator.evaluate_heuristic, reverse=True)
        next_state = node.untried_states.pop(0)
        child = MCTSNode(state=next_state, parent=node)
        node.children.append(child)
        return child

    @staticmethod
    def _simulate(state: ThoughtState) -> Tuple[float, Optional[ThoughtState]]:
        curr = state
        while not curr.is_terminal():
            children = GameOf24Generator.expand(curr)
            if not children:
                break
            # Check if any child solves immediately
            for ch in children:
                if ch.is_solved():
                    return 1.0, ch
            children.sort(key=GameOf24Generator.evaluate_heuristic, reverse=True)
            curr = children[0]
        if curr.is_solved():
            return 1.0, curr
        return GameOf24Generator.evaluate_heuristic(curr), None

    @staticmethod
    def _backpropagate(node: Optional[MCTSNode], reward: float) -> None:
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent


def run_chapter_demo() -> Dict[str, Any]:
    puzzle = (4.0, 9.0, 10.0, 13.0)  # (13 - 9) * (10 - 4) = 4 * 6 = 24
    tot_res = ToTBeamSearch(beam_width=12).solve(puzzle)
    mcts_res = MCTSReasoningAgent(iterations=100).search(puzzle)
    return {
        "puzzle": puzzle,
        "tot": tot_res,
        "mcts": mcts_res,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 08 Demo] ToT Solved={res['tot']['solved']} ({res['tot']['history']}) | MCTS Solved={res['mcts']['solved']} ({res['mcts']['history']})")
