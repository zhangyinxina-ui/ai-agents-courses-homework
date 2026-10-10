#!/usr/bin/env python3
"""
Chapter 1: Introduction to AI Agents (初识智能体)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 1: Foundations

Implements:
1. POMDP / Environment-Agent Perception-Action Loop (`Observation`, `Action`, `StepResult`)
2. Deterministic Grid/Smart-Room Environment (`SmartHomeEnv`)
3. Paradigms of Agents:
   - ReflexAgent (Rule-based reactive policy)
   - StatefulGoalAgent (Internal world state + goal-directed planner + LLM policy wrapper)
4. Episode rollout runner with trajectory logging and reward accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    """Sensory observation emitted by the environment at step t."""
    step: int
    location: str
    temperature_c: float
    lights_on: bool
    dirty_rooms: Tuple[str, ...]
    message: str = ""


@dataclass(frozen=True)
class Action:
    """Discrete action selected by the agent."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of executing an Action in the Environment."""
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


class SmartHomeEnv:
    """
    A multi-room smart-home environment modeling a Partially Observable Markov Decision Process.
    Goal: Clean all dirty rooms, regulate temperature to target range [21.0, 24.0] C,
    and turn off lights before calling FINISH.
    """

    ROOMS = ("living_room", "kitchen", "study")

    def __init__(
        self,
        initial_room: str = "living_room",
        initial_temp: float = 28.0,
        dirty_rooms: Tuple[str, ...] = ("living_room", "kitchen"),
        max_steps: int = 15,
    ) -> None:
        self.initial_room = initial_room
        self.initial_temp = initial_temp
        self.initial_dirty = set(dirty_rooms)
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> Observation:
        self.step_count = 0
        self.current_room = self.initial_room
        self.temperature_c = self.initial_temp
        self.lights_on = True
        self.dirty_rooms = set(self.initial_dirty)
        self.done = False
        return self._get_obs("Environment reset.")

    def _get_obs(self, msg: str = "") -> Observation:
        return Observation(
            step=self.step_count,
            location=self.current_room,
            temperature_c=round(self.temperature_c, 1),
            lights_on=self.lights_on,
            dirty_rooms=tuple(sorted(self.dirty_rooms)),
            message=msg,
        )

    def is_goal_satisfied(self) -> bool:
        temp_ok = 21.0 <= self.temperature_c <= 24.0
        clean_ok = len(self.dirty_rooms) == 0
        lights_ok = not self.lights_on
        return temp_ok and clean_ok and lights_ok

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated environment.")

        self.step_count += 1
        reward = -0.1  # step living penalty to encourage efficiency
        msg = ""

        if action.name == "MOVE":
            target = action.args.get("room", "")
            if target in self.ROOMS and target != self.current_room:
                self.current_room = target
                msg = f"Moved to {target}."
            else:
                reward -= 0.5
                msg = f"Invalid or redundant move to '{target}'."

        elif action.name == "CLEAN":
            if self.current_room in self.dirty_rooms:
                self.dirty_rooms.remove(self.current_room)
                reward += 2.0
                msg = f"Cleaned {self.current_room}."
            else:
                reward -= 0.3
                msg = f"Room {self.current_room} is already clean."

        elif action.name == "SET_TEMP":
            target_temp = float(action.args.get("target_c", 22.5))
            self.temperature_c = target_temp
            if 21.0 <= self.temperature_c <= 24.0:
                reward += 1.5
                msg = f"Thermostat set to {self.temperature_c:.1f}C (optimal)."
            else:
                reward -= 0.5
                msg = f"Thermostat set to {self.temperature_c:.1f}C (out of comfort range)."

        elif action.name == "SWITCH_LIGHTS":
            state = bool(action.args.get("on", False))
            self.lights_on = state
            reward += 0.5 if not state and len(self.dirty_rooms) == 0 else -0.1
            msg = f"Lights turned {'ON' if state else 'OFF'}."

        elif action.name == "FINISH":
            self.done = True
            if self.is_goal_satisfied():
                reward += 5.0
                msg = "Goal achieved! All rooms clean, temperature optimal, lights off."
            else:
                reward -= 3.0
                msg = "Premature FINISH: goal conditions not yet satisfied."
        else:
            reward -= 1.0
            msg = f"Unknown action '{action.name}'."

        truncated = self.step_count >= self.max_steps and not self.done
        if truncated:
            self.done = True

        obs = self._get_obs(msg)
        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            terminated=self.done and not truncated,
            truncated=truncated,
            info={"goal_satisfied": self.is_goal_satisfied()},
        )


class BaseAgent:
    """Abstract base class for an Agent in the Perception-Thought-Action loop."""

    def reset(self) -> None:
        pass

    def perceive_and_act(self, obs: Observation) -> Tuple[str, Action]:
        """Returns (thought_trace, selected_action)."""
        raise NotImplementedError


class StatefulLLMAgentWrapper(BaseAgent):
    """
    A stateful agent wrapper that maintains an episodic interaction history,
    constructs structured prompts from observations, invokes a policy backend
    (either a deterministic mock LLM or custom callable), and parses actions.
    """

    def __init__(
        self,
        policy_fn: Optional[Callable[[Observation, List[Dict[str, Any]]], Tuple[str, Action]]] = None,
    ) -> None:
        self.policy_fn = policy_fn or self._default_structured_policy
        self.memory: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.memory.clear()

    @staticmethod
    def _default_structured_policy(
        obs: Observation, memory: List[Dict[str, Any]]
    ) -> Tuple[str, Action]:
        # 1. Clean current room if dirty
        if obs.location in obs.dirty_rooms:
            thought = f"Current room '{obs.location}' is dirty. I should clean it first."
            return thought, Action("CLEAN")

        # 2. Move to another dirty room if any remain
        if obs.dirty_rooms:
            next_room = obs.dirty_rooms[0]
            thought = f"Room '{obs.location}' is clean, but '{next_room}' is still dirty. Moving there."
            return thought, Action("MOVE", {"room": next_room})

        # 3. Regulate temperature if outside [21.0, 24.0]
        if not (21.0 <= obs.temperature_c <= 24.0):
            thought = f"All rooms clean. Temperature is {obs.temperature_c}C; adjusting to 22.5C."
            return thought, Action("SET_TEMP", {"target_c": 22.5})

        # 4. Turn off lights before finishing
        if obs.lights_on:
            thought = "Rooms clean and temperature optimal. Turning off lights to save energy."
            return thought, Action("SWITCH_LIGHTS", {"on": False})

        # 5. All goals met -> FINISH
        thought = "All goal conditions satisfied. Issuing FINISH."
        return thought, Action("FINISH")

    def perceive_and_act(self, obs: Observation) -> Tuple[str, Action]:
        thought, action = self.policy_fn(obs, self.memory)
        self.memory.append(
            {
                "step": obs.step,
                "observation": obs,
                "thought": thought,
                "action": action,
            }
        )
        return thought, action


def run_episode(env: SmartHomeEnv, agent: BaseAgent) -> Dict[str, Any]:
    """Executes a complete perception-action episode loop and returns trajectory metrics."""
    obs = env.reset()
    agent.reset()
    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0

    while True:
        thought, action = agent.perceive_and_act(obs)
        result = env.step(action)
        total_reward += result.reward
        trajectory.append(
            {
                "step": obs.step,
                "location": obs.location,
                "thought": thought,
                "action": action.name,
                "args": action.args,
                "reward": result.reward,
                "message": result.observation.message,
            }
        )
        obs = result.observation
        if result.terminated or result.truncated:
            return {
                "success": bool(result.info.get("goal_satisfied", False)),
                "steps": len(trajectory),
                "total_reward": round(total_reward, 4),
                "trajectory": trajectory,
            }


def run_chapter_demo() -> Dict[str, Any]:
    env = SmartHomeEnv(
        initial_room="living_room",
        initial_temp=28.5,
        dirty_rooms=("living_room", "kitchen"),
        max_steps=15,
    )
    agent = StatefulLLMAgentWrapper()
    summary = run_episode(env, agent)
    return summary


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 01 Demo] Episode Success={res['success']}, Steps={res['steps']}, Total Reward={res['total_reward']}")
    for step_log in res["trajectory"]:
        print(f"  Step {step_log['step']}: Thought='{step_log['thought']}' -> Action={step_log['action']}({step_log['args']}) | R={step_log['reward']}")
