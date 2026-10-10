#!/usr/bin/env python3
"""
Chapter 7: Tool Calling & Action Space Grounding (工具调用与外部环境交互)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 2: Core Architectures

Implements:
1. Automatic Python Function Inspection -> JSON Schema Tool Registry (`ToolRegistry`)
2. Type-Validated JSON Tool Dispatcher with Error-Recovery Hints (`ToolExecutionResult`)
3. Self-Correcting Tool-Use Agent that repairs malformed arguments using schema feedback.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, get_type_hints


PYTHON_TO_JSON_TYPE: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolExecutionResult:
    tool_name: str
    success: bool
    output: Any
    error_message: Optional[str] = None
    recovery_hint: Optional[str] = None


class ToolRegistry:
    """Registers Python callables, generates OpenAI/MCP-compatible JSON schemas, and dispatches calls."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        name = func.__name__
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        doc = (inspect.getdoc(func) or "").strip()

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            py_type = type_hints.get(param_name, str)
            json_type = PYTHON_TO_JSON_TYPE.get(py_type, "string")
            properties[param_name] = {"type": json_type, "description": f"Parameter '{param_name}'"}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        self._tools[name] = func
        self._schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return func

    def get_schemas(self) -> List[Dict[str, Any]]:
        return list(self._schemas.values())

    def dispatch(self, tool_name: str, raw_arguments: Dict[str, Any]) -> ToolExecutionResult:
        if tool_name not in self._tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error_message=f"Tool '{tool_name}' is not registered.",
                recovery_hint=f"Available tools: {sorted(self._tools.keys())}",
            )

        func = self._tools[tool_name]
        schema = self._schemas[tool_name]["function"]["parameters"]
        type_hints = get_type_hints(func)

        # Check required fields
        for req_field in schema["required"]:
            if req_field not in raw_arguments:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error_message=f"Missing required argument '{req_field}'.",
                    recovery_hint=f"Required arguments for '{tool_name}': {schema['required']}",
                )

        # Validate argument types
        for arg_k, arg_v in raw_arguments.items():
            if arg_k not in schema["properties"]:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error_message=f"Unexpected argument '{arg_k}'.",
                    recovery_hint=f"Allowed arguments: {list(schema['properties'].keys())}",
                )
            expected_py_type = type_hints.get(arg_k, str)
            # Note: bool is a subclass of int in Python, guard explicitly
            if expected_py_type is int and isinstance(arg_v, bool):
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error_message=f"Argument '{arg_k}' expected int, got bool.",
                    recovery_hint=f"Pass an integer value for '{arg_k}'.",
                )
            if not isinstance(arg_v, expected_py_type):
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error_message=f"Argument '{arg_k}' expected {expected_py_type.__name__}, got {type(arg_v).__name__}.",
                    recovery_hint=f"Convert '{arg_k}' to {expected_py_type.__name__} before calling '{tool_name}'.",
                )

        try:
            result = func(**raw_arguments)
            return ToolExecutionResult(tool_name=tool_name, success=True, output=result)
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error_message=f"{type(exc).__name__}: {exc}",
                recovery_hint="Check domain constraints (e.g. non-zero divisor or valid identifier).",
            )


def run_chapter_demo() -> Dict[str, Any]:
    registry = ToolRegistry()

    @registry.register
    def compute_gpu_memory_bandwidth_utilization(
        achieved_gbps: float, peak_gbps: float, kernel_name: str = "gemm_fp16"
    ) -> Dict[str, Any]:
        """Computes normalized GPU HBM bandwidth utilization percentage."""
        if peak_gbps <= 0:
            raise ValueError("peak_gbps must be strictly positive.")
        pct = round((achieved_gbps / peak_gbps) * 100.0, 2)
        return {"kernel": kernel_name, "utilization_pct": pct}

    # Attempt 1: Malformed call (string passed instead of float) -> triggers structured error recovery
    attempt1 = registry.dispatch(
        "compute_gpu_memory_bandwidth_utilization",
        {"achieved_gbps": "2400.0", "peak_gbps": 3350.0},
    )

    # Attempt 2: Repaired call using recovery hint
    attempt2 = registry.dispatch(
        "compute_gpu_memory_bandwidth_utilization",
        {"achieved_gbps": 2680.0, "peak_gbps": 3350.0, "kernel_name": "flash_attn_v3"},
    )

    return {
        "registered_schemas": registry.get_schemas(),
        "attempt1_failed_as_expected": not attempt1.success,
        "attempt1_recovery_hint": attempt1.recovery_hint,
        "attempt2_success": attempt2.success,
        "attempt2_output": attempt2.output,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 07 Demo] Attempt 1 Recovery Hint='{res['attempt1_recovery_hint']}' -> Attempt 2 Output={res['attempt2_output']}")
