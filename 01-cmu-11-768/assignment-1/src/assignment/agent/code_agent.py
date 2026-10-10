"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        self.tools = [EXECUTE_TOOL, SEND_MESSAGE_TOOL] + list(self.tools)

        sys_info = json.dumps(
            {
                "machine": getattr(self.env, "machine", ""),
                "release": getattr(self.env, "release", ""),
                "system": getattr(self.env, "system", ""),
                "version": getattr(self.env, "version", ""),
            },
            indent=2,
            sort_keys=True,
        )
        self.system_prompt = (
            "You are an autonomous software engineering agent working in a bash terminal "
            f"environment (working directory: {getattr(self.env, 'cwd', '/testbed')}).\n"
            "Use the `execute` tool to inspect files, reproduce issues, make minimal targeted "
            "code edits, and run tests to verify your changes. When finished, use `send_message` "
            "to report your summary.\n\n"
            "<system_information>\n"
            f"{sys_info}\n"
            "</system_information>"
        )
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )

        self.task_prompt = self.task

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        observations: list[dict[str, str]] = []
        registered_names = {
            tool.get("function", {}).get("name")
            for tool in self.tools
            if isinstance(tool, dict)
        }

        for call in tool_calls:
            call_id = str(call.get("id", "")) if isinstance(call, dict) else ""
            func = call.get("function") if isinstance(call, dict) else None
            if not isinstance(func, dict):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(
                            {"error": "Malformed tool call: missing function specification."}
                        ),
                    }
                )
                continue

            name = func.get("name", "")
            raw_args = func.get("arguments", "")

            if name not in registered_names:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(
                            {"error": f"Unknown tool: {name}"}
                        ),
                    }
                )
                continue

            if not isinstance(raw_args, str):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(
                            {"error": "Tool arguments must be a JSON string."}
                        ),
                    }
                )
                continue

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(
                            {"error": f"Malformed JSON arguments: {exc}"}
                        ),
                    }
                )
                continue

            if not isinstance(args, dict):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(
                            {"error": "Decoded tool arguments must be a JSON object."}
                        ),
                    }
                )
                continue

            if name == "execute":
                allowed_keys = {"command", "shell", "cwd", "timeout", "env"}
                extra_keys = set(args.keys()) - allowed_keys
                if "command" not in args or extra_keys:
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": format_tool_output(
                                {"error": "Invalid arguments for execute tool."}
                            ),
                        }
                    )
                    continue

                command = args["command"]
                valid_command = isinstance(command, str) or (
                    isinstance(command, list)
                    and len(command) > 0
                    and all(isinstance(item, str) for item in command)
                )
                if not valid_command:
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": format_tool_output(
                                {
                                    "error": "'command' must be a string or a non-empty list of strings."
                                }
                            ),
                        }
                    )
                    continue

                exec_kwargs: dict[str, Any] = {}
                arg_error = None
                if "shell" in args and args["shell"] is not None:
                    if not isinstance(args["shell"], bool):
                        arg_error = "'shell' must be a boolean or null."
                    else:
                        exec_kwargs["shell"] = args["shell"]
                if "cwd" in args and args["cwd"] is not None:
                    if not isinstance(args["cwd"], str):
                        arg_error = "'cwd' must be a string or null."
                    else:
                        exec_kwargs["cwd"] = args["cwd"]
                if "timeout" in args and args["timeout"] is not None:
                    if isinstance(args["timeout"], bool) or not isinstance(
                        args["timeout"], (int, float)
                    ):
                        arg_error = "'timeout' must be a number or null."
                    else:
                        exec_kwargs["timeout"] = args["timeout"]
                if "env" in args and args["env"] is not None:
                    if not isinstance(args["env"], dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in args["env"].items()
                    ):
                        arg_error = "'env' must be a mapping of string keys to string values."
                    else:
                        exec_kwargs["env"] = args["env"]

                if arg_error:
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": format_tool_output({"error": arg_error}),
                        }
                    )
                    continue

                raw_result = self.env.execute(command, **exec_kwargs)
                formatted_dict: dict[str, Any] = {
                    "output": raw_result.get("output", ""),
                    "returncode": raw_result.get("returncode", 0),
                }
                if raw_result.get("exception_info"):
                    formatted_dict["exception_info"] = raw_result["exception_info"]
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output(formatted_dict),
                    }
                )

            elif name == "send_message":
                if set(args.keys()) != {"summary"} or not isinstance(
                    args.get("summary"), str
                ):
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": format_tool_output(
                                {
                                    "error": "send_message requires a single string argument 'summary'."
                                }
                            ),
                        }
                    )
                    continue

                summary = args["summary"]
                try:
                    patch_res = self.env.execute("cat patch.txt")
                    if (
                        isinstance(patch_res, dict)
                        and patch_res.get("returncode", 0) == 0
                    ):
                        self.submitted_patch = patch_res.get("output", "")
                except Exception:
                    pass
                self.finished = True
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": format_tool_output({"summary": summary}),
                    }
                )

            elif name == "invoke_skill":
                if set(args.keys()) != {"name"} or not isinstance(
                    args.get("name"), str
                ):
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": format_tool_output(
                                {
                                    "error": "invoke_skill requires a single string argument 'name'."
                                }
                            ),
                        }
                    )
                    continue

                skill_name = args["name"].strip()
                if skill_name not in self.skills:
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": format_tool_output(
                                {"error": f"Unknown skill: {skill_name}"}
                            ),
                        }
                    )
                    continue

                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": self.skills[skill_name]["content"],
                    }
                )

        return observations
