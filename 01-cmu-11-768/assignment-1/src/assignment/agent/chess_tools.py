"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    try:
        if not isinstance(arguments, str):
            raise ValueError("Arguments must be a JSON string.")
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")

        allowed_keys = {"fen", "move"}
        extra_keys = set(args.keys()) - allowed_keys
        if extra_keys:
            raise ValueError(f"Unexpected arguments: {', '.join(sorted(extra_keys))}")
        if "fen" not in args:
            raise ValueError("Missing required argument: fen.")

        fen = args["fen"]
        if not isinstance(fen, str) or not fen.strip():
            raise ValueError("Argument 'fen' must be a non-empty string.")

        move = args.get("move", None)
        if move is not None and (not isinstance(move, str) or not move.strip()):
            raise ValueError("Argument 'move' must be a non-empty string or null.")

        payload: dict[str, Any] = {"fen": fen}
        if move is not None:
            payload["move"] = move
        state = _request_state(client, "POST", "/api/simulate", json=payload)
        return json.dumps(state)
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    try:
        if not isinstance(arguments, str):
            raise ValueError("Arguments must be a JSON string.")
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")

        extra_keys = set(args.keys()) - {"move"}
        if extra_keys:
            raise ValueError(f"Unexpected arguments: {', '.join(sorted(extra_keys))}")
        if "move" not in args:
            raise ValueError("Missing required argument: move.")

        move = args["move"]
        if not isinstance(move, str) or not move.strip():
            raise ValueError("Argument 'move' must be a non-empty string.")

        state = _request_state(client, "POST", "/api/move", json={"move": move})
        return json.dumps(state)
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    try:
        if not isinstance(arguments, str):
            raise ValueError("Arguments must be a JSON string.")
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")

        extra_keys = set(args.keys()) - {"code"}
        if extra_keys:
            raise ValueError(f"Unexpected arguments: {', '.join(sorted(extra_keys))}")
        if "code" not in args:
            raise ValueError("Missing required argument: code.")

        code = args["code"]
        if not isinstance(code, str):
            raise ValueError("Argument 'code' must be a string.")

        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        cmd = f"python /opt/assignment/sandbox_python.py {port} {encoded}"
        result = env.execute(cmd)
        if not isinstance(result, dict):
            raise RuntimeError("Sandbox execute did not return a result dictionary.")

        returncode = result.get("returncode", 0)
        if returncode != 0:
            err_detail = (
                result.get("exception_info")
                or result.get("stderr")
                or result.get("output")
                or f"Sandbox command failed with exit code {returncode}"
            )
            return f"<chess_error>{str(err_detail).strip()}</chess_error>"

        raw_out = (
            result["stdout"]
            if "stdout" in result and result["stdout"]
            else result.get("output", "")
        )
        if not isinstance(raw_out, str) or not raw_out.strip():
            raise RuntimeError("Sandbox runner produced empty output.")

        parsed_out = json.loads(raw_out)
        if not isinstance(parsed_out, dict) or not {"stdout", "stderr", "error"}.issubset(
            parsed_out.keys()
        ):
            raise RuntimeError("Sandbox runner output missing required keys.")
        return raw_out
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    try:
        if not isinstance(arguments, str):
            raise ValueError("Arguments must be a JSON string.")
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")

        extra_keys = set(args.keys()) - {"name"}
        if extra_keys:
            raise ValueError(f"Unexpected arguments: {', '.join(sorted(extra_keys))}")
        if "name" not in args:
            raise ValueError("Missing required argument: name.")

        name = args["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Argument 'name' must be a non-empty string.")

        skill_name = name.strip()
        if not skills or skill_name not in skills:
            raise ValueError(f"Unknown skill: {skill_name}")
        return skills[skill_name]["content"]
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
