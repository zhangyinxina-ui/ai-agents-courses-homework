#!/usr/bin/env python3
"""
Chapter 15: Agent Interoperability Protocols — MCP & A2A (智能体通信与互联协议：MCP与A2A)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 4: Frontiers

Implements:
1. Model Context Protocol (`MCPServer`) JSON-RPC 2.0 Lifecycle:
   - `initialize` (protocol version & capability negotiation)
   - `tools/list` (discovering structured tool schemas)
   - `tools/call` (executing tools and returning structured content blocks)
   - `resources/read` (reading URI-addressed context resources)
2. Agent-to-Agent Protocol (`A2APeerAgent`, `A2ARouter`):
   - `AgentCard` capability discovery (`/.well-known/agent.json` semantics)
   - `tasks/send` asynchronous/synchronous task delegation between peer agents, where an
     A2A specialist agent internally invokes an MCP server to fulfill a delegated task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class MCPServer:
    """
    Standards-compliant Model Context Protocol (MCP) JSON-RPC 2.0 Server implementation
    supporting `initialize`, `tools/list`, `tools/call`, and `resources/read`.
    """

    PROTOCOL_VERSION = "2025-03-26"

    def __init__(self, server_name: str = "sjtu-systems-mcp-server") -> None:
        self.server_name = server_name
        self.initialized = False
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._resources: Dict[str, str] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }

    def register_resource(self, uri: str, content: str) -> None:
        self._resources[uri] = content

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        if request.get("jsonrpc") != "2.0":
            return self._error(req_id, -32600, "Invalid JSON-RPC version; expected '2.0'.")

        if method == "initialize":
            self.initialized = True
            return self._result(
                req_id,
                {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "serverInfo": {"name": self.server_name, "version": "1.0.0"},
                    "capabilities": {"tools": {}, "resources": {}},
                },
            )

        if not self.initialized:
            return self._error(req_id, -32002, "Server not initialized. Call 'initialize' first.")

        if method == "tools/list":
            tool_list = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                }
                for t in self._tools.values()
            ]
            return self._result(req_id, {"tools": tool_list})

        if method == "tools/call":
            t_name = params.get("name", "")
            t_args = params.get("arguments", {}) or {}
            if t_name not in self._tools:
                return self._error(req_id, -32601, f"MCP tool '{t_name}' not found.")
            try:
                out = self._tools[t_name]["handler"](t_args)
                return self._result(
                    req_id,
                    {
                        "content": [{"type": "text", "text": str(out)}],
                        "isError": False,
                    },
                )
            except Exception as exc:
                return self._result(
                    req_id,
                    {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                )

        if method == "resources/read":
            uri = params.get("uri", "")
            if uri not in self._resources:
                return self._error(req_id, -32602, f"Resource URI '{uri}' not found.")
            return self._result(
                req_id,
                {
                    "contents": [
                        {"uri": uri, "mimeType": "text/plain", "text": self._resources[uri]}
                    ]
                },
            )

        return self._error(req_id, -32601, f"Method '{method}' not found.")

    @staticmethod
    def _result(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


@dataclass(frozen=True)
class AgentCard:
    """A2A AgentCard metadata advertised for peer agent capability discovery."""
    agent_id: str
    name: str
    description: str
    skills: List[str]
    endpoint: str


class A2APeerAgent:
    """
    An Agent-to-Agent (A2A) specialist agent that exposes an `AgentCard` and handles
    `tasks/send` JSON-RPC requests by delegating tool execution to its bound `MCPServer`.
    """

    def __init__(self, card: AgentCard, mcp_server: MCPServer) -> None:
        self.card = card
        self.mcp = mcp_server

    def handle_a2a_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        if method == "agent/card":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "agent_id": self.card.agent_id,
                    "name": self.card.name,
                    "skills": self.card.skills,
                    "endpoint": self.card.endpoint,
                },
            }

        if method == "tasks/send":
            task_id = params.get("task_id", "task-001")
            skill = params.get("skill", "")
            payload = params.get("input", {}) or {}

            # Ensure internal MCP client session is initialized
            if not self.mcp.initialized:
                self.mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 0, "method": "initialize"})

            mcp_resp = self.mcp.handle_jsonrpc(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "tools/call",
                    "params": {"name": skill, "arguments": payload},
                }
            )
            content_blocks = mcp_resp.get("result", {}).get("content", [])
            artifact_text = content_blocks[0]["text"] if content_blocks else ""

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "task_id": task_id,
                    "status": "COMPLETED",
                    "responder_agent": self.card.agent_id,
                    "artifact": artifact_text,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unsupported A2A method '{method}'"},
        }


class A2ARouter:
    """Registry and message router for discovering and invoking A2A peer agents by skill."""

    def __init__(self) -> None:
        self.agents: Dict[str, A2APeerAgent] = {}

    def register_agent(self, agent: A2APeerAgent) -> None:
        self.agents[agent.card.agent_id] = agent

    def discover_by_skill(self, required_skill: str) -> Optional[AgentCard]:
        for agent in self.agents.values():
            if required_skill in agent.card.skills:
                return agent.card
        return None

    def send_task(
        self, agent_id: str, task_id: str, skill: str, task_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        if agent_id not in self.agents:
            raise KeyError(f"Unknown A2A agent '{agent_id}'")
        return self.agents[agent_id].handle_a2a_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": task_id,
                "method": "tasks/send",
                "params": {"task_id": task_id, "skill": skill, "input": task_input},
            }
        )


def run_chapter_demo() -> Dict[str, Any]:
    # 1. Configure an MCP Server with a tool and a URI resource
    mcp = MCPServer("rtl-synthesis-mcp")
    mcp.register_tool(
        name="estimate_ppa",
        description="Estimates area and delay for an RTL adder bitwidth.",
        input_schema={"type": "object", "properties": {"bits": {"type": "integer"}}, "required": ["bits"]},
        handler=lambda args: f"PPA(bits={args['bits']}): Area={args['bits']*4.2:.1f}um2, Delay={0.18 + 0.01*args['bits']:.2f}ns",
    )
    mcp.register_resource("mcp://specs/alu_v1", "ALU Specification: 64-bit RISC-V RV64I")

    # Initialize MCP and list tools
    init_resp = mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    tools_resp = mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    res_resp = mcp.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "mcp://specs/alu_v1"}}
    )

    # 2. Wrap MCP Server inside an A2A Specialist Agent and route a task via A2ARouter
    card = AgentCard(
        agent_id="hw-eda-specialist",
        name="Hardware PPA Specialist Agent",
        description="Evaluates RTL PPA metrics via Yosys/OpenROAD MCP tools",
        skills=["estimate_ppa"],
        endpoint="a2a://agents/hw-eda-specialist",
    )
    peer = A2APeerAgent(card=card, mcp_server=mcp)
    router = A2ARouter()
    router.register_agent(peer)

    discovered = router.discover_by_skill("estimate_ppa")
    assert discovered is not None
    a2a_resp = router.send_task(
        agent_id=discovered.agent_id,
        task_id="T-900",
        skill="estimate_ppa",
        task_input={"bits": 64},
    )

    return {
        "mcp_protocol_version": init_resp["result"]["protocolVersion"],
        "mcp_listed_tools": [t["name"] for t in tools_resp["result"]["tools"]],
        "mcp_resource_text": res_resp["result"]["contents"][0]["text"],
        "a2a_discovered_agent": discovered.agent_id,
        "a2a_task_status": a2a_resp["result"]["status"],
        "a2a_artifact": a2a_resp["result"]["artifact"],
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 15 Demo] MCP Tools={res['mcp_listed_tools']} | A2A Agent='{res['a2a_discovered_agent']}' -> Artifact='{res['a2a_artifact']}'")
