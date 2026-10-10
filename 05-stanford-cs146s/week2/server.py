"""GitHub Issues & Release Workflow MCP Server (CS146S Week 2).

Exposes 4 composing tools designed for agent ergonomics:
1. `search_project_issues` (read-only): Search issues by state/severity/label -> returns `issue_number` IDs.
2. `get_issue_context` (read-only): Fetch shaped issue details, comments, and linked PRs for an `issue_number`.
3. `draft_release_notes` (read-only): Aggregate multiple `issue_numbers` into categorized release notes.
4. `publish_issue_triage` (write tool with `dry_run: bool = True` brake): Apply labels, milestone, and
   triage/release comment to an `issue_number` after previewing the mutation when `dry_run=True`.

Supports both offline deterministic mock mode (`GITHUB_MCP_OFFLINE_MOCK=1`) for reproducible testing
and live GitHub REST API v3 calls authenticated via `GitHubOAuthManager`.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from week2.oauth_client import CachedTokenSet, GitHubOAuthManager, OAuthToolError
except ImportError:
    from oauth_client import CachedTokenSet, GitHubOAuthManager, OAuthToolError


SERVER_NAME = "github-release-workflow-mcp"
SERVER_VERSION = "1.0.0"
SERVER_INSTRUCTIONS = (
    "Workflow guidance for GitHub Issues & Release Triage:\n"
    "1. Start with `search_project_issues(repo, state, severity, sort_by)` to discover relevant `issue_number` integers.\n"
    "2. Pass each candidate `issue_number` into `get_issue_context(repo, issue_number)` to inspect root-cause comments, "
    "linked pull requests, and suggested triage labels.\n"
    "3. To prepare a release summary, pass the verified `issue_numbers` list into `draft_release_notes(repo, milestone_or_tag, issue_numbers)`.\n"
    "4. Before mutating any issue with `publish_issue_triage`, ALWAYS call it first with `dry_run=True` (the default) "
    "to inspect `planned_mutations`, and only set `dry_run=False` after verifying the preview."
)


# ---------------------------------------------------------------------------
# Strict Pydantic Input Schemas (Constraints in the Schema, Not Just Prose)
# ---------------------------------------------------------------------------


class SearchIssuesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
        description="Target GitHub repository in 'owner/name' format (e.g. 'stanford-oval/storm').",
    )
    state: Literal["open", "closed", "all"] = Field(
        default="open",
        description="Filter issues by lifecycle state.",
    )
    severity: Literal["critical", "high", "medium", "low", "any"] = Field(
        default="any",
        description="Filter by severity label tier ('any' disables severity filtering).",
    )
    query: str = Field(
        default="",
        max_length=200,
        description="Optional substring keyword to match against issue title or body.",
    )
    sort_by: Literal["updated", "created", "comments"] = Field(
        default="updated",
        description="Sort order for returned issue summaries.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of shaped issue summaries to return (1-50).",
    )


class GetIssueContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
        description="Target GitHub repository in 'owner/name' format.",
    )
    issue_number: int = Field(
        ge=1,
        description="The positive integer `issue_number` returned by `search_project_issues`.",
    )


class DraftReleaseNotesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
        description="Target GitHub repository in 'owner/name' format.",
    )
    milestone_or_tag: str = Field(
        min_length=2,
        max_length=64,
        description="Target release version tag or milestone title (e.g. 'v2.4.0').",
    )
    issue_numbers: list[int] = Field(
        min_length=1,
        max_length=30,
        description="List of integer `issue_number` IDs obtained from `search_project_issues` / `get_issue_context`.",
    )
    format_style: Literal["keep-a-changelog", "executive-summary", "markdown-bullets"] = Field(
        default="keep-a-changelog",
        description="Output formatting template for the synthesized release notes.",
    )


class PublishIssueTriageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
        description="Target GitHub repository in 'owner/name' format.",
    )
    issue_number: int = Field(
        ge=1,
        description="The integer `issue_number` from `search_project_issues` / `get_issue_context` to update.",
    )
    add_labels: list[Literal["bug", "enhancement", "security", "breaking-change", "triaged", "release-ready"]] = Field(
        min_length=1,
        description="Canonical triage labels to attach to the issue.",
    )
    milestone: str | None = Field(
        default=None,
        description="Optional milestone title (e.g. 'v2.4.0') to assign.",
    )
    comment_body: str = Field(
        min_length=10,
        max_length=4000,
        description="Structured triage or release-note verification comment to post on the issue.",
    )
    new_state: Literal["keep_current", "open", "closed"] = Field(
        default="keep_current",
        description="Whether to transition the issue state or keep its current state.",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "Safety brake: when True (default), computes and returns `planned_mutations` without modifying "
            "GitHub state. Set to False ONLY after inspecting the dry_run preview."
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic Mock Store (Used when GITHUB_MCP_OFFLINE_MOCK=1)
# ---------------------------------------------------------------------------

DEFAULT_MOCK_ISSUES: dict[tuple[str, int], dict[str, Any]] = {
    ("stanford-oval/storm", 101): {
        "issue_number": 101,
        "title": "Fix citation deduplication crash when two snippets share identical URLs",
        "state": "open",
        "severity": "high",
        "labels": ["bug", "severity:high"],
        "author": "alice-dev",
        "created_at": "2026-10-01T10:15:00Z",
        "updated_at": "2026-10-08T14:20:00Z",
        "comments_count": 2,
        "milestone": "v2.4.0",
        "body": (
            "When `StormArticleGeneration` merges citations from two perspective agents that retrieved the same URL, "
            "`citation_index[url]` raises a `KeyError` if normalization strips query parameters."
        ),
        "comments": [
            {"author": "bob-maint", "body": "Reproduced on main. Root cause is in `src/utils/citations.py:L84`."},
            {"author": "alice-dev", "body": "Opened PR #118 with URL canonicalization fix and regression test."},
        ],
        "linked_pull_requests": [{"pr_number": 118, "title": "Canonicalize URLs before citation merge", "merged": True}],
    },
    ("stanford-oval/storm", 104): {
        "issue_number": 104,
        "title": "Add token-budget cutoff telemetry tolit-search retriever loop",
        "state": "open",
        "severity": "medium",
        "labels": ["enhancement", "severity:medium"],
        "author": "carol-ml",
        "created_at": "2026-10-03T09:00:00Z",
        "updated_at": "2026-10-09T11:30:00Z",
        "comments_count": 1,
        "milestone": "v2.4.0",
        "body": "Expose per-hop token consumption and search API latency in `run_metadata.json` for cost auditing.",
        "comments": [
            {"author": "bob-maint", "body": "Merged in PR #121. Ready for v2.4.0 release notes."},
        ],
        "linked_pull_requests": [{"pr_number": 121, "title": "Add retriever telemetry metrics", "merged": True}],
    },
    ("stanford-oval/storm", 109): {
        "issue_number": 109,
        "title": "Drop deprecated legacy v1 prompt template configuration keys",
        "state": "closed",
        "severity": "critical",
        "labels": ["breaking-change", "severity:critical"],
        "author": "bob-maint",
        "created_at": "2026-09-28T16:45:00Z",
        "updated_at": "2026-10-07T18:05:00Z",
        "comments_count": 1,
        "milestone": "v2.4.0",
        "body": "Removes `legacy_v1_prompt_path` from `StormConfig`. Callers must use `prompt_template_registry`.",
        "comments": [
            {"author": "bob-maint", "body": "Breaking change documented in migration guide; landed in PR #115."},
        ],
        "linked_pull_requests": [{"pr_number": 115, "title": "Remove legacy_v1_prompt_path", "merged": True}],
    },
}


class GitHubWorkflowService:
    """Backend service backing the 4 MCP tools with OAuth enforcement and shaped outputs."""

    def __init__(
        self,
        oauth_manager: GitHubOAuthManager | None = None,
        *,
        offline_mock: bool | None = None,
    ) -> None:
        if offline_mock is None:
            offline_mock = os.environ.get("GITHUB_MCP_OFFLINE_MOCK", "1") == "1"
        self.offline_mock = offline_mock
        self.oauth_manager = oauth_manager or GitHubOAuthManager(
            client_id=os.environ.get("GITHUB_OAUTH_CLIENT_ID", "mock-client-id"),
            client_secret=os.environ.get("GITHUB_OAUTH_CLIENT_SECRET", "mock-client-secret"),
        )
        self._issues: dict[tuple[str, int], dict[str, Any]] = copy.deepcopy(DEFAULT_MOCK_ISSUES)
        self._simulated_rate_limit_remaining: int = 0

    def simulate_rate_limit(self, retry_after_seconds: int = 42) -> None:
        """Test helper to provoke a structured retryable HTTP 429 rate-limit response."""
        self._simulated_rate_limit_remaining = retry_after_seconds

    def _check_auth_and_rate_limit(self) -> str:
        if self._simulated_rate_limit_remaining > 0:
            wait_s = self._simulated_rate_limit_remaining
            self._simulated_rate_limit_remaining = 0
            raise OAuthToolError(
                error_code="GITHUB_RATE_LIMITED",
                message=f"GitHub API secondary rate limit exceeded. Retry after {wait_s}s.",
                retryable=True,
                retry_after_seconds=wait_s,
                remediation=f"Transient rate limit: wait {wait_s} seconds and retry the same tool call.",
            )
        if self.offline_mock and not self.oauth_manager.cache_path.is_file():
            # Provision a default valid token in mock mode if none exists yet
            now = time.time()
            self.oauth_manager._save_cache(
                CachedTokenSet(
                    access_token="gho_mock_valid_access_token",
                    refresh_token="ghr_mock_valid_refresh_token",
                    token_type="Bearer",
                    scope="repo:status public_repo",
                    expires_at=now + 3600,
                    obtained_at=now,
                )
            )
        return self.oauth_manager.get_valid_access_token()

    def search_project_issues(self, params: SearchIssuesInput) -> dict[str, Any]:
        """Search repository issues and return shaped summaries with `issue_number` IDs."""
        self._check_auth_and_rate_limit()
        matches: list[dict[str, Any]] = []
        q_lower = params.query.strip().lower()

        for (repo_name, _), issue in self._issues.items():
            if repo_name != params.repo:
                continue
            if params.state != "all" and issue["state"] != params.state:
                continue
            if params.severity != "any" and issue.get("severity") != params.severity:
                continue
            if q_lower and q_lower not in issue["title"].lower() and q_lower not in issue["body"].lower():
                continue
            matches.append(
                {
                    "issue_number": issue["issue_number"],
                    "title": issue["title"],
                    "state": issue["state"],
                    "severity": issue.get("severity", "unspecified"),
                    "labels": list(issue["labels"]),
                    "milestone": issue.get("milestone"),
                    "comments_count": issue["comments_count"],
                    "updated_at": issue["updated_at"],
                }
            )

        sort_key = {
            "updated": lambda x: x["updated_at"],
            "created": lambda x: x["issue_number"],
            "comments": lambda x: x["comments_count"],
        }[params.sort_by]
        matches.sort(key=sort_key, reverse=True)
        sliced = matches[: params.limit]

        return {
            "ok": True,
            "repo": params.repo,
            "total_matched": len(sliced),
            "issues": sliced,
            "next_step_hint": (
                "Pass any `issue_number` from `issues` into `get_issue_context` to inspect full details, "
                "or pass multiple `issue_number` values into `draft_release_notes`."
            ),
        }

    def get_issue_context(self, params: GetIssueContextInput) -> dict[str, Any]:
        """Return shaped context for a single `issue_number` discovered via `search_project_issues`."""
        self._check_auth_and_rate_limit()
        key = (params.repo, params.issue_number)
        issue = self._issues.get(key)
        if issue is None:
            available = sorted(num for (r, num) in self._issues if r == params.repo)
            return {
                "ok": False,
                "error_code": "ISSUE_NOT_FOUND",
                "error": f"Issue #{params.issue_number} does not exist in repository '{params.repo}'.",
                "retryable": False,
                "remediation": (
                    f"Do NOT retry with issue_number={params.issue_number}. "
                    f"Call `search_project_issues(repo='{params.repo}', state='all')` to find valid issue numbers "
                    f"(known IDs in '{params.repo}': {available})."
                ),
            }

        suggested_labels: list[str] = ["triaged"]
        if "bug" in issue["labels"]:
            suggested_labels.append("bug")
        if any(pr.get("merged") for pr in issue.get("linked_pull_requests", [])):
            suggested_labels.append("release-ready")

        return {
            "ok": True,
            "repo": params.repo,
            "issue_number": issue["issue_number"],
            "title": issue["title"],
            "state": issue["state"],
            "severity": issue.get("severity", "unspecified"),
            "labels": list(issue["labels"]),
            "milestone": issue.get("milestone"),
            "author": issue["author"],
            "body": issue["body"],
            "recent_comments": list(issue["comments"]),
            "linked_pull_requests": list(issue["linked_pull_requests"]),
            "suggested_triage": {
                "recommended_labels": suggested_labels,
                "recommended_milestone": issue.get("milestone") or "v2.4.0",
            },
        }

    def draft_release_notes(self, params: DraftReleaseNotesInput) -> dict[str, Any]:
        """Compose structured release notes from a list of `issue_numbers`."""
        self._check_auth_and_rate_limit()
        resolved: list[dict[str, Any]] = []
        missing: list[int] = []
        for num in params.issue_numbers:
            issue = self._issues.get((params.repo, num))
            if issue is None:
                missing.append(num)
            else:
                resolved.append(issue)

        if missing:
            return {
                "ok": False,
                "error_code": "INVALID_ISSUE_NUMBERS",
                "error": f"Cannot draft release notes: issue_numbers {missing} not found in '{params.repo}'.",
                "retryable": False,
                "remediation": (
                    "Remove the invalid issue numbers and pass only `issue_number` values returned by "
                    "`search_project_issues`."
                ),
            }

        breaking = [i for i in resolved if "breaking-change" in i["labels"]]
        bugs = [i for i in resolved if "bug" in i["labels"] and "breaking-change" not in i["labels"]]
        enhancements = [
            i for i in resolved if "bug" not in i["labels"] and "breaking-change" not in i["labels"]
        ]

        if params.format_style == "keep-a-changelog":
            lines = [f"## [{params.milestone_or_tag}] - {time.strftime('%Y-%m-%d')}"]
            if breaking:
                lines.append("\n### Breaking Changes")
                for i in breaking:
                    lines.append(f"- **#{i['issue_number']}**: {i['title']} (@{i['author']})")
            if enhancements:
                lines.append("\n### Added & Improved")
                for i in enhancements:
                    lines.append(f"- **#{i['issue_number']}**: {i['title']} (@{i['author']})")
            if bugs:
                lines.append("\n### Fixed")
                for i in bugs:
                    lines.append(f"- **#{i['issue_number']}**: {i['title']} (@{i['author']})")
            markdown = "\n".join(lines)
        elif params.format_style == "executive-summary":
            markdown = (
                f"# Release {params.milestone_or_tag} Summary ({params.repo})\n"
                f"Includes {len(resolved)} resolved items ({len(breaking)} breaking, "
                f"{len(enhancements)} enhancements, {len(bugs)} bug fixes):\n"
                + "\n".join(f"- #{i['issue_number']}: {i['title']}" for i in resolved)
            )
        else:
            markdown = "\n".join(f"* #{i['issue_number']} ({i['state']}): {i['title']}" for i in resolved)

        return {
            "ok": True,
            "repo": params.repo,
            "milestone_or_tag": params.milestone_or_tag,
            "format_style": params.format_style,
            "included_issue_numbers": [i["issue_number"] for i in resolved],
            "counts": {
                "total": len(resolved),
                "breaking_changes": len(breaking),
                "enhancements": len(enhancements),
                "bug_fixes": len(bugs),
            },
            "markdown_release_notes": markdown,
            "next_step_hint": (
                "To record triage/release readiness on an issue, call `publish_issue_triage` with `dry_run=True` "
                "first to preview the mutation."
            ),
        }

    def publish_issue_triage(self, params: PublishIssueTriageInput) -> dict[str, Any]:
        """Mutate issue labels, milestone, state, and comments (guarded by `dry_run: bool = True`)."""
        self._check_auth_and_rate_limit()
        key = (params.repo, params.issue_number)
        issue = self._issues.get(key)
        if issue is None:
            return {
                "ok": False,
                "error_code": "ISSUE_NOT_FOUND",
                "error": f"Cannot triage non-existent issue #{params.issue_number} in '{params.repo}'.",
                "retryable": False,
                "remediation": "Verify `issue_number` using `search_project_issues` before calling `publish_issue_triage`.",
            }

        existing_labels = list(issue["labels"])
        merged_labels = sorted(set(existing_labels) | set(params.add_labels))
        target_milestone = params.milestone if params.milestone is not None else issue.get("milestone")
        target_state = issue["state"] if params.new_state == "keep_current" else params.new_state

        planned_mutations = {
            "repo": params.repo,
            "issue_number": params.issue_number,
            "labels_before": existing_labels,
            "labels_after": merged_labels,
            "milestone_before": issue.get("milestone"),
            "milestone_after": target_milestone,
            "state_before": issue["state"],
            "state_after": target_state,
            "comment_to_append": params.comment_body,
        }

        if params.dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "applied": False,
                "planned_mutations": planned_mutations,
                "commit_instruction": (
                    "No state was modified because `dry_run=True`. Review `planned_mutations` and re-invoke "
                    "`publish_issue_triage` with the same arguments plus `dry_run=False` to commit."
                ),
            }

        # Apply real mutation when dry_run=False
        issue["labels"] = merged_labels
        issue["milestone"] = target_milestone
        issue["state"] = target_state
        issue["comments"].append({"author": "mcp-release-bot", "body": params.comment_body})
        issue["comments_count"] = len(issue["comments"])
        issue["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return {
            "ok": True,
            "dry_run": False,
            "applied": True,
            "mutated_issue": {
                "repo": params.repo,
                "issue_number": issue["issue_number"],
                "state": issue["state"],
                "labels": list(issue["labels"]),
                "milestone": issue.get("milestone"),
                "comments_count": issue["comments_count"],
                "latest_comment": issue["comments"][-1],
            },
        }


# ---------------------------------------------------------------------------
# Tool Definitions & Protocol Dispatcher (MCP JSON-RPC 2.0 over stdio)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_project_issues",
        "description": (
            "Search GitHub issues in a repository by state, severity tier, and keyword. "
            "Returns shaped issue summaries containing the integer `issue_number` required by "
            "`get_issue_context`, `draft_release_notes`, and `publish_issue_triage`."
        ),
        "inputSchema": SearchIssuesInput.model_json_schema(),
        "annotations": {
            "title": "Search Project Issues",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "name": "get_issue_context",
        "description": (
            "Fetch full context (description, root-cause comments, linked pull requests, and suggested triage labels) "
            "for a single `issue_number` obtained from `search_project_issues`."
        ),
        "inputSchema": GetIssueContextInput.model_json_schema(),
        "annotations": {
            "title": "Get Issue Context",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "name": "draft_release_notes",
        "description": (
            "Generate structured release notes for a milestone or tag from a list of integer `issue_numbers` "
            "discovered via `search_project_issues` and verified via `get_issue_context`."
        ),
        "inputSchema": DraftReleaseNotesInput.model_json_schema(),
        "annotations": {
            "title": "Draft Release Notes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "publish_issue_triage",
        "description": (
            "Write tool: attach canonical triage labels, assign a milestone, post a triage/release comment, "
            "and optionally close/reopen an `issue_number` from `search_project_issues`. "
            "Defaults to `dry_run=True` (preview mode); set `dry_run=False` only after reviewing `planned_mutations`."
        ),
        "inputSchema": PublishIssueTriageInput.model_json_schema(),
        "annotations": {
            "title": "Publish Issue Triage (Write Tool with Dry-Run Brake)",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
]


class MCPServerRuntime:
    """Protocol-level MCP JSON-RPC 2.0 server supporting stdio and in-memory test transport."""

    def __init__(self, service: GitHubWorkflowService | None = None) -> None:
        self.service = service or GitHubWorkflowService()

    def invoke_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Validate arguments with Pydantic, invoke the tool, and return structured MCP tool result."""
        args = arguments or {}
        try:
            if name == "search_project_issues":
                parsed = SearchIssuesInput.model_validate(args)
                data = self.service.search_project_issues(parsed)
            elif name == "get_issue_context":
                parsed = GetIssueContextInput.model_validate(args)
                data = self.service.get_issue_context(parsed)
            elif name == "draft_release_notes":
                parsed = DraftReleaseNotesInput.model_validate(args)
                data = self.service.draft_release_notes(parsed)
            elif name == "publish_issue_triage":
                parsed = PublishIssueTriageInput.model_validate(args)
                data = self.service.publish_issue_triage(parsed)
            else:
                data = {
                    "ok": False,
                    "error_code": "UNKNOWN_TOOL",
                    "error": f"Unknown tool '{name}'. Available tools: {[t['name'] for t in TOOL_DEFINITIONS]}",
                    "retryable": False,
                    "remediation": "Select one of the registered tool names from tools/list.",
                }
        except ValidationError as exc:
            data = {
                "ok": False,
                "error_code": "SCHEMA_VALIDATION_ERROR",
                "error": "Tool input arguments failed schema validation.",
                "validation_errors": [
                    {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"], "type": err["type"]}
                    for err in exc.errors()
                ],
                "retryable": False,
                "remediation": "Fix the invalid field values according to the tool's JSON Schema and re-call.",
            }
        except OAuthToolError as exc:
            data = exc.to_dict()

        is_error = not data.get("ok", True)
        return {
            "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
            "structuredContent": data,
            "isError": is_error,
        }

    def handle_jsonrpc_message(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single MCP JSON-RPC 2.0 message and return the response envelope (or None for notifications)."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if req_id is None and method.startswith("notifications/"):
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": SERVER_INSTRUCTIONS,
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOL_DEFINITIONS},
            }
        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments") or {}
            result = self.invoke_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    def serve_stdio(self) -> None:
        """Run the MCP server over standard input/output (line-delimited JSON-RPC 2.0)."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
                continue

            resp = self.handle_jsonrpc_message(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    runtime = MCPServerRuntime()
    runtime.serve_stdio()


if __name__ == "__main__":
    main()
