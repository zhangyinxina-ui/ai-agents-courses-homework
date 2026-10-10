"""Unit and MCP Protocol-Level Integration Tests for CS146S Week 2 MCP Server."""

from __future__ import annotations

import hashlib
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week2.oauth_client import (
    CachedTokenSet,
    GitHubOAuthManager,
    OAuthToolError,
    generate_pkce_challenge,
)
from week2.server import (
    GitHubWorkflowService,
    MCPServerRuntime,
)


def test_pkce_s256_generation_and_code_exchange(tmp_path: Path) -> None:
    """Verify RFC 7636 S256 PKCE code challenge, code exchange, and 0600 file caching."""
    pkce = generate_pkce_challenge()
    expected_digest = hashlib.sha256(pkce.code_verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
    assert pkce.code_challenge == expected_challenge
    assert pkce.code_challenge_method == "S256"

    recorded_posts: list[tuple[str, dict[str, str]]] = []

    def fake_http_post(url: str, payload: dict[str, str]) -> dict[str, Any]:
        recorded_posts.append((url, payload))
        return {
            "access_token": "gho_initial_token_123",
            "refresh_token": "ghr_refresh_token_456",
            "token_type": "Bearer",
            "scope": "repo:status public_repo",
            "expires_in": 3600,
        }

    cache_file = tmp_path / "token_cache.json"
    mgr = GitHubOAuthManager(
        client_id="cid_test",
        client_secret="csecret_test",
        cache_path=cache_file,
        http_post_fn=fake_http_post,
        clock_fn=lambda: 1000.0,
    )

    auth_url = mgr.build_authorization_url(pkce, write_access=True)
    assert "code_challenge=" + pkce.code_challenge in auth_url
    assert "code_challenge_method=S256" in auth_url

    token_set = mgr.exchange_code_for_token("auth_code_xyz", pkce, received_state=pkce.state)
    assert token_set.access_token == "gho_initial_token_123"
    assert cache_file.is_file()
    assert (cache_file.stat().st_mode & 0o777) == 0o600
    assert recorded_posts[0][1]["code_verifier"] == pkce.code_verifier


def test_oauth_silent_refresh_and_mid_session_expiry_without_browser(tmp_path: Path) -> None:
    """Verify automatic silent refresh on expiry and actionable error (never opening a browser) when refresh fails."""
    cache_file = tmp_path / "token_cache.json"
    current_time = [1000.0]
    should_fail_refresh = [False]

    def fake_http_post(url: str, payload: dict[str, str]) -> dict[str, Any]:
        assert payload["grant_type"] == "refresh_token"
        if should_fail_refresh[0]:
            return {"error": "bad_refresh_token", "error_description": "The refresh token has been revoked."}
        return {
            "access_token": "gho_refreshed_token_999",
            "refresh_token": "ghr_new_refresh_888",
            "token_type": "Bearer",
            "scope": "repo:status public_repo",
            "expires_in": 7200,
        }

    mgr = GitHubOAuthManager(
        client_id="cid_test",
        client_secret="csecret_test",
        cache_path=cache_file,
        http_post_fn=fake_http_post,
        clock_fn=lambda: current_time[0],
    )
    mgr._save_cache(
        CachedTokenSet(
            access_token="gho_soon_expiring",
            refresh_token="ghr_old_refresh",
            token_type="Bearer",
            scope="repo:status public_repo",
            expires_at=1030.0,  # Within 60s margin at t=1000.0 -> triggers silent refresh
            obtained_at=500.0,
        )
    )

    # 1. Silent refresh succeeds transparently
    tok = mgr.get_valid_access_token()
    assert tok == "gho_refreshed_token_999"

    # 2. Advance clock past new expiry (1000 + 7200 = 8200) and simulate mid-session revoked refresh token
    current_time[0] = 9000.0
    should_fail_refresh[0] = True

    service = GitHubWorkflowService(oauth_manager=mgr, offline_mock=False)
    runtime = MCPServerRuntime(service=service)
    result = runtime.invoke_tool("search_project_issues", {"repo": "stanford-oval/storm"})
    assert result["isError"] is True
    payload = result["structuredContent"]
    assert payload["error_code"] == "OAUTH_REAUTH_REQUIRED"
    assert payload["retryable"] is False
    assert "oauth_client.py --login" in payload["remediation"]


def test_end_to_end_four_tool_composition_and_dry_run_brake(tmp_path: Path) -> None:
    """Verify the 4 tools compose (search -> get_context -> draft_release_notes -> publish_issue_triage)."""
    cache_file = tmp_path / "token_cache.json"
    mgr = GitHubOAuthManager(client_id="cid", client_secret="sec", cache_path=cache_file)
    service = GitHubWorkflowService(oauth_manager=mgr, offline_mock=True)
    runtime = MCPServerRuntime(service=service)

    # Step 1: Search open issues
    r1 = runtime.invoke_tool(
        "search_project_issues",
        {"repo": "stanford-oval/storm", "state": "open", "severity": "any", "sort_by": "updated"},
    )
    assert r1["isError"] is False
    issues = r1["structuredContent"]["issues"]
    issue_ids = [i["issue_number"] for i in issues]
    assert 101 in issue_ids and 104 in issue_ids

    # Step 2: Get context for issue #101 discovered in Step 1
    r2 = runtime.invoke_tool("get_issue_context", {"repo": "stanford-oval/storm", "issue_number": 101})
    assert r2["isError"] is False
    ctx = r2["structuredContent"]
    assert ctx["issue_number"] == 101
    assert "release-ready" in ctx["suggested_triage"]["recommended_labels"]

    # Step 3: Draft release notes across open + closed issues [101, 104, 109]
    r3 = runtime.invoke_tool(
        "draft_release_notes",
        {
            "repo": "stanford-oval/storm",
            "milestone_or_tag": "v2.4.0",
            "issue_numbers": [101, 104, 109],
            "format_style": "keep-a-changelog",
        },
    )
    assert r3["isError"] is False
    notes = r3["structuredContent"]
    assert notes["counts"] == {"total": 3, "breaking_changes": 1, "enhancements": 1, "bug_fixes": 1}
    assert "### Breaking Changes" in notes["markdown_release_notes"]

    # Step 4a: Call write tool `publish_issue_triage` with default dry_run=True brake
    r4_preview = runtime.invoke_tool(
        "publish_issue_triage",
        {
            "repo": "stanford-oval/storm",
            "issue_number": 101,
            "add_labels": ["triaged", "release-ready"],
            "milestone": "v2.4.0",
            "comment_body": "Verified fix in PR #118; queued for v2.4.0 release notes.",
            "new_state": "closed",
        },
    )
    assert r4_preview["isError"] is False
    preview_data = r4_preview["structuredContent"]
    assert preview_data["dry_run"] is True
    assert preview_data["applied"] is False
    # Verify underlying issue state was NOT mutated during dry_run
    assert service._issues[("stanford-oval/storm", 101)]["state"] == "open"

    # Step 4b: Commit mutation with dry_run=False
    r4_commit = runtime.invoke_tool(
        "publish_issue_triage",
        {
            "repo": "stanford-oval/storm",
            "issue_number": 101,
            "add_labels": ["triaged", "release-ready"],
            "milestone": "v2.4.0",
            "comment_body": "Verified fix in PR #118; queued for v2.4.0 release notes.",
            "new_state": "closed",
            "dry_run": False,
        },
    )
    assert r4_commit["isError"] is False
    commit_data = r4_commit["structuredContent"]
    assert commit_data["applied"] is True
    assert commit_data["mutated_issue"]["state"] == "closed"
    assert "release-ready" in commit_data["mutated_issue"]["labels"]


def test_structured_errors_distinguish_non_retryable_vs_retryable(tmp_path: Path) -> None:
    """Verify errors are returned as actionable structured data with accurate `retryable` flags."""
    cache_file = tmp_path / "token_cache.json"
    mgr = GitHubOAuthManager(client_id="cid", client_secret="sec", cache_path=cache_file)
    service = GitHubWorkflowService(oauth_manager=mgr, offline_mock=True)
    runtime = MCPServerRuntime(service=service)

    # 1. Non-retryable bad ID error (404 equivalent)
    bad_id_res = runtime.invoke_tool("get_issue_context", {"repo": "stanford-oval/storm", "issue_number": 9999})
    assert bad_id_res["isError"] is True
    assert bad_id_res["structuredContent"]["error_code"] == "ISSUE_NOT_FOUND"
    assert bad_id_res["structuredContent"]["retryable"] is False

    # 2. Non-retryable schema validation error (invalid Literal value)
    bad_schema_res = runtime.invoke_tool(
        "search_project_issues",
        {"repo": "stanford-oval/storm", "state": "archived_invalid"},
    )
    assert bad_schema_res["isError"] is True
    assert bad_schema_res["structuredContent"]["error_code"] == "SCHEMA_VALIDATION_ERROR"
    assert bad_schema_res["structuredContent"]["retryable"] is False

    # 3. Retryable rate-limit error (429 equivalent with retry_after_seconds)
    service.simulate_rate_limit(retry_after_seconds=42)
    rate_limit_res = runtime.invoke_tool("search_project_issues", {"repo": "stanford-oval/storm"})
    assert rate_limit_res["isError"] is True
    assert rate_limit_res["structuredContent"]["error_code"] == "GITHUB_RATE_LIMITED"
    assert rate_limit_res["structuredContent"]["retryable"] is True
    assert rate_limit_res["structuredContent"]["retry_after_seconds"] == 42


def test_mcp_protocol_over_stdio_subprocess() -> None:
    """Exercise the server end-to-end through the MCP JSON-RPC 2.0 stdio transport."""
    server_script = Path(__file__).resolve().parent / "server.py"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_project_issues",
                "arguments": {"repo": "stanford-oval/storm", "state": "open", "severity": "high"},
            },
        },
    ]
    stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, str(server_script)],
        input=stdin_payload,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    lines = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
    assert len(lines) == 3  # notifications/initialized produces no response

    # Response 1: initialize
    assert lines[0]["id"] == 1
    assert lines[0]["result"]["serverInfo"]["name"] == "github-release-workflow-mcp"

    # Response 2: tools/list
    assert lines[1]["id"] == 2
    tool_names = [t["name"] for t in lines[1]["result"]["tools"]]
    assert tool_names == [
        "search_project_issues",
        "get_issue_context",
        "draft_release_notes",
        "publish_issue_triage",
    ]

    # Response 3: tools/call
    assert lines[2]["id"] == 3
    call_res = lines[2]["result"]
    assert call_res["isError"] is False
    assert call_res["structuredContent"]["issues"][0]["issue_number"] == 101
