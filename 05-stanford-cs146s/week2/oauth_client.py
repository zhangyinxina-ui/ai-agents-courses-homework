"""OAuth 2.0 Authorization Code + PKCE client with local caching and silent refresh.

Designed for long-running MCP servers (CS146S Week 2):
- Full Authorization Code + PKCE (RFC 7636 S256) generation and exchange.
- Minimal scope enforcement (`read:org`, `public_repo` / `issues:write`).
- Local file cache (`0600` permissions) with proactive silent refresh before expiry.
- Safe mid-session expiry handling: NEVER opens a browser during an active MCP tool call;
  instead raises `OAuthToolError` with structured remediation instructions for the agent/user.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_SCOPES: tuple[str, ...] = ("repo:status", "public_repo")
REFRESH_MARGIN_SECONDS: int = 60


class OAuthToolError(Exception):
    """Structured OAuth error designed to be serialized as data, never a raw traceback."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: int | None = None,
        remediation: str = "",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error_code": self.error_code,
            "error": self.message,
            "retryable": self.retryable,
            "remediation": self.remediation,
        }
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        return payload


@dataclass
class PKCEChallenge:
    """RFC 7636 PKCE verifier and S256 challenge pair."""

    code_verifier: str
    code_challenge: str
    code_challenge_method: str = "S256"
    state: str = field(default_factory=lambda: secrets.token_urlsafe(24))


@dataclass
class CachedTokenSet:
    """OAuth 2.0 token record persisted on disk with expiry metadata."""

    access_token: str
    refresh_token: str
    token_type: str
    scope: str
    expires_at: float
    obtained_at: float

    def is_expired(self, now: float | None = None, margin_seconds: int = REFRESH_MARGIN_SECONDS) -> bool:
        current = time.time() if now is None else now
        return current + margin_seconds >= self.expires_at

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "CachedTokenSet":
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data.get("refresh_token", "")),
            token_type=str(data.get("token_type", "Bearer")),
            scope=str(data.get("scope", "")),
            expires_at=float(data["expires_at"]),
            obtained_at=float(data.get("obtained_at", time.time())),
        )


def generate_pkce_challenge() -> PKCEChallenge:
    """Generate a cryptographically random RFC 7636 code_verifier and S256 code_challenge."""
    verifier_bytes = secrets.token_bytes(32)
    code_verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return PKCEChallenge(
        code_verifier=code_verifier,
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )


class GitHubOAuthManager:
    """Manages OAuth 2.0 token acquisition, local caching, and silent refresh."""

    AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str = "http://127.0.0.1:8765/callback",
        cache_path: Path | None = None,
        scopes: tuple[str, ...] = DEFAULT_SCOPES,
        http_post_fn: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        clock_fn: Callable[[], float] = time.time,
    ) -> None:
        self.client_id = client_id if client_id is not None else os.environ.get("GITHUB_OAUTH_CLIENT_ID", "")
        self.client_secret = (
            client_secret if client_secret is not None else os.environ.get("GITHUB_OAUTH_CLIENT_SECRET", "")
        )
        self.redirect_uri = redirect_uri
        default_cache = Path(
            os.environ.get("GITHUB_MCP_TOKEN_CACHE", str(Path.home() / ".cache" / "cs146s_github_mcp_token.json"))
        )
        self.cache_path = cache_path or default_cache
        self.scopes = scopes
        self._http_post_fn = http_post_fn or self._default_http_post
        self._clock = clock_fn

    def build_authorization_url(self, pkce: PKCEChallenge, *, write_access: bool = False) -> str:
        """Build the OAuth 2.0 authorization URL with minimal required scopes and PKCE S256."""
        if not self.client_id:
            raise OAuthToolError(
                error_code="OAUTH_CONFIG_MISSING",
                message="GITHUB_OAUTH_CLIENT_ID is not set in the environment.",
                retryable=False,
                remediation="Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET in .mcp.json env.",
            )
        requested_scopes = list(self.scopes) if write_access else ["repo:status"]
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(requested_scopes),
            "state": pkce.state,
            "response_type": "code",
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": pkce.code_challenge_method,
        }
        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(
        self,
        code: str,
        pkce: PKCEChallenge,
        received_state: str,
    ) -> CachedTokenSet:
        """Exchange an authorization code + PKCE code_verifier for an access/refresh token pair."""
        if not secrets.compare_digest(pkce.state, received_state):
            raise OAuthToolError(
                error_code="OAUTH_STATE_MISMATCH",
                message="OAuth callback state parameter did not match the initiated PKCE request.",
                retryable=False,
                remediation="Re-run 'python -m week2.oauth_client --login' to start a fresh OAuth handshake.",
            )
        if not self.client_id or not self.client_secret:
            raise OAuthToolError(
                error_code="OAUTH_CONFIG_MISSING",
                message="Missing GITHUB_OAUTH_CLIENT_ID or GITHUB_OAUTH_CLIENT_SECRET.",
                retryable=False,
                remediation="Provide OAuth client credentials via environment variables.",
            )

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": pkce.code_verifier,
            "grant_type": "authorization_code",
        }
        raw = self._http_post_fn(self.TOKEN_URL, payload)
        token_set = self._parse_token_response(raw)
        self._save_cache(token_set)
        return token_set

    def refresh_access_token(self, refresh_token: str) -> CachedTokenSet:
        """Silently refresh an expired access token using the cached refresh_token."""
        if not refresh_token:
            raise OAuthToolError(
                error_code="OAUTH_REAUTH_REQUIRED",
                message="Access token expired and no refresh_token is available for silent refresh.",
                retryable=False,
                remediation=(
                    "Do NOT retry this tool call. Ask the user to run "
                    "`python3 week2/oauth_client.py --login` in their terminal to re-authenticate."
                ),
            )
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        try:
            raw = self._http_post_fn(self.TOKEN_URL, payload)
        except OAuthToolError:
            raise
        except Exception as exc:
            raise OAuthToolError(
                error_code="OAUTH_REFRESH_FAILED",
                message=f"Silent OAuth token refresh failed: {exc}",
                retryable=False,
                remediation=(
                    "The refresh token was revoked or expired mid-session. The server never opens a browser "
                    "during a tool call. Run `python3 week2/oauth_client.py --login` out-of-band."
                ),
            ) from exc

        if "error" in raw:
            raise OAuthToolError(
                error_code="OAUTH_REAUTH_REQUIRED",
                message=f"OAuth provider rejected refresh_token ({raw.get('error')}): {raw.get('error_description', '')}",
                retryable=False,
                remediation=(
                    "Token died mid-session (refresh token revoked/expired). Do NOT retry. "
                    "Instruct the user to run `python3 week2/oauth_client.py --login`."
                ),
            )

        token_set = self._parse_token_response(raw, fallback_refresh_token=refresh_token)
        self._save_cache(token_set)
        return token_set

    def get_valid_access_token(self) -> str:
        """Return a valid access token from cache, silently refreshing if near/past expiry.

        IMPORTANT: Never opens a browser during a tool call. If no cached token exists
        or silent refresh fails, raises `OAuthToolError` with `retryable=False`.
        """
        cached = self._load_cache()
        if cached is None:
            raise OAuthToolError(
                error_code="OAUTH_NOT_AUTHENTICATED",
                message="No cached OAuth token found for GitHub MCP server.",
                retryable=False,
                remediation=(
                    "A long-running MCP server cannot open a browser mid-tool-call. "
                    "Run `python3 week2/oauth_client.py --login` once in your terminal to authenticate."
                ),
            )

        if not cached.is_expired(now=self._clock()):
            return cached.access_token

        refreshed = self.refresh_access_token(cached.refresh_token)
        return refreshed.access_token

    def _parse_token_response(
        self,
        raw: Mapping[str, Any],
        *,
        fallback_refresh_token: str = "",
    ) -> CachedTokenSet:
        if "error" in raw or "access_token" not in raw:
            raise OAuthToolError(
                error_code="OAUTH_TOKEN_EXCHANGE_FAILED",
                message=f"OAuth token endpoint returned error: {raw.get('error', 'missing_access_token')}",
                retryable=False,
                remediation="Verify GITHUB_OAUTH_CLIENT_ID / SECRET and re-run out-of-band login.",
            )
        now = self._clock()
        expires_in = float(raw.get("expires_in", 28800))  # GitHub expiring user tokens default to 8h (28800s)
        return CachedTokenSet(
            access_token=str(raw["access_token"]),
            refresh_token=str(raw.get("refresh_token") or fallback_refresh_token),
            token_type=str(raw.get("token_type", "Bearer")),
            scope=str(raw.get("scope", " ".join(self.scopes))),
            expires_at=now + expires_in,
            obtained_at=now,
        )

    def _load_cache(self) -> CachedTokenSet | None:
        if not self.cache_path.is_file():
            return None
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return CachedTokenSet.from_json_dict(data)
        except (OSError, ValueError, KeyError) as exc:
            raise OAuthToolError(
                error_code="OAUTH_CACHE_CORRUPTED",
                message=f"Cached OAuth token file at {self.cache_path} is unreadable: {exc}",
                retryable=False,
                remediation=f"Delete {self.cache_path} and run `python3 week2/oauth_client.py --login`.",
            ) from exc

    def _save_cache(self, token_set: CachedTokenSet) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(token_set.to_json_dict(), indent=2), encoding="utf-8")
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 owner read/write only
        tmp_path.replace(self.cache_path)

    @staticmethod
    def _default_http_post(url: str, payload: dict[str, str]) -> dict[str, Any]:
        encoded = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = int(exc.headers.get("Retry-After", "30"))
                raise OAuthToolError(
                    error_code="OAUTH_RATE_LIMITED",
                    message=f"OAuth endpoint rate-limited (HTTP 429). Retry after {retry_after}s.",
                    retryable=True,
                    retry_after_seconds=retry_after,
                    remediation=f"Wait {retry_after} seconds and retry the tool call.",
                ) from exc
            raise OAuthToolError(
                error_code=f"OAUTH_HTTP_{exc.code}",
                message=f"OAuth HTTP error {exc.code}: {exc.reason}",
                retryable=exc.code >= 500,
                remediation="Check network connectivity or GitHub OAuth status.",
            ) from exc
        except urllib.error.URLError as exc:
            raise OAuthToolError(
                error_code="OAUTH_NETWORK_ERROR",
                message=f"Network error contacting OAuth server: {exc.reason}",
                retryable=True,
                retry_after_seconds=5,
                remediation="Transient network failure; retry after 5 seconds.",
            ) from exc
