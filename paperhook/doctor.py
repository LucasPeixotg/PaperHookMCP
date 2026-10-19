"""`paperhook-mcp --doctor`: startup diagnostics run in a terminal.

Over stdio the server is headless, so whatever breaks it (an unwritable cache, a
blocked network, model weights that won't download) reaches the user as
"server disconnected". Each check here exercises one of those pieces directly
and prints a line, with a hint when it fails. Output goes to stdout: in doctor
mode nothing speaks MCP over it.
"""

import asyncio
import importlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import settings

_MARKS = {True: "✓", False: "✗", None: "-"}


@dataclass(frozen=True)
class CheckResult:
    # None means the check was skipped, which is not a failure.
    ok: bool | None
    label: str
    detail: str = ""
    hint: str = ""

    def render(self) -> str:
        line = f"[{_MARKS[self.ok]}] {self.label}"
        if self.detail:
            line += f" ({self.detail})"
        if self.hint:
            line += f"\n    → {self.hint}"
        return line


def _display_path(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def check_cache_dir() -> CheckResult:
    if not settings.CACHE_ENABLED:
        return CheckResult(None, "Cache disabled", "CACHE_ENABLED=false")

    cache_dir = Path(settings.CACHE_DIR)
    shown = _display_path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir):
            pass

        # Opening the store as the server does catches SQLite locking and corruption,
        # which a plain file write would not.
        import diskcache

        with diskcache.Cache(str(cache_dir)) as cache:
            cache.set("doctor:probe", 1)
            if cache.get("doctor:probe") != 1:
                raise OSError("probe value did not read back")
            cache.delete("doctor:probe")
    except Exception as error:
        return CheckResult(
            False,
            "Cache directory not writable",
            shown,
            f"{error}. Fix its permissions, point CACHE_DIR elsewhere, or set "
            "CACHE_ENABLED=false (the server silently runs uncached otherwise)",
        )
    return CheckResult(True, "Cache directory writable", shown)


def check_openalex(transport: httpx.BaseTransport | None = None) -> CheckResult:
    # A bare client rather than OpenAlexClient: its pointer cache could answer without
    # touching the network, and its retries take ~40s to give up on a blocked one.
    headers = {"Accept": "application/json"}
    token = settings.OPENALEX_API_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"
    key_status = "API key set" if token else "no API key: anonymous budget"
    label = "OpenAlex API"

    try:
        with httpx.Client(
            base_url=settings.OPENALEX_API_BASE_URL,
            headers=headers,
            timeout=10.0,
            transport=transport,
        ) as client:
            response = client.get("/works", params={"per-page": 1, "select": "id"})
    except httpx.RequestError as error:
        return CheckResult(
            False,
            f"{label} unreachable",
            settings.OPENALEX_API_BASE_URL,
            f"{type(error).__name__}: {error}. Network blocked? Check proxy/firewall "
            "settings, or OPENALEX_API_BASE_URL",
        )

    status = response.status_code
    if status == 200:
        return CheckResult(True, f"{label} reachable", key_status)
    if status in (401, 403):
        return CheckResult(
            False,
            f"{label} rejected OPENALEX_API_TOKEN",
            f"HTTP {status}",
            "Check the key, or unset it to use the anonymous budget",
        )
    if status == 429:
        return CheckResult(
            False,
            f"{label} daily budget exhausted",
            f"HTTP 429, {key_status}",
            "Wait for the budget to reset, or set OPENALEX_API_TOKEN to raise it",
        )
    return CheckResult(False, f"{label} returned an error", f"HTTP {status}")


def check_reranker() -> CheckResult:
    model_id = settings.RERANK_MODEL_ID
    try:
        # Importing the module loads the weights, downloading them on a first run.
        reranker = importlib.import_module("paperhook.services.models.reranker")
        reranker._model.predict([("probe", "probe")])
    except Exception as error:
        return CheckResult(
            False,
            "Cross-encoder failed to load",
            model_id,
            f"{error}. A first run downloads the weights from Hugging Face: check "
            "the network, HF_HOME and free disk space",
        )
    return CheckResult(True, "Cross-encoder weights loaded", model_id)


def check_stdio() -> CheckResult:
    try:
        # The whole tool import chain, as the server runs it.
        from .server import mcp

        tools = asyncio.run(mcp.list_tools())
    except Exception as error:
        return CheckResult(False, "MCP server failed to start", hint=str(error))

    for name in ("stdin", "stdout"):
        stream = getattr(sys, name)
        if stream is None or stream.closed:
            return CheckResult(False, "MCP stdio transport unavailable", f"{name} is closed")

    return CheckResult(True, "MCP stdio transport ready", f"{len(tools)} tools registered")


# Cheapest first, so the quick answers print before the model load.
CHECKS = [check_cache_dir, check_openalex, check_reranker, check_stdio]


def run() -> int:
    """Run every check, printing each as it finishes. Returns the exit code."""
    failed = False
    for check in CHECKS:
        try:
            result = check()
        except Exception as error:
            # One check crashing must not hide the ones after it.
            result = CheckResult(False, f"{check.__name__} crashed", hint=repr(error))
        print(result.render(), flush=True)
        failed |= result.ok is False
    return 1 if failed else 0
