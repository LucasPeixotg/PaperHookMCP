"""Tunables for the services.

Values below are defaults; any of them can be overridden via environment
variables or the ``.env`` file at the repo root (e.g. ``DEFAULT_TOP_K=5`` in the
environment takes precedence over the default here).
"""

from pathlib import Path
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, resolved from this file so paths work from any working directory
ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # RESEARCH APIs ACCESS

    # SEMANTIC SCHOLAR
    SEMANTIC_SCHOLAR_API_TOKEN: str = ""
    SEMANTIC_SCHOLAR_API_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"

    # OPENALEX
    # The token is optional here too — a key only raises the daily budget.
    OPENALEX_API_TOKEN: str = ""
    OPENALEX_API_BASE_URL: str = "https://api.openalex.org"

    # RERANKING

    # The cross-encoder that reorders search candidates in
    # src/services/models/reranker.py.
    RERANK_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # The cost knob: every candidate in the pool is one cross-encoder forward
    # pass over a title and a ~1,000-character abstract. Raising it past 50 buys
    # nothing from OpenAlex alone — `OpenAlexClient.SEMANTIC_MAX_LIMIT` caps the
    # semantic endpoint there and `_search_works` clamps the request to it.
    RERANK_CANDIDATE_POOL: int = 40

    # CACHE

    # The per-paper store in src/services/cache/paper_store.py: every paper a search
    # or listing returns, keyed by `paper_id` with its DOI as an alias, and every
    # request as a list of pointers to those records. Anchored to the repo root for
    # the same reason as `.env` below; `.data/` is gitignored.
    CACHE_ENABLED: bool = True
    CACHE_DIR: Path = ROOT_DIR / ".data" / "paper_cache"
    # Citation counts and URLs drift, so records expire rather than live forever.
    CACHE_TTL_SECONDS: int = 30 * 24 * 3600
    # Which papers a query or listing returns drifts faster than any one paper, so a
    # request's pointers expire well before the records they point at.
    REQUEST_CACHE_TTL_SECONDS: int = 24 * 3600
    CACHE_SIZE_LIMIT_MB: int = 512

    model_config = SettingsConfigDict(
        # Anchored to the repo root so the same config works from any working
        # directory — an MCP client launches src/server.py with a cwd of its own.
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def masked_dsn(dsn: str) -> str:
    """A DSN with the credentials stripped, safe to print in logs and errors."""
    parsed = urlparse(dsn)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    user = f"{parsed.username}@" if parsed.username else ""
    return f"{parsed.scheme}://{user}{host}{port}{parsed.path}"


# Singleton instance — import this everywhere
settings = Settings()
