# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"                  # ~2 GB: sentence-transformers pulls in torch
cp .env.example .env

paperhook-mcp                            # MCP server over stdio (or: python -m paperhook.server)
uvx --from . paperhook-mcp               # same, as users run it (git+https://… URL in README)
pytest                                   # all tests (no network)
pytest tests/test_reranker.py            # one file
pytest tests/test_reranker.py -k tie     # one test by name
```

Python 3.13. No linter/formatter is configured. `pyproject.toml` (hatchling) is the single source of dependencies and defines the `paperhook-mcp` console script (`paperhook.server:main`).

## Package layout

The code is the `paperhook/` package at the repo root (flat layout). Inside it, modules import each other **relatively** (`from ..config import settings`, `from ...common import PaperData`); tests import absolutely (`from paperhook.common import PaperData`, `importlib.import_module("paperhook.tools.get_citations")`). pytest's `pythonpath = ["."]` in `pyproject.toml` makes `paperhook` importable without installing. Keep new imports in that style — an absolute `from config import …` breaks once installed.

Installed (e.g. via uvx), `config.ROOT_DIR` points into site-packages, so nothing may be written there: the cache defaults to `~/.cache/paperhook/paper_cache` (`$XDG_CACHE_HOME` honored), and `.env` is only read in a checkout — installed users set env vars through the MCP client.

## Architecture

FastMCP server (`paperhook/server.py`) exposing tools. `find_relevant_papers`, `get_citations`, `get_references`, `get_author_top_works` and `fetch_full_paper` are registered; the rest of `paperhook/tools/` (`fetch_paper_artifacts`, etc.) are stubs that raise `ToolError("Not Yet Implemented")`, and their imports in `server.py` are commented out. No local corpus or database — every call hits public research APIs live.

`find_relevant_papers` (`paperhook/tools/find_relevant_papers/tool.py`) is a pipeline:
1. `remove_wildcards` strips `*`/`?` (search APIs treat them as operators).
2. OpenAlex semantic search, over-fetching `max(top_k*2, RERANK_CANDIDATE_POOL)`. OpenAlex caps this endpoint at 50 (`OpenAlexClient.SEMANTIC_MAX_LIMIT`). Semantic Scholar and OpenAlex keyword search are wired as clients but their calls are commented out in `tool.py`.
3. `remove_duplicates` — DOI match, or fuzzy title+abstract match (rapidfuzz); a published copy beats a preprint.
4. `rerank` (`paperhook/services/models/reranker.py`) — cross-encoder scores against the **raw** (uncleaned) query, source `relevance_score` breaks ties.
5. Split: top `top_k` slots go only to papers with an abstract; higher-ranked abstract-less papers are returned in `relevant_without_abstract` instead of being dropped.

Output is JSON `{papers, relevant_without_abstract}`; `relevance_score` is stripped from the response because its scale differs per source and search mode.

`get_citations` (`paperhook/tools/get_citations.py`) accepts only an OpenAlex ID (bare `W…` or `https://openalex.org/W…`; DOIs and arXiv IDs are rejected) and normalizes it with `to_openalex_id` for a `/works/{id}` lookup (in `paperhook/utils/paper_id.py`, alongside the `bare_doi` / `bare_openalex_id` strippers the OpenAlex client maps through), then lists citing works with `filter=cites:W…`. Unlike the searches, these client methods raise `APIError`, and the tool turns it into a `ToolError`, so "paper not found" never looks like "no citations".

`get_author_top_works` (`paperhook/tools/get_author_top_works.py`) takes an OpenAlex author ID (`A…`), an ORCID, or a name; `to_openalex_author_key` (`paperhook/utils/paper_id.py`) turns the first two into a `/authors/{key}` lookup and returns None for a name. A name is searched via `/authors` and must match exactly one author — several matches raise a `ToolError` listing the candidate IDs, since a namesake's list looks just as plausible. Works come from `filter=author.id:A…`, sorted by citations, with an optional `publication_year_min`.

`fetch_full_paper` (`paperhook/tools/fetch_full_paper.py`) takes an OpenAlex ID like `get_citations`. It returns the full text cached by `PaperStore.get_full_paper` if there is one; otherwise `OpenAlexClient.get_paper` fetches the one record (answered from the store when that record already has an `oa_pdf`), and `PdfClient` (`paperhook/services/fulltext/pdf_client.py`) downloads `oa_pdf` and extracts it with pymupdf. Successful text is stored with `put_full_paper`. A paper with no `oa_pdf` is a `ToolError` that hands the caller its `oa_url` (labelled as the open-access URL) or, failing that, its `url` (saying there is no open-access URL) to fetch manually, and failed downloads and blank (scanned) text are never cached.

Key shared pieces:
- `PaperData` (`paperhook/common/paper_data.py`) — the record every source maps into. Semantic Scholar has no relevance score, so its papers default to 0.0 and lose ties; revisit when re-enabling it. `url` is the best link to any copy (it may fall back to the publisher's); `oa_pdf` / `oa_url` are the open-access copy's direct PDF and landing page only (OpenAlex `best_oa_location`, S2 `openAccessPdf`), blank when there is none — the links `fetch_full_paper` should use.
- `ApiClient` (`paperhook/services/api/api_client.py`) — base class for source clients: `x-api-key`/bearer headers, tenacity retries on network/timeout errors in `_send`, translation to `APIError` in `_request`.
- `PaperStore` (`paperhook/services/cache/paper_store.py`) — diskcache-backed store of paper records at `CACHE_DIR` (`~/.cache/paperhook/paper_cache/`), keyed by `paper_id` with the DOI (casefolded) as the only alias; titles are never a key. `OpenAlexClient._search_works` / `_list_linked_works` pass every mapped paper through `absorb`, which fills blank fields from the cached copy (only the abstract when reached via DOI under another `paper_id`) and writes it back; `relevance_score` is never stored. Records may also carry `full_paper` (not a `PaperData` field), written and read only via `put_full_paper` / `get_full_paper`, for `fetch_full_paper` to use once implemented. `get_work("W…")` is answered from the store when the record has a title and `cited_by_count`. Every `OpenAlexClient` request is also cached by **pointer**: `request_key` hashes method, endpoint and all params (so changing a `select` retires old entries), and the entry holds only the ordered ids plus per-request data (`relevance_score`, listing total). Hits are rebuilt from the shared records, and a pointer to any missing record is a miss. Pointers expire after `REQUEST_CACHE_TTL_SECONDS` (1 day), records after `CACHE_TTL_SECONDS`. Failures are never cached, but successful empty results are. A DOI `get_work` is answered only via its own pointer, never via the DOI alias. Authors are stored under `author:` keys with an ORCID alias, and author searches are cached by pointer the same way. It's built lazily by `get_paper_store()` and is off when `CACHE_ENABLED=false`; cache errors are logged and degrade to no cache. Semantic Scholar is not wired in: its ids would need their own namespace.
- `settings` (`paperhook/config.py`) — pydantic-settings singleton; `.env` is resolved from the repo root so it works whatever cwd the MCP client uses.

Side effects at import time: clients in `tool.py` and the `CrossEncoder` in `reranker.py` (module-level `_model`, downloads weights on first run) are created when the module is imported.

## Tests

No test touches the network. API client tests use `httpx.MockTransport`; reranker tests inject a stub scorer rather than loading the real model. Test modules open with docstrings explaining which contracts they pin — read those before changing the corresponding code.
