# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # ~2 GB: sentence-transformers pulls in torch
cp .env.example .env

python src/server.py                     # MCP server over stdio
pytest                                   # all tests (no network)
pytest tests/test_reranker.py            # one file
pytest tests/test_reranker.py -k tie     # one test by name
```

Python 3.13. No linter/formatter is configured, and there is no `pyproject.toml`.

## Import roots

Imports are rooted at `src/` (`from config import settings`, `from common import PaperData`), not at the repo root. `python src/server.py` gets this for free; `tests/conftest.py` inserts `src/` on `sys.path` to match, and `.vscode/settings.json` adds it to Pylance. Keep new imports in that style.

## Architecture

FastMCP server (`src/server.py`) exposing tools. `find_relevant_papers` and `get_citations` are registered; the other modules in `src/tools/` (`fetch_full_paper`, `get_references`, etc.) are stubs that raise `ToolError("Not Yet Implemented")`, and their imports in `server.py` are commented out. No local corpus or database — every call hits public research APIs live.

`find_relevant_papers` (`src/tools/find_relevant_papers/tool.py`) is a pipeline:
1. `remove_wildcards` strips `*`/`?` (search APIs treat them as operators).
2. OpenAlex semantic search, over-fetching `max(top_k*2, RERANK_CANDIDATE_POOL)`. OpenAlex caps this endpoint at 50 (`OpenAlexClient.SEMANTIC_MAX_LIMIT`). Semantic Scholar and OpenAlex keyword search are wired as clients but their calls are commented out in `tool.py`.
3. `remove_duplicates` — DOI match, or fuzzy title+abstract match (rapidfuzz); a published copy beats a preprint.
4. `rerank` (`src/services/models/reranker.py`) — cross-encoder scores against the **raw** (uncleaned) query, source `relevance_score` breaks ties.
5. Split: top `top_k` slots go only to papers with an abstract; higher-ranked abstract-less papers are returned in `relevant_without_abstract` instead of being dropped.

Output is JSON `{papers, relevant_without_abstract}`; `relevance_score` is stripped from the response because its scale differs per source and search mode.

`get_citations` (`src/tools/get_citations.py`) accepts only an OpenAlex ID (bare `W…` or `https://openalex.org/W…`; DOIs and arXiv IDs are rejected) and normalizes it with `to_openalex_id` for a `/works/{id}` lookup (in `src/utils/paper_id.py`, alongside the `bare_doi` / `bare_openalex_id` strippers the OpenAlex client maps through), then lists citing works with `filter=cites:W…`. Unlike the searches, these client methods raise `APIError`, and the tool turns it into a `ToolError`, so "paper not found" never looks like "no citations".

`get_author_top_works` (`src/tools/get_author_top_works.py`) takes an OpenAlex author ID (`A…`), an ORCID, or a name; `to_openalex_author_key` (`src/utils/paper_id.py`) turns the first two into a `/authors/{key}` lookup and returns None for a name. A name is searched via `/authors` and must match exactly one author — several matches raise a `ToolError` listing the candidate IDs, since a namesake's list looks just as plausible. Works come from `filter=author.id:A…`, sorted by citations, with an optional `publication_year_min`.

Key shared pieces:
- `PaperData` (`src/common/paper_data.py`) — the record every source maps into. Semantic Scholar has no relevance score, so its papers default to 0.0 and lose ties; revisit when re-enabling it.
- `ApiClient` (`src/services/api/api_client.py`) — base class for source clients: `x-api-key`/bearer headers, tenacity retries on network/timeout errors in `_send`, translation to `APIError` in `_request`.
- `PaperStore` (`src/services/cache/paper_store.py`) — diskcache-backed store of paper records at `.data/paper_cache/`, keyed by `paper_id` with the DOI (casefolded) as the only alias; titles are never a key. `OpenAlexClient._search_works` / `_list_linked_works` pass every mapped paper through `absorb`, which fills blank fields from the cached copy (only the abstract when reached via DOI under another `paper_id`) and writes it back; `relevance_score` is never stored. Records may also carry `full_paper` (not a `PaperData` field), written and read only via `put_full_paper` / `get_full_paper`, for `fetch_full_paper` to use once implemented. `get_work("W…")` is answered from the store when the record has a title and `cited_by_count`. It's built lazily by `get_paper_store()` and is off when `CACHE_ENABLED=false`; cache errors are logged and degrade to no cache. Semantic Scholar is not wired in: its ids would need their own namespace.
- `settings` (`src/config.py`) — pydantic-settings singleton; `.env` is resolved from the repo root so it works whatever cwd the MCP client uses.

Side effects at import time: clients in `tool.py` and the `CrossEncoder` in `reranker.py` (module-level `_model`, downloads weights on first run) are created when the module is imported.

## Tests

No test touches the network. API client tests use `httpx.MockTransport`; reranker tests inject a stub scorer rather than loading the real model. Test modules open with docstrings explaining which contracts they pin — read those before changing the corresponding code.
