<img src="./logo.svg" alt="Paper Hook MCP Logo" width="250"/>

# Paper Hook MCP

An MCP server that finds the academic papers most relevant to a natural-language
research question, then lets you follow them: forward to the work that cites them,
back to the work they cite, and over to an author's most-cited publications.

It holds no local corpus. Every query is answered by searching public research APIs
live, then merging and reranking the candidates locally with a cross-encoder before
handing back a short, ordered list. Responses are cached on disk so repeat lookups
stay cheap.

## Demo

[https://github.com/LucasPeixotg/PaperHookMCP/raw/main/assets/example_usage.mp4](https://github.com/user-attachments/assets/112bab8e-8bc4-4e4c-bfe7-7f2228ad036c)

## How it works

A call to `find_relevant_papers` runs one pipeline
(`paperhook/tools/find_relevant_papers/tool.py`):

1. **Clean the query** — `*` and `?` are stripped, since search APIs read them as
   wildcard operators rather than as language.
2. **Search** — OpenAlex's semantic (embedding) endpoint, deliberately over-fetching
   `max(top_k * 2, RERANK_CANDIDATE_POOL)` candidates. A reranker can only improve on
   the order it is given, so it needs a pool wider than the slots it fills.
3. **Deduplicate** — two records are the same work if their DOIs match, or if title
   and abstract both clear a fuzzy-match threshold. A published copy beats a preprint
   of the same paper.
4. **Rerank** — a cross-encoder scores each candidate against the *raw* query, using
   the source's own relevance score to break ties.
5. **Split** — the top `top_k` slots are filled only with papers that actually carry
   an abstract; higher-ranked papers whose source publishes no abstract are returned
   separately rather than dropped.

### Following citations and authors

`get_citations`, `get_references` and `get_author_top_works` list works straight from
OpenAlex, with no reranking:

- `get_citations` (forward) and `get_references` (backward) take only an OpenAlex work
  ID, bare (`W2626778328`) or as a URL (`https://openalex.org/W2626778328`). DOIs and
  arXiv IDs are rejected. The target work is looked up first, so a paper that does not
  exist raises an error rather than looking like one with no citations.
- `get_author_top_works` takes an OpenAlex author ID (`A5001234567`), an ORCID, or a
  name. A name must match exactly one OpenAlex author. If it matches several, the error
  lists up to five candidates with their IDs, since picking the top hit would silently
  mix up namesakes.

### Caching

Every paper the tools see is stored on disk (diskcache, at `~/.cache/paperhook/paper_cache/`,
or under `$XDG_CACHE_HOME` when set), keyed by `paper_id` with its DOI as an alias. Each API request is cached
separately as a list of pointers to those records, so a paper shared by several queries
is stored once and stays consistent across them. Records expire after 30 days and request
pointers after one day, because which papers a query returns drifts faster than any one
paper does. Failures are never cached. If the cache errors, it is logged and the tools
carry on without it. Set `CACHE_ENABLED=false` to turn it off.

## Requirements

- Python 3.13
- ~1 GB of disk for dependencies (`sentence-transformers` pulls in PyTorch, installed
  as the CPU-only build), plus a one-time download of the cross-encoder weights on
  first run.

## Install and run

The server speaks MCP over stdio. With [uv](https://docs.astral.sh/uv/) installed,
register it with an MCP client and `uvx` fetches and runs it, no checkout needed:

```json
{
  "mcpServers": {
    "paperhook": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/LucasPeixotg/PaperHookMCP", "paperhook-mcp"],
      "env": {
        "OPENALEX_API_TOKEN": ""
      }
    }
  }
}
```

Settings (see [Configuration](#configuration)) go in `env`. The first launch
downloads ~1 GB of dependencies plus the cross-encoder weights, which can outlast a
client's startup timeout. To pay that cost up front, install it as a tool once and
point the client at the installed command instead:

```bash
uv tool install git+https://github.com/LucasPeixotg/PaperWeightMCP
```

```json
{ "mcpServers": { "paperhook": { "command": "paperhook-mcp" } } }
```

### From a checkout

```bash
uv sync --extra dev                # or, with pip (see below): pip install -e ".[dev]"
cp .env.example .env
uv run paperhook-mcp               # or: python -m paperhook.server
```

uv takes torch from PyTorch's CPU-only index (`[tool.uv.sources]` in `pyproject.toml`).
pip ignores that, and on Linux would install the CUDA build (~4 GB more), so install
torch first:

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### Troubleshooting

An MCP client usually reports a broken server only as "disconnected". Run the
diagnostics in a terminal instead, with the same environment variables the client sets:

```bash
uvx --from git+https://github.com/LucasPeixotg/PaperHookMCP paperhook-mcp --doctor
```

```
[✓] Cache directory writable (~/.cache/paperhook/paper_cache)
[✓] OpenAlex API reachable (API key set)
[✓] Cross-encoder weights loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)
[✓] MCP stdio transport ready (5 tools registered)
```

A failing check prints `[✗]` and a hint, and the command exits with status 1.

## Tools

| Tool | Signature | Returns |
| --- | --- | --- |
| `find_relevant_papers` | `(query: str, top_k: int = 5) -> str` | JSON with `papers` and `relevant_without_abstract` |
| `get_citations` | `(paper_id: str, limit: int = 10, sort: "most_cited" \| "newest" = "most_cited") -> str` | JSON list of papers citing the target |
| `get_references` | `(paper_id: str, limit: int = 10, sort: "most_cited" \| "newest" = "most_cited") -> str` | JSON list of papers the target cites |
| `get_author_top_works` | `(author: str, limit: int = 10, publication_year_min: int \| None = None) -> str` | JSON list of the author's works, most cited first |

`limit` is capped at 200. The `paper_id` that `find_relevant_papers` returns is an
OpenAlex work ID, so it can be passed straight to `get_citations` or `get_references`.

### `find_relevant_papers`

Each paper in either list carries the same seven fields:

| Field | Meaning |
| --- | --- |
| `paper_id` | The source's identifier for the work |
| `doi` | DOI, when the source publishes one |
| `title` | Paper title |
| `year` | Publication year |
| `abstract` | Abstract text (empty in `relevant_without_abstract`) |
| `url` | Best available link: open-access PDF, else landing page, else DOI |
| `license` | License of the location `url` points at, when known |

`papers` holds up to `top_k` matches that come with an abstract — the ones worth
reading now. `relevant_without_abstract` holds matches that ranked among the best but
whose abstract the source does not publish. That often means the work is not openly
available, though not always; some publishers restrict the abstract alone. They are
still worth following by DOI or URL.

### `get_citations`, `get_references`, `get_author_top_works`

Each paper carries the fields above except `abstract` (no query ranked these works,
and abstracts are not fetched for them), plus two fields for judging a work without
reading it:

| Field | Meaning |
| --- | --- |
| `venue` | Journal or conference it appeared in, when known |
| `cited_by_count` | How many works cite it |

`get_references` lists only the references OpenAlex could match to a work, so the list
may be shorter than the paper's printed bibliography.

## Configuration

Settings live in `paperhook/config.py` and can be overridden through the environment
(the MCP client's `env` block) or, in a checkout, the `.env` file at the repo root. Both API tokens are optional — they only raise
rate limits and daily budgets.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENALEX_API_BASE_URL` | `https://api.openalex.org` | OpenAlex endpoint |
| `OPENALEX_API_TOKEN` | *(empty)* | Optional; sent as a bearer token |
| `SEMANTIC_SCHOLAR_API_BASE_URL` | `https://api.semanticscholar.org/graph/v1` | Semantic Scholar Graph API endpoint |
| `SEMANTIC_SCHOLAR_API_TOKEN` | *(empty)* | Optional; sent as `x-api-key` |
| `RERANK_MODEL_ID` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking |
| `RERANK_CANDIDATE_POOL` | `40` | Candidates to rerank; the cost knob — one forward pass each. Past 50 buys nothing, since OpenAlex caps its semantic endpoint there. |
| `CACHE_ENABLED` | `true` | Turns the disk cache on or off |
| `CACHE_DIR` | `~/.cache/paperhook/paper_cache` | Where the cache lives (`$XDG_CACHE_HOME/paperhook/paper_cache` when that is set) |
| `CACHE_TTL_SECONDS` | `2592000` (30 days) | How long a paper record is kept |
| `REQUEST_CACHE_TTL_SECONDS` | `86400` (1 day) | How long a cached request's result list is kept |
| `CACHE_SIZE_LIMIT_MB` | `512` | Size cap for the cache on disk |

## Layout

```
pyproject.toml                     Package metadata, dependencies, `paperhook-mcp` entry point
paperhook/
  server.py                        FastMCP instance, tool registration, `main()`
  config.py                        Settings, loaded from the environment or .env
  common/paper_data.py             PaperData — the record every source maps into
  services/
    api/api_client.py              Shared HTTP client: headers, retries, error wrapping
    api/open_alex_client.py        OpenAlex search, work/author lookups, citation links
    api/semantic_scholar_client.py Semantic Scholar Graph API search (not wired in)
    api/arxiv_api_client.py        Placeholder, empty
    cache/paper_store.py           Disk cache of paper records and request pointers
    models/reranker.py             Cross-encoder reranking
  utils/
    paper_id.py                    Normalising OpenAlex work/author IDs, DOIs, ORCIDs
    response.py                    Shaping records into tool responses
  tools/
    find_relevant_papers/          Search tool, plus its query cleaner and deduplicator
    get_citations.py               Forward citations
    get_references.py              Backward citations
    get_author_top_works.py        An author's most-cited works
    fetch_full_paper.py            Stub, not registered
    fetch_paper_artifacts.py       Stub, not registered
tests/
```

## Development

```bash
pytest
```

Modules inside `paperhook/` import each other relatively; tests import
`paperhook.…` absolutely, and pytest's `pythonpath` (in `pyproject.toml`) puts the repo
root on `sys.path`, so the suite runs with or without the package installed. No test touches the network — the API client suites drive `httpx.MockTransport`,
and the reranker suite stubs the cross-encoder out.

## Status

Four tools are registered: `find_relevant_papers`, `get_citations`, `get_references`
and `get_author_top_works`. Planned but not yet implemented (they exist as stubs that
raise "Not Yet Implemented"):

- `fetch_full_paper`
- `fetch_paper_artifacts`

and more ...

## License

MIT — see [LICENSE](LICENSE).
