"""SemanticScholarClient against a mocked transport — no network, no real API key.

The mock is installed by swapping `httpx.MockTransport` into the client *after* it
is built, so `base_url`, the headers and the timeout are exactly the ones production
assembles: every assertion below is made against the request that would really have
gone out.
"""

import json
import logging

import httpx
import pytest
from tenacity import wait_none

from paperhook.common import PaperData
from paperhook.config import settings
from paperhook.services.api.api_client import APIError, ApiClient
from paperhook.services.api.semantic_scholar_client import SemanticScholarClient

# Deliberately not the real host: a test that somehow escapes the mock should fail
# loudly rather than quietly querying Semantic Scholar.
BASE_URL = "https://api.example.test/graph/v1"

MAX_LIMIT = SemanticScholarClient.MAX_LIMIT
SEARCH_FIELDS = SemanticScholarClient.SEARCH_FIELDS


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Strip the exponential wait so the retry tests don't sleep ~6s each."""
    monkeypatch.setattr(ApiClient._send.retry, "wait", wait_none())


def recorder(*results):
    """A MockTransport handler returning `results` in order, plus the requests it saw.

    A result that is an exception is raised instead of returned, which is how a
    timeout or a refused connection is simulated. The last result repeats, so a
    single response covers a call of any arity.
    """
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        result = results[min(len(calls), len(results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    return handler, calls


def make_client(monkeypatch, handler, *, token=""):
    """A client pointed at BASE_URL with `handler` standing in for the network."""
    # settings is a module-level singleton loaded from the repo's .env at import
    # time, so both values are pinned here or the assertions vary per machine.
    monkeypatch.setattr(settings, "SEMANTIC_SCHOLAR_API_BASE_URL", BASE_URL)
    monkeypatch.setattr(settings, "SEMANTIC_SCHOLAR_API_TOKEN", token)

    client = SemanticScholarClient()
    client._client._transport = httpx.MockTransport(handler)
    return client


def ok(*items) -> httpx.Response:
    return httpx.Response(200, json={"data": list(items)})


def logged_api_errors(caplog) -> list[APIError]:
    """The APIErrors search_papers swallowed.

    The client logs the exception object itself (`logger.error(error)`), so the
    record's `msg` *is* the APIError — which is how a test proves one was raised
    upstream even though the caller only ever sees an empty list.
    """
    return [r.msg for r in caplog.records if isinstance(r.msg, APIError)]


def item(**overrides) -> dict:
    base = {
        "paperId": "p1",
        "externalIds": {"DOI": "10.48550/arXiv.1706.03762", "ArXiv": "1706.03762"},
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": "The dominant sequence transduction models...",
        "url": "https://www.semanticscholar.org/paper/p1",
        "openAccessPdf": None,
    }
    return base | overrides


# --- the request that goes out ------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_makes_no_request(monkeypatch, query):
    """A blank query is a guaranteed 400, so it must not cost a round trip."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    assert client.search_papers(query) == []
    assert calls == []


def test_search_hits_the_graph_search_endpoint(monkeypatch):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.search_papers("transformers")

    request = calls[0]
    assert request.method == "GET"
    # httpx appends to the base_url's path rather than replacing it, so /graph/v1
    # survives — the whole reason the endpoint is written as a bare "/paper/search".
    assert str(request.url).startswith(f"{BASE_URL}/paper/search?")
    assert request.url.params["query"] == "transformers"
    assert request.url.params["fields"] == SEARCH_FIELDS


@pytest.mark.parametrize(
    "limit, sent",
    [(0, 1), (-5, 1), (7, 7), (MAX_LIMIT, MAX_LIMIT), (500, MAX_LIMIT)],
)
def test_limit_is_clamped_to_the_api_range(monkeypatch, limit, sent):
    """The endpoint 400s above 100 and below 1, so the client clamps rather than relay."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.search_papers("q", limit=limit)

    assert calls[0].url.params["limit"] == str(sent)


def test_api_token_becomes_an_x_api_key_header(monkeypatch):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler, token="secret-token")

    client.search_papers("q")

    assert calls[0].headers["x-api-key"] == "secret-token"
    assert calls[0].headers["accept"] == "application/json"


def test_no_token_sends_no_key_header(monkeypatch):
    """Unauthenticated requests are legal here — just rate limited — so no empty key."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler, token="")

    client.search_papers("q")

    assert "x-api-key" not in calls[0].headers
    # Bearer auth belongs to the OpenAlex client, never to this one.
    assert "authorization" not in calls[0].headers


# --- turning the payload into PaperData ---------------------------------------


def test_maps_a_full_record_to_paper_data(monkeypatch):
    handler, _ = recorder(ok(item()))
    client = make_client(monkeypatch, handler)

    assert client.search_papers("q") == [
        PaperData(
            paper_id="p1",
            doi="10.48550/arXiv.1706.03762",
            title="Attention Is All You Need",
            year=2017,
            abstract="The dominant sequence transduction models...",
            url="https://www.semanticscholar.org/paper/p1",
            license="",
        )
    ]


def test_every_record_in_the_payload_is_mapped(monkeypatch):
    handler, _ = recorder(ok(item(paperId="a"), item(paperId="b"), item(paperId="c")))
    client = make_client(monkeypatch, handler)

    assert [p.paper_id for p in client.search_papers("q")] == ["a", "b", "c"]


def test_missing_fields_coerce_to_empties():
    """Every search field is nullable upstream while PaperData requires all seven.

    Note the consequence for `year`: a missing year and a genuine year of 0 are
    indistinguishable downstream.
    """
    empty = SemanticScholarClient._to_paper_data(
        {
            "paperId": None,
            "externalIds": None,
            "title": None,
            "year": None,
            "abstract": None,
            "url": None,
        }
    )

    assert empty == PaperData(
        paper_id="", doi="", title="", year=0, abstract="", url="", license=""
    )


@pytest.mark.parametrize(
    "external_ids",
    [None, {}, {"ArXiv": "1706.03762"}],
    ids=["null", "empty", "no-doi-key"],
)
def test_a_record_without_a_doi_maps_to_an_empty_string(external_ids):
    """Plenty of records carry other external ids but no DOI — preprints, theses."""
    paper = SemanticScholarClient._to_paper_data(item(externalIds=external_ids))

    assert paper.doi == ""


def test_the_doi_comes_from_external_ids():
    """There is no top-level `doi` in a Graph API record; it is nested and upper-cased."""
    paper = SemanticScholarClient._to_paper_data(
        item(externalIds={"DOI": "10.1038/s41586-020-2649-2"})
    )

    assert paper.doi == "10.1038/s41586-020-2649-2"


def test_open_access_pdf_wins_over_the_landing_page():
    """The point of asking for openAccessPdf: prefer a URL the caller can actually read."""
    paper = SemanticScholarClient._to_paper_data(
        item(
            openAccessPdf={
                "url": "https://arxiv.org/pdf/1706.03762",
                "license": "CCBY",
            }
        )
    )

    assert paper.url == "https://arxiv.org/pdf/1706.03762"
    assert paper.license == "CCBY"
    assert paper.oa_pdf == "https://arxiv.org/pdf/1706.03762"
    assert paper.oa_url == ""


@pytest.mark.parametrize("open_access", [None, {}, {"url": None, "license": None}])
def test_landing_page_is_the_fallback_url(open_access):
    paper = SemanticScholarClient._to_paper_data(item(openAccessPdf=open_access))

    assert paper.url == "https://www.semanticscholar.org/paper/p1"
    assert paper.license == ""
    assert paper.oa_pdf == ""


def test_url_is_empty_when_neither_is_offered():
    paper = SemanticScholarClient._to_paper_data(item(url=None, openAccessPdf=None))

    assert paper.url == ""


@pytest.mark.parametrize("payload", [{}, {"data": None}, {"data": []}])
def test_empty_payloads_yield_no_papers(monkeypatch, payload):
    """A query with no matches omits `data` entirely on some responses."""
    handler, _ = recorder(httpx.Response(200, json=payload))
    client = make_client(monkeypatch, handler)

    assert client.search_papers("nonsense") == []


# --- failures -----------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 404, 429, 500])
def test_http_error_yields_an_empty_list(monkeypatch, caplog, status):
    """A failed search is empty, not an exception: the APIError stops at the client.

    Status errors are still not retried — a 400 stays a 400 however often it's sent.
    """
    handler, calls = recorder(httpx.Response(status, text="upstream said no"))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        assert client.search_papers("q") == []

    assert len(calls) == 1
    errors = logged_api_errors(caplog)
    assert len(errors) == 1
    assert f"API status {status}" in str(errors[0])


def test_status_error_carries_the_upstream_body(monkeypatch, caplog):
    """The caller loses the failure, so the log has to keep what upstream said."""
    handler, _ = recorder(httpx.Response(429, text="Too Many Requests"))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        assert client.search_papers("q") == []

    assert "Too Many Requests" in str(logged_api_errors(caplog)[0])


def test_transient_timeout_is_retried_then_succeeds(monkeypatch, caplog):
    """The reason the retry exists: one flaky attempt must not fail the search."""
    handler, calls = recorder(httpx.ReadTimeout("timed out"), ok(item()))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        papers = client.search_papers("q")

    assert len(calls) == 2
    assert [p.paper_id for p in papers] == ["p1"]
    # A recovered attempt is not a failure, so nothing is logged.
    assert logged_api_errors(caplog) == []


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("timed out"), httpx.ConnectError("connection refused")],
    ids=["timeout", "connect"],
)
def test_network_failure_gives_up_after_three_attempts(monkeypatch, caplog, error):
    handler, calls = recorder(error)
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        assert client.search_papers("q") == []

    assert len(calls) == 3
    errors = logged_api_errors(caplog)
    assert len(errors) == 1
    assert "Network failure" in str(errors[0])


def test_malformed_json_body_is_not_wrapped(monkeypatch):
    """The boundary of the `except APIError` clause: only APIError is swallowed.

    A 200 carrying junk escapes as a JSON decode error, so the caller sees it
    rather than an empty list pretending the search simply found nothing.
    """
    handler, _ = recorder(httpx.Response(200, text="<html>maintenance</html>"))
    client = make_client(monkeypatch, handler)

    with pytest.raises(json.JSONDecodeError):
        client.search_papers("q")


def test_unexpected_error_propagates_to_the_caller(monkeypatch, caplog):
    """Anything that isn't an APIError is a bug, not a failed search — let it out."""
    handler, calls = recorder(RuntimeError("boom"))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="boom"):
        client.search_papers("q")

    # Not a retryable httpx error, so it escapes on the first attempt, unlogged.
    assert len(calls) == 1
    assert logged_api_errors(caplog) == []


# --- lifecycle ----------------------------------------------------------------


def test_context_manager_closes_the_client(monkeypatch):
    handler, _ = recorder(ok())
    client = make_client(monkeypatch, handler)

    with client as entered:
        assert entered is client
        assert not client._client.is_closed

    assert client._client.is_closed
