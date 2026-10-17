"""OpenAlexClient against a mocked transport — no network, no real API key.

Mirrors tests/test_semantic_scholar_client.py: the mock is swapped into the client
*after* it is built, so `base_url`, the headers and the timeout are exactly the ones
production assembles.

The client exposes two searches — keyword (`search`) and embedding-based
(`search.semantic`) — over one shared request path, `_search_works`. Tests of what the
two share take the `search` fixture below and therefore run once per entry point; tests
of what differs (the parameter carrying the query, the per-page ceiling, the query
truncation) name their method directly.
"""

import json
import logging

import httpx
import pytest
from tenacity import wait_none

from common import CitingPaperData, PaperData
from config import settings
from services.api.api_client import ApiClient, APIError
from services.api.open_alex_client import OpenAlexClient

# Deliberately not the real host: a test that somehow escapes the mock should fail
# loudly rather than quietly querying OpenAlex.
BASE_URL = "https://api.example.test/openalex"

MAX_LIMIT = OpenAlexClient.MAX_LIMIT
SEMANTIC_MAX_LIMIT = OpenAlexClient.SEMANTIC_MAX_LIMIT
SEMANTIC_MAX_QUERY_CHARS = OpenAlexClient.SEMANTIC_MAX_QUERY_CHARS
SEARCH_FIELDS = OpenAlexClient.SEARCH_FIELDS


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Strip the exponential wait so the retry tests don't sleep ~6s each."""
    monkeypatch.setattr(ApiClient._send.retry, "wait", wait_none())


@pytest.fixture(params=["search_papers", "semantic_search_papers"],
                ids=["keyword", "semantic"])
def search(request):
    """Both public searches, so the machinery they share is proven through each.

    The two differ only in which parameter carries the query and how high the per-page
    ceiling goes. Everything beneath them — the endpoint, the retry, the swallowing of
    APIError and the mapping to PaperData — is `_search_works`, and a test that reaches
    it through one caller says nothing about the other.
    """
    return request.param


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
    monkeypatch.setattr(settings, "OPENALEX_API_BASE_URL", BASE_URL)
    monkeypatch.setattr(settings, "OPENALEX_API_TOKEN", token)

    client = OpenAlexClient()
    client._client._transport = httpx.MockTransport(handler)
    return client


def ok(*items) -> httpx.Response:
    return httpx.Response(200, json={"results": list(items)})


def logged_api_errors(caplog) -> list[APIError]:
    """The APIErrors a search swallowed.

    The client logs the exception object itself (`logger.error(error)`), so the
    record's `msg` *is* the APIError — which is how a test proves one was raised
    upstream even though the caller only ever sees an empty list.
    """
    return [r.msg for r in caplog.records if isinstance(r.msg, APIError)]


# "The dominant sequence models are dominant" as OpenAlex would publish it.
INVERTED_ABSTRACT = {
    "The": [0],
    "dominant": [1, 5],
    "sequence": [2],
    "models": [3],
    "are": [4],
}
PLAIN_ABSTRACT = "The dominant sequence models are dominant"


def item(**overrides) -> dict:
    base = {
        "id": "https://openalex.org/W2626778328",
        "doi": "https://doi.org/10.48550/arxiv.1706.03762",
        "display_name": "Attention Is All You Need",
        "publication_year": 2017,
        "abstract_inverted_index": dict(INVERTED_ABSTRACT),
        "best_oa_location": None,
        "primary_location": None,
        "relevance_score": 1.2024246507243719,
    }
    return base | overrides


def location(**overrides) -> dict:
    base = {"pdf_url": None, "landing_page_url": None, "license": None}
    return base | overrides


# --- the request that goes out ------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_makes_no_request(monkeypatch, query):
    """A blank search matches every work in OpenAlex, so it must not be sent."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    assert client.search_papers(query) == []
    assert calls == []


def test_search_hits_the_works_endpoint(monkeypatch):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.search_papers("transformers")

    request = calls[0]
    assert request.method == "GET"
    # httpx appends to the base_url's path rather than replacing it, which is why the
    # endpoint is written as a bare "/works".
    assert str(request.url).startswith(f"{BASE_URL}/works?")
    assert request.url.params["search"] == "transformers"
    assert request.url.params["select"] == SEARCH_FIELDS


def test_relevance_score_is_among_the_selected_fields(monkeypatch, search):
    """The assertions above compare `select` to the constant, so both move together.

    This one names the field, because leaving it out of `select` is the one way this
    breaks in silence: OpenAlex does not reject an unselected field, it just omits it,
    the mapping reads a missing key as 0.0, and the reranker's tiebreak quietly stops
    doing anything. Nothing else in the suite would notice.
    """
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    getattr(client, search)("transformers")

    assert "relevance_score" in calls[0].url.params["select"].split(",")


@pytest.mark.parametrize(
    "limit, sent",
    [(0, 1), (-5, 1), (7, 7), (MAX_LIMIT, MAX_LIMIT), (500, MAX_LIMIT)],
)
def test_limit_is_clamped_to_the_api_range(monkeypatch, limit, sent):
    """A per-page outside 1..200 is a Pagination error, so the client clamps."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.search_papers("q", limit=limit)

    assert calls[0].url.params["per-page"] == str(sent)


def test_api_token_becomes_a_bearer_header(monkeypatch):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler, token="secret-token")

    client.search_papers("q")

    assert calls[0].headers["authorization"] == "Bearer secret-token"
    assert calls[0].headers["accept"] == "application/json"


def test_no_token_sends_no_auth_header(monkeypatch):
    """Unauthenticated requests are legal — a key only raises the daily budget."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler, token="")

    client.search_papers("q")

    assert "authorization" not in calls[0].headers
    # x-api-key belongs to the Semantic Scholar client, never to this one.
    assert "x-api-key" not in calls[0].headers


# --- semantic search ----------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_semantic_query_makes_no_request(monkeypatch, query):
    """A blank query has nothing to embed; OpenAlex would return the whole index."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    assert client.semantic_search_papers(query) == []
    assert calls == []


def test_semantic_search_hits_the_works_endpoint(monkeypatch):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.semantic_search_papers("how do transformers attend")

    request = calls[0]
    assert request.method == "GET"
    assert str(request.url).startswith(f"{BASE_URL}/works?")
    assert request.url.params["search.semantic"] == "how do transformers attend"
    assert request.url.params["select"] == SEARCH_FIELDS


def test_semantic_search_sends_no_other_search_parameter(monkeypatch):
    """The API rejects a request naming more than one of search/search.exact/semantic."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.semantic_search_papers("q")

    params = calls[0].url.params
    assert "search" not in params
    assert "search.exact" not in params


def test_keyword_search_sends_no_semantic_parameter(monkeypatch):
    """The mirror of the rule above — the shared helper must not leak between paths."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.search_papers("q")

    assert "search.semantic" not in calls[0].url.params


@pytest.mark.parametrize(
    "limit, sent",
    [
        (0, 1),
        (-5, 1),
        (7, 7),
        (SEMANTIC_MAX_LIMIT, SEMANTIC_MAX_LIMIT),
        # 200 is legal for the keyword endpoint but not for this one.
        (MAX_LIMIT, SEMANTIC_MAX_LIMIT),
        (500, SEMANTIC_MAX_LIMIT),
    ],
)
def test_semantic_limit_is_clamped_to_the_vector_ceiling(monkeypatch, limit, sent):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    client.semantic_search_papers("q", limit=limit)

    assert calls[0].url.params["per-page"] == str(sent)


@pytest.mark.parametrize(
    "length, sent_length",
    [
        (SEMANTIC_MAX_QUERY_CHARS - 1, SEMANTIC_MAX_QUERY_CHARS - 1),
        (SEMANTIC_MAX_QUERY_CHARS, SEMANTIC_MAX_QUERY_CHARS),
        (SEMANTIC_MAX_QUERY_CHARS + 1, SEMANTIC_MAX_QUERY_CHARS),
        (SEMANTIC_MAX_QUERY_CHARS + 500, SEMANTIC_MAX_QUERY_CHARS),
    ],
    ids=["just under", "exactly at", "one over", "well over"],
)
def test_a_long_query_is_truncated_before_it_goes_out(
    monkeypatch, length, sent_length
):
    """Only the first 2000 characters are embedded, so the rest never leaves here."""
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    query = "a" * length
    client.semantic_search_papers(query)

    sent = calls[0].url.params["search.semantic"]
    assert len(sent) == sent_length
    assert sent == query[:sent_length]


def test_a_query_under_the_cap_is_sent_verbatim(monkeypatch):
    handler, calls = recorder(ok())
    client = make_client(monkeypatch, handler)

    query = "an abstract-length paragraph about protein folding. " * 10
    client.semantic_search_papers(query)

    assert calls[0].url.params["search.semantic"] == query


# --- turning the payload into PaperData ---------------------------------------


def test_maps_a_full_record_to_paper_data(monkeypatch, search):
    handler, _ = recorder(
        ok(item(best_oa_location=location(pdf_url="https://arxiv.org/pdf/1706.03762",
                                          license="cc-by")))
    )
    client = make_client(monkeypatch, handler)

    assert getattr(client, search)("q") == [
        PaperData(
            paper_id="W2626778328",
            doi="10.48550/arxiv.1706.03762",
            title="Attention Is All You Need",
            year=2017,
            abstract=PLAIN_ABSTRACT,
            url="https://arxiv.org/pdf/1706.03762",
            license="cc-by",
            relevance_score=1.2024246507243719,
        )
    ]


def test_paper_id_is_the_bare_id_not_the_url():
    """`id` arrives as https://openalex.org/W… — callers want the W… on its own."""
    paper = OpenAlexClient._to_paper_data(item(id="https://openalex.org/W123"))

    assert paper.paper_id == "W123"


def test_doi_is_the_bare_identifier_not_the_resolver_url():
    """`doi` arrives as https://doi.org/10.… — callers want the 10.… on its own."""
    paper = OpenAlexClient._to_paper_data(item())

    assert paper.doi == "10.48550/arxiv.1706.03762"


def test_relevance_score_is_carried_through_as_a_float():
    """The upstream ranking, which the reranker breaks its own ties with."""
    paper = OpenAlexClient._to_paper_data(item(relevance_score=1.185))

    assert paper.relevance_score == 1.185


def test_a_null_relevance_score_is_zero():
    assert OpenAlexClient._to_paper_data(item(relevance_score=None)).relevance_score == 0.0


def test_a_record_with_no_relevance_score_at_all_is_zero():
    """OpenAlex omits the key on any request carrying no search parameter.

    It is also exactly what comes back if the field ever falls out of `select`, which
    is why the absent key is worth a case of its own rather than only the null one.
    """
    payload = item()
    del payload["relevance_score"]

    assert OpenAlexClient._to_paper_data(payload).relevance_score == 0.0


def test_a_missing_doi_is_an_empty_string():
    paper = OpenAlexClient._to_paper_data(item(doi=None))

    assert paper.doi == ""


def test_every_record_in_the_payload_is_mapped(monkeypatch, search):
    """Payload order is preserved, which is what the caller ranks by.

    OpenAlex returns results sorted by relevance_score descending. The score now
    rides along on each record, so a reordering here would no longer lose the
    ranking outright — but the pipeline still reads position as the order to rank
    in, and `remove_duplicates` restores that order rather than the score's.
    """
    handler, _ = recorder(
        ok(
            item(id="https://openalex.org/W1"),
            item(id="https://openalex.org/W2"),
            item(id="https://openalex.org/W3"),
        )
    )
    client = make_client(monkeypatch, handler)

    papers = getattr(client, search)("q")

    assert [paper.paper_id for paper in papers] == ["W1", "W2", "W3"]


def test_missing_fields_coerce_to_empties():
    """Every field is nullable upstream while PaperData requires all seven.

    Note the consequence for `year`: a missing year and a genuine year of 0 are
    indistinguishable downstream.
    """
    empty = OpenAlexClient._to_paper_data(
        {
            "id": None,
            "doi": None,
            "display_name": None,
            "publication_year": None,
            "abstract_inverted_index": None,
            "best_oa_location": None,
            "primary_location": None,
            "relevance_score": None,
        }
    )

    assert empty == PaperData(
        paper_id="",
        doi="",
        title="",
        year=0,
        abstract="",
        url="",
        license="",
        relevance_score=0.0,
    )


# --- picking the URL and the license ------------------------------------------


def test_open_access_pdf_wins_over_everything_else():
    """The point of asking for best_oa_location: a URL the caller can actually read."""
    paper = OpenAlexClient._to_paper_data(
        item(
            best_oa_location=location(
                pdf_url="https://arxiv.org/pdf/1706.03762",
                landing_page_url="http://arxiv.org/abs/1706.03762",
                license="cc-by",
            ),
            primary_location=location(pdf_url="https://publisher.test/paywalled.pdf"),
        )
    )

    assert paper.url == "https://arxiv.org/pdf/1706.03762"
    assert paper.license == "cc-by"


def test_landing_page_is_the_fallback_within_a_location():
    """Observed live: an OA location with a landing page but no PDF."""
    paper = OpenAlexClient._to_paper_data(
        item(
            best_oa_location=location(
                landing_page_url="https://figshare.test/article/1", license="cc-by"
            ),
            primary_location=location(pdf_url="https://publisher.test/paywalled.pdf"),
        )
    )

    assert paper.url == "https://figshare.test/article/1"
    assert paper.license == "cc-by"


@pytest.mark.parametrize("best", [None, {}, location()])
def test_primary_location_is_the_fallback(best):
    paper = OpenAlexClient._to_paper_data(
        item(
            best_oa_location=best,
            primary_location=location(
                landing_page_url="https://publisher.test/article", license="cc-by-nc"
            ),
        )
    )

    assert paper.url == "https://publisher.test/article"
    assert paper.license == "cc-by-nc"


def test_the_doi_url_fallback_leaves_the_doi_field_bare():
    """The two representations meet here: `url` stays a link, `doi` stays an id."""
    paper = OpenAlexClient._to_paper_data(item())

    assert paper.url == "https://doi.org/10.48550/arxiv.1706.03762"
    assert paper.doi == "10.48550/arxiv.1706.03762"


def test_doi_is_the_last_resort_and_carries_no_license():
    paper = OpenAlexClient._to_paper_data(item())

    assert paper.url == "https://doi.org/10.48550/arxiv.1706.03762"
    assert paper.license == ""


def test_url_is_empty_when_nothing_is_offered():
    paper = OpenAlexClient._to_paper_data(item(doi=None))

    assert paper.url == ""


def test_a_null_license_coerces_to_empty():
    """Observed live: a best_oa_location with a pdf_url but license: null."""
    paper = OpenAlexClient._to_paper_data(
        item(best_oa_location=location(pdf_url="https://arxiv.org/pdf/1901.00596"))
    )

    assert paper.url == "https://arxiv.org/pdf/1901.00596"
    assert paper.license == ""


# --- reconstructing the abstract ----------------------------------------------


def test_inverted_index_is_rebuilt_in_order():
    assert OpenAlexClient._abstract_from_inverted_index(INVERTED_ABSTRACT) == (
        PLAIN_ABSTRACT
    )


def test_a_repeated_word_lands_at_every_position():
    assert OpenAlexClient._abstract_from_inverted_index(
        {"all": [0, 2], "of": [1], "it": [3]}
    ) == "all of all it"


@pytest.mark.parametrize("index", [None, {}, {"word": []}])
def test_an_absent_abstract_is_an_empty_string(index):
    assert OpenAlexClient._abstract_from_inverted_index(index) == ""


def test_gaps_in_the_index_are_dropped():
    """A truncated index must not leave blank slots doubling up the separators."""
    assert OpenAlexClient._abstract_from_inverted_index(
        {"first": [0], "last": [4]}
    ) == "first last"


@pytest.mark.parametrize("payload", [{}, {"results": None}, {"results": []}])
def test_empty_payloads_yield_no_papers(monkeypatch, payload, search):
    handler, _ = recorder(httpx.Response(200, json=payload))
    client = make_client(monkeypatch, handler)

    assert getattr(client, search)("nonsense") == []


# --- failures -----------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_http_error_yields_an_empty_list(monkeypatch, caplog, status, search):
    """A failed search is empty, not an exception: the APIError stops at the client.

    Status errors are still not retried — a 400 stays a 400 however often it's sent.
    """
    handler, calls = recorder(httpx.Response(status, text="upstream said no"))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        assert getattr(client, search)("q") == []

    assert len(calls) == 1
    errors = logged_api_errors(caplog)
    assert len(errors) == 1
    assert f"API status {status}" in str(errors[0])


def test_status_error_carries_the_upstream_body(monkeypatch, caplog, search):
    """The caller loses the failure, so the log has to keep what upstream said."""
    handler, _ = recorder(
        httpx.Response(403, text='{"error":"Pagination error."}')
    )
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        assert getattr(client, search)("q") == []

    assert "Pagination error." in str(logged_api_errors(caplog)[0])


def test_transient_timeout_is_retried_then_succeeds(monkeypatch, caplog, search):
    """The reason the retry exists: one flaky attempt must not fail the search."""
    handler, calls = recorder(httpx.ReadTimeout("timed out"), ok(item()))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        papers = getattr(client, search)("q")

    assert len(calls) == 2
    assert [p.paper_id for p in papers] == ["W2626778328"]
    # A recovered attempt is not a failure, so nothing is logged.
    assert logged_api_errors(caplog) == []


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("timed out"), httpx.ConnectError("connection refused")],
    ids=["timeout", "connect"],
)
def test_network_failure_gives_up_after_three_attempts(
    monkeypatch, caplog, error, search
):
    handler, calls = recorder(error)
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR):
        assert getattr(client, search)("q") == []

    assert len(calls) == 3
    errors = logged_api_errors(caplog)
    assert len(errors) == 1
    assert "Network failure" in str(errors[0])


def test_malformed_json_body_is_not_wrapped(monkeypatch, search):
    """The boundary of the `except APIError` clause: only APIError is swallowed.

    A 200 carrying junk escapes as a JSON decode error, so the caller sees it
    rather than an empty list pretending the search simply found nothing.
    """
    handler, _ = recorder(httpx.Response(200, text="<html>maintenance</html>"))
    client = make_client(monkeypatch, handler)

    with pytest.raises(json.JSONDecodeError):
        getattr(client, search)("q")


def test_unexpected_error_propagates_to_the_caller(monkeypatch, caplog, search):
    """Anything that isn't an APIError is a bug, not a failed search — let it out."""
    handler, calls = recorder(RuntimeError("boom"))
    client = make_client(monkeypatch, handler)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="boom"):
        getattr(client, search)("q")

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


# --- citations ----------------------------------------------------------------
#
# Unlike the searches, `get_work` and `get_citing_works` raise APIError rather than
# swallowing it: get_citations turns a 404 into a "paper not found" the caller sees.


def citing(**overrides) -> dict:
    base = {
        "id": "https://openalex.org/W3000000001",
        "doi": "https://doi.org/10.1000/follow-up",
        "display_name": "A Follow-up",
        "publication_year": 2021,
        "best_oa_location": None,
        "primary_location": {
            **location(landing_page_url="https://example.test/paper"),
            "source": {"display_name": "NeurIPS"},
        },
        "cited_by_count": 42,
    }
    return base | overrides


def citing_page(*items, count=None) -> httpx.Response:
    return httpx.Response(
        200,
        json={"meta": {"count": len(items) if count is None else count},
              "results": list(items)},
    )


def test_citing_works_filter_on_the_target(monkeypatch):
    handler, calls = recorder(citing_page())
    client = make_client(monkeypatch, handler)

    client.get_citing_works("W2626778328")

    (request,) = calls
    assert request.url.path == "/openalex/works"
    assert request.url.params["filter"] == "cites:W2626778328"
    assert request.url.params["select"] == OpenAlexClient.CITATION_FIELDS
    assert "search" not in request.url.params


@pytest.mark.parametrize("sort, sent", [
    ("most_cited", "cited_by_count:desc"),
    ("newest", "publication_date:desc"),
])
def test_citing_works_sort_is_translated(monkeypatch, sort, sent):
    handler, calls = recorder(citing_page())
    client = make_client(monkeypatch, handler)

    client.get_citing_works("W1", sort=sort)

    assert calls[0].url.params["sort"] == sent


@pytest.mark.parametrize("limit, sent", [(0, 1), (10, 10), (500, MAX_LIMIT)])
def test_citing_works_limit_is_clamped(monkeypatch, limit, sent):
    handler, calls = recorder(citing_page())
    client = make_client(monkeypatch, handler)

    client.get_citing_works("W1", limit=limit)

    assert calls[0].url.params["per-page"] == str(sent)


def test_citing_works_map_venue_and_count_and_return_the_total(monkeypatch):
    handler, _ = recorder(citing_page(citing(), count=1234))
    client = make_client(monkeypatch, handler)

    papers, total = client.get_citing_works("W1")

    assert total == 1234
    assert papers == [CitingPaperData(
        paper_id="W3000000001",
        doi="10.1000/follow-up",
        title="A Follow-up",
        year=2021,
        abstract="",
        url="https://example.test/paper",
        license="",
        venue="NeurIPS",
        cited_by_count=42,
    )]


@pytest.mark.parametrize("primary", [None, location(), {**location(), "source": None}])
def test_a_missing_source_leaves_the_venue_empty(primary):
    paper = OpenAlexClient._to_citing_paper(
        citing(primary_location=primary, cited_by_count=None)
    )

    assert paper.venue == ""
    assert paper.cited_by_count == 0


def test_citing_works_raise_on_http_error(monkeypatch):
    handler, _ = recorder(httpx.Response(503, text="busy"))
    client = make_client(monkeypatch, handler)

    with pytest.raises(APIError, match="503"):
        client.get_citing_works("W1")


def test_get_work_resolves_an_external_id(monkeypatch):
    handler, calls = recorder(httpx.Response(200, json={
        "id": "https://openalex.org/W2626778328",
        "display_name": "Attention Is All You Need",
        "cited_by_count": 100000,
    }))
    client = make_client(monkeypatch, handler)

    work = client.get_work("doi:10.48550/arXiv.1706.03762")

    assert calls[0].url.path == "/openalex/works/doi:10.48550/arXiv.1706.03762"
    assert calls[0].url.params["select"] == OpenAlexClient.WORK_SUMMARY_FIELDS
    assert work == {
        "paper_id": "W2626778328",
        "title": "Attention Is All You Need",
        "cited_by_count": 100000,
    }


def test_get_work_raises_on_404(monkeypatch):
    handler, _ = recorder(httpx.Response(404, text="not found"))
    client = make_client(monkeypatch, handler)

    with pytest.raises(APIError, match="404"):
        client.get_work("W0")
