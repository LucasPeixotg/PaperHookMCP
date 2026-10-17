"""get_author_top_works against a stubbed OpenAlex client — no network.

Identifier parsing itself is pinned in tests/test_paper_id.py; here only how each kind
is resolved: an OpenAlex author id or ORCID is looked up directly with no search, and a
name is searched and must match exactly one author. Several matches are a ToolError
listing the candidates' ids, raised before any works are fetched, because silently
picking a namesake yields a plausible but wrong publication list.

Beyond that this pins the response shape — a bare list of the works, without their
`abstract` and `relevance_score` — and that every APIError reaches the caller as a
ToolError rather than an empty list, since "no such author" and "no works" must not
look alike.
"""

import importlib
import json

import pytest
from fastmcp.exceptions import ToolError

from common import CitingPaperData
from services.api.api_client import APIError

# `tools.get_author_top_works` the attribute is the function (re-exported by
# tools/__init__), so the module is fetched by name.
module = importlib.import_module("tools.get_author_top_works")


def candidate(author_id, **overrides):
    return {
        "author_id": author_id, "name": "Ada Lovelace", "orcid": "",
        "institution": "", "works_count": 12, "cited_by_count": 3400,
    } | overrides


class StubClient:
    def __init__(self, *, candidates=None, author_error=None, search_error=None,
                 works_error=None):
        self.candidates = [candidate("A1")] if candidates is None else candidates
        self.author_error = author_error
        self.search_error = search_error
        self.works_error = works_error
        self.calls = []

    def get_author(self, key):
        self.calls.append(("get_author", key))
        if self.author_error:
            raise self.author_error
        return candidate("A1")

    def search_authors(self, name, limit):
        self.calls.append(("search_authors", name))
        if self.search_error:
            raise self.search_error
        return self.candidates

    def get_author_works(self, author_id, limit, publication_year_min):
        self.calls.append(("get_author_works", author_id, limit, publication_year_min))
        if self.works_error:
            raise self.works_error
        paper = CitingPaperData(
            paper_id="W2", doi="10.1/x", title="Notes", year=1843, abstract="An abstract",
            url="https://example.test", license="cc-by", relevance_score=0.0,
            venue="Taylor's Scientific Memoirs", cited_by_count=3,
        )
        return [paper], 1


def use(monkeypatch, client):
    monkeypatch.setattr(module, "open_alex_client", client)
    return client


@pytest.fixture
def stub(monkeypatch):
    return use(monkeypatch, StubClient())


@pytest.mark.parametrize("author, key", [
    ("https://openalex.org/a1", "A1"),
    ("0000-0002-1825-0097", "orcid:0000-0002-1825-0097"),
])
def test_an_identifier_is_looked_up_without_searching(stub, author, key):
    module.get_author_top_works(author, limit=5, publication_year_min=2020)

    assert stub.calls == [
        ("get_author", key),
        ("get_author_works", "A1", 5, 2020),
    ]


def test_a_name_with_one_match_is_used(stub):
    module.get_author_top_works("Ada Lovelace")

    assert stub.calls == [
        ("search_authors", "Ada Lovelace"),
        ("get_author_works", "A1", 10, None),
    ]


def test_an_ambiguous_name_lists_the_candidates(monkeypatch):
    client = use(monkeypatch, StubClient(candidates=[
        candidate("A1", orcid="0000-0002-1825-0097", institution="University of London"),
        candidate("A2"),
    ]))

    with pytest.raises(ToolError, match="matches several") as raised:
        module.get_author_top_works("Ada Lovelace")

    message = str(raised.value)
    assert "A1 — Ada Lovelace (ORCID 0000-0002-1825-0097; University of London; " \
           "12 works, 3400 citations)" in message
    assert "A2 — Ada Lovelace (12 works, 3400 citations)" in message
    assert [call[0] for call in client.calls] == ["search_authors"]


def test_a_name_with_no_match_is_not_found(monkeypatch):
    use(monkeypatch, StubClient(candidates=[]))

    with pytest.raises(ToolError, match="Author not found"):
        module.get_author_top_works("Nobody At All")


def test_response_shape(stub):
    result = json.loads(module.get_author_top_works("A1"))

    assert result == [{
        "paper_id": "W2", "doi": "10.1/x", "title": "Notes", "year": 1843,
        "url": "https://example.test", "license": "cc-by",
        "venue": "Taylor's Scientific Memoirs", "cited_by_count": 3,
    }]


def test_an_unknown_author_is_a_tool_error(monkeypatch):
    use(monkeypatch, StubClient(author_error=APIError("API status 404: nope")))

    with pytest.raises(ToolError, match="Author not found") as raised:
        module.get_author_top_works("A1")

    assert "nope" not in str(raised.value)


def test_a_failed_lookup_is_not_reported_as_not_found(monkeypatch):
    use(monkeypatch, StubClient(author_error=APIError("Network failure: refused")))

    with pytest.raises(ToolError, match="Could not look up .*refused"):
        module.get_author_top_works("A1")


def test_a_failed_search_is_not_reported_as_not_found(monkeypatch):
    use(monkeypatch, StubClient(search_error=APIError("API status 503: busy")))

    with pytest.raises(ToolError, match="Could not search for author .*busy"):
        module.get_author_top_works("Ada Lovelace")


def test_a_failed_works_fetch_is_a_tool_error(monkeypatch):
    use(monkeypatch, StubClient(works_error=APIError("API status 503: busy")))

    with pytest.raises(ToolError, match="Could not fetch works"):
        module.get_author_top_works("A1")
