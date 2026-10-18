"""get_citations against a stubbed OpenAlex client — no network.

Identifier parsing itself is pinned in tests/test_paper_id.py; here only its failure
reaching the caller as a ToolError, before any request goes out. Beyond that this pins
the response shape — a bare list of the citing papers, without the target paper or
their `abstract` and `relevance_score` — and that every
APIError reaches the caller as a ToolError rather than an empty list, since "no such
paper" and "no citations" must not look alike.
"""

import importlib
import json

import pytest
from fastmcp.exceptions import ToolError

from common import CitingPaperData
from services.api.api_client import APIError

# `tools.get_citations` the attribute is the function (re-exported by tools/__init__),
# so the module is fetched by name.
module = importlib.import_module("tools.get_citations")


def test_an_unrecognised_identifier_is_a_tool_error(stub):
    with pytest.raises(ToolError, match="Unrecognised paper identifier"):
        module.get_citations("attention is all you need")

    assert stub.calls == []


def test_a_doi_is_rejected(stub):
    with pytest.raises(ToolError, match="DOIs and arXiv IDs are not accepted"):
        module.get_citations("10.1038/x")

    assert stub.calls == []


class StubClient:
    def __init__(self, *, work_error=None, citing_error=None):
        self.work_error = work_error
        self.citing_error = citing_error
        self.calls = []

    def get_work(self, key):
        self.calls.append(("get_work", key))
        if self.work_error:
            raise self.work_error
        return {"paper_id": "W1", "title": "Target", "cited_by_count": 7}

    def get_citing_works(self, work_id, limit, sort):
        self.calls.append(("get_citing_works", work_id, limit, sort))
        if self.citing_error:
            raise self.citing_error
        paper = CitingPaperData(
            paper_id="W2", doi="10.1/x", title="Follow-up", year=2022, abstract="",
            url="https://example.test", license="cc-by",
            oa_pdf="https://example.test/oa.pdf", oa_url="https://example.test/oa",
            relevance_score=0.0,
            venue="ICML", cited_by_count=3,
        )
        return [paper], 7


@pytest.fixture
def stub(monkeypatch):
    client = StubClient()
    monkeypatch.setattr(module, "open_alex_client", client)
    return client


def test_citations_are_fetched_for_the_resolved_id(stub):
    module.get_citations("https://openalex.org/w1", limit=5, sort="newest")

    assert stub.calls == [
        ("get_work", "W1"),
        ("get_citing_works", "W1", 5, "newest"),
    ]


def test_response_shape(stub):
    result = json.loads(module.get_citations("W1"))

    assert result == [{
        "paper_id": "W2", "doi": "10.1/x", "title": "Follow-up", "year": 2022,
        "url": "https://example.test", "license": "cc-by",
        "oa_pdf": "https://example.test/oa.pdf", "oa_url": "https://example.test/oa",
        "venue": "ICML", "cited_by_count": 3,
    }]


def test_an_unknown_paper_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(module, "open_alex_client",
                        StubClient(work_error=APIError("API status 404: nope")))

    with pytest.raises(ToolError, match="Paper not found") as raised:
        module.get_citations("W1")

    assert "nope" not in str(raised.value)


def test_a_failed_lookup_is_not_reported_as_not_found(monkeypatch):
    monkeypatch.setattr(module, "open_alex_client",
                        StubClient(work_error=APIError("Network failure: refused")))

    with pytest.raises(ToolError, match="Could not look up .*refused"):
        module.get_citations("W1")


def test_a_failed_citation_fetch_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(module, "open_alex_client",
                        StubClient(citing_error=APIError("API status 503: busy")))

    with pytest.raises(ToolError, match="Could not fetch citations"):
        module.get_citations("W1")
