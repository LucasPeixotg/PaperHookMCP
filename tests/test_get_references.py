"""get_references against a stubbed OpenAlex client — no network.

The mirror of tests/test_get_citations.py, pinning the same contracts for the backward
direction: identifier parsing failing as a ToolError before any request goes out, a
bare list of the referenced papers without the target paper or their `abstract` and
`relevance_score`, and every APIError reaching the caller as a ToolError rather than
an empty list, since "no such paper" and "no references" must not look alike.
"""

import importlib
import json

import pytest
from fastmcp.exceptions import ToolError

from common import CitingPaperData
from services.api.api_client import APIError

# `tools.get_references` the attribute is the function (re-exported by tools/__init__),
# so the module is fetched by name.
module = importlib.import_module("tools.get_references")


def test_an_unrecognised_identifier_is_a_tool_error(stub):
    with pytest.raises(ToolError, match="Unrecognised paper identifier"):
        module.get_references("attention is all you need")

    assert stub.calls == []


def test_a_doi_is_rejected(stub):
    with pytest.raises(ToolError, match="DOIs and arXiv IDs are not accepted"):
        module.get_references("10.1038/x")

    assert stub.calls == []


class StubClient:
    def __init__(self, *, work_error=None, referenced_error=None):
        self.work_error = work_error
        self.referenced_error = referenced_error
        self.calls = []

    def get_work(self, key):
        self.calls.append(("get_work", key))
        if self.work_error:
            raise self.work_error
        return {"paper_id": "W1", "title": "Target", "cited_by_count": 7}

    def get_referenced_works(self, work_id, limit, sort):
        self.calls.append(("get_referenced_works", work_id, limit, sort))
        if self.referenced_error:
            raise self.referenced_error
        paper = CitingPaperData(
            paper_id="W0", doi="10.1/x", title="Prior Work", year=2015, abstract="",
            url="https://example.test", license="cc-by",
            oa_pdf="https://example.test/oa.pdf", oa_url="https://example.test/oa",
            relevance_score=0.0,
            venue="NeurIPS", cited_by_count=900,
        )
        return [paper], 30


@pytest.fixture
def stub(monkeypatch):
    client = StubClient()
    monkeypatch.setattr(module, "open_alex_client", client)
    return client


def test_references_are_fetched_for_the_resolved_id(stub):
    module.get_references("https://openalex.org/w1", limit=5, sort="newest")

    assert stub.calls == [
        ("get_work", "W1"),
        ("get_referenced_works", "W1", 5, "newest"),
    ]


def test_response_shape(stub):
    result = json.loads(module.get_references("W1"))

    assert result == [{
        "paper_id": "W0", "doi": "10.1/x", "title": "Prior Work", "year": 2015,
        "url": "https://example.test", "license": "cc-by",
        "oa_pdf": "https://example.test/oa.pdf", "oa_url": "https://example.test/oa",
        "venue": "NeurIPS", "cited_by_count": 900,
    }]


def test_an_unknown_paper_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(module, "open_alex_client",
                        StubClient(work_error=APIError("API status 404: nope")))

    with pytest.raises(ToolError, match="Paper not found") as raised:
        module.get_references("W1")

    assert "nope" not in str(raised.value)


def test_a_failed_lookup_is_not_reported_as_not_found(monkeypatch):
    monkeypatch.setattr(module, "open_alex_client",
                        StubClient(work_error=APIError("Network failure: refused")))

    with pytest.raises(ToolError, match="Could not look up .*refused"):
        module.get_references("W1")


def test_a_failed_reference_fetch_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(module, "open_alex_client",
                        StubClient(referenced_error=APIError("API status 503: busy")))

    with pytest.raises(ToolError, match="Could not fetch references"):
        module.get_references("W1")
