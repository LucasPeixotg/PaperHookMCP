"""fetch_full_paper against a stubbed OpenAlex client and PDF client — no network.

Pins the order of work: a full text already in the paper store is returned with no
request at all; otherwise the paper is looked up for its `oa_pdf`, which is downloaded,
extracted and stored. A paper with no `oa_pdf` is a ToolError naming its `oa_url`, or
failing that its `url`, for the caller to fetch manually; so is every failure on the
way — none of which is cached.
"""

import importlib
import json

import diskcache
import pytest
from fastmcp.exceptions import ToolError

from paperhook.common import PaperData
from paperhook.services.api.api_client import APIError
from paperhook.services.cache import PaperStore

# `tools.fetch_full_paper` the attribute is the function, so the module is fetched by name.
module = importlib.import_module("paperhook.tools.fetch_full_paper")

PDF_URL = "https://oa.example.test/paper.pdf"


def paper(**overrides) -> PaperData:
    base = dict(paper_id="W1", doi="10.1/x", title="Target", year=2020, abstract="",
                url=PDF_URL, license="cc-by", oa_pdf=PDF_URL, oa_url="")
    return PaperData(**(base | overrides))


class StubOpenAlex:
    def __init__(self, result=None, error=None):
        self.result = result or paper()
        self.error = error
        self.calls = []

    def get_paper(self, work_id):
        self.calls.append(work_id)
        if self.error:
            raise self.error
        return self.result


class StubPdf:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def download(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        return b"%PDF"


@pytest.fixture
def store(tmp_path, monkeypatch):
    cache = diskcache.Cache(str(tmp_path))
    paper_store = PaperStore(cache)
    monkeypatch.setattr(module, "get_paper_store", lambda: paper_store)
    yield paper_store
    cache.close()


@pytest.fixture
def stubs(monkeypatch, store):
    """Stubbed clients, with extraction returning a fixed text."""
    open_alex, pdf = StubOpenAlex(), StubPdf()
    monkeypatch.setattr(module, "open_alex_client", open_alex)
    monkeypatch.setattr(module, "pdf_client", pdf)
    monkeypatch.setattr(module, "extract_text", lambda _: "Full text.")
    return open_alex, pdf


def test_an_unrecognised_identifier_is_a_tool_error(stubs):
    with pytest.raises(ToolError, match="Unrecognised paper identifier"):
        module.fetch_full_paper("10.1/x")

    assert stubs[0].calls == []


def test_the_pdf_is_fetched_extracted_and_stored(stubs, store):
    open_alex, pdf = stubs

    result = json.loads(module.fetch_full_paper("https://openalex.org/w1"))

    assert open_alex.calls == ["W1"]
    assert pdf.calls == [PDF_URL]
    assert result == {"paper_id": "W1", "doi": "10.1/x", "title": "Target",
                      "source_url": PDF_URL, "full_text": "Full text."}
    assert store.get_full_paper("W1") == "Full text."


def test_a_cached_full_text_makes_no_request(stubs, store):
    store.absorb([paper()])
    store.put_full_paper("W1", "Cached text.")

    result = json.loads(module.fetch_full_paper("W1"))

    assert stubs[0].calls == stubs[1].calls == []
    assert result == {"paper_id": "W1", "doi": "10.1/x", "title": "Target",
                      "source_url": PDF_URL, "full_text": "Cached text."}


def test_the_second_call_is_answered_from_the_store(stubs, store):
    # The real `get_paper` writes the record it returns into the store.
    store.absorb([paper()])

    first = module.fetch_full_paper("W1")
    second = module.fetch_full_paper("W1")

    assert first == second
    assert len(stubs[0].calls) == len(stubs[1].calls) == 1


def test_no_oa_pdf_is_a_tool_error_naming_the_oa_url(stubs, store):
    stubs[0].result = paper(oa_pdf="", oa_url="https://oa.example.test/landing")

    with pytest.raises(ToolError, match="No open-access PDF") as error:
        module.fetch_full_paper("W1")

    assert "open-access URL (oa_url) is https://oa.example.test/landing" in str(error.value)
    assert stubs[1].calls == []


def test_without_an_oa_url_the_error_names_the_other_url(stubs, store):
    stubs[0].result = paper(oa_pdf="", oa_url="", url="https://publisher.example.test/x")

    with pytest.raises(ToolError, match="No open-access PDF") as error:
        module.fetch_full_paper("W1")

    message = str(error.value)
    assert "no open-access URL (oa_url)" in message
    assert "https://publisher.example.test/x" in message
    assert stubs[1].calls == []


def test_with_no_url_at_all_the_error_says_so(stubs, store):
    stubs[0].result = paper(oa_pdf="", oa_url="", url="")

    with pytest.raises(ToolError, match="No URL is known"):
        module.fetch_full_paper("W1")


def test_an_unknown_paper_is_not_found(stubs):
    stubs[0].error = APIError("API status 404: <html>")

    with pytest.raises(ToolError, match="Paper not found in OpenAlex"):
        module.fetch_full_paper("W1")


def test_a_failed_lookup_is_a_tool_error(stubs):
    stubs[0].error = APIError("Network failure: boom")

    with pytest.raises(ToolError, match="Could not look up"):
        module.fetch_full_paper("W1")


def test_a_failed_download_is_a_tool_error_and_not_cached(stubs, store):
    stubs[1].error = APIError("The link did not serve a PDF")

    with pytest.raises(ToolError, match="did not serve a PDF"):
        module.fetch_full_paper("W1")

    assert store.get_full_paper("W1") is None


def test_a_pdf_without_text_is_a_tool_error_and_not_cached(monkeypatch, stubs, store):
    monkeypatch.setattr(module, "extract_text", lambda _: "")

    with pytest.raises(ToolError, match="no extractable text"):
        module.fetch_full_paper("W1")

    assert store.get_full_paper("W1") is None


def test_without_a_store_the_text_is_still_returned(monkeypatch, stubs):
    monkeypatch.setattr(module, "get_paper_store", lambda: None)

    result = json.loads(module.fetch_full_paper("W1"))

    assert result["full_text"] == "Full text."
