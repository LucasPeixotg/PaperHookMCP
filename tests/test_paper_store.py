"""PaperStore over a real diskcache under tmp_path — no network.

The contracts pinned here:

* A paper is keyed by its `paper_id`, with its DOI (any case) as the only alias. A
  matching title alone identifies nothing.
* `relevance_score` is never stored: it belongs to the query that produced it.
* Enrichment fills only blanks, and fresh non-blank fields win. A record reached
  through the DOI under a different `paper_id` lends its abstract and nothing else.
* `absorb` never mutates its input, and a failing cache degrades to no cache.
"""

import time

import diskcache
import pytest

from common import CitingPaperData, PaperData
from services.cache import PaperStore


@pytest.fixture
def store(tmp_path):
    cache = diskcache.Cache(str(tmp_path))
    yield PaperStore(cache)
    cache.close()


def paper(**overrides) -> PaperData:
    base = dict(
        paper_id="W1",
        doi="10.1000/Example.1",
        title="Attention Is All You Need",
        year=2017,
        abstract="The dominant sequence models",
        url="https://example.test/w1.pdf",
        license="cc-by",
        relevance_score=1.5,
    )
    return PaperData(**(base | overrides))


def test_absorbed_paper_is_found_by_id_and_doi_without_its_score(store):
    store.absorb([paper()])

    record = store.get("W1")
    assert record["title"] == "Attention Is All You Need"
    assert "relevance_score" not in record

    by_doi = store.find(paper(paper_id="", doi="10.1000/EXAMPLE.1"))
    assert by_doi == record


def test_same_id_fills_blanks_and_fresh_fields_win(store):
    store.absorb([paper()])

    [fresh] = store.absorb([paper(abstract="", url="https://example.test/new.pdf")])

    assert fresh.abstract == "The dominant sequence models"
    assert fresh.url == "https://example.test/new.pdf"
    assert store.get("W1")["url"] == "https://example.test/new.pdf"
    assert store.get("W1")["abstract"] == "The dominant sequence models"


def test_relevance_score_is_passed_through(store):
    store.absorb([paper(relevance_score=0.3)])

    [fresh] = store.absorb([paper(relevance_score=0.9)])

    assert fresh.relevance_score == 0.9


def test_doi_match_under_another_id_lends_only_the_abstract(store):
    store.absorb([paper()])

    [other] = store.absorb([paper(paper_id="W2", abstract="", url="", license="")])

    assert other.abstract == "The dominant sequence models"
    assert other.url == ""
    assert other.license == ""


def test_title_alone_does_not_enrich(store):
    store.absorb([paper()])

    [other] = store.absorb([paper(paper_id="W2", doi="", abstract="")])

    assert other.abstract == ""


def test_input_is_not_mutated(store):
    store.absorb([paper()])
    incoming = paper(abstract="")

    store.absorb([incoming])

    assert incoming.abstract == ""


def test_citing_paper_keeps_its_fresh_zero_count(store):
    store.absorb([CitingPaperData(**vars(paper()), venue="NeurIPS", cited_by_count=5)])

    [fresh] = store.absorb(
        [CitingPaperData(**vars(paper()), venue="", cited_by_count=0)]
    )

    assert fresh.venue == "NeurIPS"
    assert fresh.cited_by_count == 0
    assert store.get("W1")["cited_by_count"] == 0


def test_put_summary_merges_into_the_record(store):
    store.absorb([paper()])

    store.put_summary("W1", "Attention Is All You Need", 100)

    record = store.get("W1")
    assert record["cited_by_count"] == 100
    assert record["abstract"] == "The dominant sequence models"


def test_expired_entries_are_gone(tmp_path):
    with diskcache.Cache(str(tmp_path)) as cache:
        store = PaperStore(cache, ttl_seconds=0.01)
        store.absorb([paper()])
        time.sleep(0.05)

        assert store.get("W1") is None


class BrokenCache:
    def get(self, *args, **kwargs):
        raise OSError("database is locked")

    def set(self, *args, **kwargs):
        raise OSError("database is locked")


def test_broken_cache_leaves_papers_unchanged():
    store = PaperStore(BrokenCache())
    incoming = [paper(), paper(paper_id="W2")]

    assert store.absorb(incoming) == incoming
    assert store.get("W1") is None
