"""PaperStore over a real diskcache under tmp_path — no network.

The contracts pinned here:

* A paper is keyed by its `paper_id`, with its DOI (any case) as the only alias. A
  matching title alone identifies nothing.
* `relevance_score` is never stored: it belongs to the query that produced it.
* Enrichment fills only blanks, and fresh non-blank fields win. A record reached
  through the DOI under a different `paper_id` lends its abstract and nothing else.
* `absorb` never mutates its input, and a failing cache degrades to no cache.
* `full_paper` is reached by `paper_id` or DOI, survives later merges into the record,
  and never appears on a PaperData.
* A request is cached as pointers: its ids in order, with the per-request
  `relevance_score` and total, rebuilt from the shared records on a hit. A pointer to
  any missing record is a miss, and a paper without an id leaves the request uncached.
* Authors are kept by `author_id` with their ORCID (any case) as an alias, and author
  searches by pointer the same way.
"""

import time

import diskcache
import pytest

from common import CitingPaperData, PaperData
from services.cache import PaperStore, request_key


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
        oa_pdf="https://example.test/w1-oa.pdf",
        oa_url="https://example.test/w1-oa",
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

    [fresh] = store.absorb(
        [paper(abstract="", oa_pdf="", url="https://example.test/new.pdf")]
    )

    assert fresh.abstract == "The dominant sequence models"
    assert fresh.oa_pdf == "https://example.test/w1-oa.pdf"
    assert fresh.url == "https://example.test/new.pdf"
    assert store.get("W1")["url"] == "https://example.test/new.pdf"
    assert store.get("W1")["abstract"] == "The dominant sequence models"


def test_relevance_score_is_passed_through(store):
    store.absorb([paper(relevance_score=0.3)])

    [fresh] = store.absorb([paper(relevance_score=0.9)])

    assert fresh.relevance_score == 0.9


def test_doi_match_under_another_id_lends_only_the_abstract(store):
    store.absorb([paper()])

    [other] = store.absorb([paper(paper_id="W2", abstract="", url="", license="",
                                  oa_pdf="", oa_url="")])

    assert other.abstract == "The dominant sequence models"
    assert other.url == ""
    assert other.license == ""
    assert (other.oa_pdf, other.oa_url) == ("", "")


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


FULL_TEXT = "1 Introduction\nThe dominant sequence models..."


def test_full_paper_is_found_by_id_and_doi(store):
    store.put_full_paper("W1", FULL_TEXT, doi="10.1000/Example.1")

    assert store.get_full_paper("W1") == FULL_TEXT
    assert store.get_full_paper(doi="10.1000/EXAMPLE.1") == FULL_TEXT


def test_full_paper_survives_absorb_and_summary(store):
    store.put_full_paper("W1", FULL_TEXT)

    store.absorb([paper()])
    store.put_summary("W1", "Attention Is All You Need", 100)

    assert store.get_full_paper("W1") == FULL_TEXT


def test_full_paper_never_lands_on_a_paper(store):
    store.absorb([paper()])
    store.put_full_paper("W1", FULL_TEXT)

    [fresh] = store.absorb([paper(abstract="")])

    assert not hasattr(fresh, "full_paper")


def test_missing_full_paper_is_none(store):
    store.absorb([paper()])

    assert store.get_full_paper("W1") is None
    assert store.get_full_paper("W404") is None
    assert store.get_full_paper() is None


def test_broken_cache_full_paper_degrades():
    store = PaperStore(BrokenCache())

    store.put_full_paper("W1", FULL_TEXT, doi="10.1/x")

    assert store.get_full_paper("W1", doi="10.1/x") is None


# --- request pointers -----------------------------------------------------------

SEARCH = request_key("GET", "/works", {"search": "attention", "per-page": 10})
LISTING = request_key("GET", "/works", {"filter": "cites:W9", "per-page": 10})


def test_request_key_ignores_parameter_order_but_not_values():
    assert request_key("GET", "/works", {"a": 1, "b": 2}) == request_key(
        "get", "/works", {"b": 2, "a": 1}
    )
    assert request_key("GET", "/works", {"a": 1}) != request_key(
        "GET", "/works", {"a": 2}
    )
    assert request_key("GET", "/works", {"a": 1}) != request_key(
        "GET", "/authors", {"a": 1}
    )


def test_pointers_round_trip_in_order_with_their_scores(store):
    papers = store.absorb([paper(paper_id="W2", relevance_score=0.9), paper()])

    store.put_pointers(SEARCH, papers)

    hit, total = store.get_pointers(SEARCH)
    assert [p.paper_id for p in hit] == ["W2", "W1"]
    assert [p.relevance_score for p in hit] == [0.9, 1.5]
    assert hit[1] == paper()
    assert total is None


def test_a_record_without_a_field_rebuilds_it_blank(store):
    store.put_pointers(SEARCH, store.absorb([paper(abstract="", license="", oa_url="")]))

    [hit], _ = store.get_pointers(SEARCH)

    assert hit.abstract == ""
    assert hit.license == ""
    assert hit.oa_url == ""


def test_requests_sharing_a_paper_share_its_record(store):
    store.put_pointers(SEARCH, store.absorb([paper()]))
    listed = store.absorb(
        [CitingPaperData(**vars(paper(relevance_score=0.0)), cited_by_count=5)]
    )
    store.put_pointers(LISTING, listed, total=40)

    store.put_summary("W1", "Attention Is All You Need", 99)

    [searched], _ = store.get_pointers(SEARCH, PaperData)
    [cited], total = store.get_pointers(LISTING, CitingPaperData)
    assert type(searched) is PaperData
    assert searched.relevance_score == 1.5
    assert cited.cited_by_count == 99
    assert total == 40


def test_an_empty_result_is_a_hit(store):
    store.put_pointers(SEARCH, [])

    assert store.get_pointers(SEARCH) == ([], None)


def test_an_unknown_request_is_a_miss(store):
    assert store.get_pointers(SEARCH) is None


def test_a_pointer_to_a_missing_record_is_a_miss(store):
    store.put_pointers(SEARCH, [paper(), paper(paper_id="W2")])
    store.absorb([paper()])

    assert store.get_pointers(SEARCH) is None


def test_a_paper_without_an_id_leaves_the_request_uncached(store):
    store.put_pointers(SEARCH, store.absorb([paper(), paper(paper_id="")]))

    assert store.get_pointers(SEARCH) is None


def test_full_paper_never_lands_on_a_rebuilt_paper(store):
    store.put_pointers(SEARCH, store.absorb([paper()]))
    store.put_full_paper("W1", FULL_TEXT)

    [hit], _ = store.get_pointers(SEARCH)

    assert not hasattr(hit, "full_paper")


def test_pointers_expire_on_their_own_ttl(tmp_path):
    with diskcache.Cache(str(tmp_path)) as cache:
        store = PaperStore(cache, request_ttl_seconds=0.01)
        store.put_pointers(SEARCH, store.absorb([paper()]))
        time.sleep(0.05)

        assert store.get_pointers(SEARCH) is None
        assert store.get("W1") is not None


def test_work_pointer_resolves_to_its_paper_id(store):
    key = request_key("GET", "/works/doi:10.1/x", {})
    store.put_work_pointer(key, "W1")

    assert store.get_work_pointer(key) == "W1"
    assert store.get_work_pointer(SEARCH) is None


def test_broken_cache_pointers_degrade():
    store = PaperStore(BrokenCache())

    store.put_pointers(SEARCH, [paper()])

    assert store.get_pointers(SEARCH) is None


# --- authors ----------------------------------------------------------------------

ADA = {
    "author_id": "A1",
    "name": "Ada Lovelace",
    "orcid": "0000-0002-1825-009X",
    "institution": "University of London",
    "works_count": 12,
    "cited_by_count": 3400,
}
AUTHORS = request_key("GET", "/authors", {"search": "ada"})


def test_author_is_found_by_id_and_orcid(store):
    store.put_authors([ADA])

    assert store.get_author("A1") == ADA
    assert store.get_author("orcid:0000-0002-1825-009x") == ADA
    assert store.get_author("orcid:0000-0000-0000-0000") is None
    assert store.get_author("A404") is None


def test_author_pointers_round_trip_and_miss_on_a_gap(store):
    babbage = ADA | {"author_id": "A2", "name": "Charles Babbage", "orcid": ""}
    store.put_authors([ADA])
    store.put_author_pointers(AUTHORS, [babbage, ADA])

    assert store.get_author_pointers(AUTHORS) is None

    store.put_authors([babbage])
    assert store.get_author_pointers(AUTHORS) == [babbage, ADA]
