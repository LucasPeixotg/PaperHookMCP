"""remove_duplicates — pure logic, no transport, no fixtures beyond conftest.

The interesting cases are the ones the function exists for: the same paper arriving
from two sources under different DOI casing, the same paper arriving with no DOI or
with two different ones — caught by its title and abstract together, which never match
byte for byte — the preprint that must lose to the version that came out of review, and
the many records missing one of those fields, which must never be judged duplicates on
information they do not carry.
"""

import pytest

from paperhook.common import PaperData
from paperhook.tools.find_relevant_papers.paper_deduplicator import remove_duplicates

# The same work as OpenAlex and as Semantic Scholar publish it. Identical DOIs by
# specification, different strings by casing — the pair the function has to catch.
OPENALEX_DOI = "10.48550/arxiv.1706.03762"
SEMANTIC_SCHOLAR_DOI = "10.48550/arXiv.1706.03762"

# A real-length abstract, because similarity is a ratio: an edit that a journal version
# genuinely makes has to stay above the threshold, and on a one-line stub nothing does.
ABSTRACT = (
    "The dominant sequence transduction models are based on complex recurrent or "
    "convolutional neural networks that include an encoder and a decoder. The best "
    "performing models also connect the encoder and decoder through an attention "
    "mechanism. We propose a new simple network architecture, the Transformer, based "
    "solely on attention mechanisms, dispensing with recurrence and convolutions "
    "entirely."
)

# The same abstract after review: punctuation drift plus a sentence of results.
REVISED_ABSTRACT = (
    ABSTRACT.replace("a new simple network", "a new, simple network")
    + " We report state of the art results."
)

# A real title pair from OpenAlex: the AAAI version tags the venue, the preprint does
# not. Its length is the point — the same tag on a short title would score far lower.
STUDENT_ABSTRACT_TITLE = (
    "Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an "
    "Alternative to Attention Layers in Transformers"
)

PUBLISHED_DOI = "10.1000/journal.1"
PUBLISHED_URL = "https://journals.example.org/10.1000/journal.1"


def paper(**overrides) -> PaperData:
    """A complete PaperData so a test can name only the field it cares about.

    Defaults to the arXiv copy — an arXiv DOI on an arXiv URL — so a test that says
    nothing about publication status is talking about a preprint.
    """
    base = {
        "paper_id": "W1",
        "doi": OPENALEX_DOI,
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": ABSTRACT,
        "url": "https://arxiv.org/pdf/1706.03762",
        "license": "cc-by",
    }
    return PaperData(**(base | overrides))


def published(**overrides) -> PaperData:
    """The journal copy of the same work: a publisher DOI, no preprint host."""
    return paper(**({"doi": PUBLISHED_DOI, "url": PUBLISHED_URL} | overrides))


def test_two_copies_of_one_work_collapse_to_the_first_on_a_tie():
    """With nothing to separate two preprints, position decides — it is the ranking."""
    papers = [paper(paper_id="first"), paper(paper_id="second")]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["first"]


def test_order_is_preserved():
    """Position in the list is the ranking, so a duplicate must not reshuffle it."""
    papers = [
        paper(paper_id="W1", doi="10.1/a", title="a"),
        paper(paper_id="W2", doi="10.1/b", title="b"),
        paper(paper_id="W3", doi="10.1/a", title="c"),
        paper(paper_id="W4", doi="10.1/c", title="d"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1", "W2", "W4"]


def test_doi_matching_is_case_insensitive():
    """The cross-source case: OpenAlex lowercases, Semantic Scholar does not."""
    papers = [
        paper(paper_id="W2626778328", doi=OPENALEX_DOI, title="a"),
        paper(paper_id="p1", doi=SEMANTIC_SCHOLAR_DOI, title="b"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W2626778328"]


def test_papers_with_distinct_dois_all_survive():
    papers = [
        paper(paper_id="W1", doi="10.1/a", title="a"),
        paper(paper_id="W2", doi="10.1/b", title="b"),
    ]

    assert remove_duplicates(papers) == papers


# --- the title + abstract pair ------------------------------------------------


def test_identical_papers_without_a_doi_are_deduplicated_by_title_and_abstract():
    """The preprint case: no DOI to match on, so the pair is all there is."""
    papers = [paper(paper_id="first", doi=""), paper(paper_id="second", doi="")]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["first"]


def test_the_pair_deduplicates_across_differing_dois():
    """Preprint versus published: one work, two DOIs. Matching is DOI *or* pair."""
    papers = [
        paper(paper_id="preprint", doi=OPENALEX_DOI),
        published(paper_id="published"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["published"]


def test_the_same_title_with_a_different_abstract_is_not_a_duplicate():
    """The pair matches as a unit — a shared title alone proves nothing."""
    papers = [
        paper(paper_id="W1", doi="", abstract="one"),
        paper(paper_id="W2", doi="", abstract="two"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1", "W2"]


def test_the_same_abstract_with_a_different_title_is_not_a_duplicate():
    """The other half of the same rule."""
    papers = [
        paper(paper_id="W1", doi="", title="one"),
        paper(paper_id="W2", doi="", title="two"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1", "W2"]


def test_pair_matching_ignores_case_and_surrounding_whitespace():
    """Sources disagree on casing and leave stray whitespace behind in extraction."""
    papers = [
        paper(paper_id="W1", doi="", title="Attention Is All You Need", abstract=ABSTRACT),
        paper(
            paper_id="p1",
            doi="",
            title="  attention is all you need\n",
            abstract=f"\t{ABSTRACT.upper()}  ",
        ),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1"]


# --- near-identical, which is as close as two sources ever get ------------------


def test_a_revised_abstract_is_still_the_same_work():
    """Review adds a results sentence and moves a comma; the work does not change."""
    papers = [
        paper(paper_id="preprint", doi="", abstract=ABSTRACT),
        paper(paper_id="p1", doi="", abstract=REVISED_ABSTRACT),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["preprint"]


def test_a_title_differing_only_in_punctuation_and_word_order_still_matches():
    """OpenAlex and Semantic Scholar render the same title differently."""
    papers = [
        paper(paper_id="W1", doi="", title="Attention Is All You Need"),
        paper(paper_id="p1", doi="", title="Attention is all you need."),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1"]


def test_a_venue_tag_appended_to_the_title_is_still_the_same_work():
    """Seen live: AAAI carries "(Student Abstract)", its arXiv preprint does not."""
    papers = [
        paper(paper_id="preprint", title=STUDENT_ABSTRACT_TITLE),
        published(
            paper_id="published",
            title=f"{STUDENT_ABSTRACT_TITLE} (Student Abstract)",
        ),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["published"]


def test_a_title_that_gained_a_qualifier_is_not_a_duplicate():
    """A subset is not a match — the trap a set-based ratio would fall into."""
    papers = [
        paper(paper_id="W1", doi="", title="Attention Is All You Need"),
        paper(paper_id="W2", doi="", title="Attention Is All You Need for Graph Networks"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1", "W2"]


# --- the published copy wins ---------------------------------------------------


def test_the_published_copy_wins_when_the_preprint_ranks_higher():
    """The whole point: the better DOI, URL and license beat the better position."""
    papers = [paper(paper_id="preprint"), published(paper_id="published")]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["published"]


def test_the_published_copy_wins_when_it_ranks_higher():
    """The same answer from the other direction."""
    papers = [published(paper_id="published"), paper(paper_id="preprint")]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["published"]


def test_the_surviving_copy_keeps_its_own_position():
    """The preprint is dropped rather than replaced, so its slot goes with it."""
    papers = [
        paper(paper_id="preprint"),
        paper(paper_id="other", doi="10.1/z", title="Something Else", abstract="other"),
        published(paper_id="published"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["other", "published"]


def test_a_record_with_no_evidence_either_way_outranks_a_preprint():
    """No DOI and no preprint host is unknown, and unknown beats a known preprint."""
    papers = [
        paper(paper_id="preprint"),
        paper(paper_id="unknown", doi="", url="https://example.org/paper"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["unknown"]


def test_a_cold_spring_harbor_journal_beats_its_biorxiv_preprint():
    """10.1101 registers both; only the date-shaped suffix is the preprint."""
    papers = [
        paper(
            paper_id="biorxiv",
            doi="10.1101/2020.05.01.123456",
            url="https://www.biorxiv.org/content/10.1101/2020.05.01.123456v1",
        ),
        paper(
            paper_id="genome-research",
            doi="10.1101/gr.123456.789",
            url="https://genome.cshlp.org/content/30/7/1000",
        ),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["genome-research"]


def test_matching_is_transitive():
    """A DOI link and a pair link make one group, so the best copy of it can win.

    The middle record shares its DOI with the first and its abstract with the last.
    Single linkage is what lets the published copy beat a preprint it never matched
    directly.
    """
    papers = [
        paper(paper_id="preprint-a", doi="10.48550/arxiv.1", abstract=""),
        paper(paper_id="preprint-b", doi="10.48550/arxiv.1", abstract=ABSTRACT),
        published(paper_id="published", abstract=REVISED_ABSTRACT),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["published"]


# --- absent identifiers prove nothing -----------------------------------------


def test_papers_without_a_doi_are_all_kept():
    """A missing DOI is missing information, not evidence that two papers match."""
    papers = [
        paper(paper_id="a", doi="", title="a"),
        paper(paper_id="b", doi="", title="b"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["a", "b"]


def test_a_shared_title_with_no_abstract_is_not_a_duplicate():
    """Half a pair is not a pair — otherwise every abstract-less record collapses."""
    papers = [
        paper(paper_id="a", doi="", abstract=""),
        paper(paper_id="b", doi="", abstract=""),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["a", "b"]


def test_papers_with_no_identifiers_at_all_are_all_kept():
    """Nothing to judge them by, so they must all survive."""
    papers = [
        paper(paper_id="a", doi="", title="", abstract=""),
        paper(paper_id="b", doi="", title="", abstract=""),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["a", "b"]


def test_a_paper_without_a_doi_survives_alongside_duplicates():
    """The empty-DOI path and the dedup path must not interfere."""
    papers = [
        paper(paper_id="W1", doi="10.1/a", title="a"),
        paper(paper_id="no-doi", doi="", title="b"),
        paper(paper_id="W2", doi="10.1/a", title="c"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["W1", "no-doi"]


# --- shape of the function ----------------------------------------------------


@pytest.mark.parametrize("papers", [[], iter([])], ids=["list", "iterator"])
def test_an_empty_input_stays_empty(papers):
    assert remove_duplicates(papers) == []


def test_any_iterable_is_accepted():
    """The signature promises an Iterable, so a generator has to work too."""
    papers = (paper(paper_id=pid) for pid in ("first", "second"))

    assert [p.paper_id for p in remove_duplicates(papers)] == ["first"]


def test_the_input_list_is_not_mutated():
    papers = [paper(paper_id="first"), paper(paper_id="second")]

    remove_duplicates(papers)

    assert [p.paper_id for p in papers] == ["first", "second"]


def test_a_journal_doi_beats_an_arxiv_open_access_link():
    """OpenAlex often points a published paper at its arXiv PDF; the DOI decides."""
    papers = [
        paper(paper_id="preprint"),
        published(paper_id="published", url="https://arxiv.org/pdf/1706.03762"),
    ]

    assert [p.paper_id for p in remove_duplicates(papers)] == ["published"]
