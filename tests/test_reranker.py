"""rerank — pure logic, with the cross-encoder stubbed out.

The real model is ~90MB of weights fetched from HuggingFace and a torch install
the rest of this tree does not need, so every test here injects a scorer instead.
That is not only about speed: what the function actually owns is the ordering
contract around the scores, and a fake scorer is the only way to state a case
like "these two papers tie" and know it holds.

Selection is the other half of what the function owns, and it is not just a
slice: `top_k` counts papers that carry an abstract, and the abstract-less ones
passed on the way to filling it come back in a second list rather than being
dropped. A stubbed scorer is what lets a test state "this paper outranks that
one and has no abstract" and know which list each lands in.

The cases worth the file are the ones the wiring can get wrong. Sorting the
(paper, score) pairs directly rather than by key reads fine and works right up
until two papers tie, where it compares PaperData and raises — so both the tie
and the ordering are pinned. The guards are checked for *not loading the model*
rather than only for their return value, since skipping that download on a blank
query is the whole reason they run first. And the document handed to the model is
asserted field by field: a paper missing an abstract is common enough in OpenAlex
that a stray separator would be the normal case, not the edge one.
"""

import pytest

from paperhook.common import PaperData
from paperhook.services.models import reranker
from paperhook.services.models.reranker import RerankedPapers, rerank

ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely "
    "on attention mechanisms, dispensing with recurrence and convolutions entirely."
)


def paper(**overrides) -> PaperData:
    """A complete PaperData so a test can name only the field it cares about."""
    base = {
        "paper_id": "W1",
        "doi": "10.48550/arxiv.1706.03762",
        "title": "Attention Is All You Need",
        "year": 2017,
        "abstract": ABSTRACT,
        "url": "https://arxiv.org/abs/1706.03762",
        "license": "",
        "relevance_score": 0.0,
    }

    return PaperData(**(base | overrides))


class StubModel:
    """A cross-encoder that returns scores chosen by the test.

    Records the pairs it was given, so a test can assert both what the model was
    asked and — by an empty record — that it was never asked at all.
    """

    def __init__(self, scores=()):
        self.scores = list(scores)
        self.calls = []

    def predict(self, pairs, batch_size=None):
        pairs = list(pairs)
        self.calls.append(pairs)

        # A test that cares about ordering supplies scores; one that does not
        # gets a flat zero per pair rather than an index error.
        return self.scores or [0.0] * len(pairs)


@pytest.fixture
def model(monkeypatch):
    """Install a stub in place of the lazily-loaded singleton.

    `_get_model` returns `_model` untouched when it is already set, so seeding it
    is enough to keep sentence-transformers out of the test run entirely.
    """

    def install(*scores):
        stub = StubModel(scores)
        monkeypatch.setattr(reranker, "_model", stub)
        return stub

    return install


def test_papers_come_back_in_descending_score_order(model):
    """The reordering is the entire point: input order must not survive scoring."""
    first, second, third = paper(paper_id="A"), paper(paper_id="B"), paper(paper_id="C")
    model(0.1, 0.9, 0.5)

    ranked = rerank("attention mechanisms", [first, second, third], top_k=3).papers

    assert [p.paper_id for p in ranked] == ["B", "C", "A"]


def test_only_top_k_are_returned(model):
    """The pool is deliberately wider than the answer — the surplus is dropped here."""
    papers = [paper(paper_id=str(index)) for index in range(5)]
    model(0.1, 0.2, 0.3, 0.4, 0.5)

    ranked = rerank("attention", papers, top_k=2).papers

    assert [p.paper_id for p in ranked] == ["4", "3"]


def test_a_top_k_larger_than_the_pool_returns_every_paper(model):
    """Asking for more than exists is not an error; dedup can shrink the pool."""
    papers = [paper(paper_id="A"), paper(paper_id="B")]
    model(0.2, 0.8)

    assert len(rerank("attention", papers, top_k=10).papers) == 2


def test_a_score_tie_is_broken_by_the_source_relevance_score(model):
    """What the second sort key is for: the ranking the source itself gave.

    The cross-encoder ties in practice — most often on the empty document, since
    OpenAlex ships a null abstract often enough that title-less records collide.
    The relevance order here runs *against* the input order, so a sort that only
    leans on stability returns the input unchanged and fails.
    """
    papers = [
        paper(paper_id="A", relevance_score=1.0),
        paper(paper_id="B", relevance_score=1.2),
        paper(paper_id="C", relevance_score=1.1),
    ]
    model(0.5, 0.5, 0.5)

    ranked = rerank("attention", papers, top_k=3).papers

    assert [p.paper_id for p in ranked] == ["B", "C", "A"]


def test_the_model_score_outranks_the_relevance_score(model):
    """`relevance_score` breaks ties; it does not get a vote otherwise.

    The upstream bi-encoder is the stage the cross-encoder exists to correct, so a
    strong upstream score must never pull a paper past one the model scored higher.
    """
    papers = [
        paper(paper_id="A", relevance_score=99.0),
        paper(paper_id="B", relevance_score=0.1),
    ]
    model(0.1, 0.9)

    ranked = rerank("attention", papers, top_k=2).papers

    assert [p.paper_id for p in ranked] == ["B", "A"]


def test_papers_tied_on_both_scores_keep_their_input_order(model):
    """The last fallback, once neither score separates two papers.

    This is also what catches a sort that compares the (paper, score) pairs
    themselves: PaperData has no ordering, so a tie would raise TypeError rather
    than quietly misorder.
    """
    papers = [paper(paper_id="A"), paper(paper_id="B"), paper(paper_id="C")]
    model(0.5, 0.5, 0.5)

    ranked = rerank("attention", papers, top_k=3).papers

    assert [p.paper_id for p in ranked] == ["A", "B", "C"]


def test_the_model_scores_title_and_abstract_together(model):
    """Both fields are one passage — a title alone rarely settles relevance."""
    stub = model(0.5)

    rerank("attention mechanisms", [paper()], top_k=1)

    (query, document), = stub.calls[0]
    assert query == "attention mechanisms"
    assert document == f"Attention Is All You Need\n\n{ABSTRACT}"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"abstract": ""}, "Attention Is All You Need"),
        ({"title": ""}, ABSTRACT),
        ({"title": "", "abstract": ""}, ""),
    ],
    ids=["no abstract", "no title", "neither"],
)
def test_a_missing_field_leaves_no_stray_separator(model, overrides, expected):
    """PaperData coerces absent values to "", and OpenAlex omits abstracts often.

    Joining unconditionally would open most real documents with a blank line.
    """
    stub = model(0.5)

    rerank("attention", [paper(**overrides)], top_k=1)

    (_, document), = stub.calls[0]
    assert document == expected


def test_a_blank_query_keeps_the_upstream_order_without_loading_the_model(model):
    """There is nothing to score against, and loading weights to learn that is waste."""
    stub = model(0.9, 0.1)
    papers = [paper(paper_id="A"), paper(paper_id="B"), paper(paper_id="C")]

    ranked = rerank("   ", papers, top_k=2).papers

    assert [p.paper_id for p in ranked] == ["A", "B"]
    assert stub.calls == []


def test_a_blank_query_ranks_by_the_source_relevance_score(model):
    """Nothing is scored, so every paper ties and the tiebreak is the whole ranking.

    Input order stands in for that ranking only while one source fills the pool;
    sorting is what keeps it true once a second `extend()` makes position mean
    concatenation order instead.
    """
    stub = model()
    papers = [
        paper(paper_id="A", relevance_score=1.0),
        paper(paper_id="B", relevance_score=1.2),
        paper(paper_id="C", relevance_score=1.1),
    ]

    ranked = rerank("   ", papers, top_k=2).papers

    assert [p.paper_id for p in ranked] == ["B", "C"]
    assert stub.calls == []


@pytest.mark.parametrize(
    "papers, top_k",
    [([], 5), ([paper()], 0), ([paper()], -1)],
    ids=["empty pool", "top_k of zero", "negative top_k"],
)
def test_nothing_to_rank_returns_empty_without_loading_the_model(model, papers, top_k):
    """A cold cache would otherwise pay a ~90MB download to return an empty list."""
    stub = model(0.5)

    assert rerank("attention", papers, top_k) == RerankedPapers([], [])
    assert stub.calls == []


def test_the_input_list_is_not_mutated(model):
    """Callers hold the pool afterwards — the deduplicator hands over its own list."""
    papers = [paper(paper_id="A"), paper(paper_id="B")]
    model(0.1, 0.9)

    rerank("attention", papers, top_k=2)

    assert [p.paper_id for p in papers] == ["A", "B"]


def test_top_k_skips_past_papers_with_no_abstract(model):
    """The point of the split: a title-only record must not spend a `top_k` slot.

    G and H are only reached because B, C and F were skipped — a plain
    `ranked[:top_k]` would have returned two papers the caller cannot read.
    """
    papers = [
        paper(paper_id="A"),
        paper(paper_id="B", abstract=""),
        paper(paper_id="C", abstract=""),
        paper(paper_id="D"),
        paper(paper_id="E"),
        paper(paper_id="F", abstract=""),
        paper(paper_id="G"),
        paper(paper_id="H"),
    ]
    model(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

    reranked = rerank("attention", papers, top_k=5)

    assert [p.paper_id for p in reranked.papers] == ["A", "D", "E", "G", "H"]


def test_the_papers_skipped_while_filling_top_k_come_back_separately(model):
    """Skipped is not discarded — the model rated these over papers that made the cut.

    F is the case that matters: it falls outside the *original* top five, so only a
    walk that tracks what it passed reports it. It still outranks G and H, which the
    caller is being handed.
    """
    papers = [
        paper(paper_id="A"),
        paper(paper_id="B", abstract=""),
        paper(paper_id="C", abstract=""),
        paper(paper_id="D"),
        paper(paper_id="E"),
        paper(paper_id="F", abstract=""),
        paper(paper_id="G"),
        paper(paper_id="H"),
    ]
    model(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

    reranked = rerank("attention", papers, top_k=5)

    assert [p.paper_id for p in reranked.relevant_without_abstract] == ["B", "C", "F"]


def test_an_abstractless_paper_below_the_cut_is_dropped(model):
    """It displaced nothing, so there is nothing to report about it.

    The walk stops the moment `top_k` abstracts are collected. Reporting C would
    mean reporting the whole tail of the pool, which is the ranking's business.
    """
    papers = [
        paper(paper_id="A"),
        paper(paper_id="B"),
        paper(paper_id="C", abstract=""),
    ]
    model(0.9, 0.8, 0.7)

    reranked = rerank("attention", papers, top_k=2)

    assert [p.paper_id for p in reranked.papers] == ["A", "B"]
    assert reranked.relevant_without_abstract == []


def test_a_whitespace_only_abstract_counts_as_missing(model):
    """It says exactly as much as the empty string, and `_as_document` drops it too."""
    papers = [paper(paper_id="A", abstract="   \n  "), paper(paper_id="B")]
    model(0.9, 0.1)

    reranked = rerank("attention", papers, top_k=1)

    assert [p.paper_id for p in reranked.papers] == ["B"]
    assert [p.paper_id for p in reranked.relevant_without_abstract] == ["A"]


def test_a_pool_short_on_abstracts_returns_fewer_than_top_k(model):
    """Asking for five when two are readable is not an error — the pool is what it is.

    The walk consumes the whole pool looking for a fifth abstract, so every
    abstract-less paper in it is reported. That is bounded by the pool size.
    """
    papers = [
        paper(paper_id="A"),
        paper(paper_id="B", abstract=""),
        paper(paper_id="C", abstract=""),
        paper(paper_id="D"),
    ]
    model(0.9, 0.8, 0.7, 0.6)

    reranked = rerank("attention", papers, top_k=5)

    assert [p.paper_id for p in reranked.papers] == ["A", "D"]
    assert [p.paper_id for p in reranked.relevant_without_abstract] == ["B", "C"]


def test_a_blank_query_splits_too_without_loading_the_model(model):
    """The guard skips scoring, not selection — both paths share the same walk."""
    stub = model(0.9, 0.1)
    papers = [
        paper(paper_id="A", abstract="", relevance_score=1.2),
        paper(paper_id="B", relevance_score=1.1),
        paper(paper_id="C", relevance_score=1.0),
    ]

    reranked = rerank("   ", papers, top_k=1)

    assert [p.paper_id for p in reranked.papers] == ["B"]
    assert [p.paper_id for p in reranked.relevant_without_abstract] == ["A"]
    assert stub.calls == []
