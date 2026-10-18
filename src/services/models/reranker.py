"""Second-stage reranking of search candidates with a cross-encoder."""

import logging
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder

from common import PaperData
from config import settings

logger = logging.getLogger(__name__)

# Candidates are scored in batches of this size. The pool is ~40 papers
# (settings.RERANK_CANDIDATE_POOL), so this is one or two forward passes rather
# than a knob worth tuning; it exists to bound memory if a caller passes a much
# longer list than the tool does.
BATCH_SIZE = 32

# Title and abstract are one document to the model, not two fields. The blank
# line is what the MS MARCO training data puts between a passage's parts, and it
# survives the tokenizer as a paragraph break rather than a word.
_FIELD_SEPARATOR = "\n\n"

logger.info("Loading reranker model %s", settings.RERANK_MODEL_ID)
_model = CrossEncoder(settings.RERANK_MODEL_ID)


@dataclass(frozen=True)
class RerankedPapers:
    """The reranker's answer, split by whether the source published an abstract.

    The split is not a detail of presentation. An abstract is the payload — it is
    what a calling model reads to decide whether a paper answers the question — so
    a record carrying only a title spends a `top_k` slot on nothing. Those papers
    are still real leads, though, and they arrive with a DOI and a URL, so they
    come back beside the answer rather than being dropped from it.
    """

    papers: list[PaperData] = field(default_factory=list)
    relevant_without_abstract: list[PaperData] = field(default_factory=list)


def rerank(query: str, papers: list[PaperData], top_k: int) -> RerankedPapers:
    """Reorder papers by how well a cross-encoder thinks each answers the query.

    The papers arrive already ranked, by OpenAlex's bi-encoder: it embeds the
    query and every work's title+abstract *independently* and compares the
    vectors, which is the only thing that scales to millions of works but cannot
    weigh a query term against a particular sentence of a particular abstract.
    A cross-encoder runs the query and one document through the transformer
    *together*, so it can — at a cost that only makes sense on a few dozen
    candidates. Hence the two stages: OpenAlex narrows millions to a pool, this
    reorders the pool.

    Both scored fields can be empty — ``PaperData`` requires its text fields,
    so both clients coerce an absent value to ``""``, and OpenAlex ships a null
    ``abstract_inverted_index`` often enough to matter. A paper missing one
    field is still scored on the other; a paper missing both scores as the empty
    document, which is the right answer for a record that says nothing. Where
    those papers land is the selection's business, not the scoring's — see
    ``_split_on_abstract``.

    Args:
        query: The natural language query to score relevance against. Pass it
            raw — wildcard cleaning is for search APIs that read ``?`` and ``*``
            as operators, and to a cross-encoder that punctuation is signal.
        papers: The candidates to reorder. Left untouched.
        top_k: How many papers to fill ``papers`` with.

    Returns:
        A ``RerankedPapers`` holding at most ``top_k`` papers that carry an
        abstract, most relevant first, and beside them every abstract-less paper
        that outranked one of them. Papers the model scores equally are broken
        apart by ``relevance_score``, the ranking the source itself gave them;
        papers tied on both keep their input order.

    Example:
        >>> rerank("attention mechanisms", papers, top_k=5)  # doctest: +SKIP
        RerankedPapers(papers=[PaperData(paper_id='W2626778328', ...), ...],
                       possible_non_open_papers=[...])
    """
    # Both guards return before the model is asked for anything, so neither an
    # empty pool nor a blank query pays for a forward pass.
    if top_k <= 0 or not papers:
        return RerankedPapers()

    # Nothing to score against, so every paper ties and the upstream ranking is the
    # whole answer — the same reasoning the clients apply to a blank search. Sorting
    # is what makes that true across sources; input order only carries the ranking
    # while a single source fills the pool.
    if not query.strip():
        return _split_on_abstract(
            sorted(papers, key=lambda paper: paper.relevance_score, reverse=True),
            top_k,
        )

    documents = [_as_document(paper) for paper in papers]
    scores = _model.predict(
        [(query, document) for document in documents],
        batch_size=BATCH_SIZE,
    )

    # `key` is load-bearing, not style: sorting the pairs themselves would fall
    # through to comparing PaperData on a score tie, and the dataclass carries no
    # `order=True`, so that raises TypeError. Both members of the key are floats, so
    # the tuple never reaches the paper either. The cross-encoder does tie in
    # practice — most often on the empty document, since OpenAlex ships a null
    # abstract often enough that title-less records collide — and `relevance_score`
    # is what settles those. The sort stays stable, so a tie on both keys still
    # falls back to input order.
    ranked = sorted(
        zip(papers, scores),
        key=lambda entry: (entry[1], entry[0].relevance_score),
        reverse=True,
    )

    # The whole pool, ordered — the `top_k` cut belongs to the split, which has to
    # look past the k-th paper to fill k slots with abstracts.
    return _split_on_abstract([paper for paper, _ in ranked], top_k)


def _split_on_abstract(ordered: list[PaperData], top_k: int) -> RerankedPapers:
    """Take the best ``top_k`` papers that have an abstract, and what they passed.

    Walking rather than filtering is what earns the second list. A filter would
    answer "the top k with abstracts" and say nothing about the rest; the walk
    knows *which* abstract-less papers were ranked above a paper that made the
    cut, and those are exactly the ones worth reporting — the model rated them
    over papers the caller is being handed.

    The walk stops the moment ``top_k`` abstracts are collected, so an
    abstract-less paper below the last returned one is dropped, as it would have
    been before: it displaced nothing. If the pool never yields ``top_k``
    abstracts the walk consumes all of it, which is bounded — the pool is at most
    ``OpenAlexClient.SEMANTIC_MAX_LIMIT`` papers, and a record with no abstract is
    a handful of short fields.
    """
    with_abstract: list[PaperData] = []
    without_abstract: list[PaperData] = []

    for paper in ordered:
        if len(with_abstract) == top_k:
            break

        # `.strip()` because "missing" reaches here as the empty string — the
        # clients coerce a null abstract to "" rather than None — and a record
        # holding only whitespace says exactly as much as one holding nothing.
        if paper.abstract.strip():
            with_abstract.append(paper)
        else:
            without_abstract.append(paper)

    return RerankedPapers(with_abstract, without_abstract)


def _as_document(paper: PaperData) -> str:
    """The paper as the single passage the model scores.

    The title leads deliberately. The model's window is 512 tokens and
    sentence-transformers truncates whatever overflows it, so a long abstract
    loses its tail rather than the title.
    """
    return _FIELD_SEPARATOR.join(part for part in (paper.title, paper.abstract) if part)
