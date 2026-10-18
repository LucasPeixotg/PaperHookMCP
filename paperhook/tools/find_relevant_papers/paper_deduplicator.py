
import re
from collections.abc import Iterable

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

from ...common import PaperData

# Two records of one work are never byte-identical across sources: OpenAlex rebuilds
# its abstract from an inverted index and loses the original spacing, Semantic Scholar
# keeps the publisher's, and a published version routinely revises a sentence of the
# preprint's. So sameness is scored, not compared.
#
# The abstract is the discriminating half of the pair — two different papers do not
# share one — so the title is scored the more leniently of the two. Live OpenAlex
# results bear that out: a conference version tagged "(Student Abstract)" scores 93
# against its preprint, an initial-caps variant 94, and nothing in that band matched
# an abstract without being the same work.
TITLE_SIMILARITY_THRESHOLD = 90
ABSTRACT_SIMILARITY_THRESHOLD = 90

# DOI prefixes belonging to preprint servers. A work carrying one is a preprint
# whatever else it says about itself.
PREPRINT_DOI_PREFIXES = frozenset({
    "10.48550",  # arXiv
    "10.21203",  # Research Square
    "10.26434",  # ChemRxiv
    "10.31219",  # OSF Preprints
    "10.31234",  # PsyArXiv
    "10.20944",  # Preprints.org
    "10.36227",  # TechRxiv
    "10.31235",  # SocArXiv
    "10.22541",  # Authorea, which hosts TechRxiv and ESSOAr
    "10.32388",  # Qeios
    "10.2139",   # SSRN
})

# 10.1101 cannot join the set above: Cold Spring Harbor registers both the bioRxiv and
# medRxiv preprints *and* its own journals (Genome Research is 10.1101/gr.…) under it.
# The preprints are the ones whose suffix opens with the submission date.
BIORXIV_DOI_PREFIX = "10.1101"
BIORXIV_SUFFIX = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.")

# Hosts that only ever serve preprints. These decide the records that arrive with no
# DOI at all, and only those: a published paper's open-access copy frequently *is* the
# arXiv PDF, so a preprint host alongside a journal DOI says where the file is, not
# what the record is.
PREPRINT_HOSTS = (
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "researchsquare.com",
    "chemrxiv.org",
    "psyarxiv.com",
    "osf.io/preprints",
    "preprints.org",
    "techrxiv.org",
    "ssrn.com",
    "authorea.com",
    "qeios.com",
    "essoar.org",
)

# Publication ranks, compared as numbers: the higher one wins its cluster.
_PUBLISHED = 2
_UNKNOWN = 1
_PREPRINT = 0


def remove_duplicates(papers: Iterable[PaperData]) -> list[PaperData]:
    """Collapse papers that are the same work down to the best copy of it.

    The same paper reaches us once per source that indexes it, and ``paper_id``
    cannot spot that — it is an OpenAlex ``W…`` from one client and a Semantic
    Scholar hash from the other. Two identifiers the sources do agree on stand in
    for it, and two papers are the same work when *either* matches:

    * their **DOIs**, when both have one;
    * their **titles and abstracts together**, when both have both — matched by
      similarity rather than equality, since near-identical is as close as two
      sources ever get.

    The pair catches what the DOI alone cannot: preprints, which frequently carry
    no DOI at all, and the preprint-versus-published pair of the same work, which
    carries two different ones.

    Four behaviours worth knowing:

    * **The published copy wins, and it keeps its own position.** A preprint and
      the version that came out of review are one work, and the published record
      carries the better DOI, URL and license, so it survives even when the
      preprint ranked higher. Papers with no evidence either way outrank
      preprints; a tie goes to the first occurrence, since the caller hands the
      papers over in the order they should be ranked and the surviving copy takes
      its own position in it. Ranking is not this function's business — it neither
      reads nor rewrites ``relevance_score``, which travels on each record for the
      reranker to break its ties with.
    * **An absent identifier is never evidence of sameness.** A paper with no DOI
      is not judged by DOI, and one missing a title or an abstract is not judged
      by the pair. Missing information is not proof that two papers match, and
      treating it as such would collapse every abstract-less record into one.
    * **Matching ignores case, punctuation and word order in titles.** DOIs are
      case-insensitive by specification, and the sources genuinely disagree in
      practice: OpenAlex lowercases everything while Semantic Scholar preserves
      the registrant's casing. Titles and abstracts differ the same way, plus the
      stray whitespace that survives extraction.
    * **Matching is transitive.** Papers are grouped by single linkage — a paper
      joins a group when it matches *any* member — because the best copy cannot
      be chosen until the whole group is known.

    Args:
        papers: The papers to filter, in the order they should be ranked.

    Returns:
        A new list holding the best copy of each work, in input order. The input
        is left untouched.

    Example:
        >>> papers = [
        ...     PaperData("W1", "10.48550/arxiv.1706.03762", "Attention", 2017, "a", "", ""),
        ...     PaperData("p1", "10.1000/journal.1", "Attention", 2017, "a", "", ""),
        ... ]
        >>> [paper.paper_id for paper in remove_duplicates(papers)]
        ['p1']
    """
    # Each cluster is one work, holding every record of it as (index, paper).
    clusters: list[list[tuple[int, PaperData]]] = []

    for index, paper in enumerate(papers):
        for cluster in clusters:
            if any(_is_same_work(paper, member) for _, member in cluster):
                cluster.append((index, paper))
                break
        else:
            clusters.append([(index, paper)])

    # Best copy per cluster: highest publication rank, earliest position on a tie.
    winners = [
        max(cluster, key=lambda entry: (_publication_rank(entry[1]), -entry[0]))
        for cluster in clusters
    ]

    # Clusters come out in first-appearance order, but a winner need not be its
    # cluster's first member, so the ranking is restored by input position.
    return [paper for _, paper in sorted(winners, key=lambda entry: entry[0])]


def _is_same_work(paper: PaperData, other: PaperData) -> bool:
    """Whether two records describe one work. Only present fields are ever judged."""
    if paper.doi and other.doi and paper.doi.casefold() == other.doi.casefold():
        return True

    # Half a pair identifies nothing, so both records must carry both fields.
    if not (paper.title and paper.abstract and other.title and other.abstract):
        return False

    # The title is the cheap test and gates the expensive one. `token_sort_ratio`
    # tolerates reordered words; `token_set_ratio` would not do here — being
    # subset-tolerant it scores "Attention Is All You Need" against "Attention Is
    # All You Need for Graphs" at 100, and those are different papers.
    if not fuzz.token_sort_ratio(
        paper.title,
        other.title,
        processor=default_process,
        score_cutoff=TITLE_SIMILARITY_THRESHOLD,
    ):
        return False

    return bool(
        fuzz.ratio(
            paper.abstract,
            other.abstract,
            processor=default_process,
            score_cutoff=ABSTRACT_SIMILARITY_THRESHOLD,
        )
    )


def _publication_rank(paper: PaperData) -> int:
    """How much a record looks like the published version. Higher wins."""
    if _is_preprint(paper):
        return _PREPRINT

    # A registered DOI that belongs to no preprint server is the evidence there is.
    return _PUBLISHED if paper.doi else _UNKNOWN


def _is_preprint(paper: PaperData) -> bool:
    if paper.doi:
        prefix, _, suffix = paper.doi.casefold().partition("/")

        # bioRxiv and medRxiv share their prefix with Cold Spring Harbor's journals.
        return prefix in PREPRINT_DOI_PREFIXES or (
            prefix == BIORXIV_DOI_PREFIX and bool(BIORXIV_SUFFIX.match(suffix))
        )

    url = paper.url.casefold()

    return any(host in url for host in PREPRINT_HOSTS)
