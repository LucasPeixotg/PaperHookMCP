import json

from common import PaperData
from config import settings
from services.api import OpenAlexClient, SemanticScholarClient
from services.models import rerank
from utils import as_response_dict

from .paper_deduplicator import remove_duplicates
from .query_cleaner import remove_wildcards

semantic_scholar_client = SemanticScholarClient()
open_alex_client = OpenAlexClient()

# `relevance_score` is a ranking input, not an answer. Its scale is per-source and
# per-search-mode — OpenAlex keyword scores run into the thousands where its semantic
# ones sit around 1 — so a caller could only misread it. Dropping it keeps the returned
# object the fields the docstring promises, and keeps both lists the same shape.
_DROPPED_FIELDS = {"relevance_score"}

def find_relevant_papers(query: str, top_k: int = 5) -> str:
    """
    Finds the most relevant papers through multiple sources that could answer a natural language query.

    Args:
        query: The natural language query to search for.
        top_k: The maximum number of top matching paper abstracts to retrieve (default is 5).

    Returns:
        JSON string containing an object with two lists of papers, each paper
        carrying its paper ID, DOI, title, publication year, abstract, direct
        paper URL, license, and open-access links (`oa_pdf`, a direct PDF,
        and `oa_url`, a landing page; blank when there is none).

        `papers` holds up to top_k matches that come with an abstract — the ones
        worth reading now.

        `relevant_without_abstract` holds matches that ranked among the best but
        whose abstract the source does not publish, so they carry a title and
        little else. That often means the work is not openly available, though
        not always; some publishers restrict the abstract alone. They are still
        worth following by DOI or URL, and their `abstract` field is empty.
    """

    relevant_papers: list[PaperData] = []

    ## searching through multiple sources (parallel in near future)
    # semantic_scholar_papers = semantic_scholar_client.search_papers(query)

    cleaned_query = remove_wildcards(query)

    # open_alex_papers = open_alex_client.search_papers(cleaned_query)
    # Deliberately over-fetched: the reranker can only improve on the order it is
    # given, so it needs a pool wider than the `top_k` slots it fills — wider still
    # now that filling them skips papers with no abstract.
    open_alex_semantic_papers = open_alex_client.semantic_search_papers(
        cleaned_query, limit=max(top_k*2, settings.RERANK_CANDIDATE_POOL)
    )

    ## extending all results
    # relevant_papers.extend(semantic_scholar_papers)
    # relevant_papers.extend(open_alex_papers)
    relevant_papers.extend(open_alex_semantic_papers)

    ## post processing (deduplicate, complement)
    post_processed = remove_duplicates(relevant_papers)

    ## reranking — scored against the raw query, since wildcard cleaning exists
    ## for search APIs that read `?` and `*` as operators, not for a model that
    ## reads the punctuation as language.
    reranked = rerank(query, post_processed, top_k)

    result = json.dumps(
        {
            "papers": [
                as_response_dict(paper, _DROPPED_FIELDS) for paper in reranked.papers
            ],
            "relevant_without_abstract": [
                as_response_dict(paper, _DROPPED_FIELDS)
                for paper in reranked.relevant_without_abstract
            ],
        }
    )
    return result
