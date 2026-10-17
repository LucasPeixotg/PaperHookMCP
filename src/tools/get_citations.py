import json
from typing import Literal

from fastmcp.exceptions import ToolError

from services.api import OpenAlexClient
from services.api.api_client import APIError
from utils import as_response_dict, to_openalex_id

open_alex_client = OpenAlexClient()

# Fields that are ranking or reading material elsewhere but noise in a citation list:
# no query ranked these works, and abstracts are not fetched for them.
_DROPPED_FIELDS = {"abstract", "relevance_score"}


def get_citations(
    paper_id: str,
    limit: int = 10,
    sort: Literal["most_cited", "newest"] = "most_cited",
) -> str:
    """
    Retrieves a list of papers that have cited the target publication (forward citation search).

    Use this tool when you need to track how a paper's ideas evolved, find recent
    benchmarks building on it, or discover modern follow-up studies.

    Args:
        paper_id: The OpenAlex paper ID ('W2626778328'), as returned in `paper_id`
                  by find_relevant_papers. DOIs and arXiv IDs are not accepted.
        limit: Maximum number of citing papers to return (default is 10, at most 200).
        sort: 'most_cited' (default) puts the most influential follow-ups first;
              'newest' puts the most recently published first.

    Returns:
        JSON list of the citing papers, each with its paper ID, DOI, title,
        publication year, venue, citation count, direct paper URL and license.
        The target paper itself is not included.
    """
    try:
        work_id = to_openalex_id(paper_id)
    except ValueError as error:
        raise ToolError(str(error)) from error

    try:
        target = open_alex_client.get_work(work_id)
    except APIError as error:
        # A 404's body is an HTML error page — useless to the caller, so it is dropped.
        if str(error).startswith("API status 404"):
            raise ToolError(f"Paper not found in OpenAlex: {paper_id!r}") from error
        raise ToolError(f"Could not look up {paper_id!r}: {error}") from error

    try:
        citing, _ = open_alex_client.get_citing_works(target["paper_id"], limit, sort)
    except APIError as error:
        raise ToolError(f"Could not fetch citations for {paper_id!r}: {error}") from error

    return json.dumps([as_response_dict(paper, _DROPPED_FIELDS) for paper in citing])

