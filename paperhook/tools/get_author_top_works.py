import json

from fastmcp.exceptions import ToolError

from ..services.api import OpenAlexClient
from ..services.api.api_client import APIError
from ..utils import as_response_dict, to_openalex_author_key

open_alex_client = OpenAlexClient()

# Fields that are ranking or reading material elsewhere but noise in a list of an
# author's works: no query ranked these works, and abstracts are not fetched for them.
_DROPPED_FIELDS = {"abstract", "relevance_score"}

# How many namesakes a name search fetches, and so how many an ambiguity error lists.
_AUTHOR_CANDIDATES = 5


def get_author_top_works(
    author: str,
    limit: int = 10,
    publication_year_min: int | None = None,
) -> str:
    """
    Retrieves the most influential or highly cited publications by a specific researcher or lab lead.

    Use this tool when exploring a major research lab's output, checking an
    author's domain expertise, or searching for key works by a specific scientist.

    Args:
        author: Author name (e.g. 'Geoffrey Hinton'), ORCID ('0000-0002-1825-0097'),
                or OpenAlex author ID ('A5001234567'). A name must match exactly one
                OpenAlex author; if it matches several, the error lists the candidates
                with their IDs — call again with the right one.
        limit: Maximum number of top publications to return (default is 10, at most 200).
        publication_year_min: Optional filter to only include works published from this year onwards.

    Returns:
        JSON list of the author's publications, most cited first, each with its
        paper ID, DOI, title, publication year, venue, citation count, direct
        paper URL, license, and open-access links (`oa_pdf`, a direct PDF,
        and `oa_url`, a landing page; blank when there is none).
    """
    author_id = _resolve_author(author)

    try:
        works, _ = open_alex_client.get_author_works(author_id, limit, publication_year_min)
    except APIError as error:
        raise ToolError(f"Could not fetch works for {author!r}: {error}") from error

    return json.dumps([as_response_dict(paper, _DROPPED_FIELDS) for paper in works])


def _resolve_author(author: str) -> str:
    """The bare OpenAlex author id `author` names, or a ToolError saying why not."""
    if key := to_openalex_author_key(author):
        try:
            return open_alex_client.get_author(key)["author_id"]
        except APIError as error:
            # A 404's body is an HTML error page — useless to the caller, so it is dropped.
            if str(error).startswith("API status 404"):
                raise ToolError(f"Author not found in OpenAlex: {author!r}") from error
            raise ToolError(f"Could not look up {author!r}: {error}") from error

    try:
        candidates = open_alex_client.search_authors(author, _AUTHOR_CANDIDATES)
    except APIError as error:
        raise ToolError(f"Could not search for author {author!r}: {error}") from error

    if not candidates:
        raise ToolError(f"Author not found in OpenAlex: {author!r}")

    # Picking the top hit would silently mix up namesakes — and a wrong author's
    # publication list looks every bit as plausible as the right one's.
    if len(candidates) > 1:
        listing = "\n".join(f"- {_describe(candidate)}" for candidate in candidates)
        raise ToolError(
            f"{author!r} matches several OpenAlex authors. Call again with one of "
            f"these author IDs (or an ORCID):\n{listing}"
        )

    return candidates[0]["author_id"]


def _describe(candidate: dict) -> str:
    details = [
        f"ORCID {candidate['orcid']}" if candidate["orcid"] else "",
        candidate["institution"],
        f"{candidate['works_count']} works, {candidate['cited_by_count']} citations",
    ]
    # Some display names carry stray newlines, which would break the one-per-line list.
    name = " ".join(candidate["name"].split())
    return (
        f"{candidate['author_id']} — {name} "
        f"({'; '.join(detail for detail in details if detail)})"
    )
