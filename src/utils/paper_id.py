"""Paper identifier formatting: parsing what callers pass in, and stripping the URL
forms sources publish down to the bare identifiers other APIs take back."""

import re

# OpenAlex publishes DOIs in resolver-URL form; the bare identifier is what the other
# APIs take back, so the prefix comes off in `bare_doi`. Where the DOI serves as a link
# rather than an identifier (a paper's `url` fallback), keep the URL form instead.
DOI_URL_PREFIX = "https://doi.org/"

_OPENALEX_ID = re.compile(r"^(?:https?://openalex\.org/)?(W\d+)$", re.IGNORECASE)


def to_openalex_id(paper_id: str) -> str:
    """The bare OpenAlex work id (`W…`) in `paper_id`, for a `/works/{id}` lookup.

    Accepts only an OpenAlex id, bare or URL, in any case. DOIs, arXiv ids and anything
    else raise ValueError.
    """
    if match := _OPENALEX_ID.match(paper_id.strip()):
        return match.group(1).upper()

    raise ValueError(
        f"Unrecognised paper identifier {paper_id!r}: expected an OpenAlex paper ID "
        "('W2626778328'); DOIs and arXiv IDs are not accepted."
    )


def bare_doi(value: str | None) -> str:
    """`https://doi.org/10.x/y` → `10.x/y`; empty for a missing DOI."""
    if not value:
        return ""

    return value.removeprefix(DOI_URL_PREFIX)


def bare_openalex_id(value: str | None) -> str:
    """`https://openalex.org/W123` → `W123`; empty for a missing id."""
    if not value:
        return ""

    return value.rsplit("/", 1)[-1]
