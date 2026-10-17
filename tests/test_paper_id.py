"""utils.paper_id — the one place paper identifiers are parsed and normalised.

Pins that `to_openalex_id` accepts only OpenAlex work ids (bare or URL, any case) and
returns the bare upper-cased `W…`. DOIs, arXiv ids and anything else are a ValueError,
not a ToolError, so the module stays free of fastmcp; tools translate it themselves.

Also pins the two strippers sources map through: `bare_doi` and `bare_openalex_id` turn
the URL forms OpenAlex publishes into the bare ids other APIs take back, and map a
missing value to "" because every PaperData field is a required non-null string.

`to_openalex_author_key` never raises: an OpenAlex author id or an ORCID becomes a
`/authors/{key}` key, and anything else is None, meaning "search for it as a name".
"""

import pytest

from utils import (
    bare_doi,
    bare_openalex_id,
    bare_orcid,
    to_openalex_author_key,
    to_openalex_id,
)


@pytest.mark.parametrize("paper_id", [
    "W2626778328",
    "w2626778328",
    "https://openalex.org/W2626778328",
    "  W2626778328  ",
])
def test_openalex_id_forms_map_to_the_bare_id(paper_id):
    assert to_openalex_id(paper_id) == "W2626778328"


@pytest.mark.parametrize("paper_id", [
    "",
    "attention is all you need",
    "649def34f8be52c8b66281af98ae884c09aef38b",  # a Semantic Scholar id
    "10.1038/s41586-020-2649-2",
    "doi:10.1038/s41586-020-2649-2",
    "https://doi.org/10.1038/s41586-020-2649-2",
    "1706.03762",
    "arXiv:2301.12345",
    "hep-th/9711200",
])
def test_unrecognised_identifiers_are_a_value_error(paper_id):
    with pytest.raises(ValueError, match="Unrecognised paper identifier"):
        to_openalex_id(paper_id)


@pytest.mark.parametrize("value, bare", [
    ("https://doi.org/10.48550/arxiv.1706.03762", "10.48550/arxiv.1706.03762"),
    ("10.48550/arxiv.1706.03762", "10.48550/arxiv.1706.03762"),
    ("", ""),
    (None, ""),
])
def test_bare_doi(value, bare):
    assert bare_doi(value) == bare


@pytest.mark.parametrize("value, bare", [
    ("https://openalex.org/W2626778328", "W2626778328"),
    ("W2626778328", "W2626778328"),
    ("", ""),
    (None, ""),
])
def test_bare_openalex_id(value, bare):
    assert bare_openalex_id(value) == bare


@pytest.mark.parametrize("author, key", [
    ("A5001234567", "A5001234567"),
    ("a5001234567", "A5001234567"),
    ("https://openalex.org/A5001234567", "A5001234567"),
    ("  A5001234567  ", "A5001234567"),
    ("0000-0002-1825-0097", "orcid:0000-0002-1825-0097"),
    ("https://orcid.org/0000-0002-1825-0097", "orcid:0000-0002-1825-0097"),
    ("0000-0002-1694-233x", "orcid:0000-0002-1694-233X"),
])
def test_author_identifiers_map_to_a_lookup_key(author, key):
    assert to_openalex_author_key(author) == key


@pytest.mark.parametrize("author", [
    "Geoffrey Hinton",
    "",
    "W2626778328",  # a work, not an author
    "0000-0002-1825",  # a truncated ORCID
])
def test_anything_else_is_a_name(author):
    assert to_openalex_author_key(author) is None


@pytest.mark.parametrize("value, bare", [
    ("https://orcid.org/0000-0002-1825-0097", "0000-0002-1825-0097"),
    ("0000-0002-1825-0097", "0000-0002-1825-0097"),
    ("", ""),
    (None, ""),
])
def test_bare_orcid(value, bare):
    assert bare_orcid(value) == bare
