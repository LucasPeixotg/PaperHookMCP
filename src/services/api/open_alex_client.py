import dataclasses
import logging

from common import CitingPaperData, PaperData
from config import settings
from utils import bare_doi, bare_openalex_id

from .api_client import ApiClient, APIError

logger = logging.getLogger(__name__)

class OpenAlexClient(ApiClient):
    # Fields requested via `select` — the OpenAlex counterpart of Semantic Scholar's
    # `fields`. Anything not listed here is left out of the response, `relevance_score`
    # included: unselected it is simply absent, with no error, which would leave every
    # paper at the default 0.0 and silently disable the reranker's tiebreak.
    SEARCH_FIELDS = (
        "id,doi,display_name,publication_year,"
        "abstract_inverted_index,best_oa_location,primary_location,relevance_score"
    )

    # Works reached by a citation link, in either direction, are listed, not searched:
    # there is no query to rank against, so no `relevance_score`, and the abstract stays
    # off the wire because a page of up to MAX_LIMIT reconstructed abstracts is far more
    # than a caller skimming follow-ups or a bibliography needs. `cited_by_count` and the
    # location's source supply what they read instead.
    CITATION_FIELDS = (
        "id,doi,display_name,publication_year,"
        "best_oa_location,primary_location,cited_by_count"
    )

    # Just enough of the target work to name it in the response and hand its id to
    # the `cites:` filter, which accepts nothing but an OpenAlex id.
    WORK_SUMMARY_FIELDS = "id,display_name,cited_by_count"

    CITATION_SORTS = {
        "most_cited": "cited_by_count:desc",
        "newest": "publication_date:desc",
    }

    # The API rejects a per-page outside 1..200 with a "Pagination error". Note that
    # OpenAlex's own LLM quick reference claims a maximum of 100; that is stale — 200
    # is what the endpoint actually serves, and what its rejection message names.
    MAX_LIMIT = 200

    # Semantic search runs against a vector index rather than the inverted one, and caps
    # far lower than the keyword endpoint's 200.
    SEMANTIC_MAX_LIMIT = 50

    # Only the first 2000 characters are embedded; the rest is dropped server-side, so
    # there is no reason to put it on the wire.
    SEMANTIC_MAX_QUERY_CHARS = 2000

    def __init__(self, timeout = 10):
        base_url = settings.OPENALEX_API_BASE_URL
        bearer_token = settings.OPENALEX_API_TOKEN

        super().__init__(base_url, timeout=timeout, bearer_token=bearer_token)

    def search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        # Unlike Semantic Scholar, a blank search is a 200 here — it just matches every
        # work in OpenAlex. That makes the request worse than useless, so skip it.
        if not query.strip():
            return []

        return self._search_works({"search": query}, limit, self.MAX_LIMIT)

    def semantic_search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        """Rank works by embedding similarity rather than by keyword overlap.

        OpenAlex embeds every work's title and abstract with GTE Large EN and compares
        the query against them by cosine similarity, so this reaches papers that answer
        the query in vocabulary it never used — "predicting drug toxicity from molecular
        structure" finds work that only ever says "QSAR". It rewards long, prose-like
        input: an abstract or a paragraph beats a handful of keywords.
        """
        # Nothing to embed, and OpenAlex would answer with every work in the index —
        # the same reasoning as the keyword path.
        if not query.strip():
            return []

        return self._search_works(
            {"search.semantic": query[: self.SEMANTIC_MAX_QUERY_CHARS]},
            limit,
            self.SEMANTIC_MAX_LIMIT,
        )

    def get_work(self, key: str) -> dict:
        """Look up one work by OpenAlex id (`W123`) or external id (`doi:10.x/...`).

        Unlike the searches this raises APIError instead of swallowing it: a 404 here
        means the caller named a paper OpenAlex does not know, which is an answer they
        need to hear, not an empty result to paper over.
        """
        payload = self._request(
            "GET", f"/works/{key}", params={"select": self.WORK_SUMMARY_FIELDS}
        )

        return {
            "paper_id": bare_openalex_id(payload.get("id")),
            "title": payload.get("display_name") or "",
            "cited_by_count": payload.get("cited_by_count") or 0,
        }

    def get_citing_works(
        self, work_id: str, limit: int = 10, sort: str = "most_cited"
    ) -> tuple[list[CitingPaperData], int]:
        """The works citing `work_id`, plus how many there are in total.

        `work_id` must be a bare OpenAlex id — `cites:` takes no DOI — so resolve
        anything else through `get_work` first. Raises APIError, as `get_work` does.
        """
        return self._list_linked_works(f"cites:{work_id}", limit, sort)

    def get_referenced_works(
        self, work_id: str, limit: int = 10, sort: str = "most_cited"
    ) -> tuple[list[CitingPaperData], int]:
        """The works `work_id` cites (its reference list), plus how many there are.

        The mirror of `get_citing_works`: `cited_by:` lists the entries of the work's
        `referenced_works`, and likewise takes nothing but a bare OpenAlex id. Raises
        APIError, as `get_work` does.
        """
        return self._list_linked_works(f"cited_by:{work_id}", limit, sort)

    def _list_linked_works(
        self, filter_value: str, limit: int, sort: str
    ) -> tuple[list[CitingPaperData], int]:
        """List the works one citation filter selects, plus how many there are in total.

        Both directions of the citation graph share everything but the filter: the
        endpoint, sorts, selected fields, mapping and the APIError left to propagate.
        """
        payload = self._request(
            "GET",
            "/works",
            params={
                "filter": filter_value,
                "sort": self.CITATION_SORTS[sort],
                "per-page": max(1, min(limit, self.MAX_LIMIT)),
                "select": self.CITATION_FIELDS,
            },
        )

        papers = [self._to_citing_paper(item) for item in payload.get("results") or []]
        total = (payload.get("meta") or {}).get("count") or len(papers)
        return papers, total

    def _search_works(
        self, search_param: dict, limit: int, max_limit: int
    ) -> list[PaperData]:
        """Issue one /works search and map the payload.

        The keyword and semantic paths differ only in which search parameter carries the
        query and how high the per-page ceiling goes; everything else — the endpoint, the
        selected fields, the error handling and the mapping — is shared. `search_param`
        holds exactly one entry because the API rejects a request naming more than one of
        `search`, `search.exact` and `search.semantic`.
        """
        try:
            payload = self._request(
                "GET",
                "/works",
                params={
                    **search_param,
                    "per-page": max(1, min(limit, max_limit)),
                    "select": self.SEARCH_FIELDS,
                },
            )
        except APIError as error:
            logger.error(error)
            return []

        return [self._to_paper_data(item) for item in payload.get("results") or []]

    @staticmethod
    def _to_paper_data(item: dict) -> PaperData:
        # Every field is nullable while PaperData requires all seven, so each value is
        # coerced to its empty equivalent.
        best = item.get("best_oa_location") or {}
        primary = item.get("primary_location") or {}

        # A PDF the caller can actually read beats a landing page, and the open-access
        # location beats the publisher's. Whichever wins also supplies the license.
        url, license_ = "", ""
        for location in (best, primary):
            url = location.get("pdf_url") or location.get("landing_page_url") or ""
            if url:
                license_ = location.get("license") or ""
                break
        else:
            # Deliberately the resolver-URL form: here the DOI is a link, not an id.
            url = item.get("doi") or ""

        return PaperData(
            # Both published as URLs ("https://openalex.org/W2626778328"); callers
            # want the bare identifiers.
            paper_id=bare_openalex_id(item.get("id")),
            doi=bare_doi(item.get("doi")),
            title=item.get("display_name") or "",
            year=item.get("publication_year") or 0,
            abstract=OpenAlexClient._abstract_from_inverted_index(
                item.get("abstract_inverted_index")
            ),
            url=url,
            license=license_,
            # Absent on any request carrying no search parameter, and null-able even
            # when present; both mean "this source did not rank this record".
            relevance_score=float(item.get("relevance_score") or 0.0),
        )

    @staticmethod
    def _to_citing_paper(item: dict) -> CitingPaperData:
        source = (item.get("primary_location") or {}).get("source") or {}

        return CitingPaperData(
            **dataclasses.asdict(OpenAlexClient._to_paper_data(item)),
            venue=source.get("display_name") or "",
            cited_by_count=item.get("cited_by_count") or 0,
        )

    @staticmethod
    def _abstract_from_inverted_index(index: dict | None) -> str:
        """Rebuild the plain-text abstract from OpenAlex's inverted index.

        Abstracts are only ever published as ``{"word": [positions...]}`` — there is no
        plain `abstract` field, and `select=abstract` is rejected outright — so this
        reconstruction is mandatory rather than an optimisation.
        """
        if not index:
            return ""

        positions = [position for slots in index.values() for position in slots]
        if not positions:
            return ""

        # A word can occupy several positions, and the index may skip some entirely;
        # the gaps are dropped rather than left as blanks.
        words = [""] * (max(positions) + 1)
        for word, slots in index.items():
            for position in slots:
                words[position] = word

        return " ".join(word for word in words if word)
