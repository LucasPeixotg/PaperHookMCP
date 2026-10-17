import dataclasses
import logging

from common import CitingPaperData, PaperData
from config import settings
from services.cache import PaperStore, get_paper_store, request_key
from utils import bare_doi, bare_openalex_id, bare_orcid

from .api_client import ApiClient, APIError

logger = logging.getLogger(__name__)

# Tells "use the process-wide store" apart from an explicit None, which means none.
_DEFAULT_STORE = object()

class OpenAlexClient(ApiClient):
    # Fields requested via `select` — the OpenAlex counterpart of Semantic Scholar's
    # `fields`. Anything not listed here is left out of the response, `relevance_score`
    # included: unselected it is simply absent, with no error, which would leave every
    # paper at the default 0.0 and silently disable the reranker's tiebreak.
    SEARCH_FIELDS = (
        "id,doi,display_name,publication_year,"
        "abstract_inverted_index,best_oa_location,primary_location,relevance_score"
    )

    # Works reached by a citation link, in either direction, or by authorship are
    # listed, not searched:
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

    # Enough to tell namesakes apart when a name matches several authors, and to name
    # the one picked otherwise.
    AUTHOR_FIELDS = (
        "id,display_name,orcid,works_count,cited_by_count,last_known_institutions"
    )

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

    def __init__(self, timeout = 10, paper_store: PaperStore | None = _DEFAULT_STORE):
        base_url = settings.OPENALEX_API_BASE_URL
        bearer_token = settings.OPENALEX_API_TOKEN

        super().__init__(base_url, timeout=timeout, bearer_token=bearer_token)

        self._paper_store = (
            get_paper_store() if paper_store is _DEFAULT_STORE else paper_store
        )

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

        A bare `W…` already in the paper store, with a title and a citation count, is
        answered from there without a request. An external id is answered only once
        this lookup has resolved it before: the DOI alias alone may lead to another
        OpenAlex record of the work, so it is OpenAlex's own answer that is cached.
        """
        endpoint = f"/works/{key}"
        params = {"select": self.WORK_SUMMARY_FIELDS}
        cache_key = request_key("GET", endpoint, params)

        if cached := self._cached_work(key, cache_key):
            return cached

        payload = self._request("GET", endpoint, params=params)

        work = {
            "paper_id": bare_openalex_id(payload.get("id")),
            "title": payload.get("display_name") or "",
            "cited_by_count": payload.get("cited_by_count") or 0,
        }
        if self._paper_store:
            self._paper_store.put_summary(**work)
            if key.startswith("doi:"):
                self._paper_store.put_work_pointer(cache_key, work["paper_id"])
        return work

    def _cached_work(self, key: str, cache_key: str) -> dict | None:
        if not self._paper_store:
            return None

        paper_id = (
            self._paper_store.get_work_pointer(cache_key)
            if key.startswith("doi:") else key
        )
        cached = self._paper_store.get(paper_id) if paper_id else None
        if cached and cached.get("title") and "cited_by_count" in cached:
            return {
                "paper_id": cached["paper_id"],
                "title": cached["title"],
                "cited_by_count": cached["cited_by_count"],
            }
        return None

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

    def get_author(self, key: str) -> dict:
        """Look up one author by OpenAlex id (`A123`) or ORCID (`orcid:0000-…`).

        Raises APIError, as `get_work` does: a 404 means the caller named an author
        OpenAlex does not know.
        """
        if self._paper_store and (cached := self._paper_store.get_author(key)):
            return cached

        payload = self._request(
            "GET", f"/authors/{key}", params={"select": self.AUTHOR_FIELDS}
        )

        author = self._to_author_summary(payload)
        if self._paper_store:
            self._paper_store.put_authors([author])
        return author

    def search_authors(self, name: str, limit: int = 5) -> list[dict]:
        """The authors whose name best matches `name`, most relevant first.

        Raises APIError rather than swallowing it like the work searches: an empty list
        here reads as "no such author", which a failed request must not pass for.
        """
        # A blank search matches every author in OpenAlex — as for works, skip it.
        if not name.strip():
            return []

        params = {
            "search": name,
            "per-page": max(1, min(limit, self.MAX_LIMIT)),
            "select": self.AUTHOR_FIELDS,
        }
        cache_key = request_key("GET", "/authors", params)
        if self._paper_store and (
            cached := self._paper_store.get_author_pointers(cache_key)
        ) is not None:
            return cached

        payload = self._request("GET", "/authors", params=params)

        authors = [
            self._to_author_summary(item) for item in payload.get("results") or []
        ]
        if self._paper_store:
            self._paper_store.put_authors(authors)
            self._paper_store.put_author_pointers(cache_key, authors)
        return authors

    def get_author_works(
        self, author_id: str, limit: int = 10, publication_year_min: int | None = None
    ) -> tuple[list[CitingPaperData], int]:
        """The works `author_id` authored, most cited first, plus how many there are.

        `author_id` must be a bare OpenAlex author id — `author.id:` takes no ORCID —
        so resolve anything else through `get_author` first. Raises APIError.
        """
        filter_value = f"author.id:{author_id}"
        if publication_year_min is not None:
            filter_value += f",publication_year:>{publication_year_min - 1}"

        return self._list_linked_works(filter_value, limit, "most_cited")

    def _list_linked_works(
        self, filter_value: str, limit: int, sort: str
    ) -> tuple[list[CitingPaperData], int]:
        """List the works one filter selects, plus how many there are in total.

        Both directions of the citation graph and an author's works share everything
        but the filter: the endpoint, sorts, selected fields, mapping and the APIError
        left to propagate.
        """
        params = {
            "filter": filter_value,
            "sort": self.CITATION_SORTS[sort],
            "per-page": max(1, min(limit, self.MAX_LIMIT)),
            "select": self.CITATION_FIELDS,
        }
        cache_key = request_key("GET", "/works", params)
        if self._paper_store and (
            cached := self._paper_store.get_pointers(cache_key, CitingPaperData)
        ):
            return cached

        payload = self._request("GET", "/works", params=params)

        papers = self._absorb(
            [self._to_citing_paper(item) for item in payload.get("results") or []]
        )
        total = (payload.get("meta") or {}).get("count") or len(papers)
        if self._paper_store:
            self._paper_store.put_pointers(cache_key, papers, total)
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
        params = {
            **search_param,
            "per-page": max(1, min(limit, max_limit)),
            "select": self.SEARCH_FIELDS,
        }
        cache_key = request_key("GET", "/works", params)
        if self._paper_store and (
            cached := self._paper_store.get_pointers(cache_key, PaperData)
        ):
            papers, _ = cached
            return papers

        try:
            payload = self._request("GET", "/works", params=params)
        except APIError as error:
            # Never cached: the empty list stands in for a failure, not an answer.
            logger.error(error)
            return []

        papers = self._absorb(
            [self._to_paper_data(item) for item in payload.get("results") or []]
        )
        if self._paper_store:
            self._paper_store.put_pointers(cache_key, papers)
        return papers

    def _absorb(self, papers: list) -> list:
        """`papers` written into the paper store and enriched from it, if there is one."""
        return self._paper_store.absorb(papers) if self._paper_store else papers

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
    def _to_author_summary(item: dict) -> dict:
        # A list, since an author can hold several affiliations; the first is enough
        # to tell namesakes apart.
        institution = next(iter(item.get("last_known_institutions") or []), None) or {}

        return {
            "author_id": bare_openalex_id(item.get("id")),
            "name": item.get("display_name") or "",
            "orcid": bare_orcid(item.get("orcid")),
            "institution": institution.get("display_name") or "",
            "works_count": item.get("works_count") or 0,
            "cited_by_count": item.get("cited_by_count") or 0,
        }

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
