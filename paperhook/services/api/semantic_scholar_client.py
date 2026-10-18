import logging

from ...common import PaperData
from ...config import settings

from .api_client import APIError, ApiClient

logger = logging.getLogger(__name__)

class SemanticScholarClient(ApiClient):
    # Fields requested from the Graph API — anything not listed here comes back absent.
    SEARCH_FIELDS = "paperId,externalIds,title,year,abstract,url,openAccessPdf"

    # The relevance search endpoint rejects a limit above 100.
    MAX_LIMIT = 100

    def __init__(self, timeout = 10):
        base_url = settings.SEMANTIC_SCHOLAR_API_BASE_URL
        api_x_key = settings.SEMANTIC_SCHOLAR_API_TOKEN

        super().__init__(base_url, timeout=timeout, api_x_key=api_x_key)

    def search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        # The API answers a blank query with a 400, so don't spend a request on it.
        if not query.strip():
            return []

        try:
            payload = self._request(
                "GET",
                "/paper/search",
                params={
                    "query": query,
                    "limit": max(1, min(limit, self.MAX_LIMIT)),
                    "fields": self.SEARCH_FIELDS,
                },
            )
        except APIError as error:
            logger.error(error)
            return []

        return [self._to_paper_data(item) for item in payload.get("data") or []]

    @staticmethod
    def _to_paper_data(item: dict) -> PaperData:
        # Every field is nullable in a search response, while PaperData requires
        # its fields, so each value is coerced to its empty equivalent.
        open_access = item.get("openAccessPdf") or {}

        # The Graph API has no top-level doi; it lives under externalIds, keyed
        # upper-case, alongside ArXiv, MAG and the rest. It arrives bare already.
        external_ids = item.get("externalIds") or {}

        return PaperData(
            paper_id=item.get("paperId") or "",
            doi=external_ids.get("DOI") or "",
            title=item.get("title") or "",
            year=item.get("year") or 0,
            abstract=item.get("abstract") or "",
            url=open_access.get("url") or item.get("url") or "",
            license=open_access.get("license") or "",
            # The Graph API publishes only a PDF link for the open-access copy.
            oa_pdf=open_access.get("url") or "",
        )
