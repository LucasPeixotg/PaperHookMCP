"""A disk-backed store of paper records, keyed by paper and not by request.

Searches and lists still go live — only the API knows which papers match a query — but
every paper they return is written here, once, under its `paper_id`, with its DOI as
an alias. The store then answers identifier lookups without a request and fills in
what a fresh record left blank from the cached copy of the same work.

A record may also carry `full_paper`, the extracted full text. It is not a PaperData
field, so it never rides along on search results; only `put_full_paper` and
`get_full_paper` touch it.
"""

import dataclasses
import logging

import diskcache

from common import PaperData
from config import settings

logger = logging.getLogger(__name__)

# Query-specific: a score from one search must never be served for another.
_UNCACHED_FIELDS = {"relevance_score"}

# Filled from the cache when the fresh record has them blank. `cited_by_count` is
# absent on purpose: a listing that carries it always selected it, so its 0 is a
# real count, not a gap.
_FILLABLE_FIELDS = ("doi", "title", "year", "abstract", "url", "license", "venue")


def _is_blank(value) -> bool:
    return value in ("", 0, None)


def _doi_key(doi: str) -> str:
    # DOIs are case-insensitive by specification, and the sources disagree on casing.
    return f"doi:{doi.casefold()}"


def _paper_key(paper_id: str) -> str:
    return f"paper:{paper_id}"


class PaperStore:
    def __init__(self, cache: diskcache.Cache, ttl_seconds: int | None = None):
        self._cache = cache
        # Citation counts and URLs drift, so records expire rather than live forever.
        self._ttl = ttl_seconds

    def get(self, paper_id: str) -> dict | None:
        """The cached record for `paper_id`, or None."""
        if not paper_id:
            return None
        return self._safe(lambda: self._cache.get(_paper_key(paper_id)))

    def find(self, paper: PaperData) -> dict | None:
        """The cached record of `paper`, by its `paper_id` and then by its DOI."""
        return self._resolve(paper.paper_id, paper.doi)

    def get_full_paper(self, paper_id: str = "", doi: str = "") -> str | None:
        """The cached full text of the paper, by `paper_id` and then by DOI, or None."""
        record = self._resolve(paper_id, doi)
        return record.get("full_paper") if record else None

    def put_full_paper(self, paper_id: str, full_paper: str, doi: str = "") -> None:
        """Store the full text on the record of `paper_id`, aliased by `doi` if given."""
        if not paper_id:
            return
        self._merge(paper_id, {"paper_id": paper_id, "full_paper": full_paper})
        if doi:
            self._safe(lambda: self._cache.set(
                _doi_key(doi), paper_id, expire=self._ttl
            ))

    def absorb(self, papers: list[PaperData]) -> list[PaperData]:
        """`papers` enriched from the store, each merged back into it.

        A record cached under the same `paper_id` fills every blank fillable field; one
        reached only through the DOI is another OpenAlex record of the work, so it lends
        the abstract alone — its URL and license describe a different copy. Non-blank
        fresh fields always win. The input records are left untouched.
        """
        absorbed = []
        for paper in papers:
            enriched = self._enrich(paper)
            self._put(enriched)
            absorbed.append(enriched)
        return absorbed

    def put_summary(self, paper_id: str, title: str, cited_by_count: int) -> None:
        """Merge a work lookup's fields into the record of `paper_id`."""
        if not paper_id:
            return
        self._merge(paper_id, {"paper_id": paper_id, "title": title,
                               "cited_by_count": cited_by_count})

    def _enrich(self, paper: PaperData) -> PaperData:
        cached = self.find(paper)
        if not cached:
            return paper

        same_record = cached.get("paper_id") == paper.paper_id
        # `hasattr` below also keeps `full_paper`, which PaperData lacks, off the paper.
        fields = _FILLABLE_FIELDS if same_record else ("abstract",)

        filled = {
            name: cached[name]
            for name in fields
            if hasattr(paper, name)
            and _is_blank(getattr(paper, name))
            and not _is_blank(cached.get(name))
        }
        return dataclasses.replace(paper, **filled) if filled else paper

    def _put(self, paper: PaperData) -> None:
        if not paper.paper_id:
            return

        record = {
            key: value
            for key, value in dataclasses.asdict(paper).items()
            if key not in _UNCACHED_FIELDS
        }
        self._merge(paper.paper_id, record)

        if paper.doi:
            self._safe(lambda: self._cache.set(
                _doi_key(paper.doi), paper.paper_id, expire=self._ttl
            ))

    def _resolve(self, paper_id: str, doi: str) -> dict | None:
        if record := self.get(paper_id):
            return record
        if not doi:
            return None

        aliased_id = self._safe(lambda: self._cache.get(_doi_key(doi)))
        return self.get(aliased_id) if aliased_id else None

    def _merge(self, paper_id: str, fresh: dict) -> None:
        # Starts from the cached record, so fields the fresh one lacks — `full_paper`
        # above all — survive every later absorb or summary.
        merged = dict(self.get(paper_id) or {})
        merged.update({
            key: value
            for key, value in fresh.items()
            if not _is_blank(value) or key == "cited_by_count"
        })
        self._safe(lambda: self._cache.set(
            _paper_key(paper_id), merged, expire=self._ttl
        ))

    @staticmethod
    def _safe(operation):
        # A locked or corrupt cache degrades to no cache; it never fails a tool.
        try:
            return operation()
        except Exception as error:
            logger.warning("Paper cache unavailable: %s", error)
            return None


_store: PaperStore | None = None


def get_paper_store() -> PaperStore | None:
    """The process-wide store, or None when `CACHE_ENABLED` is off.

    Built on first use, not at import, so importing a tool module creates no
    directories.
    """
    global _store
    if not settings.CACHE_ENABLED:
        return None

    if _store is None:
        try:
            cache = diskcache.Cache(
                str(settings.CACHE_DIR),
                size_limit=settings.CACHE_SIZE_LIMIT_MB * 1024 * 1024,
            )
        except Exception as error:
            logger.warning("Paper cache unavailable: %s", error)
            return None
        _store = PaperStore(cache, ttl_seconds=settings.CACHE_TTL_SECONDS)

    return _store
