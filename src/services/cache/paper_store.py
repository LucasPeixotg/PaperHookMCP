"""A disk-backed store of paper records, plus pointers from requests to them.

Every paper a search or listing returns is written here, once, under its `paper_id`,
with its DOI as an alias. The store answers identifier lookups without a request and
fills in what a fresh record left blank from the cached copy of the same work.

Requests are cached by pointer, not by value: a request's entry holds only the ordered
ids it returned, with what belongs to that request alone — each paper's
`relevance_score`, a listing's total. A hit rebuilds the papers from their records, so
two requests returning one paper share one copy of it, and a fresher record shows up
in every request that points at it. Pointers expire sooner than records — result sets
drift faster than a paper does — and one whose records are gone is a miss.

Authors are kept the same way, under `author:` keys with their ORCID as an alias.

A record may also carry `full_paper`, the extracted full text. It is not a PaperData
field, so it never rides along on search results; only `put_full_paper` and
`get_full_paper` touch it.
"""

import dataclasses
import hashlib
import json
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
_FILLABLE_FIELDS = (
    "doi", "title", "year", "abstract", "url", "license", "oa_pdf", "oa_url", "venue"
)


def _is_blank(value) -> bool:
    return value in ("", 0, None)


def _doi_key(doi: str) -> str:
    # DOIs are case-insensitive by specification, and the sources disagree on casing.
    return f"doi:{doi.casefold()}"


def _paper_key(paper_id: str) -> str:
    return f"paper:{paper_id}"


def _author_key(author_id: str) -> str:
    return f"author:{author_id}"


def _orcid_key(orcid: str) -> str:
    return f"orcid:{orcid.casefold()}"


# What a rebuilt paper takes for a field its record lacks: `_merge` drops blank values,
# so a paper stored without an abstract comes back with no `abstract` key at all.
_BLANKS = {"doi": "", "title": "", "year": 0, "abstract": "", "url": "", "license": "",
           "oa_pdf": "", "oa_url": ""}


def request_key(method: str, endpoint: str, params: dict) -> str:
    """The key of one API request, over every parameter it sends.

    `select` and `per-page` are hashed too, so changing the fields a client selects
    retires the pointers its old requests left behind instead of serving them.
    """
    canonical = json.dumps(
        [method.upper(), endpoint, sorted(params.items())], default=str
    )
    return f"request:{hashlib.sha256(canonical.encode()).hexdigest()}"


class PaperStore:
    def __init__(
        self,
        cache: diskcache.Cache,
        ttl_seconds: int | None = None,
        request_ttl_seconds: int | None = None,
    ):
        self._cache = cache
        # Citation counts and URLs drift, so records expire rather than live forever.
        self._ttl = ttl_seconds
        self._request_ttl = request_ttl_seconds

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

    def put_pointers(
        self, request_key: str, papers: list[PaperData], total: int | None = None
    ) -> None:
        """Point `request_key` at `papers`, in order, with their scores and `total`.

        Call it after `absorb`, so every record pointed at exists. A paper without a
        `paper_id` has no record to point at, so such a request is not cached at all.
        """
        if any(not paper.paper_id for paper in papers):
            return
        items = [[paper.paper_id, paper.relevance_score] for paper in papers]
        self._put_pointer(request_key, {"items": items, "total": total})

    def get_pointers(
        self, request_key: str, cls: type[PaperData] = PaperData
    ) -> tuple[list[PaperData], int | None] | None:
        """The papers `request_key` points at, rebuilt as `cls`, and its total.

        None when the request is not cached, or when any record it points at is gone:
        a list missing a paper would pass for the full answer.

        A record evicted and then recreated in part — by `put_summary`, say — while a
        pointer is still live comes back with blanks. The pointer TTL being far
        shorter than the record one makes that rare enough to leave unguarded.
        """
        entry = self._get_pointer(request_key)
        if entry is None:
            return None

        names = {field.name for field in dataclasses.fields(cls)}
        papers = []
        for paper_id, score in entry["items"]:
            record = self.get(paper_id)
            if record is None:
                return None
            # Filtering on `cls`'s fields keeps `full_paper` off, and the listing
            # fields off a plain PaperData.
            fields = {key: value for key, value in record.items() if key in names}
            papers.append(cls(**(_BLANKS | fields), relevance_score=score))
        return papers, entry.get("total")

    def put_work_pointer(self, request_key: str, paper_id: str) -> None:
        """Point a one-work lookup, such as a DOI's, at the record it resolved to."""
        if paper_id:
            self._put_pointer(request_key, {"items": [[paper_id, 0.0]], "total": 1})

    def get_work_pointer(self, request_key: str) -> str | None:
        """The `paper_id` a one-work lookup resolved to, or None."""
        entry = self._get_pointer(request_key)
        return entry["items"][0][0] if entry and entry["items"] else None

    def put_authors(self, authors: list[dict]) -> None:
        """Store author summaries under their `author_id`, aliased by ORCID.

        Each summary replaces the cached one whole: every author lookup selects the
        same fields, so there is nothing a fresh one lacks for an old one to fill.
        """
        for author in authors:
            author_id = author.get("author_id")
            if not author_id:
                continue
            self._safe(lambda: self._cache.set(
                _author_key(author_id), author, expire=self._ttl
            ))
            if orcid := author.get("orcid"):
                self._safe(lambda: self._cache.set(
                    _orcid_key(orcid), author_id, expire=self._ttl
                ))

    def get_author(self, key: str) -> dict | None:
        """The cached author for an `A…` id or an `orcid:…` key, or None."""
        if not key:
            return None
        if key.startswith("orcid:"):
            orcid = key.removeprefix("orcid:")
            key = self._safe(lambda: self._cache.get(_orcid_key(orcid)))
            if not key:
                return None
        return self._safe(lambda: self._cache.get(_author_key(key)))

    def put_author_pointers(self, request_key: str, authors: list[dict]) -> None:
        """Point `request_key` at `authors`, in order. Call `put_authors` first."""
        if any(not author.get("author_id") for author in authors):
            return
        items = [[author["author_id"], 0.0] for author in authors]
        self._put_pointer(request_key, {"items": items, "total": len(items)})

    def get_author_pointers(self, request_key: str) -> list[dict] | None:
        """The authors `request_key` points at, or None — as `get_pointers`."""
        entry = self._get_pointer(request_key)
        if entry is None:
            return None

        authors = []
        for author_id, _ in entry["items"]:
            author = self.get_author(author_id)
            if author is None:
                return None
            authors.append(author)
        return authors

    def _put_pointer(self, request_key: str, entry: dict) -> None:
        self._safe(lambda: self._cache.set(
            request_key, entry, expire=self._request_ttl
        ))

    def _get_pointer(self, request_key: str) -> dict | None:
        return self._safe(lambda: self._cache.get(request_key))

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
        _store = PaperStore(
            cache,
            ttl_seconds=settings.CACHE_TTL_SECONDS,
            request_ttl_seconds=settings.REQUEST_CACHE_TTL_SECONDS,
        )

    return _store
