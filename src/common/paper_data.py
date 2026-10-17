from dataclasses import dataclass


@dataclass
class PaperData:
    paper_id: str
    doi: str
    title: str
    year: int
    abstract: str
    url: str
    license: str
    # The source's own relevance ranking for the query that produced this record,
    # where it publishes one. Defaulted because Semantic Scholar does not, which
    # leaves its papers at 0.0 and losing every tie against an OpenAlex one —
    # acceptable while the OpenAlex semantic path is the only live source, but the
    # thing to revisit first when `tool.py` re-enables the others. Scales are
    # per-source and per-search-mode and are not comparable: OpenAlex keyword scores
    # run into the thousands, semantic ones sit around 1.
    relevance_score: float = 0.0


@dataclass
class CitingPaperData(PaperData):
    # A paper found by following a citation link rather than by a search, so it
    # carries the two fields a caller uses to judge a follow-up study without
    # reading it: where it appeared and how much it was picked up in turn.
    venue: str = ""
    cited_by_count: int = 0
