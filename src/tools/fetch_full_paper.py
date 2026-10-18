import json

from fastmcp.exceptions import ToolError

from common import PaperData
from services.api import OpenAlexClient
from services.api.api_client import APIError
from services.cache import get_paper_store
from services.fulltext import PdfClient, extract_text
from utils import to_openalex_id

open_alex_client = OpenAlexClient()
pdf_client = PdfClient()


def fetch_full_paper(paper_id: str) -> str:
    """
    Downloads and extracts the complete full-text content of a specific paper.

    Use this tool AFTER running `find_relevant_papers` (or `get_citations` /
    `get_references`) when you need technical details not available in the
    abstract—such as specific equations, detailed methodology, hyperparameter
    grids, or results tables. Only papers with an open-access PDF can be read;
    for any other paper the error names a URL (the open-access landing page if
    there is one) that you can try fetching manually.

    Args:
        paper_id: The OpenAlex paper ID ('W2626778328'), as returned in `paper_id`
                  by the other tools. DOIs and arXiv IDs are not accepted.

    Returns:
        JSON object with the paper's `paper_id`, `doi`, `title`, `source_url`
        (the PDF the text was extracted from) and `full_text`, the plain text of
        every page.
    """
    try:
        work_id = to_openalex_id(paper_id)
    except ValueError as error:
        raise ToolError(str(error)) from error

    store = get_paper_store()
    if store and (text := store.get_full_paper(work_id)) is not None:
        record = store.get(work_id) or {}
        return _response(work_id, record.get("doi", ""), record.get("title", ""),
                         record.get("oa_pdf", ""), text)

    try:
        paper = open_alex_client.get_paper(work_id)
    except APIError as error:
        # A 404's body is an HTML error page — useless to the caller, so it is dropped.
        if str(error).startswith("API status 404"):
            raise ToolError(f"Paper not found in OpenAlex: {paper_id!r}") from error
        raise ToolError(f"Could not look up {paper_id!r}: {error}") from error

    if not paper.oa_pdf:
        raise ToolError(_no_pdf_message(paper_id, paper))

    try:
        text = extract_text(pdf_client.download(paper.oa_pdf))
    except APIError as error:
        raise ToolError(
            f"Could not read the PDF of {paper_id!r} at {paper.oa_pdf}: {error}"
        ) from error

    # A scanned PDF has no text layer. Not cached, so a later OCR fallback gets a go.
    if not text:
        raise ToolError(
            f"The PDF of {paper_id!r} has no extractable text (likely a scanned copy)."
        )

    if store:
        store.put_full_paper(paper.paper_id, text, doi=paper.doi)
    return _response(paper.paper_id, paper.doi, paper.title, paper.oa_pdf, text)


def _no_pdf_message(paper_id: str, paper: PaperData) -> str:
    """Hands the caller the best link left to fetch by hand, saying which kind it is."""
    message = f"No open-access PDF is known for {paper_id!r}."
    if paper.oa_url:
        return (f"{message} Its open-access URL (oa_url) is {paper.oa_url} — try fetching "
                "that page manually; it is the open-access copy's landing page and may "
                "link to the full text.")
    if paper.url:
        return (f"{message} There is no open-access URL (oa_url) for it; the only known "
                f"link is {paper.url}, which may be a paywalled publisher page — you can "
                "try fetching it manually.")
    return f"{message} No URL is known for it either."


def _response(paper_id: str, doi: str, title: str, source_url: str, text: str) -> str:
    return json.dumps({
        "paper_id": paper_id,
        "doi": doi,
        "title": title,
        "source_url": source_url,
        "full_text": text,
    })
