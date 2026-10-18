"""PdfClient and extract_text — no network.

Pins that a download hands back only a PDF: an HTTP error, an HTML page served with a
200, and an oversized body all raise APIError, as a corrupt file does on extraction.
The PDFs here are real ones built with pymupdf, so extraction runs end to end.
"""

import httpx
import pymupdf
import pytest
from tenacity import wait_none

from services.api.api_client import ApiClient, APIError
from services.fulltext import PdfClient, extract_text

URL = "https://oa.example.test/paper.pdf"


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    monkeypatch.setattr(ApiClient._send.retry, "wait", wait_none())


def make_pdf(*pages: str) -> bytes:
    with pymupdf.open() as document:
        for text in pages:
            document.new_page().insert_text((72, 72), text)
        return document.tobytes()


def make_client(*responses):
    calls = []

    def handler(request):
        calls.append(request)
        return responses[min(len(calls), len(responses)) - 1]

    client = PdfClient()
    client._client._transport = httpx.MockTransport(handler)
    return client, calls


def test_a_pdf_is_downloaded_and_its_text_extracted():
    pdf = make_pdf("Attention is all you need", "Results")
    client, calls = make_client(httpx.Response(200, content=pdf))

    text = extract_text(client.download(URL))

    assert str(calls[0].url) == URL
    assert calls[0].headers["Accept"] == "application/pdf"
    assert text == "Attention is all you need\n\nResults"


def test_a_redirect_is_followed():
    pdf = make_pdf("x")
    client, calls = make_client(
        httpx.Response(302, headers={"Location": "https://cdn.example.test/p.pdf"}),
        httpx.Response(200, content=pdf),
    )

    assert client.download(URL) == pdf
    assert len(calls) == 2


def test_an_http_error_raises():
    client, _ = make_client(httpx.Response(403, text="Forbidden"))

    with pytest.raises(APIError, match="403"):
        client.download(URL)


def test_an_html_page_is_not_a_pdf():
    client, _ = make_client(httpx.Response(200, text="<html>Just a moment...</html>"))

    with pytest.raises(APIError, match="did not serve a PDF"):
        client.download(URL)


def test_an_oversized_body_raises(monkeypatch):
    monkeypatch.setattr(PdfClient, "MAX_BYTES", 10)
    client, _ = make_client(httpx.Response(200, content=make_pdf("x")))

    with pytest.raises(APIError, match="too large"):
        client.download(URL)


def test_a_corrupt_pdf_raises_on_extraction():
    with pytest.raises(APIError, match="Could not read"):
        extract_text(b"%PDF-1.7 garbage")


def test_a_pdf_without_text_extracts_blank():
    assert extract_text(make_pdf("", "")) == ""
