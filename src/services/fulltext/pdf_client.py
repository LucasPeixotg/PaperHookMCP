"""Downloading an open-access PDF and extracting its text.

Both steps raise APIError, so a tool reports a dead link, a paywall interstitial and a
corrupt file the same way it reports a failed API request.
"""

import httpx
import pymupdf

from services.api.api_client import ApiClient, APIError


class PdfClient(ApiClient):
    # A paper is a few MB at most; a body far past that is not one worth extracting.
    MAX_BYTES = 50 * 1024 * 1024

    # Some open-access hosts refuse a request with no User-Agent at all.
    HEADERS = {"Accept": "application/pdf", "User-Agent": "PaperWeightMCP"}

    def __init__(self, timeout: float = 30.0):
        # No base URL: every request names a PDF on some other host.
        super().__init__("", timeout=timeout)

    def download(self, url: str) -> bytes:
        """The PDF at `url`, following redirects. Raises APIError for anything else.

        Retries on network errors and timeouts through `ApiClient._send`; not through
        `_request`, which parses the body as JSON.
        """
        try:
            response = self._send(
                "GET", url, headers=self.HEADERS, follow_redirects=True
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise APIError(
                f"PDF download failed with status {err.response.status_code}"
            ) from err
        except httpx.RequestError as err:
            raise APIError(f"Network failure: {err}") from err

        body = response.content
        if len(body) > self.MAX_BYTES:
            raise APIError(f"PDF too large: {len(body)} bytes")
        # Hosts commonly answer a PDF link with an HTML landing page or a bot check,
        # still with a 200, so the content type alone is not trusted.
        if not body.lstrip().startswith(b"%PDF"):
            raise APIError("The link did not serve a PDF")
        return body


def extract_text(pdf: bytes) -> str:
    """The plain text of every page of `pdf`, pages separated by a blank line.

    Raises APIError when the bytes are not a readable PDF. A scanned PDF with no text
    layer comes back blank rather than raising; the caller decides what that means.
    """
    try:
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            return "\n\n".join(page.get_text().strip() for page in document).strip()
    except Exception as err:
        raise APIError(f"Could not read the PDF: {err}") from err
