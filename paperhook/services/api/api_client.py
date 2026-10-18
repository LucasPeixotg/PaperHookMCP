import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class APIError(Exception):
    """Base exception for consumer errors."""

class ApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0, *, api_x_key: str = "", bearer_token: str = ""):
        headers = {}
        headers["Accept"] = "application/json"
        if api_x_key:
            headers["x-api-key"] = api_x_key

        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        reraise=True,
    )
    def _send(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        # Transport only. The retry predicate has to see the raw httpx error, so the
        # translation to APIError happens in _request, after the attempts run out.
        return self._client.request(method, endpoint, **kwargs)

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        try:
            response = self._send(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as err:
            raise APIError(f"API status {err.response.status_code}: {err.response.text}") from err
        except httpx.RequestError as err:
            raise APIError(f"Network failure: {err}") from err
