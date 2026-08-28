"""
Client for Henrik's Unofficial Valorant API (api.henrikdev.xyz).

Used instead of the official Riot match-list endpoint, which requires a
production key. Henrik's API provides the same match data without that
restriction. Free tier allows ~30 req/min; pass an API key for higher limits.
"""

import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HENRIK_BASE = "https://api.henrikdev.xyz"

_REGION_MAP = {"na", "eu", "ap", "kr", "br", "latam"}


class HenrikClient:
    """
    Thin rate-limited wrapper around Henrik's Valorant API.

    Parameters
    ----------
    api_key : str | None
        Optional Henrik API key for higher rate limits (get one at
        https://docs.henrikdev.xyz). Leave None for free-tier access.
    region : str
        Valorant region shard (na, eu, ap, kr, br, latam).
    max_retries : int
        Retry attempts on 429 / 5xx responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "na",
        max_retries: int = 5,
    ) -> None:
        self._api_key = api_key
        self._region = region if region in _REGION_MAP else "na"
        self._max_retries = max_retries
        self._http = httpx.Client(timeout=30.0)
        self._min_interval = 20.0
        self._last_request_ts = 0.0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "HenrikClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{HENRIK_BASE}{path}"
        headers = {}
        if self._api_key:
            headers["Authorization"] = self._api_key

        backoff = 2.0
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = self._http.get(url, headers=headers, params=params)
            except httpx.RequestError as exc:
                logger.warning("Network error on attempt %d: %s", attempt + 1, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                logger.warning("Rate limited. Sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code in (500, 502, 503, 504):
                logger.warning("Server error %d on attempt %d", resp.status_code, attempt + 1)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            resp.raise_for_status()

        raise RuntimeError(f"Failed after {self._max_retries} attempts: {url}")

    def get_matches(
        self,
        name: str,
        tag: str,
        mode: str = "competitive",
        size: int = 20,
    ) -> list[dict]:
        """
        Fetch recent matches for a player. Returns full match data dicts.

        Parameters
        ----------
        name : str   Player name (the part before #).
        tag  : str   Player tag (the part after #).
        mode : str   Game mode filter. Use "competitive" for ranked.
        size : int   Number of matches to return (max 20 for free tier).
        """
        path = f"/valorant/v3/matches/{self._region}/{name}/{tag}"
        data = self._get(path, params={"mode": mode, "size": size})
        return data.get("data", [])
