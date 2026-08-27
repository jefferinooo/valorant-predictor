"""
Rate-limited synchronous client for the Riot Games API (Valorant endpoints).

Dev key limits: 20 req/s, 100 req/2 min.
We stay comfortably below both with a sliding-window limiter.
"""

import time
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Regional base URLs for match data
_REGION_HOST = {
    "na": "na.api.riotgames.com",
    "eu": "eu.api.riotgames.com",
    "ap": "ap.api.riotgames.com",
    "kr": "kr.api.riotgames.com",
    "br": "br.api.riotgames.com",
    "latam": "latam.api.riotgames.com",
}

# Cluster base URLs for account data
_CLUSTER_HOST = {
    "americas": "americas.api.riotgames.com",
    "europe": "europe.api.riotgames.com",
    "asia": "asia.api.riotgames.com",
    "esports": "esports.api.riotgames.com",
}


class _SlidingWindowLimiter:
    """Token-bucket-style limiter using a sliding window per time interval."""

    def __init__(self, per_second: int = 18, per_2min: int = 90) -> None:
        self._per_second = per_second
        self._per_2min = per_2min
        self._ts_1s: list[float] = []
        self._ts_2m: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        self._ts_1s = [t for t in self._ts_1s if now - t < 1.0]
        self._ts_2m = [t for t in self._ts_2m if now - t < 120.0]

        if len(self._ts_1s) >= self._per_second:
            delay = 1.0 - (now - self._ts_1s[0])
            if delay > 0:
                time.sleep(delay)
        if len(self._ts_2m) >= self._per_2min:
            delay = 120.0 - (now - self._ts_2m[0])
            if delay > 0:
                time.sleep(delay)

        now = time.monotonic()
        self._ts_1s.append(now)
        self._ts_2m.append(now)


class RiotClient:
    """
    Thin wrapper around the Riot API with automatic rate limiting and retry.

    Parameters
    ----------
    api_key : str
        Your RGAPI-... key from developer.riotgames.com.
    region : str
        Valorant regional shard (na, eu, ap, kr, br, latam).
    cluster : str
        Riot account cluster (americas, europe, asia, esports).
    max_retries : int
        How many times to retry on 429 / 5xx before raising.
    """

    def __init__(
        self,
        api_key: str,
        region: str = "na",
        cluster: str = "americas",
        max_retries: int = 5,
    ) -> None:
        if region not in _REGION_HOST:
            raise ValueError(f"Unknown region '{region}'. Choose from {list(_REGION_HOST)}")
        if cluster not in _CLUSTER_HOST:
            raise ValueError(f"Unknown cluster '{cluster}'. Choose from {list(_CLUSTER_HOST)}")

        self._api_key = api_key
        self._region_host = _REGION_HOST[region]
        self._cluster_host = _CLUSTER_HOST[cluster]
        self._max_retries = max_retries
        self._limiter = _SlidingWindowLimiter()
        self._http = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RiotClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    def _get(self, url: str) -> dict:
        headers = {"X-Riot-Token": self._api_key}
        backoff = 1.0
        for attempt in range(self._max_retries):
            self._limiter.wait()
            try:
                resp = self._http.get(url, headers=headers)
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
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code in (500, 502, 503, 504):
                logger.warning("Server error %d on attempt %d", resp.status_code, attempt + 1)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            # 403 = bad key, 404 = not found — no point retrying
            resp.raise_for_status()

        raise RuntimeError(f"Failed after {self._max_retries} attempts: {url}")

    # ------------------------------------------------------------------
    # Account endpoints  (cluster-routed)
    # ------------------------------------------------------------------

    def get_account_by_riot_id(self, game_name: str, tag_line: str) -> dict:
        """
        Returns account info including the player's PUUID.
        game_name and tag_line are the parts of 'Name#TAG'.
        """
        url = (
            f"https://{self._cluster_host}"
            f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        )
        return self._get(url)

    def get_account_by_puuid(self, puuid: str) -> dict:
        url = f"https://{self._cluster_host}/riot/account/v1/accounts/by-puuid/{puuid}"
        return self._get(url)

    # ------------------------------------------------------------------
    # Valorant match endpoints  (region-routed)
    # ------------------------------------------------------------------

    def get_match_list(self, puuid: str) -> dict:
        """
        Returns up to 20 recent match IDs for the given player PUUID.
        The Riot API does not support pagination for this endpoint.
        """
        url = (
            f"https://{self._region_host}"
            f"/val/match/v1/matchlists/by-puuid/{puuid}"
        )
        return self._get(url)

    def get_match(self, match_id: str) -> dict:
        """
        Returns the full match JSON for one completed match.
        This is the primary data source for in-round sequence extraction.
        """
        url = f"https://{self._region_host}/val/match/v1/matches/{match_id}"
        return self._get(url)

    def get_recent_matches(self, queue: str = "competitive") -> dict:
        """
        Returns up to 10 of the most recent match IDs for the given queue.
        Valid queue values: competitive, unrated, spikerush, deathmatch, etc.
        This endpoint requires a production key for most queues.
        """
        url = (
            f"https://{self._region_host}"
            f"/val/match/v1/recent-matches/by-queue/{queue}"
        )
        return self._get(url)
