"""
Batch match fetcher.

Strategy:
  1. Start from a set of seed player Riot IDs (e.g. "Name#TAG").
  2. Resolve each to a PUUID via the account endpoint.
  3. Fetch that player's match list (up to 20 match IDs).
  4. Download each match JSON once, deduplicating by match ID.
  5. Save raw JSON under data/raw/matches/{match_id}.json.

Run via:
    python -m src.ingestion.fetch_matches --seeds "Name#TAG" "OtherName#NA1"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
from src.ingestion.riot_client import RiotClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_seeds(client: RiotClient, seeds: list[str]) -> list[str]:
    """Convert 'Name#TAG' strings to PUUIDs."""
    puuids: list[str] = []
    for seed in seeds:
        if "#" not in seed:
            logger.warning("Skipping malformed seed (expected 'Name#TAG'): %s", seed)
            continue
        name, tag = seed.rsplit("#", 1)
        try:
            account = client.get_account_by_riot_id(name, tag)
            puuids.append(account["puuid"])
            logger.info("Resolved %s → %s", seed, account["puuid"])
        except Exception as exc:
            logger.warning("Could not resolve %s: %s", seed, exc)
    return puuids


def collect_match_ids(client: RiotClient, puuids: list[str]) -> set[str]:
    """Fetch match lists for all PUUIDs and return a deduplicated set of match IDs."""
    match_ids: set[str] = set()
    for puuid in tqdm(puuids, desc="Fetching match lists"):
        try:
            data = client.get_match_list(puuid)
            for entry in data.get("history", []):
                match_ids.add(entry["matchId"])
        except Exception as exc:
            logger.warning("Match list failed for %s: %s", puuid, exc)
    return match_ids


def download_matches(
    client: RiotClient,
    match_ids: set[str],
    out_dir: Path,
    skip_existing: bool = True,
) -> int:
    """Download each match and save as JSON. Returns count of newly saved matches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for match_id in tqdm(match_ids, desc="Downloading matches"):
        dest = out_dir / f"{match_id}.json"
        if skip_existing and dest.exists():
            continue
        try:
            match_data = client.get_match(match_id)
            dest.write_text(json.dumps(match_data), encoding="utf-8")
            saved += 1
        except Exception as exc:
            logger.warning("Failed to download %s: %s", match_id, exc)
    return saved


def run(seeds: list[str], data_dir: str | None = None) -> None:
    raw_dir = Path(data_dir or settings.data_dir) / "raw" / "matches"

    with RiotClient(
        api_key=settings.riot_api_key,
        region=settings.riot_region,
        cluster=settings.riot_cluster,
    ) as client:
        puuids = resolve_seeds(client, seeds)
        if not puuids:
            logger.error("No valid PUUIDs resolved. Aborting.")
            sys.exit(1)

        match_ids = collect_match_ids(client, puuids)
        logger.info("Collected %d unique match IDs", len(match_ids))

        saved = download_matches(client, match_ids, raw_dir)
        logger.info("Saved %d new matches to %s", saved, raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Valorant match data from the Riot API.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        metavar="NAME#TAG",
        help="Riot IDs to start from, e.g. 'Shroud#NA1'",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the DATA_DIR setting from .env",
    )
    args = parser.parse_args()
    run(seeds=args.seeds, data_dir=args.data_dir)
