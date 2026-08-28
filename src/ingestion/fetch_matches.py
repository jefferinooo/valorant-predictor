"""
Batch match fetcher using Henrik's Unofficial Valorant API.

Strategy:
  1. Start from a set of seed player Riot IDs (e.g. "Name#TAG").
  2. For each seed, fetch their recent competitive matches via Henrik's API.
     Each response already includes full match data — no separate download step.
  3. Deduplicate by match ID and save each unique match as JSON.
  4. Saves raw JSON under data/raw/matches/{match_id}.json.

Run via:
    python3 -m src.ingestion.fetch_matches --seeds "Name#TAG" "OtherName#TAG"
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

from config.settings import settings
from src.ingestion.henrik_client import HenrikClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WAIT_BETWEEN_PLAYERS = 20  # seconds


def run(seeds: list[str], data_dir: str | None = None, mode: str = "competitive") -> None:
    raw_dir = Path(data_dir or settings.data_dir) / "raw" / "matches"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    for existing in raw_dir.glob("*.json"):
        seen_ids.add(existing.stem)

    total_saved = 0
    valid_seeds = [s for s in seeds if "#" in s]

    with HenrikClient(api_key=settings.henrik_api_key or None, region=settings.riot_region) as client:
        for i, seed in enumerate(valid_seeds):
            name, tag = seed.rsplit("#", 1)
            logger.info("[%d/%d] Fetching %s ...", i + 1, len(valid_seeds), seed)

            try:
                matches = client.get_matches(name, tag, mode=mode, size=20)
            except Exception as exc:
                logger.warning("  Failed: %s", exc)
                matches = []

            new_for_seed = 0
            for match in (matches or []):
                if not match:
                    continue
                match_id: str = (match.get("metadata") or {}).get("matchid", "")
                if not match_id or match_id in seen_ids:
                    continue
                seen_ids.add(match_id)
                dest = raw_dir / f"{match_id}.json"
                dest.write_text(json.dumps(match), encoding="utf-8")
                new_for_seed += 1
                total_saved += 1

            logger.info("  Saved %d new matches (total so far: %d)", new_for_seed, total_saved)

            if i < len(valid_seeds) - 1:
                logger.info("  Waiting %ds before next player...", WAIT_BETWEEN_PLAYERS)
                time.sleep(WAIT_BETWEEN_PLAYERS)

    logger.info("Done. %d new matches saved to %s", total_saved, raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Valorant match data via Henrik's API."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        metavar="NAME#TAG",
        help="Riot IDs to fetch matches for, e.g. 'TenZ#000'",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the DATA_DIR setting from .env",
    )
    parser.add_argument(
        "--mode",
        default="competitive",
        help="Game mode filter (default: competitive)",
    )
    args = parser.parse_args()
    run(seeds=args.seeds, data_dir=args.data_dir, mode=args.mode)
