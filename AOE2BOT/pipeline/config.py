"""Pipeline-wide configuration: filters, action vocabulary, paths.

Single source of truth for what counts as "in scope" for this training
run (top players, Arabia only) and what the extractor considers a
macro-level (build-order / decision) action vs. a micro-level click.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------

AOE2BOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = AOE2BOT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
QUARANTINE_DIR = DATA_DIR / "quarantine"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

# --- Match filters -------------------------------------------------------
# A replay must pass ALL of these to be extracted. Adjust as the corpus
# (top players, Arabia-only) is curated; every rejection is logged with
# its reason to QUARANTINE_DIR / rejected.jsonl rather than silently
# dropped, so filtering-out-too-much is visible.

ALLOWED_MAPS = {"Arabia"}          # match on mgz Map.name, case-insensitive
REQUIRE_RATED = False              # top-player replays are often unrated tournament games
REQUIRE_1V1 = True                 # diplomacy_type == "1v1"
REQUIRE_COMPLETED = True           # game ended normally / not restored mid-parse
MIN_DURATION_SECONDS = 5 * 60      # drop instant-resigns / no-shows
MAX_DURATION_SECONDS = 90 * 60     # sanity cap against corrupt duration fields

# Optional curation knobs. Leave empty to skip that check.
ALLOWED_PROFILE_IDS: set[int] = set()   # e.g. {196240, 199325, ...} for named top players
MIN_RATE_SNAPSHOT: int | None = None    # e.g. 1800, if using an ELO floor instead

# --- Action vocabulary (macro / decision level) ---------------------------
# `inputs[].type` values that represent a *decision* (what to build/train/
# research/buy) rather than *micro-execution* (where to click, whom to
# target). Move/Order/Gather/Attack Move/etc. are excluded on purpose:
# their object_ids are DE-packed and unresolved by this mgz version, and
# even if resolved they dominate the stream (~49% of all inputs) without
# carrying build-order signal -- see docs/PIPELINE.md "Action vocabulary".
MACRO_ACTION_TYPES = {
    "Queue",     # train unit / queue at a building (requires the DE_QUEUE parsing fix, see docs)
    "Build",     # place building
    "Research",  # start a tech (age-ups usually surface here too)
    "Wall",      # placing wall segments is a deliberate defensive decision
    "Buy",       # market
    "Sell",      # market
}

# Age-up transitions are sourced from `match.uptimes` (reliable, one row
# per age per player) rather than sniffed out of `Research`, and injected
# into the actions table as synthetic type="Age Up" rows. If a Research
# entry's param is one of these, it's dropped to avoid double-counting.
AGE_NAMES = {"Feudal Age", "Castle Age", "Imperial Age"}

# --- Sharding --------------------------------------------------------------
MATCHES_PER_SHARD = 200
