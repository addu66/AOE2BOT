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
# research/buy), extracted into the `actions` table. A model trained on
# ONLY these types learns build-order/economy priorities -- it cannot
# control units at all (no army movement, no targeting). See
# MOVEMENT_ACTION_TYPES below for that half, and docs/PIPELINE.md "Action
# vocabulary" for the full reasoning on why these two are split into
# separate tables rather than one.
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

# --- Action vocabulary (movement / unit control) ---------------------------
# `inputs[].type` values that represent commanding existing units: who
# (object_ids), what kind of command, and where (target position or target
# object). Extracted into the separate `movements` table -- these have a
# structurally different shape (a list of object_ids + a spatial target)
# than the macro table's single entity_id + amount.
#
# CORRECTION vs. an earlier, wrong blanket claim in this file: object_ids
# for these types are NOT generally DE-packed/unresolved. Checked directly
# against Hera_vs_MBL_1.json (save_version 68.0): 0 of ~4500 object_id
# references across Move/Order/Gather/Patrol/Gather Point were anomalous.
# The corruption is real but narrowly confined to Garrison/Ungarrison/
# Unqueue (~4.7% of all object_id references in that replay, ALL within
# those three types) -- see docs/PIPELINE.md "Action vocabulary" for the
# full investigation. Garrison/Ungarrison/Unqueue are deliberately left
# out of this set for now; OBJECT_ID_SUSPECT_THRESHOLD below still guards
# against silently trusting a bad value if a future replay's Move/Order
# turns out less clean than this one.
MOVEMENT_ACTION_TYPES = {
    "Move",           # move to position, no combat stance
    "Attack Move",    # move to position, engage anything encountered
    "Order",          # attack/interact with a specific target (unit or building)
    "Gather",         # start gathering from a specific resource
    "Gather Point",   # set a building's rally point
    "Patrol",         # patrol between the current position and a target
    "Attack Ground",  # target ground (e.g. siege) rather than a unit
}

# object_ids above this are flagged (not silently dropped) as suspect --
# real instance ids observed so far top out in the low thousands per
# match; the confirmed-corrupted Garrison/Ungarrison/Unqueue values were
# consistently in the 1M-10M range with a clean gap below that. Anything
# this large gets `reliable=False` in the movements table instead of being
# trusted or discarded outright.
OBJECT_ID_SUSPECT_THRESHOLD = 100_000

# --- Sharding --------------------------------------------------------------
MATCHES_PER_SHARD = 200
