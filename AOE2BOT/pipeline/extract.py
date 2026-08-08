"""Extract .aoe2record files into compact parquet tables.

Usage:
    .venv/Scripts/python.exe -m pipeline.extract <replays_dir> [--sanity]

Produces, under data/parquet/:
    matches_XXXX.parquet     1 row per match
    players_XXXX.parquet     2 rows per match (1v1)
    timeseries_XXXX.parquet  ~365 rows per player per match (economy snapshot)
    uptimes_XXXX.parquet     up to 3 rows per player per match (age-ups)
    actions_XXXX.parquet     macro/decision-level actions (see config.MACRO_ACTION_TYPES)
    movements_XXXX.parquet   unit-control actions: who/what/where (see config.MOVEMENT_ACTION_TYPES)

And under data/quarantine/:
    rejected.jsonl   replays that parsed fine but failed a filter, with reason
    failed.jsonl      replays that raised while parsing, with best-effort version info

--sanity skips every filter in config.py (map/rated/1v1/duration) so the
existing 3 sample replays (used only to validate parsing + preprocessing,
per the project scope) still produce output even though they aren't
Arabia/rated games.
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mgz.model import parse_match

from pipeline import config


def match_id_for(match, path: Path) -> str:
    guid = getattr(match, "guid", None)
    if guid:
        return str(guid)
    # Fallback: hash path + file size (stable across re-runs, not across renames)
    h = hashlib.sha1(f"{path.name}:{path.stat().st_size}".encode()).hexdigest()
    return h[:32]


def peek_version(path: Path) -> dict:
    """Best-effort version info for a replay that failed to fully parse.

    Uses mgz.fast.header.parse_version() directly rather than the full
    header parse -- that first step (decompress + read the version fields)
    is far more robust than full parsing and succeeds even when a replay's
    save_version isn't fully supported yet (e.g. save_version 68.0 as of
    this writing -- see docs/PIPELINE.md). Getting the real save_version
    into the quarantine log, instead of None, is what makes failures
    triageable by version.
    """
    try:
        from mgz.fast import header as fast_header
        with open(path, "rb") as f:
            header = fast_header.decompress(f)
            version, game_version, save_version, log_version = fast_header.parse_version(header, f)
        return {"save_version": save_version, "log_version": log_version, "game_version": game_version}
    except Exception:
        return {"save_version": None, "log_version": None, "game_version": None}


def check_filters(match) -> list[str]:
    """Return a list of failed-filter reasons; empty list means it passes."""
    reasons = []
    map_name = (match.map.name or "").strip()
    if config.ALLOWED_MAPS and map_name.lower() not in {m.lower() for m in config.ALLOWED_MAPS}:
        reasons.append(f"map={map_name!r} not in ALLOWED_MAPS")
    if config.REQUIRE_RATED and not match.rated:
        reasons.append("not rated")
    if config.REQUIRE_1V1 and match.diplomacy_type != "1v1":
        reasons.append(f"diplomacy_type={match.diplomacy_type!r} != 1v1")
    if config.REQUIRE_COMPLETED and not match.completed:
        reasons.append("not completed")
    duration_s = match.duration.total_seconds()
    if duration_s < config.MIN_DURATION_SECONDS:
        reasons.append(f"duration={duration_s:.0f}s < MIN_DURATION_SECONDS")
    if duration_s > config.MAX_DURATION_SECONDS:
        reasons.append(f"duration={duration_s:.0f}s > MAX_DURATION_SECONDS")
    if config.ALLOWED_PROFILE_IDS:
        pids = {p.profile_id for p in match.players}
        if not pids & config.ALLOWED_PROFILE_IDS:
            reasons.append("no player in ALLOWED_PROFILE_IDS")
    if config.MIN_RATE_SNAPSHOT is not None:
        snaps = [p.rate_snapshot for p in match.players if p.rate_snapshot is not None]
        if not snaps or min(snaps) < config.MIN_RATE_SNAPSHOT:
            reasons.append(f"rate_snapshot below MIN_RATE_SNAPSHOT ({snaps})")
    return reasons


def extract_one(match, mid: str, source_file: str, rows: dict) -> None:
    winner_number = next((p.number for p in match.players if p.winner), None)

    rows["matches"].append({
        "match_id": mid,
        "guid": str(getattr(match, "guid", "") or ""),
        "map_name": match.map.name,
        "map_size": match.map.size,
        "map_id": match.map.id,
        "dataset": match.dataset,
        "game_version": match.game_version,
        "save_version": float(match.save_version) if match.save_version is not None else None,
        "build_version": match.build_version,
        "duration_s": match.duration.total_seconds(),
        "rated": bool(match.rated),
        "diplomacy_type": match.diplomacy_type,
        "completed": bool(match.completed),
        "winner_number": winner_number,
        "timestamp": str(match.timestamp) if match.timestamp else None,
        "source_file": source_file,
    })

    for p in match.players:
        rows["players"].append({
            "match_id": mid,
            "number": p.number,
            "profile_id": p.profile_id,
            "name": p.name,
            "civilization": p.civilization,
            "civilization_id": p.civilization_id,
            "color": p.color,
            "team_id": p.team_id,
            "winner": bool(p.winner),
            "eapm": p.eapm,
            "rate_snapshot": p.rate_snapshot,
        })

        for row in (p.timeseries or []):
            rows["timeseries"].append({
                "match_id": mid,
                "player_number": p.number,
                "t_s": row.timestamp.total_seconds(),
                "total_resources": row.total_resources,
                "total_objects": row.total_objects,
            })

    for up in (getattr(match, "uptimes", None) or []):
        age_name = up.age.name.replace("_", " ").title() if hasattr(up.age, "name") else str(up.age)
        rows["uptimes"].append({
            "match_id": mid,
            "player_number": up.player.number,
            "age": age_name,
            "t_s": up.timestamp.total_seconds(),
        })
        rows["actions"].append({
            "match_id": mid,
            "player_number": up.player.number,
            "t_s": up.timestamp.total_seconds(),
            "type": "Age Up",
            "param": age_name,
            "entity_id": None,
            "amount": None,
        })

    for inp in match.inputs:
        if inp.type in config.MACRO_ACTION_TYPES:
            if inp.type == "Research" and inp.param in config.AGE_NAMES:
                continue  # already represented via uptimes -> "Age Up"
            payload = inp.payload or {}
            entity_id = (
                payload.get("unit_id")
                or payload.get("building_id")
                or payload.get("technology_id")
                or payload.get("resource_id")
            )
            rows["actions"].append({
                "match_id": mid,
                "player_number": inp.player.number if inp.player else None,
                "t_s": inp.timestamp.total_seconds(),
                "type": inp.type,
                "param": inp.param,
                "entity_id": entity_id,
                "amount": payload.get("amount"),
            })

        elif inp.type in config.MOVEMENT_ACTION_TYPES:
            payload = inp.payload or {}
            object_ids = payload.get("object_ids") or []
            reliable = all(oid < config.OBJECT_ID_SUSPECT_THRESHOLD for oid in object_ids)
            rows["movements"].append({
                "match_id": mid,
                "player_number": inp.player.number if inp.player else None,
                "t_s": inp.timestamp.total_seconds(),
                "type": inp.type,
                "param": inp.param,
                "object_ids": object_ids,
                "target_id": payload.get("target_id"),
                "target_x": inp.position.x if inp.position else None,
                "target_y": inp.position.y if inp.position else None,
                "reliable": reliable,
            })


def flush_shard(rows: dict, shard_idx: int) -> None:
    config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for table_name, data in rows.items():
        if not data:
            continue
        table = pa.Table.from_pylist(data)
        out_path = config.PARQUET_DIR / f"{table_name}_{shard_idx:04d}.parquet"
        pq.write_table(table, out_path)
        print(f"  wrote {out_path} ({len(data)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays_dir", type=Path)
    ap.add_argument("--sanity", action="store_true",
                     help="skip map/rated/1v1/duration filters (sample-replay smoke test)")
    args = ap.parse_args()

    if args.sanity:
        config.ALLOWED_MAPS = set()
        config.REQUIRE_RATED = False
        config.REQUIRE_1V1 = False
        config.REQUIRE_COMPLETED = False
        config.MIN_DURATION_SECONDS = 0
        config.MAX_DURATION_SECONDS = 10**9
        config.ALLOWED_PROFILE_IDS = set()
        config.MIN_RATE_SNAPSHOT = None

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    rejected_log = open(config.QUARANTINE_DIR / "rejected.jsonl", "a", encoding="utf-8")
    failed_log = open(config.QUARANTINE_DIR / "failed.jsonl", "a", encoding="utf-8")
    manifest = open(config.MANIFEST_PATH, "a", encoding="utf-8")

    rec_files = sorted(args.replays_dir.glob("*.aoe2record"))
    if not rec_files:
        print(f"No .aoe2record files found in {args.replays_dir}")
        return

    rows = {"matches": [], "players": [], "timeseries": [], "uptimes": [], "actions": [], "movements": []}
    shard_idx = 0
    n_ok = n_rejected = n_failed = 0

    for path in rec_files:
        try:
            with open(path, "rb") as f:
                match = parse_match(f)
        except Exception as exc:
            n_failed += 1
            info = peek_version(path)
            failed_log.write(json.dumps({
                "file": str(path), "error": str(exc),
                "traceback": traceback.format_exc(limit=3), **info,
            }) + "\n")
            print(f"[FAILED]   {path.name}: {exc}")
            continue

        reasons = check_filters(match)
        mid = match_id_for(match, path)
        if reasons:
            n_rejected += 1
            rejected_log.write(json.dumps({"file": str(path), "match_id": mid, "reasons": reasons}) + "\n")
            print(f"[REJECTED] {path.name}: {'; '.join(reasons)}")
            continue

        extract_one(match, mid, str(path), rows)
        manifest.write(json.dumps({"match_id": mid, "file": str(path)}) + "\n")
        n_ok += 1
        print(f"[OK]       {path.name} -> match_id={mid}")

        if len(rows["matches"]) >= config.MATCHES_PER_SHARD:
            flush_shard(rows, shard_idx)
            shard_idx += 1
            rows = {k: [] for k in rows}

    if rows["matches"]:
        flush_shard(rows, shard_idx)

    for fh in (rejected_log, failed_log, manifest):
        fh.close()

    print(f"\nDone. ok={n_ok} rejected={n_rejected} failed={n_failed}")
    print(f"Parquet -> {config.PARQUET_DIR}")
    print(f"Quarantine logs -> {config.QUARANTINE_DIR}")


if __name__ == "__main__":
    sys.exit(main())
