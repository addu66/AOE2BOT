"""Dump parsed replays to human-readable JSON, for inspection only.

Usage:
    .venv/Scripts/python.exe -m pipeline.to_json <replays_dir_or_file> [-o OUTDIR] [--force]

This is a DEBUGGING / INSPECTION tool, not part of the training path.
It writes mgz's full serialized model -- every field, including the bulky
`map.tiles` (~14k rows) and `gaia` (~15k rows) blocks -- so files land in
the 3-15 MB range per match. Training reads `data/parquet/` instead (see
pipeline/extract.py and docs/PIPELINE.md), which drops those blocks and is
~100x smaller.

Use this when you want to eyeball what mgz actually produces for a given
replay: what an input payload looks like, whether a field is populated,
what a new save_version changed. Not for bulk processing.
"""
import argparse
import json
import sys
from pathlib import Path

from mgz.model import parse_match, serialize


def dump(rec_path: Path, out_path: Path, force: bool = False) -> bool:
    if out_path.exists() and not force:
        print(f"[SKIP]   {out_path.name} already exists (use --force to overwrite)")
        return False
    try:
        with open(rec_path, "rb") as f:
            match = parse_match(f)
    except Exception as exc:
        print(f"[FAILED] {rec_path.name}: {exc}")
        return False

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialize(match), f, indent=2)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK]     {rec_path.name} -> {out_path.name} ({size_mb:.1f} MB, "
          f"{len(match.inputs)} inputs, save_version {match.save_version})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path,
                    help="a .aoe2record file, or a directory of them")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="output directory (default: alongside each input file)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing .json files")
    args = ap.parse_args()

    if args.target.is_dir():
        rec_files = sorted(args.target.glob("*.aoe2record"))
    elif args.target.is_file():
        rec_files = [args.target]
    else:
        print(f"No such file or directory: {args.target}")
        return 1

    if not rec_files:
        print(f"No .aoe2record files found in {args.target}")
        return 1

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for rec in rec_files:
        out = (args.outdir / rec.name if args.outdir else rec).with_suffix(".json")
        if dump(rec, out, force=args.force):
            n_written += 1

    print(f"\nWrote {n_written}/{len(rec_files)} JSON file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
