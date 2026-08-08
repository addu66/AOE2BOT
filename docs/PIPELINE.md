# AoE2 DE Bot — Data & Training Pipeline

**Status:** living document — updated as phases complete. Last updated 2026-08-08.

**Scope decision:** the training corpus is replays from a curated set of **top
players**, **Arabia only**, **1v1**. `AOE2BOT/replays/Replay1-3` are **not**
part of that corpus — they are fixed sanity-check fixtures used only to
validate that parsing and preprocessing work mechanically (one is a custom
tournament map, none are guaranteed rated). Never point the real training run
at `replays/`; real corpus files land in `AOE2BOT/data/raw/` (see Phase 3).

Goal: supervised behavior-cloning bootstrap on human replays → later RL
fine-tuning. The RL environment is an open problem, tracked at the bottom —
nothing here blocks on it.

---

## 1. Environment

Working venv: **`AOE2BOT/.venv`** — native Windows Python 3.11 (Microsoft
Store install). **Do not** use the MSYS2 `mingw64` Python that's on `PATH`
(`D:\MSYS2\mingw64\bin\python.exe`) — its wheel platform tag is
`mingw_x86_64`, not `win_amd64`, so `pip install pandas/numpy/torch` falls
back to building from source and fails without a full MSVC toolchain. Always
invoke pipeline scripts as:

```
AOE2BOT/.venv/Scripts/python.exe -m pipeline.extract ...
```

Installed in `.venv`: `pandas`, `pyarrow`, `numpy`, `scikit-learn`,
`torch==2.5.1+cu121` (CUDA confirmed working against the machine's GTX 1650),
and the vendored `aoc-mgz` (editable install, **with a local patch** — see §2).

Note: a second venv exists at `C:\Users\adnan\venv`, created from WSL
(`/usr/bin/python3.12`, Linux-layout `bin/python`). It cannot execute from
Windows-side tooling (permission denied — wrong-OS binary) and is not used by
this pipeline. Only relevant from inside WSL; ignore or delete it to avoid
confusion about which env is "the" env.

**Git:** the project root (`d:/AgeProgramming`) is now a git repo (`main`
branch). `AOE2BOT/aoc-mgz` was previously *also* its own nested clone of
upstream `aoc-mgz` (predating this pipeline work) — that nested `.git` has
been removed so it's tracked as plain vendored files in the main repo
instead of silently becoming an empty "gitlink" with our patch stuck
uncommitted inside it. `AOE2BOT/aoc-mgz-fork/` is a *separate*, intentional
git clone of upstream used to prepare the PR/issue described in §2 below —
it keeps its own git history and is gitignored from the main repo on
purpose (see `.gitignore`).

---

## 2. Parsing bugs found & fixed

### 2.1 `DE_QUEUE` silently mis-parsed as `Action.ERROR` — **fixed**

- File: [`AOE2BOT/aoc-mgz/mgz/fast/actions.py`](../AOE2BOT/aoc-mgz/mgz/fast/actions.py), `parse_action_71094()`, `Action.DE_QUEUE` branch.
- **Symptom:** every `"Queue"` input (train villager / train unit) resolved to
  `type="Error"` instead and was invisible to any downstream code — verified
  509/509 `DE_QUEUE` actions in `Replay1` failed identically.
- **Root cause:** the struct format `'<h4xhhh4x'` has a spurious trailing
  4-byte skip that eats the exact bytes meant to hold the `object_ids` array.
  The following line, `unpack(f'<{selected}I', data)`, then reads past
  end-of-buffer and raises `struct.error`; the caller
  (`mgz/fast/__init__.py: action()`) catches that and downgrades the whole
  action to `Action.ERROR`.
- **Fix applied:** drop the trailing `4x` → `'<h4xhhh'`. Verified against
  known-good values recovered from an older mgz build's JSON dump of the same
  replay (`unit_id=83` "Villager", `object_ids=[2266]` "Town Center",
  `amount=1`) and cross-checked against 3 independent `DE_QUEUE` samples.
- **Impact:** this one line was silently deleting the single largest
  build-order-relevant action category (`Queue`, 1588 of the 2554 macro
  actions once restored in the 3-replay sanity set — the model would have
  had *zero* signal for unit/villager production). This is a **repo-owned
  patch, not upstream trivia** — it must ship with the vendored `aoc-mgz`.
  Don't `pip install mgz` from PyPI in place of the vendored copy, and don't
  refresh the vendored copy from upstream without re-applying/re-verifying
  this line.
- **PR prepared:** `AOE2BOT/aoc-mgz-fork/` branch `fix/de-queue-object-ids`,
  one clean commit, ready to push to a fork and open against
  `happyleavesaoc/aoc-mgz`. Draft PR body in that branch's
  `PR_DRAFT_de_queue_fix.md` (delete before/after posting — not library code).
- **Systematic audit, 2026-08-08:** instrumented `parse_action()` to record
  attempt/failure counts per `Action` type across all 3 working sample
  replays (Replay1-3, ~4500+ actions each). **Zero `struct.error` failures**
  across the 25 action types actually exercised in that data (including
  `DE_QUEUE`, now 1588/1588 successful, up from 0/1588 pre-fix). No other
  systematic padding bugs found. Coverage caveat: the `Action` enum defines
  68 members total; 43 were never exercised by these 3 replays and remain
  unaudited (rare types like `TRIBUTE`, `CREATE`, `GUARD`,
  `DE_MULTI_GATHERPOINT`, `TOWN_BELL`, ...) — re-run this audit once Phase 3
  brings in a larger, more varied corpus.

### 2.2 `save_version 68.0` header parsing — **partially fixed, blocked**

- File: [`AOE2BOT/aoc-mgz/mgz/fast/header.py`](../AOE2BOT/aoc-mgz/mgz/fast/header.py), `parse_de()`.
- **Symptom:** the 3 new top-player replays (`Hera_vs_MBL_1/2`,
  `Hera_vs_Margougou` — all `save_version 68.0`, `game_version VER 9.4`,
  `build_version 180059`, added 2026-08-08) failed to parse *at all*:
  `RuntimeError("could not parse: ")` (empty inner exception message) from
  deep inside `parse_de()`.
- **Root cause 1 (fixed):** the "custom civ pool" field's encoding changed.
  Previously, `custom_civ_count > 0` was followed by that many 4-byte civ
  ids. As of 68.0, it's followed by exactly **one** 4-byte value regardless
  of the count (possibly a hash/bitmask of the pool — not confirmed).
  Reading the old N-entry array consumes real header bytes belonging to the
  *next* field, corrupting every subsequent read for that player slot.
  Verified by byte-level hex inspection: located the real player-name string
  ("VIT | Hera") in the decompressed header and walked backward to confirm
  the exact framing.
- **Root cause 2 (fixed):** an extra 4 reserved bytes (value `0` across all
  3 samples — not game-specific data) appear right before the `rated`
  field. Verified the same way: `rated`/`allow_specs`/`spec_delay` come out
  as sane values (`1`/`1`/`120`) only with this skip in place, and the
  following `string_block()` call's leading crc + marker land exactly where
  expected afterward.
- **Both fixes gated at `save >= 68.0`.** Caveat: there are zero data points
  between `save_version 63.0` and `68.0`, so this cutoff is a guess pinned
  to the only version observed failing — it may need adjusting once a
  replay from that range surfaces.
- **Still blocked:** with both fixes applied, `parse_de()` now runs to
  completion without raising for all 3 replays (confirmed: all 8 per-player
  loop slots decode with sane values — names, profile ids, handicaps — and
  the function's first `string_block()` call locates its marker exactly
  where expected). But the cursor position *after* `parse_de()` returns is
  still wrong: the next function, `parse_metadata()`, reads `num_players=0`
  instead of 2, which cascades into `parse_players()` raising `IndexError`.
  This means at least one more undiscovered drift exists later in
  `parse_de()` — most likely inside the `for _ in range(20): strings +=
  string_block(...)` loop or the achievements section that follows.
  `string_block()`'s own loop is self-resyncing (terminates on a small-crc
  heuristic rather than a known length), which makes it harder to localize
  the same way as the two fixes above — not yet traced.
- **Issue prepared (not a PR — investigation incomplete):**
  `AOE2BOT/aoc-mgz-fork/` branch `investigate/save-version-68`, one commit
  with both fixes + a `parse_de()` docstring documenting current status.
  Draft issue body in that branch's `ISSUE_DRAFT_save_version_68.md`.
- **Practical impact on this project right now:** none of the 3 new
  top-player replays can be extracted yet. They correctly quarantine as
  `failed` (not silently dropped) in `data/quarantine/failed.jsonl`, now
  logged with the real `save_version: 68.0` (see Phase 2 note below) so
  they're easy to find and re-run once this is resolved.

### 2.3 Old `Replay1.json` / `Replay2.json` dumps are stale

Predate the vendored `aoc-mgz`'s `timeseries`/`uptimes` support entirely —
`players[].objects` in those JSONs holds only the ~20 starting units
(identical for both players, never updated), so
`supervised_training/preprocess.py`'s `state_vec` is a dead constant input in
the *existing* training code. Not used going forward; `data/parquet/` is now
the source of truth. Old `.json` files kept for now only as a before/after
diff reference — safe to delete once the new pipeline is trusted.

---

## 3. Compact schema (`AOE2BOT/data/parquet/`)

Produced by [`pipeline/extract.py`](../AOE2BOT/pipeline/extract.py) directly
from `.aoe2record` files — no intermediate multi-MB JSON. One shard set per
`MATCHES_PER_SHARD` (default 200) matches: `matches_0000.parquet`,
`players_0000.parquet`, etc.

| Table | Grain | Columns |
|---|---|---|
| `matches` | 1 row / match | `match_id` (guid), map/dataset/version fields, `duration_s`, `rated`, `diplomacy_type`, `winner_number`, `source_file` |
| `players` | 1 row / player (2 for 1v1) | `match_id`, `number`, `profile_id`, `name`, `civilization`, `team_id`, `winner`, `eapm`, `rate_snapshot` |
| `timeseries` | ~365 rows / player / match | `match_id`, `player_number`, `t_s`, `total_resources`, `total_objects` — aggregate economy snapshot, ~6.5s sampling |
| `uptimes` | ≤3 rows / player / match | `match_id`, `player_number`, `age`, `t_s` — exact age-up timestamps |
| `actions` | variable | `match_id`, `player_number`, `t_s`, `type`, `param`, `entity_id`, `amount` — macro/decision-level only, see §4 |

**Deliberately not extracted (logged decision, not an oversight):**
`map.tiles` (14,400 rows/match) and `gaia` (~15,500 rows/match) are dropped.
They're static per map/seed and irrelevant to a macro/build-order policy;
including them was the dominant cost in the old JSON (14MB → ~40KB parquot
per match without them). Revisit only if a later phase needs terrain-aware
features (e.g. wall placement, hill fighting).

**Object-id corruption, out of scope for v1:** two-thirds of `object_ids` in
raw `Move`/`Order`/etc. inputs are DE's packed selection encoding, unresolved
by this mgz version. This doesn't affect the macro vocabulary (§4), which
never needs `object_ids` for micro-execution — but it means a *full-control*
bot (as opposed to a macro/build-order policy) is blocked on either fixing
that decoding or picking a different action interface entirely. Tracked in
Phase 7.

---

## 4. Action vocabulary v1 (macro / decision-level)

`pipeline/config.py: MACRO_ACTION_TYPES` — the only `inputs[].type` values
extracted into the `actions` table:

- `Queue` — train unit (requires the fix in §2.1)
- `Build` — place building
- `Research` — start a tech
- `Wall` — placing wall segments (a deliberate defensive decision)
- `Buy` / `Sell` — market

Plus a synthetic `"Age Up"` type injected from `match.uptimes` (a reliable,
one-row-per-transition source) rather than sniffed out of `Research` entries;
`Research` rows whose `param` is an age name are dropped to avoid
double-counting.

**Deliberately excluded:** `Move`, `Order`, `Gather`, `Gather Point`,
`Attack Move`, `Target`, `Garrison`, `Ungarrison`, `Follow`, `Patrol`,
`Formation`, `Stance`, `Delete`, `Stop`, `Repair`, `Back To Work`, `Unqueue`,
`Reseed`, `Pack Trebuchet`, `Attack Ground`, `Chat`, `Resign`. These are
micro-execution, not build-order decisions, and including them is actively
harmful for a first model: `Move` alone is 48.8% of all raw inputs in the
sanity replays, so a model trained on the full raw stream can hit ~49%
"accuracy" by always predicting `Move` — a completely degenerate model that
looks deceptively good on a naive metric. Confirmed by direct measurement on
`Replay1`, see conversation history.

This vocabulary is v1 and expected to be refined once the real corpus is
extracted (e.g. `Unqueue` may be worth adding back as a cancel-decision
signal).

---

## 5. Pipeline architecture

```
acquire (.aoe2record files, top players, Arabia)
        │
        ▼
pipeline/extract.py  ── parse (vendored, patched mgz)
        │              ── filter (pipeline/config.py: map/rated/1v1/duration/…)
        │              ── on filter-reject  → data/quarantine/rejected.jsonl (reason logged, not silently dropped)
        │              ── on parse-exception → data/quarantine/failed.jsonl (best-effort save_version logged)
        ▼
data/parquet/*.parquet  (matches / players / timeseries / uptimes / actions, sharded)
        │
        ▼
dataset assembly (windowed sequences, split BY MATCH, vocab from train split only)
        │
        ▼
SL training (supervised_training/, needs the fixes in Phase 4)
        │
        ▼
evaluation (held-out top-player matches, top-k accuracy vs. majority baseline)
        │
        ▼
(future) RL fine-tuning — environment TBD, see Phase 7
```

---

## 6. TODO checklist

### Phase 0 — Environment
- [x] Native Windows venv with working wheel installs (`AOE2BOT/.venv`)
- [x] pandas / pyarrow / numpy / scikit-learn / torch (+CUDA) installed
- [x] Vendored `aoc-mgz` installed editable
- [x] Root git repo initialized (`main` branch), `.gitignore` covering venv /
      replay binaries / pipeline output / the separate `aoc-mgz-fork/` clone

### Phase 1 — Parsing fix
- [x] Diagnose and fix `DE_QUEUE` → `Action.ERROR` bug (§2.1) — PR-ready
      branch in `AOE2BOT/aoc-mgz-fork/`
- [x] Audit remaining `Action.XXX` branches in `mgz/fast/actions.py` for the
      same trailing-padding mistake — zero further systematic bugs found
      among the 25 types exercised in the 3 sample replays; 43 types
      unaudited for lack of data (§2.1)
- [x] Re-ran against a wider replay sample (3 new top-player Arabia
      replays, 2026-08-08) — surfaced a *new*, unrelated version-support
      gap: `save_version 68.0` isn't fully handled yet (§2.2). Two of an
      estimated three-or-more fixes landed and verified; parsing still
      doesn't complete end-to-end for these 3 replays.
- [ ] **Open:** find the remaining `save_version 68.0` drift inside
      `parse_de()`'s 20x `string_block()` loop / achievements section
      (§2.2) — needed before any of the 3 new top-player replays can be
      extracted
- [ ] Once resolved, re-run the Action.XXX audit (§2.1) against these 3
      replays too — new civs/action patterns may exercise previously-untested
      action types

**Upstream contribution (`AOE2BOT/aoc-mgz-fork/`, gitignored from main repo):**
- [x] `fix/de-queue-object-ids` branch — one clean, verified commit, ready
      to push to a fork and open as a PR
- [x] `investigate/save-version-68` branch — one WIP commit with the two
      confirmed fixes, ready to post as a GitHub issue (not a PR — parsing
      still incomplete)
- [ ] Fork `happyleavesaoc/aoc-mgz` on GitHub, add as a remote, push both
      branches (needs the user's GitHub credentials — not done by this
      session)
- [ ] Open the PR (using `PR_DRAFT_de_queue_fix.md` as the body) and the
      issue (using `ISSUE_DRAFT_save_version_68.md` as the body), then
      delete those two draft files from the branches

### Phase 2 — Compaction
- [x] `pipeline/extract.py`: rec → filtered, sharded parquet tables
- [x] Quarantine logging (rejected vs. failed, distinct reasons)
- [x] Sanity-verified against the 3 sample replays (`--sanity` flag)
- [x] `peek_version()` now uses `mgz.fast.header.parse_version()` directly
      instead of the full (also-failing) header parse, so quarantined
      failures log their real `save_version` instead of `null` — this is
      what made the `save_version 68.0` pattern immediately visible across
      all 3 new replays rather than 3 opaque failures
- [ ] Load-test sharding/memory behavior at real corpus scale (hundreds–low
      thousands of matches)

### Phase 3 — Acquisition (top players / Arabia corpus)
- [ ] Decide replay source(s) (aoe2insights.com, aoe2recs.com, tournament
      packs) and confirm ToS/access mechanics before scraping anything
- [ ] Curate the top-player list → `profile_id` allowlist
      (`config.ALLOWED_PROFILE_IDS`)
- [ ] Build a bulk-download script into `data/raw/`
- [ ] Run `pipeline/extract.py` over the real corpus
- [ ] Review `data/quarantine/*.jsonl` — both rejected-by-filter and
      failed-to-parse — and decide whether rejection thresholds need tuning

### Phase 4 — Dataset assembly for training
- [ ] Rewrite `supervised_training/preprocess.py` to read from
      `data/parquet/` instead of the old per-replay JSON
- [ ] Fix O(n²) prefix rebuild (`preprocess.py:22-24`) → fixed-size window
      (e.g. last 64 actions) — confirmed blowup: **5.2M ints from one replay
      alone** under the current scheme
- [ ] Add a real `<PAD>` token + `padding_idx` in the embedding
      (`train.py:24`, `model.py:9` currently collide padding with token id 0)
- [ ] Use `pack_padded_sequence` in the LSTM forward pass (`model.py:16`) —
      currently the final hidden state is read after trailing pad steps
- [ ] Split train/val **by match**, not by flattened sample
      (`train.py:33-35` currently does an 80/20 slice of a flat list, so val
      is just the tail of the last replay)
- [ ] Build the action vocabulary from the train split only

### Phase 5 — SL training loop fixes
- [ ] More than 1 epoch (`train.py:48`) + early stopping / checkpointing
- [ ] Report top-k accuracy against the majority-class baseline explicitly
      (measured 48.8% "Move" baseline on raw click-level data — the macro
      vocabulary in §4 sidesteps this, but confirm the new baseline on the
      real corpus once extracted)
- [ ] Fix `data_loader.py`'s cwd-relative default path

### Phase 6 — Evaluation
- [ ] Held out top-player matches never seen in training
- [ ] Qualitative build-order comparison (does the model's predicted opening
      resemble known openings for the civs in the corpus?)

### Phase 7 — RL environment (major open problem, not blocking SL work)
- [ ] Decide scope: macro/build-order policy only vs. full unit control
      (recommended: start macro-only — it's achievable with SL alone and is
      a real deliverable; full control needs an actual environment)
- [ ] If full control: resolve the DE-packed `object_ids` decoding (§3) or
      pick a different action interface
- [ ] Evaluate an abstracted macro simulator (fit from replay data) vs. a
      live-game hookup (memory read + input injection — real-time-only,
      ToS-adjacent, doesn't scale to RL sample counts) vs. openage
      (incomplete)
- [ ] Define `step()` / `reset()` / reward once the above is decided
