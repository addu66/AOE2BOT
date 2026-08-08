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

### 2.2 `save_version 68.0` header parsing — **fixed, all 3 replays parse end-to-end**

- Files: [`AOE2BOT/aoc-mgz/mgz/fast/header.py`](../AOE2BOT/aoc-mgz/mgz/fast/header.py) (`parse_de()`), [`AOE2BOT/aoc-mgz/mgz/model/__init__.py`](../AOE2BOT/aoc-mgz/mgz/model/__init__.py) (color/civ lookup, team assignment).
- **Symptom:** the 3 new top-player replays (`Hera_vs_MBL_1/2`,
  `Hera_vs_Margougou` — all `save_version 68.0`, `game_version VER 9.4`,
  `build_version 180059`, added 2026-08-08) failed to parse *at all*:
  `RuntimeError("could not parse: ")` from deep inside `parse_de()`.
  Fully resolved via four independent fixes, found across two rounds of
  investigation (byte-level RE, then cross-referencing upstream open PRs).

**Fix 1 — custom civ pool encoding changed (`parse_de`, own byte-level find):**
Previously, `custom_civ_count > 0` was followed by that many 4-byte civ ids.
As of `save_version` in `[63.0, 68.0)` this is unchanged; by `68.0` it's
followed by exactly **one** 4-byte value regardless of the count (possibly a
hash/bitmask of the pool — not confirmed). Reading the old N-entry array
consumes real header bytes belonging to the *next* field, corrupting every
subsequent read for that player slot. Verified by byte-level hex inspection:
located the real player-name string ("VIT | Hera") in the decompressed
header and walked backward to confirm the exact framing. Gated at
`save >= 68.0` — no data points between `63.0` and `68.0` exist to pin the
exact cutoff, so this may need adjusting.

**Fix 2 — extra 4 reserved bytes before `rated` (`parse_de`, own byte-level find):**
Value `0` across all 3 samples (not game-specific data). Verified the same
way: `rated`/`allow_specs`/`spec_delay` only come out as sane values
(`1`/`1`/`120`) with this skip in place. Same `save >= 68.0` caveat as fix 1.

**Fix 3 — 2 extra settings strings near the achievements section (`parse_de`):**
This is the fix that unblocked everything past `parse_metadata`
(`num_players` was reading `0` instead of `2`, cascading into
`parse_players()` raising `IndexError`). Found by checking upstream's open
PRs rather than continuing the manual byte hunt: **[happyleavesaoc/aoc-mgz#139](https://github.com/happyleavesaoc/aoc-mgz/pull/139)**
("Add support for The Last Chieftains in the fast parser", targeting
`save_version 67.2`) and its sibling **[#142](https://github.com/happyleavesaoc/aoc-mgz/pull/142)**
(same idea, `save_version 67.0`) both add two extra `de_string()` reads
right before the achievements' `timestamp` read. Applying that exact hunk
(gated `save >= 67.2`, which covers our `68.0` replays) got `parse_de()`
producing a correct `num_players` for the first time.
**Caveat on the other half of PR #139/#142:** both PRs *also* add an
unconditional trailing `de_string()` read at the end of **every**
per-player loop iteration (for a field unrelated to the civ pool). Tested
that against our data and it does **not** match — it corrupts player slot 0
(`MbL`), which byte-level inspection had already confirmed needs *zero*
extra bytes. Fix 1 above (the civ-pool-conditional 4-byte read) remains the
correct mechanism for `68.0`; whatever field PR #139/#142 added for `67.0`/`67.2`
appears to have been superseded again by `68.0`. Only the achievements-section
hunk (fix 3) was adopted from those PRs — not the per-player-loop hunk.

**Fix 4 — stale color reference data, not a parsing bug at all (`mgz/model/__init__.py`):**
Getting past `parse_de()` surfaced a *different* class of issue: `KeyError: '15'`
from `consts['player_colors'][str(player['color_id'])]`. Traced this by
printing the raw decoded value directly — `color_id=15` for one player,
matching the exact byte read at the verified-correct offset. This isn't a
parsing bug: `player['color_id']` is correctly decoded, but the `aocref`
reference-data package (already at its latest release, `2.0.38`) only maps
color ids `0`–`7`; DE has since added more selectable colors. Fixed with the
same defensive-lookup pattern PR #139 already uses for unknown civilization
ids: `.get(..., f"<Unknown color: {id}>")` instead of a bare `[...]` lookup,
applied to both the color and (matching PR #139) the civilization lookup.

**Fix 5 — `team_id == 0` not handled in team assignment (`mgz/model/__init__.py`):**
With fixes 1–4 in place, all 3 replays parsed, but every player showed
`winner=False` — including the one who clearly won (last input in each
replay is a `Resign` from the *other* player). Traced to team assignment:
the code only special-cased `team_id == 1` as "no team" (auto-solo-team);
one player in each of the 3 replays has `team_id == 0` instead, which fell
through both branches and silently never made it into `teams` at all — not
even as a team of one. Since `Player.winner` is only ever set by iterating
`teams`, that player's `winner` stayed at its default (`False`) regardless
of the actual outcome. Fixed by treating `team_id <= 1` the same
(solo team), with an added guard to skip empty/placeholder player slots
(`number` not in the already-built `players` dict) so they don't get pulled
into a team as a side effect of the same relaxed condition.

**Verification:** all 3 new replays now parse fully via `parse_match()` —
correct map (`Arabia`), civs, colors (with the new fallback), `winner`,
`team`, `duration`, `inputs` (3446–7985 per replay), `uptimes` (4–6 per
replay), and per-player `timeseries` (185–325 rows) — with **zero regression**
on `Replay1-3` (re-verified after every fix). `pipeline/extract.py` run
against all 6 replays: `ok=4 rejected=2(non-Arabia) failed=0`.

**Upstream status:** fixes 1, 2, 4, 5 are ours (not in any known open PR);
fix 3 is adapted from PR #139/#142 (both still open/unmerged upstream as of
2026-08-08). Given the full picture now assembled, this is worth writing up
as a genuinely complete PR rather than the WIP-issue framing from the first
investigation round — see the updated Phase 1 TODO below.

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
| `actions` | variable | `match_id`, `player_number`, `t_s`, `type`, `param`, `entity_id`, `amount` — macro/decision-level, see §4.1 |
| `movements` | variable, largest table (~3.5k rows/match) | `match_id`, `player_number`, `t_s`, `type`, `param`, `object_ids` (list), `target_id`, `target_x`, `target_y`, `reliable` — unit-control, see §4.2 |

**Deliberately not extracted (logged decision, not an oversight):**
`map.tiles` (14,400 rows/match) and `gaia` (~15,500 rows/match) are dropped.
They're static per map/seed. Revisit if a later phase needs terrain-aware
features (e.g. wall placement, hill fighting) — now more likely relevant
than originally scoped, since §4.2 below means this pipeline is no longer
build-order-only.

**Object-id corruption — narrower than first thought, and doesn't block
`movements`.** An earlier version of this doc claimed "two-thirds of
`object_ids` ... are DE-packed and unresolved" and scoped `movements`-type
actions out entirely on that basis. That claim was wrong — it generalized
from a hasty check on one older replay without verifying which action
types were actually affected. Checked properly against `Hera_vs_MBL_1`
(`save_version 68.0`): **0 of ~4500** `object_id` references across
`Move`/`Order`/`Gather`/`Patrol`/`Gather Point` were anomalous. The real
corruption (~4.7% of *all* object_id references in that replay) is
narrowly confined to `Garrison`/`Ungarrison`/`Unqueue` — all excluded from
`MOVEMENT_ACTION_TYPES` already. `movements.reliable` still flags any row
whose `object_ids` exceed `config.OBJECT_ID_SUSPECT_THRESHOLD` as a guard
against a future replay being less clean, rather than assuming this holds
everywhere.

---

## 4. Action vocabulary v1

**Why two tables, not one.** A model trained only on §4.1 (macro) learns
*what to prioritize economically* — when to build a Barracks, when to age
up — but has no way to actually play a game: it never learns to move a
villager to a resource, send an army at a target, or retreat. That's not
a viable "bot," bootstrap or otherwise — it's a build-order advisor. §4.2
(movement) is what makes unit control learnable at all: who (`object_ids`),
what kind of command, and where (`target_x`/`target_y` or `target_id`).
They're kept as separate tables because the two are used differently
downstream (macro → what-to-produce-next classifier; movement → spatial/
targeting policy), not because either alone is "the" vocabulary.

### 4.1 Macro / decision-level (`actions` table)

`pipeline/config.py: MACRO_ACTION_TYPES`:

- `Queue` — train unit (requires the fix in §2.1)
- `Build` — place building
- `Research` — start a tech
- `Wall` — placing wall segments (a deliberate defensive decision)
- `Buy` / `Sell` — market

Plus a synthetic `"Age Up"` type injected from `match.uptimes` (a reliable,
one-row-per-transition source) rather than sniffed out of `Research` entries;
`Research` rows whose `param` is an age name are dropped to avoid
double-counting.

A model trained on *only* this table (or on the full raw click stream
without separating the two) risks a degenerate majority-class collapse:
`Move` alone is 48.8% of all raw inputs in the sanity replays, so
predicting `Move` unconditionally scores ~49% "accuracy" — deceptively
good on a naive metric while learning nothing. Confirmed by direct
measurement on `Replay1`. This is exactly why movement is its own table
with its own target (predicting *where*/*who*, not just *whether* it's a
`Move`) rather than folded into one flat vocabulary.

### 4.2 Movement / unit control (`movements` table)

`pipeline/config.py: MOVEMENT_ACTION_TYPES`:

- `Move` — move to a position, no combat stance
- `Attack Move` — move to a position, engage anything encountered
- `Order` — attack/interact with a specific target (unit or building)
- `Gather` — start gathering a specific resource
- `Gather Point` — set a building's rally point
- `Patrol` — patrol between current position and a target
- `Attack Ground` — target ground (e.g. siege) rather than a unit

Each row carries `object_ids` (who's being commanded), `target_id` (if the
command targets a specific object rather than a bare position), and
`target_x`/`target_y`. **Not yet solved: unit-type attribution.** We know
*which instance* is being moved (`object_ids`) and *where*, but not
reliably *what kind of unit* it is, for anything created after the game's
opening 20 starting units — `Queue`'s `object_ids` field identifies the
*producing building*, not the *id assigned to the newly trained unit*, so
there's no direct instance→type lookup in the parsed data today. Getting
that (e.g. reconciling `Queue` completion times against the order new
`object_ids` first appear, assuming DE assigns them roughly in creation
order — untested assumption) is the next real piece of work if villager
vs. military-unit distinction matters for the movement policy, and it's a
bigger undertaking than anything fixed so far — tracked in Phase 4 below,
not started.

`Attack Move` and `Attack Ground` showed 0 rows in the replays extracted so
far (small sample) — vocabulary correctness for those is unverified, not
disproven.

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
data/parquet/*.parquet  (matches / players / timeseries / uptimes / actions / movements, sharded)
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

**Inspection side-path (not part of training):**
`pipeline/to_json.py` dumps a replay's full parsed model to readable JSON
for eyeballing — every field mgz produces, including the bulky `map.tiles`
and `gaia` blocks. 6-10 MB per match vs. ~40 KB of parquet, so it's for
debugging one replay at a time, never for bulk processing.

---

## 6. Scripts: what's live, what was removed

**Live (the pipeline):**

| File | Role |
|---|---|
| `pipeline/config.py` | filters, action vocabularies, paths — single source of truth |
| `pipeline/extract.py` | rec → filtered, sharded parquet. **This is the parser used for training data.** |
| `pipeline/to_json.py` | rec → full readable JSON, inspection only |

**Kept but currently broken — scheduled for rewrite, not deletion:**
`supervised_training/{preprocess,train,model,data_loader,evaluate}.py`.
None of these run today: they read the *old* per-replay JSON layout, and
carry the specific bugs itemized in Phase 4 (O(n²) prefix rebuild, padding
colliding with a real token id, unpacked LSTM sequences, split-by-sample
leakage; `evaluate.py` additionally does `from train import model, ...`,
which re-runs training as an import side effect). They're kept because
Phase 4 rewrites them against `data/parquet/` and the model architecture is
a reasonable starting point — deleting them would orphan ~8 TODO items that
cite specific lines.

**Removed 2026-08-09** (all recoverable from git history — they were
committed in `0bf88b4`; every claim below was verified against the parsed
data before deleting, not assumed):

| File | Why removed |
|---|---|
| `my_parser.py` | Generated the old JSON dumps. Superseded by `pipeline/to_json.py`, which does the same thing with CLI args, error handling, and no import-time side effects (this one ran a parse on import). |
| `Aoe2RecParser.py` | `process_frames()` iterates `match['frames']` — **that key does not exist** in mgz output (verified). Never worked. `get_army_composition()` counted `player['objects']`, which holds only the ~20 *starting* units, so it reported the same trivial result for every match regardless of what was built. |
| `parse_frame_actions.py` | Same starting-objects mistake: "explored_tiles" derived from the 20 initial objects reports ~10 tiles as map exploration. Also ran `KMeans(n_clusters=3)` over 2 players' worth of vectors. |
| `Bot1.py` | Could not run: depended on `tslearn`/`stable_baselines3`/`gym`/`matplotlib` (none installed), used the wrong import path for `DummyVecEnv`, and called `run_rl_in_aoe2()` at module scope (infinite loop on import). Every replay field it read — `resources`, `army_composition`, a flat numeric `actions` array — is absent from mgz output (verified). Its RL *design intent* is preserved in Phase 7 below. |

---

## 7. TODO checklist

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
      gap: `save_version 68.0` (§2.2). **Fully resolved** across 5 fixes
      (2 own byte-level finds in `parse_de`, 1 adapted from upstream PRs
      #139/#142, 2 more in `mgz/model/__init__.py` found after `parse_de`
      itself was fixed: stale `aocref` color data, and a `team_id == 0`
      gap that silently broke `winner` detection). All 3 new replays now
      parse end-to-end with correct map/civs/colors/winner/team/actions.
- [x] Re-ran the Action.XXX audit (§2.1) implicitly via full extraction of
      the 3 new replays (2347 macro actions extracted, civs Armenians/
      Britons/Vikings/Romans) — no new parse failures surfaced. Formal
      per-type attempt/failure counts (like the original audit) not yet
      re-run against these specifically; low priority given zero observed
      failures.

**Upstream contribution (`AOE2BOT/aoc-mgz-fork/`, gitignored from main repo):**
- [x] `fix/de-queue-object-ids` branch — one clean, verified commit, ready
      to push to a fork and open as a PR
- [ ] `investigate/save-version-68` branch is now **stale** — it reflects
      the mid-investigation, 2-of-5-fixes state and frames this as a WIP
      issue rather than a complete fix. Needs redoing as a proper PR branch
      with all 5 fixes, crediting PR #139/#142 for the achievements-section
      hunk (fix 3) and explicitly noting the per-player-loop hunk from
      those PRs does NOT apply to `68.0` (see §2.2). Delete
      `ISSUE_DRAFT_save_version_68.md`, write a PR description instead.
- [ ] Push `fix/de-queue-object-ids` and the redone save-68 branch to
      `github.com/addu66/aoc-mgz` (remote already added by the user) and
      open both as PRs against `happyleavesaoc/aoc-mgz`
- [ ] Consider commenting on PR #139/#142 directly (rather than only opening
      a new PR) since they're addressing the same underlying problem and a
      maintainer may prefer consolidating

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
- [x] Added `movements` table (§4.2) — unit-control actions (`Move`/
      `Attack Move`/`Order`/`Gather`/`Gather Point`/`Patrol`/`Attack Ground`)
      with `object_ids`, target position/object, and a `reliable` flag.
      Corrects an earlier wrong claim that these types' `object_ids` were
      broadly DE-packed/unresolved — verified clean (0 anomalies in ~4500
      refs) against `Hera_vs_MBL_1`.

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
- [ ] **Unit-type attribution for `movements` rows** (§4.2, not started) —
      reconcile `Queue` completion times against the order new `object_ids`
      first appear in the action stream, to tag each movement with a unit
      type (villager vs. specific military unit). Needed if the movement
      policy should condition on/predict unit type, not just bare
      instance ids. Untested assumption to validate first: whether DE
      assigns `object_ids` in roughly creation order.
- [ ] Decide how macro (`actions`) and movement (`movements`) tables
      combine into one training example — interleaved single sequence
      (harder, one vocabulary) vs. two coupled heads/models sharing a
      state encoder (more like AlphaStar's split) — not decided yet

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
- [ ] Decide scope: macro/build-order policy only vs. full unit control.
      Note this is now *less* constrained than originally written — the
      `movements` table (§4.2) means unit-control data is available, so
      full control is a data-supported option, not just an aspiration.
      The environment, not the data, is the remaining blocker.
- [ ] Evaluate an abstracted macro simulator (fit from replay data) vs. a
      live-game hookup (memory read + input injection — real-time-only,
      ToS-adjacent, doesn't scale to RL sample counts) vs. openage
      (incomplete)
- [ ] Define `step()` / `reset()` / reward once the above is decided

**Salvaged design intent from the deleted `Bot1.py`** (see §6): that script
was an early, non-functional sketch of exactly this phase. It could never
run (depended on `tslearn`/`stable_baselines3`/`gym`, none installed; wrong
import path for `DummyVecEnv`; called `run_rl_in_aoe2()` at module scope,
so importing it would hang; and every field it read from the replay JSON
— `resources`, `army_composition`, a flat numeric `actions` array — does
not exist in mgz's output). Its assumptions are recorded here so the ideas
aren't lost with the file:
- **Algorithm:** PPO via `stable_baselines3`, `MlpPolicy`, 100k timesteps.
- **Observation space:** `Box(11,)` = 4 resources + exploration% + 3 own
  army counts + 3 enemy army counts. Note the enemy-army terms assume full
  observability, which a real agent would not have (no fog of war) —
  reconsider before reusing.
- **Action space:** `Discrete(6)` = train infantry/cavalry/archers, gather
  food/wood/gold. Far coarser than the §4.1 macro vocabulary already
  extracted, and with no unit-control actions at all.
- **Reward:** `own_army_strength - enemy_army_strength + resources/100`.
  Naive (a pure army-count proxy ignores composition counters, and the
  resource term rewards hoarding, which is anti-correlated with good play)
  — treat as a starting point to argue with, not a baseline.
- **IPC hack:** read game state from `aoe2_state.txt`/`aoe2_army.txt`,
  write actions to `rl_commands.txt`. Nothing ever produced those files;
  this was the placeholder for the unsolved "how do we actually talk to
  the game" problem, which remains the core blocker above.
