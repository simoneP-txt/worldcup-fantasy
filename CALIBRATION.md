# Calibration procedure

Re-tune the player-points priors against the points players have actually
scored in the World Cup so far. Safe to run any time; it changes nothing
until you pass `--write`, and even then only after enough data has
accumulated. **Intended first real run: end of round 2 (2026-06-24).** A
dry-run peek after round 1 (2026-06-18) is fine.

## TL;DR

```powershell
python -m wcfantasy fetch          # make the local feeds current
python -m wcfantasy calibrate      # dry run: prints the proposal, writes nothing
python -m wcfantasy calibrate --write   # persist data/state/calibration.json
```

Once `data/state/calibration.json` exists, `advise` and `live` load it
automatically (their header prints `calibration: ON`). Delete the file to
revert to the built-in defaults.

## How it works

- The model's quality prior is a blend of a **price prior** (linear in the
  fantasy price + ownership) and a **club-stats projection** (FBref, only
  for matched players). Observed WC form gradually takes over per player.
- `calibrate` reads the latest local feeds (no network), takes each
  played player's **observed average points per match** as the target, and
  refits a small, stable set of weights in
  [player_points.WEIGHTS](src/wcfantasy/models/player_points.py):
  - `price_slope`, `price_intercept` — least-squares of observed points on
    price;
  - `club_prior_weight` — the club/price blend weight that minimises mean
    absolute error vs observed (default 0.50).
- It deliberately leaves the rest (ownership weight, opponent spread,
  clean-sheet rate, form horizon) at defaults: two rounds is little data.
- It refuses to `--write` below `MIN_SAMPLES` observations
  ([calibrate.py](src/wcfantasy/calibrate.py)).

## What to check before writing

The dry run prints, for each prior, `{n, mae, bias, corr}` vs observed:

- **mae** (mean absolute error) should drop from the default to the refit.
- **bias** near 0 means the prior is well-centred; a large positive bias
  means the model overrates players on average (and vice versa).
- **corr** is the ranking quality; it matters more than mae for the
  optimizer, which cares about *ordering* players.

Only `--write` if the refit clearly beats the default and the sample is
reasonable. If in doubt, keep the defaults — they are sane.

## For an AI assistant doing this on 2026-06-24

1. Read [CLAUDE.md](CLAUDE.md) (architecture + rules) and this file.
2. Run `python -m wcfantasy fetch` then `python -m wcfantasy calibrate`.
3. Read the metrics. Sanity-check the proposed weights against
   `player_points._WEIGHT_DEFAULTS` (no wild jumps; slope stays positive).
4. If the refit improves mae/corr without an absurd bias, run
   `python -m wcfantasy calibrate --write`; otherwise report why not.
5. Run `pytest` to confirm the suite still passes, then run
   `python -m wcfantasy advise` and eyeball the squad for sanity.
