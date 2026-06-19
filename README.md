# worldcup-fantasy

**An advisor that helps you play the game of FIFA Fantasy of the FIFA World Cup 2026.**

![FIFA Fantasy World Cup 2026](docs/banner.png)

Run one command each round and it tells you the transfers, lineup, captain and boosters
to set in the game. It downloads the FIFA fantasy feeds updated to the last played match,
estimates every player's expected points, simulates the rest of the tournament, and
prints the moves to make by hand. It never logs into your FIFA account.

## How it decides

- **Expected points:** each player's 2025/26 club stats (FBref) run through the official
  scoring table, blended with the fantasy price and his live World Cup form.
- **Who starts:** the game's own prices encode a depth chart; confirmed lineups,
  suspensions (both from the official FIFA Match Centre feed) and injury withdrawals
  (Wikipedia) discount it — all free, no key required.
- **How far the team goes:** Elo over ~49k internationals plus a Monte Carlo of the
  official bracket, calibrated with bookmaker odds. Players from likely finalists are
  worth more.
- **The squad:** an ILP (integer linear program) picks the 15 that maximize points under
  the budget, position quotas, per-country caps and legal formations.
- **Live:** after each match it advises captain swaps and substitutions, pricing in the
  auto-sub insurance you forfeit with the first manual change.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

The first run downloads and caches everything it needs into `data/`. Availability
(lineups, suspensions, injuries) needs no key. Add a free
[The Odds API](https://the-odds-api.com) key for bookmaker odds; without it the model
runs on Elo alone.

## Web app

The simplest way to run it. One command starts a local web GUI and opens the browser:

```powershell
python -m wcfantasy serve            # http://127.0.0.1:8765
python -m wcfantasy serve --port 9000 --no-browser
```

Open **Plan round** and the advisor refreshes the FIFA feeds, rebuilds the plan, and shows
the recommended squad for the next round:

![Plan round in the web app](docs/web-plan.png)

Right below it, the transfers, captain, vice and bench order come out as a checklist you
tick off by hand in the game:

![Moves to execute, shown in the web app](docs/web-moves.png)

The web app runs the same engine as the CLI, in a browser, with no Node or npm, just the
Python venv. The slow commands (plan, fetch, odds, calibrate) run as background jobs with a
live progress log, so the UI never blocks. Screens:

- **My team** — your saved squad, XI vs bench, captain/vice, bench order, bank, boosters.
- **Plan round** — runs `advise` (refreshes the feeds first), shows the recommended squad,
  transfers, captain (with the option-value timing note), bench order, booster advice,
  availability and unverified-starter watches, and your fixtures in Italian time. A
  confirm-gated **“I executed these moves”** records the plan as your squad (`team apply`),
  with the live-round guard.
- **Ideal squad** — the unconstrained best 15 vs your squad (read-only, not applyable).
- **Live** — captain/sub/outlook advice during a round, plus a form to record what you
  actually changed in-game.
- **Odds**, **Boosters** (one-per-tournament, Wildcard irreversible behind a confirm),
  **Calibration** (dry run vs write), and **Settings** (English/Italian, source status).

Like the CLI, the web app **never logs into your FIFA account**: it shows the moves to make
by hand, then you confirm you made them.

## Terminal

The same plan, in the console:

```powershell
python -m wcfantasy advise
```

![Output of the advise command](docs/advise.png)

## Commands

All commands are `python -m wcfantasy <command>`.

| Command | What it does |
| --- | --- |
| `advise` | Refresh data and print the full plan for the next round |
| `live` | Captain switch and substitution advice during a round |
| `live --mark-changed` | Record a manual change you made in-game |
| `team apply` | Record the advised moves as executed |
| `team show` | Show your saved squad |
| `team use-booster <name>` | Mark a booster as used |
| `calibrate` | Refit the priors against real results (`--write` to persist) |
| `fetch` | Refresh the FIFA feeds only |
| `serve` | Start the local web GUI and open the browser (see “Web app”) |

## Data sources

FIFA fantasy feeds (players, prices, fixtures, scores), the official
[FIFA Match Centre feed](https://api.fifa.com/api/v3) (confirmed lineups + suspensions,
free and key-less), [Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads)
for injury withdrawals, [FBref](https://fbref.com) club stats via `soccerdata`,
[martj42 international results](https://github.com/martj42/international_results) for the
Elo ratings, [The Odds API](https://the-odds-api.com) and World Bank economics.
Everything except the FIFA feeds is optional.

## Notes

Semi-automatic: it reads public data and prints instructions; you make the moves in the
game. Prices are fixed all tournament, so price is a quality signal and a budget cost,
never a market. Unofficial, not affiliated with FIFA.
