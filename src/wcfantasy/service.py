"""Structured service layer between the engine and the web API.

Every function here returns plain JSON-friendly dicts (no rich, no console
output) by delegating to the SAME engine functions the CLI uses:
`advise.compute_advice / apply_last_advice_result / use_booster_result /
compute_odds`, `live.compute_live_state / apply_recorded_changes`,
`calibrate.compute_calibration`, `fifa_client.refresh`, `state` and
`settings`. The slow ones accept a `log` callback so the web job runner can
stream progress; they never touch the FIFA account (read-only, like the CLI).
"""

from collections.abc import Callable

from . import advise, calibrate, features, fifa_client, live, rules, settings, state
from .data_sources import odds
from .models import player_points


# --------------------------------------------------------------- read-only

def _role(pid: int, team: state.TeamState) -> str:
    """Role tag for one of my players (CAPTAIN/VICE/START/sub N/bench GK/bench)."""
    if pid == team.captain_id:
        return "CAPTAIN"
    if pid == team.vice_captain_id:
        return "VICE"
    if pid in team.starters:
        return "START"
    if pid == team.bench_gk_id:
        return "bench GK"
    if pid in team.bench_order:
        return f"sub {team.bench_order.index(pid) + 1}"
    return "bench"


_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def team_snapshot() -> dict:
    """The saved squad for the dashboard: players with role tags, bank, round,
    booster state. Degrades cleanly when no team is saved or no feeds exist
    yet (first launch), so the UI can show the right empty state."""
    team = state.load()
    snapshot: dict = {
        "has_team": bool(team.player_ids),
        "round_no": team.round_no,
        "bank": round(float(team.bank), 2),
        "transfers_carried": team.transfers_carried,
        "manual_changes_round": team.manual_changes_round,
        "boosters_used": list(team.boosters_used),
        "boosters": [
            {
                "key": key,
                "description": desc,
                "used": key in team.boosters_used,
                "irreversible": key in rules.IRREVERSIBLE_BOOSTERS,
            }
            for key, desc in rules.BOOSTERS.items()
        ],
        "feeds_available": True,
        "players": [],
    }
    if not team.player_ids:
        return snapshot

    try:
        players_raw = fifa_client.load("players")
        squads = fifa_client.load("squads")
    except FileNotFoundError:
        snapshot["feeds_available"] = False
        return snapshot

    by_id = features.player_table(players_raw, squads).set_index("id")
    players = []
    for pid in team.player_ids:
        role = _role(pid, team)
        if pid not in by_id.index:
            players.append({
                "id": int(pid), "name": str(pid), "team": "?", "position": "?",
                "price": 0.0, "role": role, "is_starter": pid in team.starters,
                "is_captain": pid == team.captain_id,
                "is_vice": pid == team.vice_captain_id,
                "is_bench_gk": pid == team.bench_gk_id,
                "bench_slot": (team.bench_order.index(pid) + 1)
                if pid in team.bench_order else None,
            })
            continue
        row = by_id.loc[pid]
        players.append({
            "id": int(pid),
            "name": str(row["name"]),
            "team": str(row["squadName"]),
            "position": str(row["position"]),
            "price": round(float(row["price"]), 1),
            "role": role,
            "is_starter": pid in team.starters,
            "is_captain": pid == team.captain_id,
            "is_vice": pid == team.vice_captain_id,
            "is_bench_gk": pid == team.bench_gk_id,
            "bench_slot": (team.bench_order.index(pid) + 1)
            if pid in team.bench_order else None,
        })
    players.sort(key=lambda p: (
        _POS_ORDER.get(p["position"], 9), not p["is_starter"], p["name"]
    ))
    snapshot["players"] = players
    return snapshot


def status_line() -> dict:
    """Lightweight source/feature status WITHOUT refreshing (for the header /
    settings row): language, whether feeds exist + their freshness, and which
    optional signals are configured. The full per-run model line is part of the
    advise/odds results."""
    ui = settings.load()
    feeds_available = True
    freshness = None
    try:
        freshness = fifa_client.freshness_report(fifa_client.load("rounds"))
    except Exception:
        feeds_available = False
    return {
        "language": ui.language,
        "supported_languages": list(settings.SUPPORTED_LANGUAGES),
        "feeds_available": feeds_available,
        "freshness": freshness,
        "calibration_on": player_points.CALIBRATION_FILE.exists(),
        "odds_key_set": bool(odds.api_key()),
    }


def get_settings() -> dict:
    ui = settings.load()
    return {
        "language": ui.language,
        "supported_languages": list(settings.SUPPORTED_LANGUAGES),
    }


def set_settings(language: str) -> dict:
    """Persist the UI language; raises ValueError on an unsupported one."""
    ui = settings.set_language(language)
    return {
        "language": ui.language,
        "supported_languages": list(settings.SUPPORTED_LANGUAGES),
    }


# ------------------------------------------------------------- slow (jobs)

def advise_plan(
    *, from_scratch: bool = False, log: Callable[[str], None] | None = None
) -> dict:
    """Full next-round plan (refreshes feeds first). Writes last_advice.json in
    applyable modes, exactly like the CLI, so `apply_advice` can promote it."""
    return advise.compute_advice(from_scratch=from_scratch, log=log).to_dict()


def odds_table(*, log: Callable[[str], None] | None = None) -> dict:
    """Championship odds (refreshes feeds + runs the Monte Carlo)."""
    return advise.compute_odds(log=log)


def predicted_bracket(*, log: Callable[[str], None] | None = None) -> dict:
    """Predicted knockout bracket (refreshes feeds + runs the Monte Carlo).

    Same single simulation as advise/odds; firms up automatically as completed
    matches pin their ties on each run."""
    return advise.compute_bracket(log=log)


def live_state(*, log: Callable[[str], None] | None = None) -> dict:
    """Intra-round advice (refreshes feeds). Read-only: records nothing; use
    `record_live` to log a change you actually made in-game."""
    return live.compute_live_state(log=log).to_dict()


def calibrate_preview(
    *, write: bool = False, log: Callable[[str], None] | None = None
) -> dict:
    """Calibration preview (dry run) or persistence (`write=True`). Reads the
    LATEST local feeds; no network here, so run a fetch/advise first."""
    return calibrate.compute_calibration(write=write, log=log)


def fetch_feeds(*, log: Callable[[str], None] | None = None) -> dict:
    """Refresh the FIFA feeds only (the `fetch` command)."""
    emit = log or (lambda _msg: None)
    emit("Refreshing FIFA feeds...")
    changed = fifa_client.refresh()
    feeds = {k: ("updated" if v else "unchanged") for k, v in changed.items()}
    emit(f"Done: {feeds}")
    return {"feeds": feeds}


# ------------------------------------------------------- sync state changes

def apply_advice(*, force: bool = False) -> dict:
    """Promote last_advice.json into the saved squad (`team apply`). Returns the
    live-round guard so the UI can confirm an override."""
    return advise.apply_last_advice_result(force=force)


def set_booster(name: str | None) -> dict:
    """Mark a booster as used (`team use-booster`); one per tournament."""
    return advise.use_booster_result(name)


def record_live(
    *,
    mark_changed: bool = False,
    captain: str | None = None,
    sub_out: str | None = None,
    sub_in: str | None = None,
    vice: str | None = None,
) -> dict:
    """Record a change actually executed in the game this round into the saved
    state, so `live` stops re-suggesting it. Synchronous (no feed refresh): it
    uses the cached feeds and the current live round, resolves names within MY
    squad and validates the swap. Mirrors the CLI `live --captain/--sub-*` path.
    """
    if not any([mark_changed, captain, sub_out, sub_in, vice]):
        return {"status": "error", "message": "Nothing to record: specify a change."}
    try:
        players_raw = fifa_client.load("players")
        rounds = fifa_client.load("rounds")
        squads = fifa_client.load("squads")
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "No FIFA data yet: run Plan or Fetch first.",
        }

    team = state.load()
    if not team.player_ids:
        return {
            "status": "error",
            "message": "No saved team: plan a round and apply it first.",
        }
    rnd = live.current_live_round(rounds)
    if rnd is None:
        return {
            "status": "error",
            "message": "No round is live right now: nothing to record.",
        }
    round_no = rnd["id"]
    df = features.player_table(players_raw, squads)
    try:
        messages = live.apply_recorded_changes(
            team, df, round_no, mark_changed=mark_changed,
            captain=captain, sub_out=sub_out, sub_in=sub_in, vice=vice,
        )
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    state.save(team)
    return {"status": "ok", "round_no": round_no, "messages": messages}
