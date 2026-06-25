"""Intra-round advisor: captain switches and manual substitutions.

`python -m wcfantasy live`, run any time while a round is being played,
answers two questions with expected-value arithmetic:

- my captain has completed his match with X points: is there an
  unlocked starter whose expected points beat X? (switching moves the
  double from a known X to an expected value);
- a starter has completed his match with X points: is there an unlocked
  bench player expected to beat X? (manual sub, formation permitting).

Both must price in the official catch: the FIRST manual change of a
round (sub or captain/vice change) cancels automatic DNP substitutions
AND the vice-captain insurance for that round. After the first change
that cost is sunk and further changes are free.

Live-feed schema verified at matchday 1 (2026-06-12): match status is
'complete' / 'scheduled' (round status 'playing' while live), and
stats.roundPoints is a dict keyed by round number as string, e.g.
{'1': 13}. The defensive fallbacks below stay as insurance.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches

from rich.console import Console

from . import feed_status, features, fifa_client, fixtures, rules, state
from .data_sources import availability as availability_mod
from .models import player_points
from .optimizer import captain as captain_mod

console = Console()


@dataclass
class LiveState:
    """Structured result of `compute_live_state`: the intra-round advice with
    no console formatting. `status` selects the case the caller renders:
    'ok' (full advice), 'no_team', 'no_live_round', or 'record_error' (a bad
    --captain/--sub name). Shared by the CLI and the web `live` view so the
    two never drift."""

    status: str
    message: str | None = None
    round_no: int | None = None
    lineup_warning: str | None = None
    recorded: list[str] = field(default_factory=list)
    recorded_error: str | None = None
    already_changed: bool = False
    change_cost: float = 0.0
    change_cost_note: str | None = None
    captain_lines: list[str] = field(default_factory=list)
    sub_lines: list[str] = field(default_factory=list)
    outlook_lines: list[str] = field(default_factory=list)
    # Structured sibling of `outlook_lines`: my still-to-play players grouped
    # by their upcoming fixture, so the web can render fixture cards (the text
    # lines stay for the CLI and as a backward-compatible fallback).
    outlook: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

# Rough chance that a starter whose match is still to play ends up with
# no minutes (late knock, surprise rotation). Used only to price the
# auto-sub insurance forfeited by the first manual change.
DNP_PROB = 0.06


def _rome_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Rome")
    except Exception:
        # No tz database: the tournament (June-July 2026) is all CEST.
        return timezone(timedelta(hours=2), "CEST")


def _rome(when: datetime) -> datetime:
    return when.astimezone(_rome_tz())


# ------------------------------------------------------------ feed parsing

def match_phase(match: dict) -> str:
    """'scheduled' | 'live' | 'completed' (defensive on status values)."""
    status = str(match.get("status") or "scheduled").lower()
    if status in feed_status.DONE:
        return "completed"
    if status in feed_status.LIVE:
        return "live"
    if match.get("homeScore") is not None and status not in ("scheduled", "postponed"):
        # has a score but an unknown status: assume in progress
        return "live"
    return "scheduled"


def current_live_round(rounds: list[dict]) -> dict | None:
    """The round being played now: some matches started, some still to go.

    Returns None between rounds (nothing started, or everything done).
    """
    for rnd in sorted(rounds, key=lambda r: r["id"]):
        phases = [match_phase(m) for m in rnd.get("tournaments", [])]
        if not phases:
            continue
        if any(p == "live" for p in phases):
            return rnd
        if any(p == "completed" for p in phases) and any(p == "scheduled" for p in phases):
            return rnd
    return None


def squad_phases(rnd: dict) -> dict[int, str]:
    """squadId -> phase of its match in this round."""
    out: dict[int, str] = {}
    for m in rnd.get("tournaments", []):
        phase = match_phase(m)
        for key in ("homeSquadId", "awaySquadId"):
            if m.get(key) is not None:
                out[m[key]] = phase
    return out


def round_kickoffs(rnd: dict) -> dict[int, datetime]:
    """squadId -> kickoff time of its match in this round (tz-aware)."""
    return fixtures.kickoffs_by_squad(rnd)


def _deadline(kickoffs: dict[int, datetime] | None, squad_id: int) -> str:
    """' Act before his kickoff (...)' - every live suggestion expires
    the moment the incoming player's match starts (he becomes locked).
    Time shown in Italian local time."""
    when = (kickoffs or {}).get(squad_id)
    return f" Act before his kickoff ({_rome(when):%d %b %H:%M}, IT)." if when else ""


def player_round_points(stats: dict | None, round_no: int) -> float | None:
    """Points scored in `round_no`, or None if the feed does not say.

    The real schema (verified matchday 1) is a dict keyed by the round
    number as a string: {'1': 13}. The list fallbacks below are kept as
    insurance against mid-tournament schema changes.
    """
    rp = (stats or {}).get("roundPoints")
    if rp is None:
        return None
    if isinstance(rp, dict):
        for key in (str(round_no), round_no):
            if key in rp:
                value = rp[key]
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, dict):
                    return _points_value(value)
        return None
    if isinstance(rp, list):
        for item in rp:
            if isinstance(item, dict):
                rid = item.get("round") or item.get("roundId") or item.get("id")
                if rid == round_no:
                    return _points_value(item)
        if rp and all(isinstance(x, (int, float)) for x in rp) and len(rp) >= round_no:
            return float(rp[round_no - 1])
    return None


def _points_value(item: dict) -> float | None:
    for key in ("points", "score", "value", "totalPoints"):
        if isinstance(item.get(key), (int, float)):
            return float(item[key])
    return None


# ------------------------------------------------------------ pure logic

def formation_valid_after_swap(
    starter_positions: list[str], out_pos: str, in_pos: str
) -> bool:
    """Would the XI still match an official formation after the swap?"""
    if (out_pos == "GK") != (in_pos == "GK"):
        return False  # GK can only be replaced by the backup GK
    if out_pos == in_pos:
        return True
    counts = {pos: starter_positions.count(pos) for pos in ("DEF", "MID", "FWD")}
    counts[out_pos] -= 1
    counts[in_pos] += 1
    return (counts["DEF"], counts["MID"], counts["FWD"]) in rules.FORMATIONS


def first_change_cost(team: state.TeamState, df, phases: dict[int, str]) -> float:
    """EV forfeited by the FIRST manual change: the auto-sub insurance.

    Every starter still to play might DNP; with auto-subs alive each DNP
    would be patched from the bench for free at round end.
    """
    by_id = df.set_index("id")
    pending = [
        pid for pid in team.starters
        if pid in by_id.index
        and phases.get(int(by_id.loc[pid, "squadId"]), "scheduled") != "completed"
    ]
    bench = [
        pid for pid in (*team.bench_order, team.bench_gk_id)
        if pid in by_id.index
        and phases.get(int(by_id.loc[pid, "squadId"]), "scheduled") == "scheduled"
    ]
    if not pending or not bench:
        return 0.0
    bench_ep = float(by_id.loc[list(bench), "ep_match"].mean())
    return len(pending) * DNP_PROB * bench_ep


def _initial_captain_lines(
    df,
    team: state.TeamState,
    phases: dict[int, str],
    kickoffs: dict[int, datetime] | None,
) -> list[str]:
    """Captain advice while the current captain has NOT played yet.

    The armband can be moved live from a completed player to one not yet
    played, so the best opening captain is usually your EARLIEST strong
    option: captain him first and roll the armband toward your higher-EP
    anchor if he disappoints (option value). This surfaces that plan before
    the round unfolds, instead of the old flat 'nothing to do'.
    """
    by_id = df.set_index("id")
    cap = by_id.loc[team.captain_id]
    candidates = [
        (
            pid,
            float(by_id.loc[pid, "ep_match"]),
            (kickoffs or {}).get(int(by_id.loc[pid, "squadId"])),
        )
        for pid in team.starters
        if pid in by_id.index
        and phases.get(int(by_id.loc[pid, "squadId"]), "scheduled") == "scheduled"
    ]
    plan = captain_mod.plan_captain(candidates)
    if plan is None:
        return [f"Captain {cap['name']} has not played yet: nothing to do."]

    if plan.initial != team.captain_id and plan.initial in by_id.index:
        target = by_id.loc[plan.initial]
        deadline = _deadline(kickoffs, int(target["squadId"]))
        if plan.premium > 0 and plan.anchor != plan.initial and plan.anchor in by_id.index:
            anchor = by_id.loc[plan.anchor]
            return [
                f"MOVE the armband to {target['name']} ({target['squadName']}, "
                f"{target['ep_match']:.1f} EP): he is your earliest strong "
                f"option, so captain him first and roll it live toward "
                f"{anchor['name']} ({anchor['ep_match']:.1f} EP) if he "
                f"disappoints (option value ~+{plan.premium:.1f} captain pts)."
                + deadline
            ]
        return [
            f"Consider captaining {target['name']} ({target['squadName']}, "
            f"{target['ep_match']:.1f} EP) instead of {cap['name']}: higher "
            "expected and not yet played." + deadline
        ]

    # Current captain is already the right opening pick: remind to re-check
    # after his match for a possible roll to a later, still-unplayed option.
    later = [
        pid for pid in plan.sequence
        if pid != team.captain_id and pid in by_id.index
    ]
    if plan.initial == team.captain_id and later:
        nxt = by_id.loc[max(later, key=lambda p: float(by_id.loc[p, "ep_match"]))]
        return [
            f"Captain {cap['name']} plays first among your strong options: keep "
            f"the armband on him, then re-run `live` after his match - if he "
            f"disappoints you can roll it to {nxt['name']} "
            f"({nxt['ep_match']:.1f} EP) before kickoff."
        ]
    return [f"Captain {cap['name']} has not played yet: nothing to do."]


def captain_advice(
    df,
    team: state.TeamState,
    phases: dict[int, str],
    points: dict[int, float | None],
    change_cost: float,
    kickoffs: dict[int, datetime] | None = None,
) -> list[str]:
    by_id = df.set_index("id")
    if team.captain_id not in by_id.index:
        return ["Captain not found in the player feed: check your saved team."]
    cap = by_id.loc[team.captain_id]
    cap_phase = phases.get(int(cap["squadId"]), "scheduled")

    if cap_phase == "scheduled":
        return _initial_captain_lines(df, team, phases, kickoffs)
    if cap_phase == "live":
        return [
            f"Captain {cap['name']} is playing now (locked): re-run after "
            "his match to evaluate a switch."
        ]

    actual = points.get(team.captain_id)
    candidates = by_id.loc[
        [
            pid for pid in team.starters
            if pid != team.captain_id
            and pid in by_id.index
            and phases.get(int(by_id.loc[pid, "squadId"]), "scheduled") == "scheduled"
        ]
    ]
    if candidates.empty:
        return [
            f"Captain {cap['name']} has finished, but no starter is still "
            "to play: no switch possible."
        ]
    best = candidates.sort_values("ep_match", ascending=False).iloc[0]

    deadline = _deadline(kickoffs, int(best["squadId"]))
    if actual is None:
        return [
            f"Captain {cap['name']} has finished but his live points are "
            "not in the feed yet. Rule of thumb: switch to "
            f"{best['name']} (expected {best['ep_match']:.1f}) only if the "
            f"captain scored fewer than ~{best['ep_match'] + change_cost:.1f} pts."
            + deadline
        ]

    gain = float(best["ep_match"]) - actual - change_cost
    if gain > 0:
        msg = (
            f"SWITCH captain to {best['name']} ({best['squadName']}): the "
            f"double moves from {actual:.0f} banked pts to an expected "
            f"{best['ep_match']:.1f} (net gain ~{gain:+.1f}"
        )
        if change_cost > 0:
            msg += f" after pricing the auto-sub forfeit ~{change_cost:.1f}"
        return [msg + ")." + deadline]
    return [
        f"KEEP captain {cap['name']}: his {actual:.0f} banked pts beat "
        f"{best['name']}'s expected {best['ep_match']:.1f}"
        + (f" once the auto-sub forfeit (~{change_cost:.1f}) is priced in"
           if change_cost > 0 else "")
        + "."
    ]


def substitution_advice(
    df,
    team: state.TeamState,
    phases: dict[int, str],
    points: dict[int, float | None],
    change_cost: float,
    kickoffs: dict[int, datetime] | None = None,
) -> list[str]:
    by_id = df.set_index("id")
    starters = [pid for pid in team.starters if pid in by_id.index]
    bench_ids = [
        pid for pid in (*team.bench_order, team.bench_gk_id)
        if pid is not None and pid in by_id.index
    ]
    starter_positions = [str(by_id.loc[pid, "position"]) for pid in starters]

    advice: list[str] = []
    first_change_pending = change_cost > 0
    for pid in starters:
        row = by_id.loc[pid]
        if phases.get(int(row["squadId"]), "scheduled") != "completed":
            continue
        actual = points.get(pid)
        candidates = [
            bid for bid in bench_ids
            if phases.get(int(by_id.loc[bid, "squadId"]), "scheduled") == "scheduled"
            and formation_valid_after_swap(
                starter_positions, str(row["position"]), str(by_id.loc[bid, "position"])
            )
        ]
        if not candidates:
            continue
        best_id = max(candidates, key=lambda b: float(by_id.loc[b, "ep_match"]))
        best = by_id.loc[best_id]
        deadline = _deadline(kickoffs, int(best["squadId"]))

        if actual is None:
            advice.append(
                f"{row['name']} has finished (points unknown in feed): bench "
                f"option is {best['name']} expected {best['ep_match']:.1f}."
                + deadline
            )
            continue
        if actual == 0:
            # Likely DNP (any appearance is worth at least 1 point).
            if first_change_pending:
                advice.append(
                    f"{row['name']} likely DNP (0 pts): if you make NO other "
                    "manual change, the round-end auto-sub fixes this for "
                    f"free; sub {best['name']} in manually only if you prefer "
                    "him over your bench priority order."
                )
            else:
                advice.append(
                    f"SUB {row['name']} (likely DNP, 0 pts) -> {best['name']} "
                    f"(expected {best['ep_match']:.1f}): auto-subs are already "
                    "off, the manual sub is free upside." + deadline
                )
            continue
        gain = float(best["ep_match"]) - actual - change_cost
        if gain > 0:
            advice.append(
                f"SUB {row['name']} ({actual:.0f} pts banked) -> {best['name']} "
                f"(expected {best['ep_match']:.1f}, net ~{gain:+.1f})."
                + deadline
            )
    return advice


def round_outlook(
    df,
    team: state.TeamState,
    phases: dict[int, str],
    availability: dict[int, float] | None,
    kickoffs: dict[int, datetime] | None = None,
    observed_starts: dict[int, float] | None = None,
) -> list[str]:
    """One line per MY player whose match is still to come this round:
    how likely he is to take the pitch (P plays = depth chart x
    availability). Expected points are decided elsewhere (lineup,
    transfers, the captain/sub advice above): here we only flag who might
    not play."""
    by_id = df.set_index("id")
    p_play = player_points.starter_probability(df, availability, observed_starts)
    p_play.index = df["id"]

    rows = []
    for pid in team.player_ids:
        if pid not in by_id.index:
            continue
        squad_id = int(by_id.loc[pid, "squadId"])
        if phases.get(squad_id, "scheduled") != "scheduled":
            continue  # already played or playing now
        when = (kickoffs or {}).get(squad_id)
        rows.append((when, pid))

    rows.sort(key=lambda r: r[0] or datetime.max.replace(tzinfo=timezone.utc))
    lines = []
    for when, pid in rows:
        row = by_id.loc[pid]
        role = "XI" if pid in team.starters else "bench"
        stamp = f"{_rome(when):%a %H:%M}  " if when else ""
        lines.append(
            f"{stamp}{row['name']} ({row['squadName']}, {role}): "
            f"{p_play[pid] * 100:.0f}% to play"
        )
    return lines


def round_outlook_matches(
    df,
    team: state.TeamState,
    rnd: dict,
    availability: dict[int, float] | None,
    kickoffs: dict[int, datetime] | None = None,
    observed_starts: dict[int, float] | None = None,
) -> list[dict]:
    """Structured sibling of `round_outlook`: my players still to play this
    round, grouped BY their upcoming fixture in chronological order.

    Same P(plays) basis as the text outlook, but each match carries the
    opponent and kickoff so the web can render the same fixture cards the Plan
    schedule uses. The kickoff is pre-formatted in Italian local time to match
    that schedule (None when the date is still TBD). `outlook_lines` stays the
    per-player text fallback for the CLI."""
    by_id = df.set_index("id")
    p_play = player_points.starter_probability(df, availability, observed_starts)
    p_play.index = df["id"]

    matches: list[dict] = []
    for m in rnd.get("tournaments", []):
        if match_phase(m) != "scheduled":
            continue  # already played or playing now
        squad_ids = {m.get("homeSquadId"), m.get("awaySquadId")}
        players = []
        for pid in team.player_ids:
            if pid not in by_id.index:
                continue
            if int(by_id.loc[pid, "squadId"]) not in squad_ids:
                continue
            row = by_id.loc[pid]
            players.append({
                "name": str(row["name"]),
                "team": str(row["squadName"]),
                "position": str(row["position"]),
                "role": "XI" if pid in team.starters else "bench",
                "p_play": round(float(p_play[pid]), 4),
                "is_captain": pid == team.captain_id,
                "is_vice": pid == team.vice_captain_id,
            })
        if not players:
            continue
        players.sort(key=lambda p: (p["role"] != "XI", -p["p_play"], p["name"]))
        when = (kickoffs or {}).get(m.get("homeSquadId")) or (
            kickoffs or {}).get(m.get("awaySquadId"))
        matches.append({
            "kickoff": f"{_rome(when):%a %d %b %H:%M}" if when else None,
            "_when": when,
            "home": str(m.get("homeSquadName") or "?"),
            "away": str(m.get("awaySquadName") or "?"),
            "players": players,
        })
    matches.sort(key=lambda mt: mt["_when"] or datetime.max.replace(tzinfo=timezone.utc))
    for mt in matches:
        mt.pop("_when", None)
    return matches


def lineup_set_for_future_round(team_round_no: int, live_round_no: int) -> bool:
    """True when my_team.json was applied for a round AFTER the one being
    played now, so the saved XI is next round's plan, not what is actually
    on the pitch (the apply/live footgun on round-transition days)."""
    return bool(team_round_no) and team_round_no > live_round_no


# ------------------------------------------------------------ recording
# `live` only suggests captain switches and subs; it never knew which one
# the user actually executed in the game, so the saved XI stayed stale and
# every re-run re-suggested the same move. These helpers record the real
# change locally (the user states what he did), so the advice stops looping.

def _label(by_id, pid: int) -> str:
    if pid in by_id.index:
        row = by_id.loc[pid]
        return f"{row['name']} ({row['squadName']})"
    return f"id {pid}"


def resolve_my_player(name: str, df, team: state.TeamState) -> int:
    """Resolve free-text `name` to exactly one of MY player ids.

    Matches only within the saved 15-man squad (you can only captain or sub
    your own players), trying exact, then case-insensitive substring, then a
    fuzzy fallback for typos. Raises ValueError listing the squad on a miss
    or an ambiguous match, so the caller aborts before mutating anything.
    """
    by_id = df.set_index("id")
    mine = [
        (pid, str(by_id.loc[pid, "name"]))
        for pid in team.player_ids
        if pid in by_id.index
    ]
    if not mine:
        raise ValueError("Your saved squad is empty: run `team apply` first.")
    target = name.strip().casefold()

    def _fail() -> None:
        roster = ", ".join(sorted(nm for _, nm in mine))
        raise ValueError(
            f"{name!r} did not match exactly one of your players. "
            f"Your squad: {roster}."
        )

    for candidates in (
        [pid for pid, nm in mine if nm.casefold() == target],            # exact
        [pid for pid, nm in mine if target and target in nm.casefold()],  # substring
    ):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            _fail()
    # Fuzzy fallback: tolerate a typo or a missing accent.
    close = get_close_matches(name, [nm for _, nm in mine], n=2, cutoff=0.6)
    fuzzy = [pid for pid, nm in mine if nm in close]
    if len(fuzzy) == 1:
        return fuzzy[0]
    _fail()


def record_substitution(team: state.TeamState, out_id: int, in_id: int, df) -> str:
    """Apply a manual sub (starter `out_id` <- bench `in_id`) to the saved
    state. Validates squad membership and that the XI stays a legal
    formation; raises ValueError otherwise WITHOUT mutating. The outgoing
    starter inherits the incoming player's bench slot."""
    by_id = df.set_index("id")
    bench = list(team.bench_order)
    if team.bench_gk_id is not None:
        bench.append(team.bench_gk_id)
    if out_id not in team.starters:
        raise ValueError(f"{_label(by_id, out_id)} is not in your starting XI.")
    if in_id not in bench:
        raise ValueError(f"{_label(by_id, in_id)} is not on your bench.")
    if out_id == in_id:
        raise ValueError("Out and in are the same player.")
    out_pos = str(by_id.loc[out_id, "position"])
    in_pos = str(by_id.loc[in_id, "position"])
    positions = [
        str(by_id.loc[pid, "position"]) for pid in team.starters if pid in by_id.index
    ]
    if not formation_valid_after_swap(positions, out_pos, in_pos):
        raise ValueError(
            f"Subbing {_label(by_id, out_id)} ({out_pos}) for "
            f"{_label(by_id, in_id)} ({in_pos}) would not leave a legal "
            "formation. Check the names."
        )
    team.starters = [in_id if pid == out_id else pid for pid in team.starters]
    if in_id == team.bench_gk_id:
        team.bench_gk_id = out_id
    else:
        team.bench_order = [out_id if pid == in_id else pid for pid in team.bench_order]
    return f"Sub recorded: {_label(by_id, out_id)} OUT, {_label(by_id, in_id)} IN."


def record_captain(
    team: state.TeamState, new_id: int, df, explicit_vice: bool = False
) -> str:
    """Set the captain to `new_id` (must be a starter). If the new captain
    was the vice and no explicit vice is being set, move the old captain
    into the vice slot so the two never coincide (the common in-game
    captain<->vice swap)."""
    by_id = df.set_index("id")
    if new_id not in team.starters:
        raise ValueError(
            f"{_label(by_id, new_id)} must be in your starting XI to be captain."
        )
    old = team.captain_id
    team.captain_id = new_id
    msg = f"Captain recorded: {_label(by_id, new_id)}."
    if (
        not explicit_vice
        and new_id == team.vice_captain_id
        and old is not None
        and old in team.starters
        and old != new_id
    ):
        team.vice_captain_id = old
        msg += f" Vice moved to {_label(by_id, old)} (the previous captain)."
    return msg


def apply_recorded_changes(
    team: state.TeamState,
    df,
    round_no: int,
    *,
    mark_changed: bool = False,
    captain: str | None = None,
    sub_out: str | None = None,
    sub_in: str | None = None,
    vice: str | None = None,
) -> list[str]:
    """Record in local state a change actually executed in the game this
    round, so `live` stops re-suggesting it. Mutates `team` in place and
    returns confirmation lines; raises ValueError on bad input (the caller
    saves only on success). Any recorded change also marks the round as
    changed - the first manual change is what cancels auto-subs and the vice
    insurance, and repeating it for further changes is idempotent."""
    by_id = df.set_index("id")
    if (sub_out is None) != (sub_in is None):
        raise ValueError("Use --sub-out and --sub-in together.")

    # Resolve every name up front so a typo aborts before any mutation.
    sub_pair = None
    if sub_out is not None:
        sub_pair = (
            resolve_my_player(sub_out, df, team),
            resolve_my_player(sub_in, df, team),
        )
    cap_id = resolve_my_player(captain, df, team) if captain else None
    vice_id = resolve_my_player(vice, df, team) if vice else None

    lines: list[str] = []
    if sub_pair is not None:
        lines.append(record_substitution(team, sub_pair[0], sub_pair[1], df))
    if cap_id is not None:
        lines.append(record_captain(team, cap_id, df, explicit_vice=vice_id is not None))
    if vice_id is not None:
        if vice_id not in team.starters:
            raise ValueError(
                f"{_label(by_id, vice_id)} must be in your starting XI to be "
                "vice-captain."
            )
        team.vice_captain_id = vice_id
        lines.append(f"Vice-captain recorded: {_label(by_id, vice_id)}.")
    if team.captain_id is not None and team.captain_id == team.vice_captain_id:
        raise ValueError("Captain and vice-captain cannot be the same player.")

    changed = sub_pair is not None or cap_id is not None or vice_id is not None
    if changed or mark_changed:
        team.manual_changes_round = round_no
    if mark_changed and not changed:
        lines.append(
            "Recorded: manual change made this round (auto-subs and vice "
            "insurance are off from now on)."
        )
    return lines


# ------------------------------------------------------------ orchestrator

def compute_live_state(
    *,
    mark_changed: bool = False,
    captain: str | None = None,
    sub_out: str | None = None,
    sub_in: str | None = None,
    vice: str | None = None,
    log: Callable[[str], None] | None = None,
) -> LiveState:
    """Intra-round advice as structured data (no console output).

    Refreshes the feeds, finds the live round, optionally records a change the
    user executed in-game (the --captain/--sub-out+--sub-in/--vice path, which
    mutates and saves the local state), then computes the captain-switch,
    substitution and round-outlook advice. `log` receives progress lines for a
    live caller (CLI or web job). Returns a LiveState whose `status` tells the
    caller which case to render.
    """
    emit = log or (lambda _msg: None)

    emit("Refreshing FIFA feeds...")
    fifa_client.refresh()
    player_points.apply_calibration()
    players_raw = fifa_client.load("players")
    rounds = fifa_client.load("rounds")
    squads = fifa_client.load("squads")

    team = state.load()
    if not team.player_ids:
        return LiveState(
            status="no_team",
            message="No saved team: run `advise` then `team apply` first.",
        )

    rnd = current_live_round(rounds)
    if rnd is None:
        return LiveState(
            status="no_live_round",
            message="No round is live right now: between rounds, run `advise` instead.",
        )
    round_no = rnd["id"]
    emit(f"Live round: {round_no}")

    lineup_warning = None
    if lineup_set_for_future_round(team.round_no, round_no):
        lineup_warning = (
            f"Warning: your saved team is set for round {team.round_no}, but "
            f"round {round_no} is live now. The XI below may be next round's "
            "plan, not what is on the pitch. On transition days run `live` "
            "BEFORE `team apply`, and apply the next round only after this one "
            "is over."
        )

    df = features.player_table(players_raw, squads)

    # Record any change actually executed in the game, so the advice below
    # reflects it and stops re-suggesting it. Names resolve against my squad.
    recorded: list[str] = []
    if any([mark_changed, captain, sub_out, sub_in, vice]):
        try:
            recorded = apply_recorded_changes(
                team, df, round_no, mark_changed=mark_changed,
                captain=captain, sub_out=sub_out, sub_in=sub_in, vice=vice,
            )
        except ValueError as exc:
            return LiveState(
                status="record_error",
                round_no=round_no,
                lineup_warning=lineup_warning,
                recorded_error=str(exc),
            )
        state.save(team)

    phases = squad_phases(rnd)
    kickoffs = round_kickoffs(rnd)
    # Injury / availability discount so a player ruled out is never
    # offered as a sub or captain target. Confirmed lineups (published
    # ~1h before kickoff) override the injuries feed when available.
    injuries = availability_mod.availability_factors(squads, df, rounds, round_no)
    lineups = availability_mod.live_lineup_factors(rnd, squads, df)
    availability = {**injuries, **lineups}
    # Same EP basis as advise: club-stats prior + the observed-starts depth
    # chart (who actually played the completed matches), not the price rank.
    observed = availability_mod.observed_start_factors(rounds, round_no, squads, df)
    club_ep = player_points.club_prior(df)
    df["ep_match"] = player_points.expected_points_per_match(
        df, club_ep, availability=availability, observed_starts=observed
    )
    points = {
        p["id"]: player_round_points(p.get("stats"), round_no) for p in players_raw
    }

    changed_already = team.manual_changes_round == round_no
    cost = 0.0 if changed_already else first_change_cost(team, df, phases)
    if changed_already:
        change_cost_note = (
            "Manual change already made this round: auto-subs and vice "
            "insurance are gone, further changes carry no hidden cost."
        )
    else:
        change_cost_note = (
            "No manual change yet this round: the first one cancels "
            f"auto-subs and the vice insurance (priced at ~{cost:.1f} pts "
            "in the advice below)."
        )

    captain_lines = captain_advice(df, team, phases, points, cost, kickoffs)
    sub_lines = substitution_advice(df, team, phases, points, cost, kickoffs) or [
        "No substitution worth making right now."
    ]
    outlook_lines = round_outlook(df, team, phases, availability, kickoffs, observed) or [
        "All your players have already played this round."
    ]
    outlook = round_outlook_matches(df, team, rnd, availability, kickoffs, observed)

    return LiveState(
        status="ok",
        round_no=round_no,
        lineup_warning=lineup_warning,
        recorded=recorded,
        already_changed=changed_already,
        change_cost=round(float(cost), 2),
        change_cost_note=change_cost_note,
        captain_lines=captain_lines,
        sub_lines=sub_lines,
        outlook_lines=outlook_lines,
        outlook=outlook,
    )


_RECORD_HINT = (
    "If you execute a change in the game, record WHAT you changed so this "
    "advice stops re-suggesting it, e.g.:\n"
    '  [bold]python -m wcfantasy live --captain "<name>"[/bold]\n'
    '  [bold]python -m wcfantasy live --sub-out "<name>" --sub-in "<name>"[/bold]\n'
    "(both imply --mark-changed; plain --mark-changed only flags the round)."
)


def _print_live(state: LiveState) -> None:
    """Render a LiveState to the console (the CLI presentation of `live`)."""
    if state.status in ("no_team", "no_live_round"):
        console.print(state.message)
        return

    console.print(f"Live round: [bold]{state.round_no}[/bold]")
    if state.lineup_warning:
        console.print(f"[bold yellow]{state.lineup_warning}[/bold yellow]\n")

    if state.status == "record_error":
        console.print(f"[bold red]Nothing recorded:[/bold red] {state.recorded_error}")
        return

    for line in state.recorded:
        console.print(f"  - {line}")
    if state.recorded:
        console.print()

    console.print(state.change_cost_note + "\n")

    console.print("[bold]Captain:[/bold]")
    for line in state.captain_lines:
        console.print(f"  - {line}")

    console.print("\n[bold]Substitutions:[/bold]")
    for line in state.sub_lines:
        console.print(f"  - {line}")

    console.print("\n[bold]Still to play this round (P plays):[/bold]")
    for line in state.outlook_lines:
        console.print(f"  - {line}")

    console.print("\n" + _RECORD_HINT)


def run(
    mark_changed: bool = False,
    *,
    captain: str | None = None,
    sub_out: str | None = None,
    sub_in: str | None = None,
    vice: str | None = None,
) -> None:
    state = compute_live_state(
        mark_changed=mark_changed, captain=captain, sub_out=sub_out,
        sub_in=sub_in, vice=vice, log=console.print,
    )
    _print_live(state)
