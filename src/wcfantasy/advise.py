"""Orchestrator: fetch fresh data -> predict -> optimize -> print the plan.

This is what `python -m wcfantasy advise` runs. It always refreshes the
FIFA feeds first so every recommendation reflects the last played match.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.table import Table

from . import feed_status, features, fifa_client, fixtures, rules, state
from .config import STATE_DIR
from .data_sources import availability, odds
from .models import player_points, progression
from .optimizer import captain as captain_mod
from .optimizer import transfers as transfers_mod
from .optimizer.squad import SquadSolution, build_squad

LAST_ADVICE_FILE = STATE_DIR / "last_advice.json"

console = Console()


@dataclass
class AdvicePlan:
    """Structured result of `compute_advice`: everything `advise` needs to
    print AND everything the web API serves, with no console formatting.

    Built once by `compute_advice`; `run()` is a thin print wrapper over it
    and the service layer reshapes it to JSON. `mode` is 'initial' (no saved
    team), 'transfer' (a plan within the allowance) or 'from_scratch_comparison'
    (the unconstrained ideal vs the saved squad, never applyable/saved).
    """

    round: int
    budget: float
    country_cap: int
    mode: str
    applyable: bool
    cost: float
    bank: float
    status: dict = field(default_factory=dict)
    squad_rows: list[dict] = field(default_factory=list)
    captain: dict = field(default_factory=dict)
    vice: dict = field(default_factory=dict)
    captain_timing_note: str | None = None
    bench_order: list[dict] = field(default_factory=list)
    bench_gk: dict = field(default_factory=dict)
    transfers: dict | None = None
    moves: list[str] = field(default_factory=list)
    availability_watch: list[dict] = field(default_factory=list)
    unverified_starters: list[str] = field(default_factory=list)
    booster_advice: list[str] = field(default_factory=list)
    schedule: list[str] = field(default_factory=list)
    comparison: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _rome_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Rome")
    except Exception:
        # No tz database available: the whole tournament (June-July
        # 2026) falls in Italian summer time anyway.
        return timezone(timedelta(hours=2), "CEST")


def fixture_schedule(
    df, solution: SquadSolution, vice_id: int, rounds: list[dict], next_round: int
) -> list[str]:
    """One line per fixture involving my players, Italian local time,
    chronological: the moments to re-run `live` during the round."""
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    if rnd is None:
        return []
    squad_rows = df[df["id"].isin(solution.squad_ids)]
    marks = {solution.captain_id: " (C)", vice_id: " (V)"}
    display = squad_rows.apply(
        lambda r: r["name"] + marks.get(r["id"], ""), axis=1
    )
    by_squad = display.groupby(squad_rows["squadId"]).apply(list)
    tz = _rome_tz()

    matches = sorted(
        rnd.get("tournaments", []),
        key=lambda m: fixtures.parse_kickoff(m.get("date"))
        or datetime.max.replace(tzinfo=timezone.utc),
    )
    lines = []
    for m in matches:
        involved = [
            sid for sid in (m.get("homeSquadId"), m.get("awaySquadId"))
            if sid in by_squad.index
        ]
        if not involved:
            continue
        when = fixtures.parse_kickoff(m.get("date"))
        stamp = f"{when.astimezone(tz):%a %d %b %H:%M}" if when else "date TBD"
        names = ", ".join(n for sid in involved for n in by_squad[sid])
        lines.append(
            f"{stamp}  {m.get('homeSquadName', '?')}-{m.get('awaySquadName', '?')}"
            f": {names}"
        )
    return lines


def next_round_number(rounds: list[dict]) -> int:
    """First round still open for transfer/lineup edits.

    Done rounds are over; a 'playing' round is locked at its first kickoff
    (feed_status.is_locked covers both). Both are skipped: transfers
    confirmed now apply from this round onward. Falls back to the last
    round when nothing is open (e.g. the final is live).
    """
    plannable = [r["id"] for r in rounds if not feed_status.is_locked(r.get("status"))]
    return min(plannable) if plannable else max(r["id"] for r in rounds)


def applying_over_live_round(live_round_id: int | None, advice_round: int) -> bool:
    """True when applying a future-round plan would clobber the lineup of
    a round still being played (so `live` would then read the wrong XI)."""
    return live_round_id is not None and live_round_id < advice_round


def plan_round_after_apply(
    open_round: int, max_round: int, forced_count: int
) -> tuple[int, str]:
    """Round to plan (and the state label) once the open round is already
    APPLIED but not yet locked.

    The free transfers for the window are spent, so:
    - forced_count > 0 (owned players who can no longer play): stay on the open
      round - those replacements are necessary and now cost -3 each ('forced');
    - otherwise nothing is left to do for the open round, so preview the next
      one ('advanced'), or, if this was the last round, there is nothing more
      ('complete').
    """
    if forced_count > 0:
        return open_round, "forced"
    if open_round < max_round:
        return open_round + 1, "advanced"
    return open_round, "complete"


def current_budget(next_round: int) -> float:
    bonus = rules.KNOCKOUT_BUDGET_BONUS if next_round >= 4 else 0.0
    return rules.BUDGET + bonus


def _squad_rows(
    df, solution: SquadSolution, vice_id: int, bench_order: list[int], bench_gk: int
) -> list[dict]:
    """One JSON-friendly row per squad player, sorted GK/DEF/MID/FWD then EP,
    each tagged with its role (CAPTAIN/VICE/START/sub N/bench GK/bench). Shared
    by the CLI table and the web squad view so the two never drift."""
    rows = df[df["id"].isin(solution.squad_ids)].copy()
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    rows = rows.sort_values(
        ["position", "ep_match"], ascending=[True, False],
        key=lambda s: s.map(pos_order) if s.name == "position" else s,
    )
    out = []
    for _, r in rows.iterrows():
        role = "bench"
        if r["id"] == bench_gk:
            role = "bench GK"
        elif r["id"] in bench_order:
            role = f"sub {bench_order.index(r['id']) + 1}"
        if r["id"] in solution.starter_ids:
            role = "START"
        if r["id"] == vice_id:
            role = "VICE"
        if r["id"] == solution.captain_id:
            role = "CAPTAIN"
        play = float(r["p_play"]) if "p_play" in r and r["p_play"] == r["p_play"] else 1.0
        out.append({
            "id": int(r["id"]),
            "position": r["position"],
            "name": r["name"],
            "team": r["squadName"],
            "price": round(float(r["price"]), 1),
            "ep_match": round(float(r["ep_match"]), 2),
            "p_play": round(play, 4),
            "role": role,
        })
    return out


def _squad_table(rows: list[dict]) -> Table:
    table = Table(title="Recommended squad")
    for col in ("Pos", "Player", "Team", "Price", "EP/match", "Play%", "Role"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["position"], r["name"], r["team"],
            f"${r['price']:.1f}m", f"{r['ep_match']:.2f}",
            f"{r['p_play'] * 100:.0f}%", r["role"],
        )
    return table


def _unverified_starter_lines(df, solution: SquadSolution) -> list[str]:
    """Lines warning about starters whose status is a PRICE guess, not observed
    minutes. [] when none. Pure core shared by the CLI watch and the web report.

    The game's price ranked the wrong keeper for several teams, so a starter
    we have neither watched play (observed_start_ids) nor pinned by hand
    (starter_overrides.json) is flagged. GKs are always listed by name (the
    acute case); before any match the whole XI is a guess, so the outfield is
    collapsed to a count to avoid an 11-line wall."""
    observed_ids = player_points.last_run_info.get("observed_start_ids", set())
    override_names: set = set()
    if player_points.STARTER_OVERRIDES_FILE.exists():
        try:
            override_names = set(json.loads(
                player_points.STARTER_OVERRIDES_FILE.read_text(encoding="utf-8")
            ))
        except (json.JSONDecodeError, OSError):
            pass

    starters = df[df["id"].isin(solution.starter_ids)]
    price_only = starters[
        ~starters["id"].isin(observed_ids) & ~starters["name"].isin(override_names)
    ]
    if price_only.empty:
        return []

    lines = ["Starter status UNVERIFIED (price guess, not observed minutes):"]
    for _, r in price_only[price_only["position"] == "GK"].iterrows():
        lines.append(
            f"  - GK {r['name']} ({r['squadName']}): price barely separates "
            "keepers — VERIFY he is the #1 before lockout."
        )
    outfield = price_only[price_only["position"] != "GK"]
    if len(price_only) > 6:
        lines.append(
            f"  - Plus {len(outfield)} outfield starters: no WC minutes played "
            "yet, so the whole XI is a price-based guess this round."
        )
    else:
        for _, r in outfield.iterrows():
            lines.append(f"  - {r['name']} ({r['squadName']}, {r['position']})")
    lines.append(
        "  Force a known case via data/external/starter_overrides.json "
        '(e.g. {"Player Name": 0.0} to bench, 0.9 to start).'
    )
    return lines


def _unverified_starter_watch(df, solution: SquadSolution) -> None:
    """CLI wrapper: print the unverified-starter warning lines (bold header)."""
    lines = _unverified_starter_lines(df, solution)
    if not lines:
        return
    console.print(f"\n[bold]{lines[0]}[/bold]")
    for line in lines[1:]:
        console.print(line)


def squad_comparison(df, ideal: SquadSolution, current_ids: list[int]) -> list[str]:
    """How the saved squad stacks up against the unconstrained ideal 15.

    Overlap, the upgrades the ideal would bring in, the players it would
    drop, and the per-round EP gap (sum of ep_match over each 15). A purely
    informational view: the ideal ignores the transfer limits, so it is a
    target to steer toward (over several windows or a Wildcard), not a plan
    to apply in one go.
    """
    by_id = df.set_index("id")
    ideal_ids = set(ideal.squad_ids)
    current = set(current_ids)
    overlap = ideal_ids & current

    def ep(pid: int) -> float:
        return float(by_id.loc[pid, "ep_match"]) if pid in by_id.index else 0.0

    def ep_sum(ids) -> float:
        return sum(ep(i) for i in ids)

    def describe(pid: int) -> str:
        if pid not in by_id.index:
            return f"id {pid} (not in feed)"
        r = by_id.loc[pid]
        return f"{r['name']} ({r['squadName']}, {r['position']}, {r['ep_match']:.1f} EP)"

    gap = ep_sum(ideal_ids) - ep_sum(current)
    lines = [
        f"You already own {len(overlap)}/{rules.SQUAD_SIZE} of the ideal squad.",
        f"Ideal EP/round {ep_sum(ideal_ids):.1f} vs yours {ep_sum(current):.1f} "
        f"(gap {gap:+.1f} pts this round).",
    ]
    add = sorted(ideal_ids - current, key=ep, reverse=True)
    drop = sorted(current - ideal_ids, key=ep)
    if add:
        lines.append("Upgrades the ideal would bring IN (you don't own):")
        lines += [f"   + {describe(i)}" for i in add]
    if drop:
        lines.append("Players of yours the ideal would drop:")
        lines += [f"   - {describe(i)}" for i in drop]
    if not add and not drop:
        lines.append("Your squad already IS the ideal for this round.")
    return lines


def _model_notes() -> dict:
    """Source-status notes after a model run (reads the modules' last_run_info).

    Returns the individual notes plus the assembled 'Model: ...' line that
    `advise` prints and the web status row reuses. Call after
    `player_points.expected_value` so the diagnostics are populated.
    """
    info = progression.last_run_info
    odds_note = (
        f"odds: {info.get('h2h_fixtures', 0)} quoted fixtures, "
        f"{info.get('outright_teams', 0)} outright teams"
        if info.get("h2h_fixtures") or info.get("outright_teams")
        else "odds: OFF (no ODDS_API_KEY; free key at the-odds-api.com)"
    )
    matched = player_points.last_run_info.get("club_matched", 0)
    club_note = (
        f"club stats: {matched} players matched"
        if matched
        else "club stats: OFF (price prior; run with soccerdata installed)"
    )
    avail_info = availability.last_run_info
    flagged = avail_info.get("flagged", 0)
    if flagged:
        breakdown = (
            f"suspensions {avail_info.get('suspensions', 0)}, "
            f"withdrawals {avail_info.get('withdrawals', 0)}, "
            f"injuries {avail_info.get('injuries', 0)}"
        )
        avail_note = f"availability: {flagged} flagged ({breakdown})"
    elif avail_info.get("fifa_reachable"):
        avail_note = "availability: none flagged (FIFA Match Centre + Wikipedia)"
    else:
        avail_note = "availability: OFF (FIFA Match Centre unreachable)"
    cal_note = (
        "calibration: ON (data/state/calibration.json)"
        if player_points.CALIBRATION_FILE.exists()
        else "calibration: OFF (built-in defaults)"
    )
    model = info.get("model", "?")
    return {
        "model": model,
        "odds_note": odds_note,
        "club_note": club_note,
        "availability_note": avail_note,
        "calibration_note": cal_note,
        "model_line": f"Model: {model} | {odds_note} | {club_note} | "
        f"{avail_note} | {cal_note}",
    }


def compute_advice(
    *, from_scratch: bool = False, log: Callable[[str], None] | None = None
) -> AdvicePlan:
    """Full next-round recommendation as structured data (no console output).

    Always refreshes the FIFA feeds first (the project requirement: every
    decision uses data updated to the last played match), then estimates
    expected points, optimizes the squad/transfers under the rules and picks
    captain / vice / bench / boosters. `log` receives progress lines so a
    caller (the CLI, or a web job) can surface them live. Writes
    last_advice.json EXACTLY as the old `run()` did, in every applyable mode
    (initial build and transfer plan) but never for the from-scratch
    comparison, so `team apply` keeps working unchanged.
    """
    emit = log or (lambda _msg: None)

    emit("Refreshing FIFA feeds...")
    changed = fifa_client.refresh()
    player_points.apply_calibration()
    feeds_changed = [k for k, v in changed.items() if v]
    emit(f"Feeds updated: {feeds_changed or 'none (cache hit)'}")

    players_raw = fifa_client.load("players")
    rounds = fifa_client.load("rounds")
    squads = fifa_client.load("squads")

    freshness = fifa_client.freshness_report(rounds)
    emit(freshness)

    team = state.load()
    rnd0 = next_round_number(rounds)
    max_round = max(r["id"] for r in rounds)

    def _ev(for_round: int):
        d = features.player_table(players_raw, squads)
        return player_points.expected_value(d, squads, rounds, for_round)

    # "Already applied" handling. When the OPEN round's squad is committed
    # (`team apply` was run for it) and the round is not locked yet, the free
    # transfers for the window are already spent: re-offering them would be
    # wrong. Replace forced-out players (eliminated/suspended -> ep 0, or gone
    # from the feed) at -3 each because those moves are NECESSARY; otherwise
    # there is nothing left to do, so preview the next round instead.
    applied_state: str | None = None
    applied_note: str | None = None
    forced_out_names: list[str] = []
    if not from_scratch and team.player_ids and team.round_no == rnd0:
        df = _ev(rnd0)
        present = df[df["id"].isin(team.player_ids)]
        unplayable = present[present["ep_match"] <= 1e-9]
        missing_ids = [pid for pid in team.player_ids if pid not in set(df["id"])]
        forced_count = len(unplayable) + len(missing_ids)
        rnd, applied_state = plan_round_after_apply(rnd0, max_round, forced_count)
        if applied_state == "forced":
            forced_out_names = [
                f"{r['name']} ({r['squadName']})" for _, r in unplayable.iterrows()
            ] + [f"id {pid} (no longer in the game)" for pid in missing_ids]
            applied_note = (
                f"You already applied your round {rnd} squad, but "
                f"{forced_count} player(s) can no longer play: "
                f"{', '.join(forced_out_names)}. Your free transfers this window "
                f"are used, so each replacement costs "
                f"-{rules.EXTRA_TRANSFER_PENALTY:.0f} pts — but these moves are "
                "NECESSARY (an unplayable player scores nothing)."
            )
        elif applied_state == "advanced":
            emit(f"Round {rnd0} already applied; previewing round {rnd}...")
            df = _ev(rnd)
            applied_note = (
                f"Round {rnd0} already applied and complete — nothing more to do "
                f"there. PREVIEW of round {rnd}: it firms up once round {rnd0} is "
                f"played and its transfer window opens when round {rnd0} starts. "
                "Reference only, not saved."
            )
        else:  # complete
            applied_note = (
                f"Round {rnd0} already applied. It is the last round, so there is "
                "nothing more to plan."
            )
    else:
        rnd = rnd0
        df = _ev(rnd)

    budget = current_budget(rnd)
    emit(f"Planning for round {rnd}, budget ${budget:.1f}m")
    if applied_note:
        emit(applied_note)

    notes = _model_notes()
    emit(notes["model_line"])

    # Official rule: hard cap on players per country, growing with the
    # stage (3 in the groups and R32, up to 8 at the Final).
    country_cap = rules.max_per_squad(rnd)
    country_rule = f"Game rule: max {country_cap} players per nation this round"
    emit(country_rule)

    carried_next = 0
    # Ideal-squad view: build from scratch even with a team saved, to show
    # how far the unconstrained optimum is from the current squad. It is a
    # reference, never written to last_advice (you cannot rebuild freely
    # without the Wildcard), so `team apply` is not offered for it.
    comparison_mode = from_scratch and bool(team.player_ids)
    header: str | None = None
    transfer_window: str | None = None
    transfers_info: dict | None = None
    if from_scratch or not team.player_ids:
        if comparison_mode:
            header = (
                "Ideal-squad view: optimal 15 from scratch, ignoring your saved "
                "team and the transfer limits (a target, not an applyable plan)."
            )
        else:
            header = "No saved team: building the initial squad."
        emit(header)
        solution = build_squad(df, budget=budget, max_per_squad=country_cap)
        moves: list[str] = (
            [] if comparison_mode else ["Create your team with the 15 players below."]
        )
    else:
        # When the round is already applied (free transfers spent), pin the
        # allowance to 0 and only propose forced, necessary moves; never
        # re-offer the spent free transfers or churn voluntary ones.
        spent = applied_state in ("forced", "complete")
        if spent:
            plan = transfers_mod.plan_transfers(
                df, team, budget, next_round=rnd, max_per_squad=country_cap,
                allowance_override=0, forced_only=True,
            )
        else:
            plan = transfers_mod.plan_transfers(
                df, team, budget, next_round=rnd, max_per_squad=country_cap
            )
        solution = plan.solution
        carried_next = plan.carried_next
        names = df.set_index("id")["name"]
        moves = [
            f"OUT: {names.get(i, i)}" for i in plan.out_ids
        ] + [
            f"IN:  {names.get(i, i)}" for i in plan.in_ids
        ] or ["No transfers: keep the current squad."]
        if spent:
            transfer_window = (
                "Free transfers this window are already used: any further change "
                f"costs -{rules.EXTRA_TRANSFER_PENALTY:.0f} pts."
            )
        elif plan.free_allowance is None:
            transfer_window = "Unlimited transfer window: every change below is free."
        else:
            transfer_window = (
                f"Free transfers this window: {plan.free_allowance}"
                + (f" (incl. {team.transfers_carried} carried over)"
                   if team.transfers_carried else "")
            )
        emit(transfer_window)
        forced_note = None
        if plan.n_forced_paid:
            forced_note = (
                f"FORCED: {plan.n_forced_paid} transfer(s) beyond the free "
                f"{plan.free_allowance}, "
                f"-{plan.n_forced_paid * rules.EXTRA_TRANSFER_PENALTY:.0f} pts at lockout. "
                "Caused by unplayable players, not a voluntary upgrade: if the gain "
                "is clearly worth it, play the Wildcard instead of paying the penalty."
            )
            moves.append(forced_note)
        voluntary_note = None
        if plan.n_voluntary_paid:
            voluntary_note = (
                f"PAID (worth it): {plan.n_voluntary_paid} voluntary transfer(s) at "
                f"-{rules.EXTRA_TRANSFER_PENALTY:.0f} pts each "
                f"(-{plan.n_voluntary_paid * rules.EXTRA_TRANSFER_PENALTY:.0f} pts total), "
                f"but they add ~+{plan.nextround_gain:.1f} next-round pts — net positive "
                "even after the penalty. The same swap would be free next window; paying "
                "now only banks this round's gain."
            )
            moves.append(voluntary_note)
        carry_note = None
        if plan.carried_next:
            carry_note = (
                f"Note: {plan.carried_next} unused transfer carries over to round 3."
            )
            moves.append(carry_note)
        transfers_info = {
            "out": [str(names.get(i, i)) for i in plan.out_ids],
            "in": [str(names.get(i, i)) for i in plan.in_ids],
            "out_ids": [int(i) for i in plan.out_ids],
            "in_ids": [int(i) for i in plan.in_ids],
            "free_allowance": plan.free_allowance,
            "unlimited": plan.free_allowance is None,
            "n_paid": plan.n_paid,
            "n_forced_paid": plan.n_forced_paid,
            "n_voluntary_paid": plan.n_voluntary_paid,
            "nextround_gain": round(float(plan.nextround_gain), 2),
            "penalty_points": round(float(plan.penalty_points), 1),
            "carried_next": plan.carried_next,
            "forced_note": forced_note,
            "voluntary_note": voluntary_note,
            "carry_note": carry_note,
        }

    # Option-value captaincy: the armband is switchable live (completed ->
    # not-yet-played), so captaining the EARLIEST strong starter first and
    # rolling it to a later, higher-EP one if he disappoints banks the better
    # of the two - worth more than captaining the top-EP starter outright even
    # at a meaningful EP gap. captain_timing_note explains it below.
    solution.captain_id = captain_mod.recommend_captain(df, solution, rounds, rnd).initial

    names = df.set_index("id")["name"]
    vice_id = captain_mod.vice_captain(df, solution)
    bench_order, bench_gk = captain_mod.ordered_bench(df, solution)
    squad_rows = _squad_rows(df, solution, vice_id, bench_order, bench_gk)
    timing = captain_mod.captain_timing_note(df, solution, rounds, rnd)

    doubtful = df[
        df["id"].isin(solution.squad_ids) & (df["p_play"] < 0.75)
    ].sort_values("p_play")
    availability_watch = [
        {
            "name": r["name"],
            "team": r["squadName"],
            "p_play": round(float(r["p_play"]), 4),
        }
        for _, r in doubtful.iterrows()
    ]

    unverified = _unverified_starter_lines(df, solution)
    comparison = squad_comparison(df, solution, team.player_ids) if comparison_mode else None
    booster = [] if comparison_mode else captain_mod.booster_advice(df, team, solution, rnd)
    schedule = [] if comparison_mode else fixture_schedule(df, solution, vice_id, rounds, rnd)

    bank = round(budget - solution.cost, 2)
    if comparison_mode:
        mode = "from_scratch_comparison"
    elif not team.player_ids:
        mode = "initial"
    elif applied_state == "advanced":
        mode = "preview_next"
    elif applied_state == "complete":
        mode = "complete"
    else:
        mode = "transfer"
    # A preview (next-round look-ahead) and a "complete" final are not applyable:
    # the open round is already committed, so they must not overwrite my_team.json.
    applyable = not comparison_mode and applied_state not in ("advanced", "complete")
    plan_obj = AdvicePlan(
        round=rnd,
        budget=budget,
        country_cap=country_cap,
        mode=mode,
        applyable=applyable,
        cost=round(float(solution.cost), 2),
        bank=bank,
        status={
            "feeds_changed": feeds_changed,
            "freshness": freshness,
            "country_rule": country_rule,
            "header": header,
            "transfer_window": transfer_window,
            "applied_state": applied_state,
            "applied_note": applied_note,
            **notes,
        },
        squad_rows=squad_rows,
        captain={"id": int(solution.captain_id), "name": str(names[solution.captain_id])},
        vice={"id": int(vice_id), "name": str(names[vice_id])},
        captain_timing_note=timing,
        bench_order=[{"id": int(i), "name": str(names[i])} for i in bench_order],
        bench_gk={"id": int(bench_gk), "name": str(names[bench_gk])},
        transfers=transfers_info,
        moves=moves,
        availability_watch=availability_watch,
        unverified_starters=unverified,
        booster_advice=booster,
        schedule=schedule,
        comparison=comparison,
    )

    if plan_obj.applyable:
        LAST_ADVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_ADVICE_FILE.write_text(json.dumps({
            "round": rnd,
            "squad_ids": solution.squad_ids,
            "starter_ids": solution.starter_ids,
            "captain_id": solution.captain_id,
            "vice_captain_id": vice_id,
            "bench_order": bench_order,
            "bench_gk_id": bench_gk,
            "bank": bank,
            "transfers_carried": carried_next,
        }, indent=2), encoding="utf-8")
    return plan_obj


def _print_advice(plan: AdvicePlan) -> None:
    """Render an AdvicePlan to the console (the CLI presentation of `advise`)."""
    s = plan.status
    comparison_mode = plan.mode == "from_scratch_comparison"

    console.print("[bold]Refreshing FIFA feeds...[/bold]")
    console.print(
        f"Feeds updated: {s['feeds_changed'] or 'none (cache hit)'}"
    )
    console.print(s["freshness"])
    console.print(
        f"Planning for round [bold]{plan.round}[/bold], budget ${plan.budget:.1f}m\n"
    )
    console.print(s["model_line"] + "\n")
    console.print(s["country_rule"])

    if s.get("header"):
        console.print(f"[bold]{s['header']}[/bold]")
    if s.get("applied_note"):
        console.print(f"\n[bold yellow]{s['applied_note']}[/bold yellow]")
    if s.get("transfer_window"):
        console.print(s["transfer_window"])

    console.print(_squad_table(plan.squad_rows))
    console.print(f"\nTotal cost: ${plan.cost:.1f}m (${plan.bank:.1f}m left)")

    if plan.availability_watch:
        console.print("\n[bold]Availability watch (P(plays) below 75%):[/bold]")
        for r in plan.availability_watch:
            console.print(
                f"  - {r['name']} ({r['team']}): {r['p_play'] * 100:.0f}% "
                f"likely to play"
            )

    if plan.unverified_starters:
        console.print(f"\n[bold]{plan.unverified_starters[0]}[/bold]")
        for line in plan.unverified_starters[1:]:
            console.print(line)

    if comparison_mode and plan.comparison is not None:
        console.print("\n[bold]Your squad vs the ideal:[/bold]")
        for line in plan.comparison:
            console.print(f"  - {line}")

    header = "Ideal lineup (reference):" if comparison_mode else "Moves to execute in the game:"
    console.print(f"\n[bold]{header}[/bold]")
    for m in plan.moves:
        console.print(f"  - {m}")
    console.print(f"  - Captain: [bold]{plan.captain['name']}[/bold]")
    console.print(f"  - Vice-captain: [bold]{plan.vice['name']}[/bold]")
    if plan.captain_timing_note:
        console.print(f"  - {plan.captain_timing_note}")
    subs = ", ".join(f"{k+1}) {b['name']}" for k, b in enumerate(plan.bench_order))
    console.print(f"  - Bench order: {subs} | GK: {plan.bench_gk['name']}")

    if comparison_mode:
        console.print(
            "\nReference only: not saved. Run [bold]python -m wcfantasy advise[/bold] "
            "(no flag) for the applyable plan within your transfer limits."
        )
        return

    for tip in plan.booster_advice:
        console.print(f"  - Booster: {tip}")

    if plan.schedule:
        console.print("\n[bold]Your players' matches (Italian time):[/bold]")
        for line in plan.schedule:
            console.print(f"  - {line}")
        console.print(
            "After each of these matches, run [bold]python -m wcfantasy live[/bold] "
            "to re-check captain and substitutions."
        )

    if plan.applyable:
        console.print("\nWhen you have applied these moves in the game, run: "
                      "[bold]python -m wcfantasy team apply[/bold]")
    elif plan.mode == "preview_next":
        console.print(
            "\nPreview only — not saved. It becomes the applyable plan once the "
            "open round is played and its transfer window opens."
        )


def run(from_scratch: bool = False) -> None:
    _print_advice(compute_advice(from_scratch=from_scratch, log=console.print))


def apply_last_advice_result(force: bool = False) -> dict:
    """Record last_advice.json into my_team.json; structured result, no print.

    Mirrors `apply_last_advice` exactly (same live-round footgun guard) but
    returns the outcome instead of printing, so both the CLI and the web
    `apply` endpoint render the SAME messages. `status` is 'no_advice',
    'blocked' (a future-round plan would clobber a live round's lineup; pass
    force=True to override) or 'applied'.
    """
    if not LAST_ADVICE_FILE.exists():
        return {
            "status": "no_advice",
            "message": "No advice to apply: run `advise` first.",
        }
    advice = json.loads(LAST_ADVICE_FILE.read_text(encoding="utf-8"))

    if not force:
        # Footgun guard: recording a future-round plan while an earlier
        # round is still being played overwrites my_team.json, so `live`
        # would then read the wrong XI for the round on the pitch. Block
        # by default; `team apply --force` overrides.
        from . import live as live_mod

        try:
            live_rnd = live_mod.current_live_round(fifa_client.load("rounds"))
        except Exception:
            live_rnd = None
        live_id = live_rnd["id"] if live_rnd else None
        if applying_over_live_round(live_id, advice["round"]):
            return {
                "status": "blocked",
                "live_round": live_id,
                "advice_round": advice["round"],
                "message": (
                    f"Not applying. Round {live_id} is still live, but this plan "
                    f"is for round {advice['round']}. Recording it now would make "
                    f"`live` read the wrong lineup for round {live_id}. Apply after "
                    f"round {live_id} ends, or force it to override."
                ),
            }

    team = state.load()
    team.player_ids = advice["squad_ids"]
    team.starters = advice["starter_ids"]
    team.captain_id = advice["captain_id"]
    team.vice_captain_id = advice.get("vice_captain_id")
    team.bench_order = advice.get("bench_order", [])
    team.bench_gk_id = advice.get("bench_gk_id")
    team.bank = advice["bank"]
    team.round_no = advice["round"]
    team.transfers_carried = advice.get("transfers_carried", 0)
    state.save(team)
    return {
        "status": "applied",
        "round": advice["round"],
        "message": f"Saved: team registered for round {advice['round']}.",
    }


def apply_last_advice(force: bool = False) -> None:
    """CLI wrapper over apply_last_advice_result (prints the outcome)."""
    result = apply_last_advice_result(force=force)
    if result["status"] == "blocked":
        console.print(
            f"[bold yellow]Not applying.[/bold yellow] Round {result['live_round']} "
            f"is still live, but this plan is for round {result['advice_round']}. "
            "Recording it now would make `live` read the wrong lineup for round "
            f"{result['live_round']}. Apply after round {result['live_round']} ends, "
            "or run [bold]team apply --force[/bold] to override."
        )
    else:
        console.print(result["message"])


def use_booster_result(name: str | None) -> dict:
    """Mark a booster as used; structured result, no print.

    `advise` can only recommend a booster, never observe which one you
    actually confirmed (and most are reversible until lockout), so the
    user marks it here. Once recorded it is no longer suggested and the
    one-per-tournament rule is respected (rules.BOOSTERS). `status` is
    'usage' (no/unknown name), 'already' or 'recorded'.
    """
    available = ", ".join(rules.BOOSTERS)
    if not name:
        return {
            "status": "usage",
            "message": f"Usage: team use-booster <name>. Available: {available}",
            "available": list(rules.BOOSTERS),
        }
    key = name.strip().lower().replace("-", "_")
    if key not in rules.BOOSTERS:
        return {
            "status": "usage",
            "message": f"Unknown booster {name!r}. Available: {available}",
            "available": list(rules.BOOSTERS),
        }
    team = state.load()
    if key in team.boosters_used:
        return {
            "status": "already",
            "booster": key,
            "message": f"Booster '{key}' was already marked as used.",
            "boosters_used": list(team.boosters_used),
        }
    team.boosters_used.append(key)
    state.save(team)
    return {
        "status": "recorded",
        "booster": key,
        "message": f"Recorded: '{key}' marked as used; it won't be suggested again.",
        "boosters_used": list(team.boosters_used),
    }


def use_booster(name: str | None) -> None:
    """CLI wrapper over use_booster_result (prints the outcome)."""
    console.print(use_booster_result(name)["message"])


def show_team() -> None:
    team = state.load()
    if not team.player_ids:
        console.print("No team saved yet.")
        return
    players_raw = fifa_client.load("players")
    squads = fifa_client.load("squads")
    names = features.player_table(players_raw, squads).set_index("id")
    console.print(f"Team as of round {team.round_no}, bank ${team.bank:.1f}m")
    for pid in team.player_ids:
        row = names.loc[pid]
        marks = []
        if pid in team.starters:
            marks.append("XI")
        if pid == team.captain_id:
            marks.append("C")
        if pid == team.vice_captain_id:
            marks.append("VC")
        if pid in team.bench_order:
            marks.append(f"sub{team.bench_order.index(pid) + 1}")
        if pid == team.bench_gk_id:
            marks.append("subGK")
        console.print(f"  {row['position']:>3} {row['name']:<28} "
                      f"{row['squadName']:<15} {' '.join(marks)}")
    if team.transfers_carried:
        console.print(f"Carried-over transfers: {team.transfers_carried}")
    if team.boosters_used:
        console.print(f"Boosters used: {', '.join(team.boosters_used)}")


def compute_odds(*, log: Callable[[str], None] | None = None) -> dict:
    """Championship odds as structured data (no console output).

    Runs the same Monte Carlo as `advise` (market-calibrated when odds are
    available): each team's P(win), the fair decimal odds (1/p) and, when a
    bookmaker key is set, the market's implied probability. Refreshes the
    feeds first so the numbers reflect the last played match. `log` receives
    progress lines.
    """
    emit = log or (lambda _msg: None)

    emit("Refreshing FIFA feeds...")
    fifa_client.refresh()
    rounds = fifa_client.load("rounds")
    squads = fifa_client.load("squads")
    emit(fifa_client.freshness_report(rounds))

    champ = progression.championship_probabilities(squads, rounds)
    if not champ:
        return {
            "available": False,
            "message": (
                "Championship odds unavailable: the Monte Carlo model could not "
                "run (check the historical-results download / network)."
            ),
            "rows": [],
            "market_on": False,
            "model": progression.last_run_info.get("model", "?"),
        }

    market = {}
    try:
        market = odds.outright_probabilities(squads) or {}
    except Exception:
        pass

    names = {s["id"]: s["name"] for s in squads}
    model = progression.last_run_info.get("model", "?")
    emit(f"Model: {model}")
    rows = []
    for sid, p in sorted(champ.items(), key=lambda kv: kv[1], reverse=True):
        if p <= 0:
            continue
        m = market.get(sid)
        rows.append({
            "team": names.get(sid, str(sid)),
            "p_win": round(float(p), 5),
            "fair_odds": round(1.0 / p, 2),
            "market": round(float(m), 5) if m else None,
        })
    return {
        "available": True,
        "message": None,
        "model": model,
        "market_on": bool(market),
        "rows": rows,
    }


def show_odds() -> None:
    """CLI wrapper over compute_odds (prints the championship-odds table)."""
    result = compute_odds(log=console.print)
    if not result["available"]:
        console.print(result["message"])
        return
    console.print(f"Model: {result['model']}\n")
    table = Table(title="Championship odds (your model)")
    table.add_column("Team")
    table.add_column("P(win)", justify="right")
    table.add_column("Fair odds", justify="right")
    table.add_column("Market", justify="right")
    for r in result["rows"]:
        table.add_row(
            r["team"],
            f"{r['p_win'] * 100:.1f}%",
            f"{r['fair_odds']:.1f}",
            f"{r['market'] * 100:.1f}%" if r["market"] else "-",
        )
    console.print(table)
    if not result["market_on"]:
        console.print(
            "\n(Market column off: set ODDS_API_KEY for the bookmaker comparison.)"
        )
