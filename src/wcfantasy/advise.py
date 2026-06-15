"""Orchestrator: fetch fresh data -> predict -> optimize -> print the plan.

This is what `python -m wcfantasy advise` runs. It always refreshes the
FIFA feeds first so every recommendation reflects the last played match.
"""

import json
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.table import Table

from . import feed_status, features, fifa_client, fixtures, rules, state
from .config import STATE_DIR
from .data_sources import availability
from .models import player_points, progression
from .optimizer import captain as captain_mod
from .optimizer import transfers as transfers_mod
from .optimizer.squad import SquadSolution, build_squad

LAST_ADVICE_FILE = STATE_DIR / "last_advice.json"

console = Console()


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


def current_budget(next_round: int) -> float:
    bonus = rules.KNOCKOUT_BUDGET_BONUS if next_round >= 4 else 0.0
    return rules.BUDGET + bonus


def _squad_table(df, solution: SquadSolution) -> Table:
    table = Table(title="Recommended squad")
    for col in ("Pos", "Player", "Team", "Price", "EP/match", "Play%", "Role"):
        table.add_column(col)
    rows = df[df["id"].isin(solution.squad_ids)].copy()
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    rows = rows.sort_values(["position", "ep_match"], ascending=[True, False],
                            key=lambda s: s.map(pos_order) if s.name == "position" else s)
    vice_id = captain_mod.vice_captain(df, solution)
    bench_order, bench_gk = captain_mod.ordered_bench(df, solution)
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
        play = r["p_play"] if "p_play" in r else 1.0
        table.add_row(
            r["position"], r["name"], r["squadName"],
            f"${r['price']:.1f}m", f"{r['ep_match']:.2f}", f"{play * 100:.0f}%",
            role,
        )
    return table


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


def run(from_scratch: bool = False) -> None:
    console.print("[bold]Refreshing FIFA feeds...[/bold]")
    changed = fifa_client.refresh()
    player_points.apply_calibration()
    console.print(f"Feeds updated: {[k for k, v in changed.items() if v] or 'none (cache hit)'}")

    players_raw = fifa_client.load("players")
    rounds = fifa_client.load("rounds")
    squads = fifa_client.load("squads")

    console.print(fifa_client.freshness_report(rounds))
    rnd = next_round_number(rounds)
    budget = current_budget(rnd)
    console.print(f"Planning for round [bold]{rnd}[/bold], budget ${budget:.1f}m\n")

    df = features.player_table(players_raw, squads)
    df = player_points.expected_value(df, squads, rounds, rnd)

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
    console.print(
        f"Model: {info.get('model', '?')} | {odds_note} | {club_note} | "
        f"{avail_note} | {cal_note}\n"
    )

    # Official rule: hard cap on players per country, growing with the
    # stage (3 in the groups and R32, up to 8 at the Final).
    country_cap = rules.max_per_squad(rnd)
    console.print(f"Game rule: max {country_cap} players per nation this round")

    team = state.load()
    carried_next = 0
    # Ideal-squad view: build from scratch even with a team saved, to show
    # how far the unconstrained optimum is from the current squad. It is a
    # reference, never written to last_advice (you cannot rebuild freely
    # without the Wildcard), so `team apply` is not offered for it.
    comparison_mode = from_scratch and bool(team.player_ids)
    if from_scratch or not team.player_ids:
        if comparison_mode:
            console.print(
                "[bold]Ideal-squad view[/bold]: optimal 15 from scratch, ignoring "
                "your saved team and the transfer limits (a target, not an "
                "applyable plan)."
            )
        else:
            console.print("[bold]No saved team: building the initial squad.[/bold]")
        solution = build_squad(df, budget=budget, max_per_squad=country_cap)
        moves: list[str] = (
            [] if comparison_mode else ["Create your team with the 15 players below."]
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
        if plan.free_allowance is None:
            console.print(
                "Unlimited transfer window: every change below is free."
            )
        else:
            console.print(
                f"Free transfers this window: {plan.free_allowance}"
                + (f" (incl. {team.transfers_carried} carried over)"
                   if team.transfers_carried else "")
            )
        if plan.n_paid:
            moves.append(
                f"FORCED: {plan.n_paid} transfer(s) beyond the free "
                f"{plan.free_allowance}, -{plan.penalty_points:.0f} pts at lockout. "
                "Caused by unplayable players, not a voluntary upgrade: if the gain "
                "is clearly worth it, play the Wildcard instead of paying the penalty."
            )
        if plan.carried_next:
            moves.append(
                f"Note: {plan.carried_next} unused transfer carries over to round 3."
            )

    # Option-value tweak: among near-equal top starters, captain whoever
    # plays earliest (the armband is switchable live to a not-yet-played
    # contender, never the reverse). captain_timing_note explains it below.
    solution.captain_id = captain_mod.captain_by_timing(df, solution, rounds, rnd)

    console.print(_squad_table(df, solution))
    console.print(f"\nTotal cost: ${solution.cost:.1f}m "
                  f"(${budget - solution.cost:.1f}m left)")

    doubtful = df[
        df["id"].isin(solution.squad_ids) & (df["p_play"] < 0.75)
    ].sort_values("p_play")
    if not doubtful.empty:
        console.print("\n[bold]Availability watch (P(plays) below 75%):[/bold]")
        for _, r in doubtful.iterrows():
            console.print(
                f"  - {r['name']} ({r['squadName']}): {r['p_play'] * 100:.0f}% "
                f"likely to play"
            )

    if comparison_mode:
        console.print("\n[bold]Your squad vs the ideal:[/bold]")
        for line in squad_comparison(df, solution, team.player_ids):
            console.print(f"  - {line}")

    names = df.set_index("id")["name"]
    header = "Ideal lineup (reference):" if comparison_mode else "Moves to execute in the game:"
    console.print(f"\n[bold]{header}[/bold]")
    for m in moves:
        console.print(f"  - {m}")
    console.print(f"  - Captain: [bold]{names[solution.captain_id]}[/bold]")
    vice_id = captain_mod.vice_captain(df, solution)
    console.print(f"  - Vice-captain: [bold]{names[vice_id]}[/bold]")
    timing = captain_mod.captain_timing_note(df, solution, rounds, rnd)
    if timing:
        console.print(f"  - {timing}")
    bench_order, bench_gk = captain_mod.ordered_bench(df, solution)
    subs = ", ".join(f"{k+1}) {names[i]}" for k, i in enumerate(bench_order))
    console.print(f"  - Bench order: {subs} | GK: {names[bench_gk]}")

    if comparison_mode:
        console.print(
            "\nReference only: not saved. Run [bold]python -m wcfantasy advise[/bold] "
            "(no flag) for the applyable plan within your transfer limits."
        )
        return

    for tip in captain_mod.booster_advice(df, team, solution, rnd):
        console.print(f"  - Booster: {tip}")

    schedule = fixture_schedule(df, solution, vice_id, rounds, rnd)
    if schedule:
        console.print("\n[bold]Your players' matches (Italian time):[/bold]")
        for line in schedule:
            console.print(f"  - {line}")
        console.print(
            "After each of these matches, run [bold]python -m wcfantasy live[/bold] "
            "to re-check captain and substitutions."
        )

    LAST_ADVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_ADVICE_FILE.write_text(json.dumps({
        "round": rnd,
        "squad_ids": solution.squad_ids,
        "starter_ids": solution.starter_ids,
        "captain_id": solution.captain_id,
        "vice_captain_id": vice_id,
        "bench_order": bench_order,
        "bench_gk_id": bench_gk,
        "bank": round(budget - solution.cost, 2),
        "transfers_carried": carried_next,
    }, indent=2), encoding="utf-8")
    console.print("\nWhen you have applied these moves in the game, run: "
                  "[bold]python -m wcfantasy team apply[/bold]")


def apply_last_advice(force: bool = False) -> None:
    if not LAST_ADVICE_FILE.exists():
        console.print("No advice to apply: run `advise` first.")
        return
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
            console.print(
                f"[bold yellow]Not applying.[/bold yellow] Round {live_id} is "
                f"still live, but this plan is for round {advice['round']}. "
                "Recording it now would make `live` read the wrong lineup for "
                f"round {live_id}. Apply after round {live_id} ends, or run "
                "[bold]team apply --force[/bold] to override."
            )
            return

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
    console.print(f"Saved: team registered for round {advice['round']}.")


def use_booster(name: str | None) -> None:
    """Record that a booster was activated in the game this round.

    `advise` can only recommend a booster, never observe which one you
    actually confirmed (and most are reversible until lockout), so the
    user marks it here. Once recorded it is no longer suggested and the
    one-per-tournament rule is respected (rules.BOOSTERS).
    """
    available = ", ".join(rules.BOOSTERS)
    if not name:
        console.print(f"Usage: team use-booster <name>. Available: {available}")
        return
    key = name.strip().lower().replace("-", "_")
    if key not in rules.BOOSTERS:
        console.print(f"Unknown booster {name!r}. Available: {available}")
        return
    team = state.load()
    if key in team.boosters_used:
        console.print(f"Booster '{key}' was already marked as used.")
        return
    team.boosters_used.append(key)
    state.save(team)
    console.print(f"Recorded: '{key}' marked as used; it won't be suggested again.")


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
