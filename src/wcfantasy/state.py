"""Local record of MY fantasy team.

The game API is authenticated and out of scope, so the app cannot read
the team back from FIFA. The user applies the advised moves in the web
app, then confirms them here (`team apply`), keeping this file the
source of truth for the next round.
"""

import json
from dataclasses import asdict, dataclass, field, fields

from .config import MY_TEAM_FILE


@dataclass
class TeamState:
    player_ids: list[int] = field(default_factory=list)  # 15 ids
    starters: list[int] = field(default_factory=list)  # 11 ids
    captain_id: int | None = None
    vice_captain_id: int | None = None
    bench_order: list[int] = field(default_factory=list)  # 3 outfield subs, priority order
    bench_gk_id: int | None = None
    bank: float = 0.0  # unspent budget, $m
    boosters_used: list[str] = field(default_factory=list)
    round_no: int = 0  # last round the team was set for
    # Group stage only: unused transfer carried into the next window.
    transfers_carried: int = 0
    # Round where a manual live change was made (0 = none). A manual
    # change cancels auto-subs and the vice insurance for that round.
    manual_changes_round: int = 0


def load() -> TeamState:
    if not MY_TEAM_FILE.exists():
        return TeamState()
    data = json.loads(MY_TEAM_FILE.read_text(encoding="utf-8"))
    # Ignore keys from older schemas (e.g. a dropped field) so a stale
    # my_team.json still loads instead of raising.
    known = {f.name for f in fields(TeamState)}
    return TeamState(**{k: v for k, v in data.items() if k in known})


def save(state: TeamState) -> None:
    MY_TEAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    MY_TEAM_FILE.write_text(
        json.dumps(asdict(state), indent=2), encoding="utf-8"
    )
