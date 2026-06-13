"""Build the player table used by models and optimizer."""

import pandas as pd


def player_table(players: list[dict], squads: list[dict]) -> pd.DataFrame:
    squad_df = pd.DataFrame(squads).rename(
        columns={"id": "squadId", "name": "squadName"}
    )
    rows = []
    for p in players:
        stats = p.get("stats") or {}
        rows.append(
            {
                "id": p["id"],
                "name": p.get("knownName")
                or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                "squadId": p["squadId"],
                "position": p["position"],
                "price": p["price"],
                "status": p["status"],
                "percentSelected": p.get("percentSelected") or 0.0,
                "totalPoints": stats.get("totalPoints") or 0,
                "avgPoints": stats.get("avgPoints") or 0.0,
                "form": stats.get("form") or 0.0,
                "matchesPlayed": len(stats.get("roundPoints") or []),
            }
        )
    df = pd.DataFrame(rows)
    return df.merge(
        squad_df[["squadId", "squadName", "group", "isEliminated"]],
        on="squadId",
        how="left",
    )
