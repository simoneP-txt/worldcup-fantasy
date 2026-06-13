"""Elo ratings for national teams + a trained match-outcome model.

One pass over ~45k historical matches produces:
- current Elo per team (standard football Elo: K scaled by match
  importance and goal margin, +home advantage when not neutral);
- a training set of (elo_diff, outcome) pairs from recent competitive
  matches, used to fit a multinomial logistic regression that converts
  an Elo gap into win/draw/loss probabilities for the simulator.

World Cup 2026 matches come from the live FIFA feed (rounds.json), not
the dataset, so ratings stay current to the last played match.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_RATING = 1500.0
HOME_ADVANTAGE = 80.0
# Training window for the outcome model: modern, competitive era.
TRAIN_FROM = pd.Timestamp("1998-01-01")

_K_BY_TOURNAMENT = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "Friendly": 20.0,
}
_K_DEFAULT = 40.0  # continental cups and their qualifiers


def _k_factor(tournament: str, goal_diff: int) -> float:
    k = _K_BY_TOURNAMENT.get(tournament, _K_DEFAULT)
    if goal_diff == 2:
        k *= 1.5
    elif goal_diff >= 3:
        k *= 1.75 + (goal_diff - 3) / 8
    return k


def expected_score(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


@dataclass
class EloModel:
    ratings: dict[str, float]  # dataset team name -> rating
    outcome_model: LogisticRegression  # P(loss/draw/win) from elo_diff

    def rating(self, team: str) -> float:
        return self.ratings.get(team, BASE_RATING)

    def outcome_probs(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        """(p_home_win, p_draw, p_away_win)."""
        diff = self.rating(home) - self.rating(away)
        if not neutral:
            diff += HOME_ADVANTAGE
        proba = self.outcome_model.predict_proba([[diff]])[0]
        by_class = dict(zip(self.outcome_model.classes_, proba))
        return by_class[1], by_class[0], by_class[-1]

    def update(self, home: str, away: str, home_score: int, away_score: int,
               tournament: str = "FIFA World Cup", neutral: bool = True) -> None:
        _apply_match(self.ratings, home, away, home_score, away_score,
                     tournament, neutral)


def _apply_match(ratings, home, away, home_score, away_score, tournament, neutral):
    r_home = ratings.get(home, BASE_RATING)
    r_away = ratings.get(away, BASE_RATING)
    diff = r_home - r_away + (0.0 if neutral else HOME_ADVANTAGE)
    exp_home = expected_score(diff)
    actual = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
    k = _k_factor(tournament, abs(home_score - away_score))
    delta = k * (actual - exp_home)
    ratings[home] = r_home + delta
    ratings[away] = r_away - delta
    return diff, actual


def fit(history: pd.DataFrame) -> EloModel:
    """Replay history to build ratings and train the outcome model."""
    ratings: dict[str, float] = {}
    X, y = [], []
    for row in history.itertuples(index=False):
        diff, actual = _apply_match(
            ratings, row.home_team, row.away_team,
            int(row.home_score), int(row.away_score),
            row.tournament, bool(row.neutral),
        )
        if row.date >= TRAIN_FROM and row.tournament != "Friendly":
            X.append([diff])
            y.append(1 if actual == 1.0 else 0 if actual == 0.5 else -1)

    model = LogisticRegression(max_iter=1000)
    model.fit(np.array(X), np.array(y))
    return EloModel(ratings=ratings, outcome_model=model)
