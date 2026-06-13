import pandas as pd

from wcfantasy.models import elo


def _history():
    # A mostly beats B, B mostly beats C: expect rating A > B > C.
    # Mix in draws and upsets so the outcome model sees all 3 classes.
    rows = []
    for i in range(20):
        rows.append(("A", "B", 2, 0) if i % 5 else ("A", "B", 1, 1))
        rows.append(("B", "C", 1, 0) if i % 4 else ("B", "C", 0, 1))
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "home_team": h, "away_team": a,
                "home_score": hs, "away_score": as_,
                "tournament": "FIFA World Cup qualification",
                "neutral": True,
            }
            for i, (h, a, hs, as_) in enumerate(rows)
        ]
    )


def test_ratings_order_and_outcome_probs():
    model = elo.fit(_history())
    assert model.rating("A") > model.rating("B") > model.rating("C")

    p_win, p_draw, p_loss = model.outcome_probs("A", "C")
    assert p_win > p_loss
    assert abs(p_win + p_draw + p_loss - 1) < 1e-9


def test_update_moves_ratings():
    model = elo.fit(_history())
    before = model.rating("C")
    model.update("C", "A", 3, 0)  # upset: C gains
    assert model.rating("C") > before
