"""The centralized feed-status vocabulary and the round-transition guards."""

from wcfantasy import advise, feed_status, live


def test_done_live_and_locked():
    assert feed_status.is_done("complete") and feed_status.is_done("COMPLETED")
    assert feed_status.is_live("playing") and feed_status.is_live("live")
    assert not feed_status.is_done("scheduled")
    assert not feed_status.is_live("scheduled")
    # locked = done or live; only a not-yet-started round is editable
    assert feed_status.is_locked("complete") and feed_status.is_locked("playing")
    assert not feed_status.is_locked("scheduled")
    assert not feed_status.is_locked(None)


def test_result_usable_matches_the_old_literal_check():
    # progression applies a score for anything that is not None / scheduled
    assert feed_status.result_usable("complete")
    assert feed_status.result_usable("postponed")  # preserved historical edge
    assert not feed_status.result_usable("scheduled")
    assert not feed_status.result_usable(None)


def test_lineup_set_for_future_round():
    assert live.lineup_set_for_future_round(5, 4) is True   # team set for r5, r4 live
    assert live.lineup_set_for_future_round(4, 4) is False
    assert live.lineup_set_for_future_round(0, 4) is False  # no team set yet


def test_applying_over_live_round():
    assert advise.applying_over_live_round(4, 5) is True    # r4 live, applying r5
    assert advise.applying_over_live_round(5, 5) is False
    assert advise.applying_over_live_round(None, 5) is False
