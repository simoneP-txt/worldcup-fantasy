"""Offline tests for the HTTP retry helper (no real network)."""

import pytest
import requests

from wcfantasy import fifa_client


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"status {self.status_code}")

    @property
    def content(self):
        return b"ok"

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(fifa_client.time, "sleep", lambda *_: None)


def _sequence(monkeypatch, responses):
    """Feed requests.get a queue of responses or exceptions to raise."""
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        item = responses[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(fifa_client.requests, "get", fake_get)
    return calls


def test_returns_first_200(monkeypatch):
    calls = _sequence(monkeypatch, [_Resp(200)])
    assert fifa_client._request_with_retry("http://x").status_code == 200
    assert calls["n"] == 1


def test_retries_transport_error_then_succeeds(monkeypatch):
    calls = _sequence(monkeypatch, [requests.ConnectionError("down"), _Resp(200)])
    assert fifa_client._request_with_retry("http://x").status_code == 200
    assert calls["n"] == 2  # retried after the transport error


def test_retries_non_200_then_succeeds(monkeypatch):
    _sequence(monkeypatch, [_Resp(503), _Resp(200)])
    assert fifa_client._request_with_retry("http://x").status_code == 200


def test_all_transport_errors_raise_runtimeerror(monkeypatch):
    _sequence(monkeypatch, [requests.ConnectionError("x")] * 3)
    with pytest.raises(RuntimeError):
        fifa_client._request_with_retry("http://x")


def test_persistent_non_200_raises_http_error(monkeypatch):
    _sequence(monkeypatch, [_Resp(403)] * 3)
    with pytest.raises(requests.HTTPError):
        fifa_client._request_with_retry("http://x")


def test_fifa_api_get_returns_json(monkeypatch):
    _sequence(monkeypatch, [_Resp(200, payload={"Results": [1, 2]})])
    assert fifa_client.fifa_api_get("/calendar") == {"Results": [1, 2]}
