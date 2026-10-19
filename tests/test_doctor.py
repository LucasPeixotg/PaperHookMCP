"""`paperhook-mcp --doctor` checks — no network, no model.

Pins that the cache check writes to the real directory and reports a skipped check
(not a failure) when the cache is off; that the OpenAlex check names the failure an
HTTP status or a transport error means, and never prints the API token; and that
`run` keeps going past a check that crashes and exits 1 if anything failed. The
reranker and stdio checks load the real model, so they are left to a manual run.
"""

import os

import httpx
import pytest

from paperhook import doctor
from paperhook.config import settings


@pytest.fixture
def cache_on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "CACHE_DIR", tmp_path / "paper_cache")
    return tmp_path / "paper_cache"


def test_a_writable_cache_dir_passes_and_is_created(cache_on):
    result = doctor.check_cache_dir()

    assert result.ok is True
    assert cache_on.is_dir()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_read_only_cache_dir_fails_with_a_hint(cache_on):
    cache_on.mkdir()
    cache_on.chmod(0o500)
    try:
        result = doctor.check_cache_dir()
    finally:
        cache_on.chmod(0o700)

    assert result.ok is False
    assert "CACHE_ENABLED=false" in result.hint


def test_a_disabled_cache_is_skipped_not_failed(monkeypatch):
    monkeypatch.setattr(settings, "CACHE_ENABLED", False)

    assert doctor.check_cache_dir().ok is None


def _openalex(monkeypatch, handler, token=""):
    monkeypatch.setattr(settings, "OPENALEX_API_TOKEN", token)
    return doctor.check_openalex(transport=httpx.MockTransport(handler))


def test_openalex_reachable_reports_the_key_status_not_the_key(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"results": []})

    result = _openalex(monkeypatch, handler, token="secret-token")

    assert result.ok is True
    assert seen["auth"] == "Bearer secret-token"
    assert "API key set" in result.render()
    assert "secret-token" not in result.render()


def test_openalex_without_a_key_reports_the_anonymous_budget(monkeypatch):
    result = _openalex(monkeypatch, lambda request: httpx.Response(200, json={}))

    assert result.ok is True
    assert "anonymous budget" in result.detail


def test_a_rejected_key_is_named(monkeypatch):
    result = _openalex(monkeypatch, lambda request: httpx.Response(403), token="bad")

    assert result.ok is False
    assert "OPENALEX_API_TOKEN" in result.label


def test_an_exhausted_budget_is_named(monkeypatch):
    result = _openalex(monkeypatch, lambda request: httpx.Response(429))

    assert result.ok is False
    assert "budget" in result.label


def test_a_blocked_network_fails_with_a_hint(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    result = _openalex(monkeypatch, handler)

    assert result.ok is False
    assert "unreachable" in result.label
    assert "firewall" in result.hint


def _stub(ok):
    def check():
        return doctor.CheckResult(ok, f"stub {ok}")
    return check


def test_run_exits_zero_when_nothing_fails(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "CHECKS", [_stub(True), _stub(None)])

    assert doctor.run() == 0
    assert capsys.readouterr().out.splitlines() == ["[✓] stub True", "[-] stub None"]


def test_run_exits_one_when_a_check_fails(monkeypatch):
    monkeypatch.setattr(doctor, "CHECKS", [_stub(True), _stub(False)])

    assert doctor.run() == 1


def test_a_crashing_check_fails_without_hiding_the_rest(monkeypatch, capsys):
    def explodes():
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor, "CHECKS", [explodes, _stub(True)])

    assert doctor.run() == 1
    out = capsys.readouterr().out
    assert "[✗] explodes crashed" in out
    assert "boom" in out
    assert "[✓] stub True" in out
