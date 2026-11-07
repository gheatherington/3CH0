"""Opt-in live check against real echo360.ca (``pytest -m live``).

Needs a signed-in profile (``3ch0 auth login``) and ``THREECH0_LIVE_URL`` set to a
classroom URL you can view. Skipped by default (pyproject addopts: ``-m 'not live'``).
"""

from __future__ import annotations

import os

import pytest

from threech0.echo360.discovery import discover
from threech0.echo360.urls import parse_classroom_url
from threech0.models import AccessState, DiscoveryOptions, NullProgress


@pytest.mark.live
def test_live_discovery(tmp_path, monkeypatch):
    url = os.environ.get("THREECH0_LIVE_URL")
    if not url:
        pytest.skip("set THREECH0_LIVE_URL to a classroom URL")
    monkeypatch.delenv("THREECH0_HOME", raising=False)  # use the real signed-in profile
    trace = tmp_path / "trace.json"
    recording, session = discover(parse_classroom_url(url),
                                  DiscoveryOptions(interactive=False, save_trace=trace),
                                  NullProgress())
    try:
        assert any(s.access is AccessState.AVAILABLE for s in recording.sources)
        assert trace.exists()
    finally:
        session.close()
