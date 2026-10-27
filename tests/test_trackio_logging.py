# Smoke tests for the TrackioRun wrapper. We do NOT exercise the real trackio
# SDK (it spawns a Gradio server); we monkeypatch it with a recording stub.
from __future__ import annotations

import sys
import types

import pytest

from train_logging import TrackioRun  # type: ignore[import-not-found]


class _Stub:
    """Minimal trackio API surface, recording every call."""
    def __init__(self):
        self.calls = []
    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
    def log(self, metrics, step=None):
        self.calls.append(("log", dict(metrics), step))
    def finish(self):
        self.calls.append(("finish",))


@pytest.fixture
def stub_trackio(monkeypatch):
    stub = _Stub()
    fake = types.ModuleType("trackio")
    fake.init = stub.init  # type: ignore[attr-defined]
    fake.log = stub.log  # type: ignore[attr-defined]
    fake.finish = stub.finish  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trackio", fake)
    return stub


def test_disabled_run_is_noop(stub_trackio):
    run = TrackioRun(project="p", name="r", config={"a": 1}, disable=True)
    run.log({"x": 1.0}, step=1)
    run.log_histogram("h", [1.0, 2.0], step=1)
    run.finish()
    assert stub_trackio.calls == [], "disabled run must not touch the SDK"


def test_enabled_run_logs_scalars_and_histograms(stub_trackio):
    run = TrackioRun(project="p", name="r", config={"a": 1})
    run.log({"loss": 0.5}, step=10)
    run.log_histogram("hist", [0.1, 0.2, 0.3], step=10)
    run.finish()
    kinds = [c[0] for c in stub_trackio.calls]
    assert kinds == ["init", "log", "log", "finish"]
    init_kwargs = stub_trackio.calls[0][1]
    assert init_kwargs["project"] == "p"
    assert init_kwargs["name"] == "r"
    assert init_kwargs["config"] == {"a": 1}


def test_missing_trackio_disables_silently(monkeypatch):
    # Force ImportError by removing the module if cached.
    monkeypatch.setitem(sys.modules, "trackio", None)
    run = TrackioRun(project="p", name="r", config={})
    # Must not raise even when trackio is absent.
    run.log({"x": 1.0}, step=1)
    run.finish()
    assert run.enabled is False
