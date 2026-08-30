"""Memory budgets must scale with the machine and stay within their bounds."""

from __future__ import annotations

import os

import pytest

from photo_editor.core import memory_budget as mb
from photo_editor.core.memory_budget import GB, MB


def test_budgets_are_within_their_bounds():
    cache = mb.render_cache_budget()
    hist = mb.history_budget()
    assert mb._RENDER_CACHE_MIN <= cache <= mb._RENDER_CACHE_MAX
    assert mb._HISTORY_MIN <= hist <= mb._HISTORY_MAX


def test_budgets_leave_room_for_the_document():
    """Together they must stay well under half of RAM -- the layer data,
    the OS and everything else still need room."""
    total = mb.total_system_memory()
    combined = mb.render_cache_budget() + mb.history_budget()
    assert combined < total * 0.5, (
        f"budgets claim {combined / total:.0%} of system memory")


@pytest.mark.parametrize("total_gb,expect_min_mb", [
    (4, 192), (8, 192), (16, 192), (64, 192),
])
def test_small_machines_still_get_a_usable_cache(monkeypatch, total_gb,
                                                 expect_min_mb):
    monkeypatch.setattr(mb, "total_system_memory", lambda: total_gb * GB)
    assert mb.render_cache_budget() >= expect_min_mb * MB


def test_large_machines_do_not_hoard(monkeypatch):
    monkeypatch.setattr(mb, "total_system_memory", lambda: 512 * GB)
    assert mb.render_cache_budget() <= mb._RENDER_CACHE_MAX
    assert mb.history_budget() <= mb._HISTORY_MAX


def test_budget_scales_with_memory(monkeypatch):
    monkeypatch.setattr(mb, "total_system_memory", lambda: 8 * GB)
    small = mb.render_cache_budget()
    monkeypatch.setattr(mb, "total_system_memory", lambda: 32 * GB)
    large = mb.render_cache_budget()
    assert large > small


def test_environment_override(monkeypatch):
    monkeypatch.setenv("BASERA_RENDER_CACHE_MB", "256")
    assert mb.render_cache_budget() == 256 * MB
    monkeypatch.setenv("BASERA_HISTORY_MB", "64")
    assert mb.history_budget() == 64 * MB


def test_invalid_override_falls_back(monkeypatch):
    monkeypatch.setenv("BASERA_RENDER_CACHE_MB", "not-a-number")
    assert mb.render_cache_budget() > 0


def test_total_system_memory_is_plausible():
    total = mb.total_system_memory()
    assert 512 * MB <= total <= 4096 * GB


def test_pipeline_and_history_use_the_budgets():
    from photo_editor.core.document import Document
    from photo_editor.engine.render_pipeline import RenderPipeline

    pipe = RenderPipeline()
    assert pipe.layer_cache._budget == mb.render_cache_budget()
    pipe.shutdown()

    doc = Document(16, 16)
    assert doc.history._budget == mb.history_budget()


def test_describe_reports_all_three():
    info = mb.describe()
    assert {"system_mb", "render_cache_mb", "history_mb"} == set(info)
    assert all(v > 0 for v in info.values())
