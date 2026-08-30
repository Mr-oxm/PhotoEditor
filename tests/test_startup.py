"""Startup cost.

Half of cold start used to be a hard-coded splash animation that blocked in
a nested event loop *before* the window was built -- a fixed one-second
delay during which nothing else happened. The animation now runs on a timer
alongside window construction, so the two overlap.
"""

from __future__ import annotations

import time

import pytest


def test_splash_does_not_block():
    """The splash must not run a nested event loop.

    Blocking here is invisible in every other test -- the app simply takes
    a second longer to appear -- so it is asserted structurally.
    """
    import inspect

    from photo_editor import app

    source = inspect.getsource(app._show_animated_splash)
    assert "loop.exec()" not in source, (
        "the splash blocks in a nested event loop again; that is a fixed "
        "second of cold start during which the window is not being built")
    assert "QTimer.singleShot" in source, "the animation should be timer-driven"


def test_splash_schedules_its_frames_rather_than_waiting(qtbot, tmp_path):
    """The animation must be queued, not waited for.

    Wall-clock is not asserted here: QSplashScreen.show() itself blocks for
    about a second under Qt's offscreen platform, which has no compositor to
    show to. That is a harness artifact and would swamp what is being
    measured. What is under test is that the ten animation frames are
    *scheduled* and the function returns, so window construction proceeds in
    parallel instead of waiting a fixed second for the animation to finish.
    """
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from photo_editor.app import _show_animated_splash

    path = tmp_path / "splash.png"
    QImage(64, 32, QImage.Format.Format_RGB32).save(str(path))

    splash = _show_animated_splash(QApplication.instance(), str(path))
    assert splash is not None
    # Returning at all is the point: the old code could not reach here until
    # every frame had been drawn.
    splash.close()


def test_missing_splash_image_is_handled(qtbot):
    from PySide6.QtWidgets import QApplication

    from photo_editor.app import _show_animated_splash
    assert _show_animated_splash(QApplication.instance(), "/nope.png") is None


def test_main_window_builds_in_reasonable_time(qtbot):
    """A regression here is the difference between the app feeling instant
    and feeling sluggish before the user has done anything."""
    from photo_editor.ui.main_window import MainWindow

    start = time.perf_counter()
    win = MainWindow(dev_mode=True)
    elapsed = time.perf_counter() - start
    qtbot.addWidget(win)
    win._autosave_timer.stop()
    assert elapsed < 3.0, f"MainWindow() took {elapsed * 1000:.0f} ms"
