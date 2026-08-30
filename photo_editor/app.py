"""Application entry point — initialises Qt and launches the main window."""

import os
import sys

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from photo_editor.ui.main_window import MainWindow


def _blend_cool_desaturated(image: QImage, amount: float) -> QImage:
    """Return a copy of ``image`` blended from a cool tint to full colour."""
    import numpy as np

    amount = max(0.0, min(1.0, amount))
    source = image.convertToFormat(QImage.Format.Format_ARGB32)

    ptr = source.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((source.height(), source.width(), 4)).copy()

    # ARGB32 channels: [B, G, R, A]
    b = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    r = arr[:, :, 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    cool_r = gray * 0.84
    cool_g = gray * 0.9
    cool_b = np.clip(gray * 1.05 + 10, 0, 255)

    arr[:, :, 0] = np.clip(cool_b + (b - cool_b) * amount, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(cool_g + (g - cool_g) * amount, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(cool_r + (r - cool_r) * amount, 0, 255).astype(np.uint8)

    result = QImage(arr.tobytes(), source.width(), source.height(), QImage.Format.Format_ARGB32)
    return result.copy()


def _show_animated_splash(app: QApplication, splash_path: str) -> QSplashScreen | None:
    """Show the splash screen, animating it from grey to colour.

    The animation runs on a timer alongside window construction rather than
    blocking in a nested event loop first. Blocking cost a *fixed* one
    second of startup -- more than half of it -- and, worse, it was time in
    which nothing else happened: the window was not built until the last
    frame had been drawn. Now the two overlap, so the splash is a view of
    real startup progress rather than a delay pretending to be one.
    """
    if not os.path.exists(splash_path):
        return None

    pixmap = QPixmap(splash_path)
    if pixmap.isNull():
        return None

    pixmap = pixmap.scaledToWidth(700, Qt.TransformationMode.SmoothTransformation)
    base_image = pixmap.toImage()
    splash = QSplashScreen(
        QPixmap.fromImage(_blend_cool_desaturated(base_image, 0.0)),
        Qt.WindowType.WindowStaysOnTopHint,
    )
    splash.show()

    screen = splash.screen() or QGuiApplication.primaryScreen()
    if screen is not None:
        splash.move(
            (screen.geometry().width() - splash.width()) // 2,
            (screen.geometry().height() - splash.height()) // 2,
        )

    app.processEvents()

    frame_count = 10
    frame_interval_ms = 40

    def update_frame(frame: int) -> None:
        # The splash may already have been finished by the window appearing;
        # setting a pixmap on a closed splash is harmless but pointless.
        if not splash.isVisible():
            return
        splash.setPixmap(
            QPixmap.fromImage(_blend_cool_desaturated(base_image, frame / frame_count))
        )

    for frame in range(1, frame_count + 1):
        QTimer.singleShot(frame * frame_interval_ms,
                          lambda f=frame: update_frame(f))
    return splash


def run() -> int:
    """Create the QApplication, show the main window, and enter the event loop."""
    # On Windows, set the AppUserModelID so the taskbar displays the custom icon
    # instead of the default Python executable icon.
    if sys.platform == "win32":
        import ctypes
        myappid = "baseraproject.basera.app.1.0"
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except AttributeError:
            pass  # Fallback for very old Windows versions

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Basera")
    app.setOrganizationName("BaseraProject")
    
    icon_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets", "app", "logo.png"
    )
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets", "app", "splash.png"
    )
    splash = _show_animated_splash(app, splash_path)

    window = MainWindow()
    window.show()

    if splash:
        splash.finish(window)

    # Offer any work a previous session left behind, once the window is up
    # so the dialog has a parent to centre on.
    QTimer.singleShot(0, window.offer_recovery)

    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
