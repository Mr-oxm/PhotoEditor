"""Selection management with marching-ants support."""

import numpy as np


class Selection:
    """Pixel-level selection mask in [0, 1] float space.

    Carries a ``version`` counter and caches its bounding box and
    non-emptiness. Both were previously recomputed from the full mask on
    every rendered frame -- ``mask.max()`` for the overlay, and two
    ``np.any(mask > 0.5, axis=...)`` reductions for the transform box. At
    4K that is four passes over 8.3 M elements, thirty times a second, to
    answer questions whose answer had not changed.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._mask: np.ndarray | None = None
        self._version: int = 0
        self._cache_version: int = -1
        self._cached_bounds: tuple[int, int, int, int] | None = None
        self._cached_nonempty: bool = False

    # ---- Change tracking ----------------------------------------------------

    @property
    def version(self) -> int:
        """Bumped on every change; UI can skip work when it is unchanged."""
        return self._version

    def touch(self) -> None:
        """Signal that the mask was modified in place."""
        self._version += 1

    def _set_mask(self, mask) -> None:
        self._mask = mask
        self._version += 1

    def _refresh_cache(self) -> None:
        if self._cache_version == self._version:
            return
        self._cache_version = self._version
        mask = self._mask
        if mask is None:
            self._cached_bounds = None
            self._cached_nonempty = False
            return
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        if rows.any() and cols.any():
            ys = np.flatnonzero(rows)
            xs = np.flatnonzero(cols)
            self._cached_bounds = (int(xs[0]), int(ys[0]),
                                   int(xs[-1]), int(ys[-1]))
            self._cached_nonempty = True
        else:
            self._cached_bounds = None
            self._cached_nonempty = bool(mask.max() > 0)

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) inclusive bounds of the selected area, cached."""
        self._refresh_cache()
        return self._cached_bounds

    @property
    def is_empty(self) -> bool:
        """True when the mask exists but selects nothing, cached."""
        if self._mask is None:
            return True
        self._refresh_cache()
        return not self._cached_nonempty

    @property
    def active(self) -> bool:
        return self._mask is not None

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    # ---- Whole-image ops ----------------------------------------------------

    def select_all(self) -> None:
        self._set_mask(np.ones((self.height, self.width), dtype=np.float32))

    def deselect(self) -> None:
        self._set_mask(None)

    def invert(self) -> None:
        if self._mask is not None:
            self._set_mask(1.0 - self._mask)

    # ---- Shape selections ---------------------------------------------------

    def select_rect(self, x: int, y: int, w: int, h: int) -> None:
        self._set_mask(np.zeros((self.height, self.width), dtype=np.float32))
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        self._mask[y1:y2, x1:x2] = 1.0

    def select_ellipse(self, cx: int, cy: int, rx: int, ry: int) -> None:
        self._set_mask(np.zeros((self.height, self.width), dtype=np.float32))
        yy, xx = np.ogrid[: self.height, : self.width]
        ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
        self._mask[ellipse <= 1.0] = 1.0

    # ---- Refinement ---------------------------------------------------------

    def feather(self, radius: int) -> None:
        if self._mask is not None and radius > 0:
            import cv2
            ksize = radius * 2 + 1
            self._set_mask(cv2.GaussianBlur(self._mask, (ksize, ksize), radius / 3.0))

    def grow(self, pixels: int) -> None:
        if self._mask is not None and pixels > 0:
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1,) * 2)
            self._set_mask(cv2.dilate(self._mask, k))

    def shrink(self, pixels: int) -> None:
        if self._mask is not None and pixels > 0:
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1,) * 2)
            self._set_mask(cv2.erode(self._mask, k))

    # ---- Application --------------------------------------------------------

    def apply_to(self, image: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return image
        if image.ndim == 3:
            return image * self._mask[..., np.newaxis]
        return image * self._mask

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        if self._mask is not None:
            import cv2
            self._set_mask(cv2.resize(self._mask, (width, height)))

    def translate(self, dx: int, dy: int) -> None:
        """Shift the selection mask by (dx, dy) pixels."""
        if self._mask is None or (dx == 0 and dy == 0):
            return
        new_mask = np.zeros_like(self._mask)
        h, w = self._mask.shape
        # source and destination slices
        sx0 = max(0, -dx)
        sy0 = max(0, -dy)
        sx1 = min(w, w - dx)
        sy1 = min(h, h - dy)
        dx0 = max(0, dx)
        dy0 = max(0, dy)
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)
        if sx1 > sx0 and sy1 > sy0:
            new_mask[dy0:dy1, dx0:dx1] = self._mask[sy0:sy1, sx0:sx1]
        self._set_mask(new_mask)
