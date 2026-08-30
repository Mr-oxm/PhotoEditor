"""Layer data model — the atomic unit of the compositing stack.

Non-destructive transforms (Affinity-style)
--------------------------------------------
Raster layers store the *original* source pixel data alongside current
transform parameters (scale, rotation).  Display pixels are always
re-derived from the source, so no matter how many times you resize or
rotate, the full-resolution original is preserved.

Destructive tools (brush, eraser, etc.) automatically *rasterize* the
transform before modifying pixels, baking the current transform into
the source.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias
from uuid import uuid4

import numpy as np

from .enums import BlendMode, LayerType

if TYPE_CHECKING:
    from ..processors import ImageProcessor

    LayerProcessor: TypeAlias = ImageProcessor
else:
    LayerProcessor = object


# Never reused, so a version value identifies content globally and for the
# life of the process.
_CONTENT_VERSION = itertools.count(1)


@dataclass
class Layer:
    """Single compositing layer with optional mask and styles."""

    name: str
    width: int
    height: int
    layer_type: LayerType = LayerType.RASTER
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    opacity: float = 1.0
    blend_mode: BlendMode = BlendMode.NORMAL
    visible: bool = True
    locked: bool = False
    position: tuple[int, int] = (0, 0)
    mask_enabled: bool = True
    clipping_mask: bool = False
    clips_parent: bool = False
    parent_id: str | None = None
    # Channel visibility toggles
    channel_r: bool = True
    channel_g: bool = True
    channel_b: bool = True
    channel_a: bool = True
    # Persistent transform tracking
    transform_angle: float = 0.0
    transform_scale_x: float = 1.0
    transform_scale_y: float = 1.0
    transform_base_w: int = 0
    transform_base_h: int = 0

    def __post_init__(self) -> None:
        # Process-wide monotonic, NOT per-layer. Restoring an undo state
        # rebuilds Layer objects from scratch, so a per-layer counter
        # restarted at 1 and a later snapshot could collide with a stale
        # (id, version) pair and share the discarded branch's pixels.
        # The render pipeline's layer cache keys off this, so any code path
        # that mutates pixels in place must call touch().
        self._content_version: int = next(_CONTENT_VERSION)
        # Copy-on-write: set when the undo history holds a reference to this
        # layer's buffers. While frozen the arrays are marked non-writeable,
        # so an in-place edit that forgot begin_write() raises immediately
        # instead of silently corrupting a stored history state.
        self._frozen: bool = False
        self._pixels = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self._mask: np.ndarray | None = None
        self._styles: list = []
        self._adjustment: LayerProcessor | None = None
        self._adjustment_params: dict = {}
        self.children: list[str] = []
        # Mask layer children — IDs of MASK-type layers attached to this layer
        self.mask_layers: list[str] = []
        # Ex-parent tracking: when a mask layer is detached from its parent,
        # this records the old parent so compositing can scope the mask
        # to only that layer (instead of affecting all layers below).
        self.ex_parent_id: str | None = None
        # --- Non-destructive transform state ---
        # Original full-quality pixel data (None = no ND transform active)
        self._source_pixels: np.ndarray | None = None
        self._source_mask: np.ndarray | None = None
        self._pixels_dirty: bool = False
        # --- Vector layer data ---
        # When layer_type == SHAPE, this holds the VectorLayer scene graph.
        # The rasterizer converts vector data → _pixels on demand.
        self._vector_data: object | None = None  # VectorLayer instance

    # ---- Pixel data ---------------------------------------------------------

    @property
    def pixels(self) -> np.ndarray:
        if self._pixels_dirty:
            self._recompute_from_source()
        return self._pixels

    @pixels.setter
    def pixels(self, value: np.ndarray) -> None:
        self._pixels = value.astype(np.float32) if value.dtype != np.float32 else value
        self.height, self.width = value.shape[:2]
        self._frozen = False
        self._content_version = next(_CONTENT_VERSION)

    # ---- Content versioning -------------------------------------------------

    @property
    def content_version(self) -> int:
        """Bumped on every content change; the render cache keys off it."""
        return self._content_version

    def touch(self) -> None:
        """Signal that pixel or mask data was mutated in place.

        Painting tools write directly into ``layer.pixels[y0:y1, x0:x1]``,
        which the property setter never sees. They must call this so cached
        renders of the layer are dropped.
        """
        self._content_version = next(_CONTENT_VERSION)

    def begin_write(self) -> None:
        """Take private ownership of the pixel buffers before editing.

        Undo snapshots reference a layer's arrays rather than copying them,
        which is what makes multi-layer undo affordable. This un-shares them
        again on the first write after a snapshot, so history keeps the old
        content and the layer gets a buffer it may mutate.

        Every in-place edit must call this *before* writing. Frozen buffers
        are non-writeable, so a missed call fails loudly and immediately.
        """
        if self._frozen:
            self._pixels = self._pixels.copy()
            if self._mask is not None and not self._mask.flags.writeable:
                self._mask = self._mask.copy()
            if self._source_pixels is not None and not self._source_pixels.flags.writeable:
                self._source_pixels = self._source_pixels.copy()
            if self._source_mask is not None and not self._source_mask.flags.writeable:
                self._source_mask = self._source_mask.copy()
            self._frozen = False
        self._content_version = next(_CONTENT_VERSION)

    def freeze(self) -> None:
        """Mark the buffers as shared with history and make them read-only."""
        self._frozen = True
        for arr in (self._pixels, self._mask,
                    self._source_pixels, self._source_mask):
            if arr is not None:
                try:
                    arr.flags.writeable = False
                except ValueError:
                    # A view whose base is already read-only, or an array
                    # that does not own its data -- already effectively frozen.
                    pass

    def _adopt(self, pixels: np.ndarray, frozen: bool) -> None:
        """Install *pixels* directly, optionally as a frozen shared buffer."""
        self._pixels = pixels
        self.height, self.width = pixels.shape[:2]
        self._frozen = frozen
        self._content_version = next(_CONTENT_VERSION)

    # ---- Non-destructive transform API --------------------------------------

    @property
    def source_pixels(self) -> np.ndarray:
        """Original untransformed pixel data, or current pixels if no ND transform."""
        if self._source_pixels is not None:
            return self._source_pixels
        return self._pixels

    @property
    def source_width(self) -> int:
        """Width of the original source data."""
        if self._source_pixels is not None:
            return self._source_pixels.shape[1]
        return self.width

    @property
    def source_height(self) -> int:
        """Height of the original source data."""
        if self._source_pixels is not None:
            return self._source_pixels.shape[0]
        return self.height

    @property
    def has_transform(self) -> bool:
        """True when a non-destructive transform is active."""
        return self._source_pixels is not None and (
            self.transform_scale_x != 1.0
            or self.transform_scale_y != 1.0
            or self.transform_angle != 0.0
        )

    def init_non_destructive(self) -> None:
        """Snapshot current pixels as source for non-destructive transforms.

        Idempotent — only captures on first call.  Subsequent transforms
        keep reusing the same high-quality source.
        """
        if self._source_pixels is None:
            self._source_pixels = self._pixels.copy()
            if self._mask is not None:
                self._source_mask = self._mask.copy()
            self._content_version = next(_CONTENT_VERSION)
            # Ensure base dimensions are initialised
            if self.transform_base_w == 0:
                self.transform_base_w = self.width
                self.transform_base_h = self.height

    def compute_display(
        self,
        scale_x: float | None = None,
        scale_y: float | None = None,
        angle: float | None = None,
        fast: bool = False,
    ) -> None:
        """Eagerly recompute display pixels from source + transform params.

        Pass explicit values to override stored params (useful during a
        drag where the param hasn't been committed yet).  ``None`` means
        "use the stored value".
        """
        if self._source_pixels is None:
            return

        from ..transforms.transform_engine import TransformEngine
        import cv2

        sx = scale_x if scale_x is not None else self.transform_scale_x
        sy = scale_y if scale_y is not None else self.transform_scale_y
        ang = angle if angle is not None else self.transform_angle

        result = self._source_pixels
        mask_result = self._source_mask

        if sx != 1.0 or sy != 1.0:
            result = TransformEngine.scale(result, sx, sy, fast=fast)
            if mask_result is not None:
                sh, sw = result.shape[:2]
                mask_interp = cv2.INTER_NEAREST if fast else cv2.INTER_LINEAR
                mask_result = cv2.resize(
                    mask_result, (sw, sh), interpolation=mask_interp
                )

        # Update unrotated display dimensions
        self.transform_base_w = result.shape[1]
        self.transform_base_h = result.shape[0]

        if ang != 0.0:
            result = TransformEngine.rotate(result, ang, expand=True, fast=fast)
            if mask_result is not None:
                mask_3d = np.stack([mask_result] * 4, axis=-1)
                mask_3d = TransformEngine.rotate(mask_3d, ang, expand=True, fast=fast)
                mask_result = mask_3d[..., 0]

        self._pixels = np.clip(result, 0.0, 1.0).astype(np.float32) if result.dtype != np.float32 else np.clip(result, 0.0, 1.0)
        if mask_result is not None:
            self._mask = np.clip(mask_result, 0.0, 1.0)
        self.height, self.width = result.shape[:2]
        self._pixels_dirty = False
        self._content_version = next(_CONTENT_VERSION)

    def invalidate_transform(self) -> None:
        """Mark display pixels as needing lazy recompute from source."""
        if self._source_pixels is not None:
            self._pixels_dirty = True

    def rasterize_transform(self) -> None:
        """Bake current transforms into pixels, discarding the source.

        Called automatically before destructive operations (painting)
        and explicitly via *Layer > Rasterize*.
        """
        if self._source_pixels is not None:
            if self._pixels_dirty:
                self._recompute_from_source()
            # Current _pixels/mask are the baked result — adopt them
            self._source_pixels = None
            self._source_mask = None
        self.transform_scale_x = 1.0
        self.transform_scale_y = 1.0
        self.transform_angle = 0.0
        self.transform_base_w = 0
        self.transform_base_h = 0
        self._pixels_dirty = False

    def _recompute_from_source(self) -> None:
        """Internal lazy recompute triggered by the ``pixels`` getter."""
        if self._source_pixels is None:
            self._pixels_dirty = False
            return
        # Delegate to compute_display with stored params
        self.compute_display()

    # ---- Mask ---------------------------------------------------------------

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray | None) -> None:
        self._mask = value
        self._content_version = next(_CONTENT_VERSION)

    def add_mask(self, fill_white: bool = True) -> None:
        val = 1.0 if fill_white else 0.0
        self._mask = np.full((self.height, self.width), val, dtype=np.float32)
        self._content_version = next(_CONTENT_VERSION)

    def remove_mask(self) -> None:
        self._mask = None
        self._content_version = next(_CONTENT_VERSION)

    # ---- Styles / Adjustments ----------------------------------------------

    @property
    def styles(self) -> list:
        return self._styles

    @property
    def adjustment(self) -> LayerProcessor | None:
        return self._adjustment

    @adjustment.setter
    def adjustment(self, value: LayerProcessor | None) -> None:
        self._adjustment = value

    @property
    def adjustment_params(self) -> dict:
        return self._adjustment_params

    @adjustment_params.setter
    def adjustment_params(self, value: dict) -> None:
        self._adjustment_params = value

    # ---- Utilities ----------------------------------------------------------

    @property
    def is_mask_layer(self) -> bool:
        """True when this layer acts as a mask (MASK type)."""
        return self.layer_type == LayerType.MASK

    def get_mask_grayscale(self) -> np.ndarray:
        """Return the grayscale mask data from this layer's pixels.

        For MASK layers the luminance of the RGB channels is used:
        white = fully visible, black = fully hidden.
        """
        # Use luminance: 0.299R + 0.587G + 0.114B
        return (
            self._pixels[..., 0] * 0.299
            + self._pixels[..., 1] * 0.587
            + self._pixels[..., 2] * 0.114
        ).astype(np.float32)

    def duplicate(self) -> "Layer":
        new = Layer(
            name=f"{self.name} copy",
            width=self.width,
            height=self.height,
            layer_type=self.layer_type,
            opacity=self.opacity,
            blend_mode=self.blend_mode,
            visible=self.visible,
            position=self.position,
            transform_angle=self.transform_angle,
            transform_scale_x=self.transform_scale_x,
            transform_scale_y=self.transform_scale_y,
            transform_base_w=self.transform_base_w,
            transform_base_h=self.transform_base_h,
            channel_r=self.channel_r,
            channel_g=self.channel_g,
            channel_b=self.channel_b,
            channel_a=self.channel_a,
        )
        new._pixels = self._pixels.copy()
        if self._mask is not None:
            new._mask = self._mask.copy()
        new._styles = list(self._styles)
        new.mask_layers = list(self.mask_layers)
        new.ex_parent_id = self.ex_parent_id
        # Copy non-destructive source data
        if self._source_pixels is not None:
            new._source_pixels = self._source_pixels.copy()
        if self._source_mask is not None:
            new._source_mask = self._source_mask.copy()
        # Copy adjustment layer data
        if self._adjustment is not None:
            new._adjustment = self._adjustment
            new._adjustment_params = dict(self._adjustment_params)
        # Copy vector layer data
        if self._vector_data is not None:
            new._vector_data = self._vector_data  # VectorLayer is serializable
        return new

    def fill(self, color: np.ndarray) -> None:
        self._pixels[:] = color.astype(np.float32)
        self._content_version = next(_CONTENT_VERSION)
