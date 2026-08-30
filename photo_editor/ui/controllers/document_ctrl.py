"""Document lifecycle, file I/O, tabs, and history."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ...core.document import Document
from ...core.enums import LayerType
from .base import ControllerBase
from ..dialogs.new_project_dialog import NewProjectDialog
from ...utils.image_io import load_image

_IMG_FLT = "Images (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp)"
_BASERA_FLT = "Basera Projects (*.basera)"


def _document_revision(doc) -> tuple:
    """A value that changes whenever the document's content changes.

    Layer content versions are process-wide monotonic and move on every
    edit, so their maximum is a reliable revision marker. History depth is
    NOT: it stops growing once the byte budget starts evicting states.
    """
    versions = [getattr(l, "content_version", 0) for l in doc.layers]
    return (max(versions) if versions else 0, len(list(doc.layers)))


class DocumentController(ControllerBase):
    """Handles new/open/save/close, tabs, and undo/redo."""

    def __init__(self) -> None:
        super().__init__()

    def wire(self, main_window) -> None:
        """Connect to main window and wire menu/tab signals."""
        super().wire(main_window)
        mw = main_window

        # File menu
        a = mw._menu.actions_map
        a["new"].triggered.connect(self.on_new)
        a["open"].triggered.connect(self.on_open)
        a["place_image"].triggered.connect(self.on_place_image)
        a["save"].triggered.connect(self.on_save)
        a["save_as"].triggered.connect(self.on_save_as)
        a["export"].triggered.connect(self.on_export)
        a["import_svg"].triggered.connect(self.on_import_svg)
        a["export_svg"].triggered.connect(self.on_export_svg)
        a["export_pdf"].triggered.connect(self.on_export_pdf)
        a["quit"].triggered.connect(mw.close)

        # Edit → Document Settings
        a["edit_document_settings"].triggered.connect(self.on_edit_document_settings)

        # History
        a["undo"].triggered.connect(self.on_undo)
        a["redo"].triggered.connect(self.on_redo)

        # Tabs
        mw._file_tabs.tab_selected.connect(self.on_tab_selected)
        mw._file_tabs.tab_close_requested.connect(self.on_tab_close)

        # History panel
        mw._history_panel.state_selected.connect(self.on_history_jump)

    def on_new(self) -> None:
        dlg = NewProjectDialog(self.mw)
        if dlg.exec():
            w, h, dpi = dlg.get_values()
            self.new_document(
                w, h, dpi,
                color_mode=dlg.get_color_mode(),
                color_profile=dlg.get_color_profile(),
                unit=dlg.get_unit(),
            )
            if hasattr(self.mw, '_activate_project'):
                self.mw._activate_project()

    @property
    def _session(self):
        return self.mw._document_session

    def _set_active_document(self, document: Document, path: str | None = None) -> None:
        self.mw._doc = document
        name = Path(path).name if path else document.name
        self.ctx.refresh()
        self.ctx.zoom_to_fit()
        self.mw._status.set_document_info(name, document.width, document.height)
        self.ctx.set_window_title(f"Basera — {name}")

    def _add_document_to_session(self, document: Document, path: str | None = None, *, title: str | None = None) -> int:
        return self._session.add(document, path, title=title)

    def new_document(
        self,
        w: int,
        h: int,
        dpi: int = 72,
        *,
        color_mode: str = "RGB",
        color_profile: str = "sRGB IEC61966-2.1",
        unit: str = "px",
    ) -> None:
        """Create a new blank document and switch to it."""
        document = Document(w, h, color_mode=color_mode, color_profile=color_profile, unit=unit)
        document.dpi = dpi
        self._add_document_to_session(document, None, title=document.name)
        self._set_active_document(document)

    def on_edit_document_settings(self) -> None:
        """Open the Document Settings dialog to edit the current document."""
        if not self.doc:
            return
        dlg = NewProjectDialog(self.mw)
        dlg.setWindowTitle("Document Settings")
        dlg._create_btn.setText("Apply Changes")
        dlg.set_from_document(self.doc)
        if not dlg.exec():
            return

        new_w, new_h, new_dpi = dlg.get_values()
        doc = self.doc

        # Resize canvas only when dimensions actually changed
        if new_w != doc.width or new_h != doc.height:
            from ...core.services.document_resize import resize_canvas
            resize_canvas(doc, new_w, new_h)

        # Update metadata
        doc.dpi = new_dpi
        doc.unit = dlg.get_unit()
        doc.color_mode = dlg.get_color_mode()
        doc.color_profile = dlg.get_color_profile()
        doc.mark_dirty()

        # Refresh status bar and rulers
        name = doc.name
        self.mw._status.set_document_info(name, doc.width, doc.height)
        self.ctx.refresh()
        self.ctx.zoom_to_fit()
        self.ctx.show_status_message("Document settings updated", 2000)

    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.mw,
            "Open File",
            "",
            f"{_IMG_FLT};;{_BASERA_FLT}",
        )
        if not path:
            return
        if path.lower().endswith(".basera"):
            self.on_open_basera(path)
            return
        img = load_image(path)
        h, w = img.shape[:2]
        document = Document(w, h, name=Path(path).stem)
        document.file_path = path
        document.layers[0].pixels = img
        document.save_snapshot("Open Image")
        self._add_document_to_session(document, path, title=Path(path).name)
        self._set_active_document(document, path)
        # Make sure the editor is visible
        if hasattr(self.mw, '_activate_project'):
            self.mw._activate_project()

    def on_open_basera(self, path: str | None = None) -> None:
        """Open and restore a .basera project file."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self.mw, "Open Basera Project", "", _BASERA_FLT
            )
            if not path:
                return
        try:
            from ...utils.project_io import load_basera_project
            from ...utils.recent_projects import add_recent_project

            document = load_basera_project(path)
            self._add_document_to_session(document, path, title=Path(path).name)
            self._set_active_document(document, path)
            if hasattr(self.mw, '_activate_project'):
                self.mw._activate_project()
            add_recent_project(path)
            if hasattr(self.mw, '_refresh_recent_projects'):
                self.mw._refresh_recent_projects()
            self.ctx.show_status_message(f"Opened {Path(path).name}", 2000)
        except Exception as exc:
            msg = str(exc)
            QMessageBox.warning(self.mw, "Open Basera Error", msg)

    def on_place_image(self) -> None:
        if not self.doc:
            return
        path, _ = QFileDialog.getOpenFileName(
            self.mw, "Place Image as Layer", "", _IMG_FLT
        )
        if path:
            img = load_image(path)
            from ...commands import PlaceImageCommand
            self.ctx.execute_command(PlaceImageCommand(img, name=Path(path).stem))

    def on_save(self) -> None:
        """Save project snapshot as .basera file."""
        if not self.doc:
            return
        # If we have an existing .basera path, save there
        if self.doc.file_path and self.doc.file_path.endswith(".basera"):
            self._save_basera(self.doc.file_path)
        else:
            self.on_save_as()

    def on_save_as(self) -> None:
        """Save/Save As — saves as .basera project snapshot."""
        if not self.doc:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.mw, "Save Project As", "", _BASERA_FLT
        )
        if path:
            if not path.endswith(".basera"):
                path += ".basera"
            self.doc.file_path = path
            self._save_basera(path)

    def _save_basera(self, path: str) -> None:
        """Save the project to *path* without blocking the UI.

        Saving used to run inline on the main thread. For a 20-layer 4K
        project that measured 32 s of DEFLATE with the whole window frozen
        and the status message unable to repaint. The v4 format cut the
        work substantially, but a large project is still seconds of I/O, so
        it belongs on a worker.

        The document is not mutated during the save, and the layer buffers
        it reads are frozen by the undo system's copy-on-write, so an edit
        made while a save is running copies rather than overwriting the
        bytes being written.
        """
        if not self.doc:
            return
        from ...utils.project_io import save_basera_project
        from ...utils.worker import Worker

        if getattr(self, "_saving", False):
            self.ctx.show_status_message("A save is already in progress…", 2000)
            return
        self._saving = True

        doc = self.doc
        name = Path(path).name
        # What the document looked like when the save started. Edits made
        # during a multi-second save must NOT be marked saved when it
        # finishes, or the user quits without a prompt and loses them.
        token = _document_revision(doc)
        # The worker reads these buffers for several seconds while the user
        # keeps painting. Freezing makes the next edit copy-on-write instead
        # of mutating underneath the writer.
        doc.freeze_for_read()
        self.ctx.show_status_message(f"Saving {name}…", 0)

        def run() -> str:
            save_basera_project(doc, path)
            return path

        def on_result(_result) -> None:
            self._saving = False
            self._after_save_succeeded(doc, path, saved_revision=token)

        def on_error(message: str) -> None:
            self._saving = False
            self.ctx.show_status_message("Save failed", 3000)
            QMessageBox.warning(self.mw, "Save Error", str(message))

        Worker.run_async(run, on_result=on_result, on_error=on_error)

    def _after_save_succeeded(self, doc, path: str,
                              saved_revision=None) -> None:
        """UI bookkeeping once a save has completed.

        *saved_revision* is the document's revision when the save started.
        The document is only marked clean if it has not changed since --
        otherwise a stroke made during a four-second save would be recorded
        as saved and lost on quit without a prompt.
        """
        from ...utils.recent_projects import add_recent_project

        doc.file_path = path
        unchanged = (saved_revision is None
                     or _document_revision(doc) == saved_revision)
        if unchanged:
            doc.mark_clean()
            # Genuinely on disk: drop the autosave so a later crash does not
            # offer this work back as if it had never been saved.
            self._forget_autosave(doc)
        else:
            self.ctx.show_status_message(
                f"Saved {Path(path).name} — later edits are still unsaved",
                3000)
        self.ctx.set_window_title(f"Basera — {Path(path).name}")
        # The tab holding THIS document, not whichever is current when the
        # background save finishes. The user may well have switched tabs
        # during a multi-second save, and renaming the active tab instead
        # rewrites the wrong document's path.
        idx = self._tab_index_for(doc)
        if idx >= 0:
            self._session.update_path(idx, path)
            self._session.update_tab_metadata(
                idx, title=Path(path).name, tooltip=path,
            )
        try:
            add_recent_project(path)
            if hasattr(self.mw, '_refresh_recent_projects'):
                self.mw._refresh_recent_projects()
        except Exception:
            pass
        if unchanged:
            self.ctx.show_status_message(f"Saved {Path(path).name}", 2000)

    def _tab_index_for(self, doc) -> int:
        """Index of the tab holding *doc*, or -1."""
        session = self._session
        if session is None:
            return -1
        for i in range(len(session)):
            entry = session.entry_at(i)
            if entry is not None and entry.document is doc:
                return i
        return -1

    def _forget_autosave(self, doc) -> None:
        """Drop *doc*'s autosave now that it is on disk."""
        manager = getattr(self.mw, "_autosave", None)
        session = getattr(self.mw, "_document_session", None)
        if manager is None or session is None:
            return
        for i in range(len(session)):
            entry = session.entry_at(i)
            if entry is not None and entry.document is doc:
                manager.forget(f"{i}-{id(doc):x}")
                return

    def _save_basera_blocking(self, doc, path: str) -> bool:
        """Save *doc* synchronously and report whether it succeeded.

        Used on the quit path, which must not return before the bytes are
        on disk -- an asynchronous save there would let the process exit
        mid-write and lose the user's work.
        """
        if doc is None:
            return False
        from ...utils.project_io import save_basera_project
        token = _document_revision(doc)
        try:
            save_basera_project(doc, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self.mw, "Save Error", str(exc))
            return False
        self._after_save_succeeded(doc, path, saved_revision=token)
        return True

    def on_export(self) -> None:
        """Open the export dialog."""
        if not self.doc:
            return
        from ..dialogs.export_dialog import ExportDialog
        dlg = ExportDialog(self.mw, document=self.doc)
        if dlg.exec():
            settings = dlg.get_export_settings()
            self._do_export(settings)

    def _do_export(self, settings: dict) -> None:
        """Perform the actual export based on dialog settings."""
        if not self.doc:
            return

        path = settings.get("path")
        fmt = settings.get("format", "png")
        quality = settings.get("quality", 85)

        # Target dimensions (may differ from document dims)
        exp_w = settings.get("width")
        exp_h = settings.get("height")
        target_size: tuple[int, int] | None = None
        if exp_w and exp_h:
            tw, th = int(exp_w), int(exp_h)
            if tw != self.doc.width or th != self.doc.height:
                target_size = (tw, th)

        # JPEG / PDF background colour for transparent areas
        jpeg_bg: tuple[int, int, int] = settings.get("jpeg_bg", (255, 255, 255))

        # JPEG chroma sub-sampling
        _subsamp_map = {"4:4:4 (Best)": 0, "4:2:2": 1, "4:2:0 (Smallest)": 2}
        subsampling: int = _subsamp_map.get(settings.get("subsampling", "4:4:4 (Best)"), 0)

        if settings.get("clipboard"):
            # Copy to clipboard
            try:
                from PySide6.QtGui import QImage
                from PySide6.QtWidgets import QApplication
                import numpy as np
                from PIL import Image

                # Own pipeline: see the note on export above.
                from ...engine.render_pipeline import RenderPipeline
                result_f = RenderPipeline().execute(self.doc)
                data = (np.clip(result_f, 0, 1) * 255).astype(np.uint8)
                pil = Image.fromarray(data, "RGBA")
                if target_size:
                    pil = pil.resize(target_size, Image.Resampling.LANCZOS)
                arr = np.asarray(pil)
                h, w = arr.shape[:2]
                qimg = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
                QApplication.clipboard().setImage(qimg)
                self.ctx.show_status_message("Copied to clipboard", 2000)
            except Exception as exc:
                QMessageBox.warning(self.mw, "Clipboard Error", str(exc))
            return

        if not path:
            return

        # Standard file export
        from ...commands import SaveDocumentCommand
        mw = self.mw

        self.ctx.show_status_message(f"Exporting {Path(path).name}…", 60_000)

        def on_success(_result: object) -> None:
            self.ctx.show_status_message(f"Exported to {Path(path).name}", 2000)

        def on_error(msg: str) -> None:
            self.ctx.show_status_message("Export failed", 3000)
            QMessageBox.warning(mw, "Export Error", msg)

        # A RenderPipeline is single-owner: it caches into shared buffers
        # and its compositor carries per-composite state. Handing the
        # canvas's pipeline to a background export raced it against every
        # interactive frame. Export gets its own.
        from ...engine.render_pipeline import RenderPipeline
        export_pipeline = RenderPipeline()
        self.ctx.execute_command_async(
            SaveDocumentCommand(
                path,
                export_pipeline,
                quality=quality,
                target_size=target_size,
                jpeg_bg=jpeg_bg,
                subsampling=subsampling,
            ),
            on_success=on_success,
            on_error=on_error,
        )

    def on_import_svg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.mw, "Import SVG", "", "SVG Files (*.svg)")
        if not path:
            return
        try:
            from ...vector.svg import import_svg, SVGGroup, SVGLeaf
            from ...vector.geometry import BBox, AffineTransform, Vec2
            from ...vector.rasterizer import rasterize_vector_layer_tight

            root_node = import_svg(path)
            if isinstance(root_node, SVGGroup) and not root_node.children:
                QMessageBox.information(
                    self.mw, "Import SVG", "No vector objects found in SVG."
                )
                return

            def _compute_bbox(node):
                bb = BBox.empty()
                if isinstance(node, SVGLeaf):
                    bb = node.object.bbox()
                elif isinstance(node, SVGGroup):
                    for child in node.children:
                        bb = bb.union(_compute_bbox(child))
                return bb

            all_bb = _compute_bbox(root_node)
            mw = self.mw

            created_document = False
            if mw._doc is None:
                w = max(int(all_bb.max_pt.x + 20), 200)
                h = max(int(all_bb.max_pt.y + 20), 200)
                mw._doc = Document(w, h, name=Path(path).stem)
                self._add_document_to_session(mw._doc, path, title=Path(path).name)
                created_document = True

            doc_w, doc_h = mw._doc.width, mw._doc.height
            content_w, content_h = all_bb.width, all_bb.height

            scale = 1.0
            if content_w > doc_w or content_h > doc_h:
                scale_w = doc_w / content_w if content_w > 0 else 1.0
                scale_h = doc_h / content_h if content_h > 0 else 1.0
                scale = min(scale_w, scale_h) * 0.9

            c_center = all_bb.center
            doc_center = Vec2(doc_w / 2, doc_h / 2)

            xf = AffineTransform.translation(-c_center.x, -c_center.y).concat(
                AffineTransform.scaling(scale)
            ).concat(
                AffineTransform.translation(doc_center.x, doc_center.y)
            )

            def _apply_transform(node, transform):
                if isinstance(node, SVGLeaf):
                    node.object.transform = transform.concat(node.object.transform)
                elif isinstance(node, SVGGroup):
                    for child in node.children:
                        _apply_transform(child, transform)

            _apply_transform(root_node, xf)

            def _create_layers(node, parent_id=None):
                if isinstance(node, SVGGroup):
                    name = node.name if node.name and node.name != "svg" else "Group"
                    layer = mw._doc.add_group(name=name)
                    if parent_id:
                        mw._doc.layers.reparent([layer.id], parent_id)
                    for child in node.children:
                        _create_layers(child, layer.id)
                elif isinstance(node, SVGLeaf):
                    obj = node.object
                    layer = mw._doc.add_vector_layer(name=node.name)
                    layer._vector_data.add(obj)
                    if parent_id:
                        mw._doc.layers.reparent([layer.id], parent_id)
                    layer.opacity = getattr(obj, "opacity", 1.0)
                    rasterize_vector_layer_tight(mw._doc, layer=layer, force=True)
                    svg_filt = getattr(obj, "svg_filter", None)
                    if svg_filt and svg_filt.get("type") == "gaussian_blur":
                        from ...registries import get_filter_name_map
                        filt_cls = get_filter_name_map().get("Gaussian Blur")
                        if filt_cls is not None:
                            filt = filt_cls()
                            std_dev = svg_filt.get("std_deviation", 0.0)
                            radius = max(0.1, std_dev * 2.0)
                            filt_layer = mw._doc.add_layer(
                                name="Gaussian Blur", layer_type=LayerType.FILTER
                            )
                            filt_layer.adjustment = filt
                            filt_layer.adjustment_params = {
                                "radius": radius,
                                "preserve_alpha": svg_filt.get("preserve_alpha", False),
                            }
                            filt_layer.parent_id = layer.id
                            layer.children.append(filt_layer.id)
                            mw._doc.layers.reposition_before(filt_layer.id, layer.id)

            _create_layers(root_node, parent_id=None)

            if created_document:
                self._set_active_document(mw._doc, path)
                if hasattr(mw, '_activate_project'):
                    mw._activate_project()
            else:
                self.ctx.refresh()
                self.ctx.zoom_to_fit()
            self.ctx.show_status_message("Imported SVG successfully", 3000)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self.mw, "Import SVG Error", str(exc))

    def on_export_svg(self) -> None:
        if not self.doc:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.mw, "Export SVG", "", "SVG Files (*.svg)"
        )
        if not path:
            return
        try:
            objects = []
            for layer in self.doc.layers:
                vl = getattr(layer, "_vector_data", None)
                if vl is not None:
                    objects.extend(vl.objects)
            if not objects:
                QMessageBox.information(
                    self.mw, "Export SVG", "No vector objects to export."
                )
                return
            from ...vector.svg import export_svg
            svg_str = export_svg(
                objects, self.doc.width, self.doc.height
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg_str)
            self.ctx.show_status_message(
                f"Exported {len(objects)} objects to SVG", 3000
            )
        except Exception as exc:
            QMessageBox.warning(self.mw, "Export SVG Error", str(exc))

    def on_export_pdf(self) -> None:
        if not self.doc:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.mw, "Export PDF", "", "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            objects = []
            for layer in self.doc.layers:
                vl = getattr(layer, "_vector_data", None)
                if vl is not None:
                    objects.extend(vl.objects)
            if not objects:
                QMessageBox.information(
                    self.mw, "Export PDF", "No vector objects to export."
                )
                return
            from ...vector.pdf import export_pdf_bytes
            pdf_data = export_pdf_bytes(
                objects, self.doc.width, self.doc.height
            )
            with open(path, "wb") as f:
                f.write(pdf_data)
            self.ctx.show_status_message(
                f"Exported {len(objects)} objects to PDF", 3000
            )
        except Exception as exc:
            QMessageBox.warning(self.mw, "Export PDF Error", str(exc))

    def on_tab_selected(self, index: int) -> None:
        entry = self._session.activate(index)
        if entry is None:
            return
        self._set_active_document(entry.document, entry.path)

    def on_tab_close(self, index: int) -> None:
        entry = self._session.entry_at(index)
        if entry is None:
            return
        doc = entry.document
        if doc.dirty:
            result = self._confirm_unsaved(doc)
            if result == "cancel":
                return
            if result == "save":
                saved = self._save_basera_sync(doc)
                if not saved:
                    return  # user cancelled the save-as dialog

        # Allow closing the last tab — return to welcome screen
        self._session.close(index)
        if len(self._session) == 0:
            # No more documents — return to welcome screen
            if hasattr(self.mw, '_on_last_tab_closed'):
                self.mw._on_last_tab_closed()
            return
        new_idx = self._session.current_index()
        if self._session.entry_at(new_idx) is not None:
            self.on_tab_selected(new_idx)

    # ---- Unsaved changes helpers --------------------------------------------

    def _confirm_unsaved(self, doc) -> str:
        """Show the 'unsaved changes' dialog.

        Returns
        -------
        'save'    – user wants to save before closing
        'discard' – user wants to close without saving
        'cancel'  – user changed their mind; do not close
        """
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self.mw)
        msg.setWindowTitle("Unsaved Changes")
        msg.setText(f'"{doc.name}" has unsaved changes.')
        msg.setInformativeText(
            "Do you want to save your changes before closing?"
        )
        msg.setIcon(QMessageBox.Icon.Warning)
        save_btn = msg.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = msg.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(save_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is save_btn:
            return "save"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    def _save_basera_sync(self, doc) -> bool:
        """Resolve a save path for *doc* and write it, blocking until done.

        Uses the document's existing file_path when it is a .basera path;
        otherwise opens a Save As dialog.  Returns True on success,
        False if the user cancelled or the save failed.

        This is the quit path: it must block, because returning early would
        let the application exit while the file is still being written.
        """
        path = doc.file_path if (doc.file_path and doc.file_path.endswith(".basera")) else None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self.mw, f"Save \"{doc.name}\"", "", _BASERA_FLT
            )
            if not path:
                return False
            if not path.endswith(".basera"):
                path += ".basera"

        return self._save_basera_blocking(doc, path)

    def on_undo(self) -> None:
        if self.doc:
            self.doc.undo()
            self.ctx.invalidate()
            self.ctx.refresh()

    def on_redo(self) -> None:
        if self.doc:
            self.doc.redo()
            self.ctx.invalidate()
            self.ctx.refresh()

    def on_history_jump(self, index: int) -> None:
        if self.doc:
            self.doc.navigate_history(index)
            self.ctx.invalidate()
            self.ctx.refresh()
