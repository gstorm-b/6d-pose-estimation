"""PySide6 + VTK app to inspect instance segmentation results.

Load the instance model (a registry bundle or a raw checkpoint), open a scene
(an .npz with points/points_camera + optional features), run segmentation, and
view the clustered point cloud colored per instance. Click an instance to isolate
it, toggle background/centroids, and re-tune the clustering thresholds live
(re-clustering reuses the cached model forward pass).

    python scripts/view_instance_segmentation.py
    python scripts/view_instance_segmentation.py --sku bending_pipe \
        --scene processed-data/pointnet2_semseg_bending_pipe_wave2/test/<sample>.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import vtkmodules.all as vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    VTK_AVAILABLE = True
    VTK_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - VTK is an optional GUI dependency.
    vtk = None
    QVTKRenderWindowInteractor = None
    VTK_AVAILABLE = False
    VTK_IMPORT_ERROR = str(exc)

VTK_RENDER_DISABLED = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"

_PALETTE = np.array(
    [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
     [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
     [128, 128, 0], [255, 215, 180], [0, 0, 128], [200, 200, 200], [255, 225, 25]],
    dtype=np.uint8,
)
_BACKGROUND_COLOR = np.array([70, 75, 82], dtype=np.uint8)


def instance_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.tile(_BACKGROUND_COLOR, (labels.shape[0], 1))
    fg = labels > 0
    colors[fg] = _PALETTE[(labels[fg] - 1) % len(_PALETTE)]
    return colors


# --------------------------------------------------------------------------- worker
class CallableWorker(QObject):
    """Runs a callable off the UI thread and emits its result (or an error)."""

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, token: int, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.token = token
        self.fn = fn

    @Slot()
    def run(self) -> None:
        try:
            self.done.emit(self.token, self.fn())
        except Exception as exc:  # noqa: BLE001 - surface failures in the GUI.
            self.failed.emit(self.token, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------- vtk panel
class VtkInstancePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.status = QLabel("No scene loaded.")
        layout.addWidget(self.status)
        self.renderer = None
        self.vtk_widget = None
        self.point_actor = None
        self.centroid_actor = None
        self._points = None
        self._base_colors = None

        if not VTK_AVAILABLE or VTK_RENDER_DISABLED:
            self.status.setText(
                "VTK rendering unavailable. Install with `pip install vtk`.\n" + VTK_IMPORT_ERROR
                if not VTK_AVAILABLE else "VTK disabled for offscreen Qt."
            )
            return
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.02, 0.025, 0.03)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vtk_widget.Initialize()

    def _polydata_from_points(self, points: np.ndarray) -> Any:
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(_np_to_vtk(points.astype(np.float32)))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        cells = vtk.vtkCellArray()
        for idx in range(points.shape[0]):
            cells.InsertNextCell(1)
            cells.InsertCellPoint(idx)
        poly.SetVerts(cells)
        return poly

    def set_scene(self, points: np.ndarray, colors: np.ndarray, point_size: int, *, keep_camera: bool) -> None:
        if self.renderer is None:
            return
        self._points = points
        self._base_colors = colors
        poly = self._polydata_from_points(points)
        self._color_array = vtk.vtkUnsignedCharArray()
        self._color_array.SetNumberOfComponents(3)
        self._color_array.SetName("colors")
        self._color_array.SetNumberOfTuples(points.shape[0])
        poly.GetPointData().SetScalars(self._color_array)
        self._apply_colors(colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)

        self.renderer.RemoveAllViewProps()
        self.centroid_actor = None
        self.renderer.AddActor(actor)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.05, 0.05, 0.05)
        self.renderer.AddActor(axes)
        self.point_actor = actor
        if not keep_camera:
            self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _apply_colors(self, colors: np.ndarray) -> None:
        flat = np.ascontiguousarray(colors.astype(np.uint8))
        for i in range(flat.shape[0]):
            self._color_array.SetTypedTuple(i, (int(flat[i, 0]), int(flat[i, 1]), int(flat[i, 2])))
        self._color_array.Modified()

    def recolor(self, colors: np.ndarray) -> None:
        if self.point_actor is None:
            return
        self._base_colors = colors
        self._apply_colors(colors)
        self.vtk_widget.GetRenderWindow().Render()

    def highlight(self, mask: np.ndarray | None) -> None:
        """Dim every point whose mask is False; None restores full colors."""
        if self.point_actor is None or self._base_colors is None:
            return
        if mask is None:
            self._apply_colors(self._base_colors)
        else:
            dimmed = (self._base_colors.astype(np.float32) * 0.18).astype(np.uint8)
            shown = self._base_colors.copy()
            shown[~mask] = dimmed[~mask]
            self._apply_colors(shown)
        self.vtk_widget.GetRenderWindow().Render()

    def set_centroids(self, centroids: np.ndarray | None, colors: np.ndarray | None, diameter: float) -> None:
        if self.renderer is None:
            return
        if self.centroid_actor is not None:
            self.renderer.RemoveActor(self.centroid_actor)
            self.centroid_actor = None
        if centroids is None or centroids.shape[0] == 0:
            self.vtk_widget.GetRenderWindow().Render()
            return
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(_np_to_vtk(centroids.astype(np.float32)))
        poly.SetPoints(pts)
        carr = vtk.vtkUnsignedCharArray()
        carr.SetNumberOfComponents(3)
        carr.SetName("colors")
        for c in colors:
            carr.InsertNextTuple3(int(c[0]), int(c[1]), int(c[2]))
        poly.GetPointData().SetScalars(carr)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(max(diameter, 1e-3) * 0.12)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(12)
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetInputData(poly)
        glyph.SetColorModeToColorByScalar()
        glyph.SetScaleModeToDataScalingOff()
        glyph.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)
        self.centroid_actor = actor
        self.vtk_widget.GetRenderWindow().Render()

    def set_point_size(self, size: int) -> None:
        if self.point_actor is not None:
            self.point_actor.GetProperty().SetPointSize(size)
            self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        if self.renderer is not None:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()


def _np_to_vtk(arr: np.ndarray) -> Any:
    from vtkmodules.util.numpy_support import numpy_to_vtk

    return numpy_to_vtk(np.ascontiguousarray(arr), deep=True)


# ------------------------------------------------------------------------- window
class InstanceSegViewer(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.setWindowTitle("Instance Segmentation Viewer")
        self.resize(1360, 860)
        self.args = args
        self.segmenter = None
        self.diameter = 0.1
        self.scene_points: np.ndarray | None = None
        self.scene_features: np.ndarray | None = None
        self.scene_name = ""
        self.forward = None
        self.result = None
        self._token = 0
        self._threads: list[QThread] = []
        self._workers: list[QObject] = []
        self._callbacks: dict[int, Callable[[Any], None]] = {}

        self._build_ui()
        if args.sku:
            idx = self.model_combo.findText(args.sku)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if args.scene:
            self._set_scene_path(Path(args.scene))

    # -- UI -------------------------------------------------------------------
    def _build_ui(self) -> None:
        from src.registry.model_registry import ModelRegistry

        central = QWidget()
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        side = QWidget()
        side_layout = QVBoxLayout(side)

        # Model + scene
        self.model_combo = QComboBox()
        try:
            for sku in ModelRegistry(self.args.registry_root).list_skus():
                self.model_combo.addItem(sku)
        except Exception:  # noqa: BLE001
            pass
        self.model_combo.addItem("From checkpoint...")
        self.load_model_btn = QPushButton("Load model")
        self.load_model_btn.clicked.connect(self.on_load_model)
        self.scene_label = QLabel("No scene")
        self.scene_label.setWordWrap(True)
        open_scene_btn = QPushButton("Open scene (.npz)...")
        open_scene_btn.clicked.connect(self.on_open_scene)
        self.run_btn = QPushButton("Run segmentation")
        self.run_btn.clicked.connect(self.on_run)
        self.run_btn.setEnabled(False)

        model_form = QFormLayout()
        model_form.addRow("Model", self.model_combo)
        side_layout.addLayout(model_form)
        side_layout.addWidget(self.load_model_btn)
        side_layout.addWidget(open_scene_btn)
        side_layout.addWidget(self.scene_label)
        side_layout.addWidget(self.run_btn)

        # Clustering controls
        self.prob_spin = QDoubleSpinBox(); self.prob_spin.setRange(0.0, 1.0); self.prob_spin.setSingleStep(0.05); self.prob_spin.setValue(0.5)
        self.eps_spin = QDoubleSpinBox(); self.eps_spin.setDecimals(4); self.eps_spin.setRange(0.0005, 0.1); self.eps_spin.setSingleStep(0.001); self.eps_spin.setValue(0.006)
        self.minpts_spin = QSpinBox(); self.minpts_spin.setRange(1, 2000); self.minpts_spin.setValue(32)
        self.recluster_btn = QPushButton("Re-cluster (cached forward)")
        self.recluster_btn.clicked.connect(self.on_recluster)
        self.recluster_btn.setEnabled(False)
        cl_form = QFormLayout()
        cl_form.addRow("object prob ≥", self.prob_spin)
        cl_form.addRow("dbscan eps (m)", self.eps_spin)
        cl_form.addRow("min cluster pts", self.minpts_spin)
        side_layout.addWidget(QLabel("— Clustering —"))
        side_layout.addLayout(cl_form)
        side_layout.addWidget(self.recluster_btn)

        # View options
        self.point_size = QSpinBox(); self.point_size.setRange(1, 20); self.point_size.setValue(4)
        self.point_size.valueChanged.connect(lambda v: self.vtk.set_point_size(v))
        self.show_bg = QCheckBox("Show background points"); self.show_bg.setChecked(True); self.show_bg.stateChanged.connect(self.refresh_view)
        self.show_centroids = QCheckBox("Show centroids"); self.show_centroids.setChecked(True); self.show_centroids.stateChanged.connect(self.refresh_view)
        reset_cam = QPushButton("Reset camera")
        view_form = QFormLayout(); view_form.addRow("Point size", self.point_size)
        side_layout.addWidget(QLabel("— View —"))
        side_layout.addLayout(view_form)
        side_layout.addWidget(self.show_bg)
        side_layout.addWidget(self.show_centroids)

        # Instance list
        side_layout.addWidget(QLabel("Instances (click to isolate)"))
        self.instance_list = QListWidget()
        self.instance_list.currentItemChanged.connect(self.on_instance_selected)
        side_layout.addWidget(self.instance_list, 1)
        reset_cam.clicked.connect(lambda: self.vtk.reset_camera())
        side_layout.addWidget(reset_cam)

        self.details = QPlainTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(120)
        side_layout.addWidget(self.details)

        self.vtk = VtkInstancePanel()
        splitter.addWidget(side)
        splitter.addWidget(self.vtk)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 980])
        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Load a model, open a scene, then Run segmentation.")

    # -- workers --------------------------------------------------------------
    def _run_async(self, fn: Callable[[], Any], on_done: Callable[[Any], None], busy: str) -> None:
        self._token += 1
        token = self._token
        self._callbacks[token] = on_done
        self.statusBar().showMessage(busy)
        self._set_busy(True)
        worker = CallableWorker(token, fn)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Connect to bound-method slots on this main-thread window. Qt then uses a
        # QUEUED connection (emitter is in the worker thread, receiver in the GUI
        # thread), so the result - and any VTK/Qt rendering it triggers - runs on
        # the GUI thread. Connecting to a plain closure would default to a DIRECT
        # connection and run VTK in the worker thread, which opens a second GL
        # window and crashes.
        worker.done.connect(self._on_async_done)
        worker.failed.connect(self._on_async_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(lambda t=thread, w=worker: self._cleanup(t, w))
        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()

    @Slot(int, object)
    def _on_async_done(self, token: int, payload: Any) -> None:
        callback = self._callbacks.pop(token, None)
        if token != self._token:
            return
        self._set_busy(False)
        if callback is not None:
            callback(payload)

    @Slot(int, str)
    def _on_async_failed(self, token: int, message: str) -> None:
        self._callbacks.pop(token, None)
        if token != self._token:
            return
        self._set_busy(False)
        self.statusBar().showMessage(f"Error: {message}")
        self.details.setPlainText(message)

    def _cleanup(self, thread: QThread, worker: QObject) -> None:
        for store in (self._threads, self._workers):
            if (thread if store is self._threads else worker) in store:
                store.remove(thread if store is self._threads else worker)
        worker.deleteLater()
        thread.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        for w in (self.load_model_btn, self.run_btn, self.recluster_btn):
            w.setEnabled(not busy)
        if not busy:
            self.run_btn.setEnabled(self.segmenter is not None and self.scene_points is not None)
            self.recluster_btn.setEnabled(self.forward is not None)

    # -- model + scene --------------------------------------------------------
    def on_load_model(self) -> None:
        choice = self.model_combo.currentText()
        device = self.args.device
        if choice == "From checkpoint...":
            path, _ = QFileDialog.getOpenFileName(self, "Instance checkpoint", str(PROJECT_ROOT), "Checkpoints (*.pt)")
            if not path:
                return
            fn = lambda: _load_from_checkpoint(path, device)  # noqa: E731
        else:
            fn = lambda: _load_from_bundle(choice, self.args.registry_root, device)  # noqa: E731
        self._run_async(fn, self._on_model_loaded, f"Loading model '{choice}'...")

    def _on_model_loaded(self, payload: Any) -> None:
        self.segmenter, self.diameter, model_id = payload
        # seed clustering controls from the loaded config
        self.prob_spin.setValue(self.segmenter.clustering.object_probability_threshold)
        self.eps_spin.setValue(self.segmenter.clustering.dbscan_eps_m)
        self.minpts_spin.setValue(int(self.segmenter.clustering.min_cluster_points))
        self.statusBar().showMessage(f"Model loaded: {model_id} (device={self.segmenter.device}).")

    def on_open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", str(PROJECT_ROOT), "NumPy (*.npz)")
        if path:
            self._set_scene_path(Path(path))

    def _set_scene_path(self, path: Path) -> None:
        try:
            points, features = _load_scene_npz(path)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Could not load scene: {exc}")
            return
        self.scene_points, self.scene_features, self.scene_name = points, features, path.name
        self.scene_label.setText(f"Scene: {path.name}\n{points.shape[0]} points, "
                                 f"features={'yes' if features is not None else 'no'}")
        self.forward = None
        self.run_btn.setEnabled(self.segmenter is not None)
        self.recluster_btn.setEnabled(False)

    # -- segmentation ---------------------------------------------------------
    def _clustering_config(self):
        from src.inference.instance_clustering import VotedCenterClusteringConfig

        return VotedCenterClusteringConfig(
            object_probability_threshold=float(self.prob_spin.value()),
            dbscan_eps_m=float(self.eps_spin.value()),
            dbscan_min_samples=int(self.segmenter.clustering.dbscan_min_samples),
            min_cluster_points=int(self.minpts_spin.value()),
        )

    def on_run(self) -> None:
        if self.segmenter is None or self.scene_points is None:
            return
        seg, pts, feats, cfg = self.segmenter, self.scene_points, self.scene_features, self._clustering_config()

        def work():
            fwd = seg.forward(pts, feats)
            return fwd, seg.cluster(fwd, cfg)

        self._run_async(work, self._on_segmented, "Running segmentation (model + clustering)...")

    def on_recluster(self) -> None:
        if self.forward is None or self.segmenter is None:
            return
        seg, fwd, cfg = self.segmenter, self.forward, self._clustering_config()
        self._run_async(lambda: (fwd, seg.cluster(fwd, cfg)), self._on_segmented, "Re-clustering...")

    def _on_segmented(self, payload: Any) -> None:
        self.forward, self.result = payload
        self.recluster_btn.setEnabled(True)
        self._populate_instances()
        self.refresh_view(reset_camera=True)
        t = self.result.timings_ms
        self.statusBar().showMessage(
            f"{self.result.instance_count} instances | "
            f"{int((self.result.instance_labels > 0).sum())}/{self.result.points_camera.shape[0]} fg points | "
            f"inference {t.get('inference_ms', 0):.0f} ms + clustering {t.get('clustering_ms', 0):.0f} ms"
        )

    def _populate_instances(self) -> None:
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        all_item = QListWidgetItem("All instances")
        all_item.setData(Qt.ItemDataRole.UserRole, -1)
        self.instance_list.addItem(all_item)
        for inst in self.result.instances:
            item = QListWidgetItem(f"#{inst.instance_id}  •  {inst.point_count} pts  •  prob {inst.mean_object_probability:.2f}")
            item.setData(Qt.ItemDataRole.UserRole, inst.instance_id)
            self.instance_list.addItem(item)
        self.instance_list.setCurrentRow(0)
        self.instance_list.blockSignals(False)

    def refresh_view(self, *_args: Any, reset_camera: bool = False) -> None:
        if self.result is None:
            return
        points = self.result.points_camera
        labels = self.result.instance_labels
        colors = instance_colors(labels)
        if self.show_bg.isChecked():
            shown_points, shown_colors, self._index_map = points, colors, np.arange(points.shape[0])
        else:
            fg = labels > 0
            shown_points, shown_colors, self._index_map = points[fg], colors[fg], np.flatnonzero(fg)
        self._shown_labels = labels[self._index_map]
        self.vtk.set_scene(shown_points, shown_colors, self.point_size.value(), keep_camera=not reset_camera)
        if self.show_centroids.isChecked() and self.result.instances:
            centroids = np.stack([i.centroid_camera for i in self.result.instances])
            ccolors = _PALETTE[(np.array([i.instance_id for i in self.result.instances]) - 1) % len(_PALETTE)]
            self.vtk.set_centroids(centroids, ccolors, self.diameter)
        else:
            self.vtk.set_centroids(None, None, self.diameter)
        self.on_instance_selected(self.instance_list.currentItem(), None)

    def on_instance_selected(self, current: QListWidgetItem | None, _prev: Any) -> None:
        if current is None or self.result is None or self.vtk.point_actor is None:
            return
        instance_id = int(current.data(Qt.ItemDataRole.UserRole))
        if instance_id < 0:
            self.vtk.highlight(None)
            self.details.setPlainText(self._summary_text())
            return
        mask = self._shown_labels == instance_id
        self.vtk.highlight(mask)
        inst = next((i for i in self.result.instances if i.instance_id == instance_id), None)
        if inst is not None:
            ext = inst.bbox_max_camera - inst.bbox_min_camera
            self.details.setPlainText(
                f"Instance #{inst.instance_id}\npoints: {inst.point_count}\n"
                f"mean object prob: {inst.mean_object_probability:.3f}\n"
                f"centroid (m): ({inst.centroid_camera[0]:.3f}, {inst.centroid_camera[1]:.3f}, {inst.centroid_camera[2]:.3f})\n"
                f"bbox extent (m): ({ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f})"
            )

    def _summary_text(self) -> str:
        if self.result is None:
            return ""
        counts = [i.point_count for i in self.result.instances]
        return (
            f"Scene: {self.scene_name}\ninstances: {self.result.instance_count}\n"
            f"foreground points: {int((self.result.instance_labels > 0).sum())} / {self.result.points_camera.shape[0]}\n"
            f"points/instance: min {min(counts) if counts else 0}, max {max(counts) if counts else 0}, "
            f"mean {int(np.mean(counts)) if counts else 0}"
        )

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        for thread in list(self._threads):
            thread.quit()
            thread.wait(1500)
        super().closeEvent(event)


# ---------------------------------------------------------------- loading helpers
def _load_from_bundle(sku: str, registry_root: str, device: str):
    from src.inference.device import resolve_device
    from src.inference.instance_segmentation import InstanceSegmenter
    from src.registry.model_registry import ModelRegistry

    registry = ModelRegistry(registry_root)
    loaded = registry.load(registry.resolve(sku), device=resolve_device(device))
    return InstanceSegmenter.from_loaded_bundle(loaded), float(loaded.diameter_m), f"{sku}:{loaded.bundle.version}"


def _load_from_checkpoint(path: str, device: str):
    from src.inference.instance_segmentation import InstanceSegmenter

    seg = InstanceSegmenter.from_checkpoint(path, device=device)
    return seg, 0.1, f"checkpoint:{Path(path).name}"


def _load_scene_npz(path: Path):
    data = np.load(path)
    keys = data.files
    points_key = next((k for k in ("points", "points_camera") if k in keys), None)
    if points_key is None:
        raise KeyError(f"{path.name} has no points/points_camera (has {keys})")
    points = np.asarray(data[points_key], dtype=np.float32)
    feature_key = next((k for k in ("features", "normal_camera", "normals") if k in keys), None)
    features = np.asarray(data[feature_key], dtype=np.float32) if feature_key else None
    if features is not None and features.shape[0] != points.shape[0]:
        features = None
    return points, features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect instance segmentation results (PySide6 + VTK)")
    parser.add_argument("--sku", default=None, help="Preselect this registry SKU.")
    parser.add_argument("--scene", default=None, help="Preload this scene .npz.")
    parser.add_argument("--registry-root", default="models")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    viewer = InstanceSegViewer(args)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
