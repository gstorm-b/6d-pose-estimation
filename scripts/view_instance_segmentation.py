"""PySide6 + VTK app to inspect instance segmentation AND 6D pose results.

Load the model (a registry bundle, or a raw instance checkpoint for segmentation
only), open a scene (an .npz with points/points_camera + optional features), then:

- "Run segmentation": cluster the scene into instances; view the cloud colored per
  instance, click an instance to isolate it, show centroids, and re-tune the
  clustering thresholds live (re-clustering reuses the cached model forward pass).
- "Run pose" (bundle only): run the full pose pipeline and overlay each object's
  posed model points + pose axes on the scene, ranked by confidence; click an
  instance to isolate its pose. This is the standard way to eyeball 6D pose: does
  the posed model snap onto the observed points?

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
    from vtkmodules.util.numpy_support import numpy_to_vtk

    VTK_AVAILABLE = True
    VTK_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - VTK is an optional GUI dependency.
    vtk = None
    QVTKRenderWindowInteractor = None
    numpy_to_vtk = None
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
_SCENE_DIM_COLOR = np.array([55, 60, 66], dtype=np.uint8)
_AXIS_COLORS = np.array([[235, 60, 60], [60, 220, 90], [70, 130, 245]], dtype=np.uint8)  # x, y, z


def palette_color(instance_id: int) -> np.ndarray:
    return _PALETTE[(instance_id - 1) % len(_PALETTE)]


def instance_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.tile(_BACKGROUND_COLOR, (labels.shape[0], 1))
    fg = labels > 0
    colors[fg] = _PALETTE[(labels[fg] - 1) % len(_PALETTE)]
    return colors


def _flip_to_positive_forward(points_metadata: np.ndarray) -> np.ndarray:
    """Metadata camera frame -> positive-forward (negate z); the flip is an involution."""
    out = np.asarray(points_metadata, dtype=np.float32).copy()
    out[..., 2] *= -1.0
    return out


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
class VtkScenePanel(QWidget):
    """Renders a scene point cloud plus optional centroids, model overlay, and axes."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.status = QLabel("No scene loaded.")
        layout.addWidget(self.status)
        self.renderer = None
        self.vtk_widget = None
        self.scene_actor = None
        self.overlay_actor = None
        self.centroid_actor = None
        self.axes_actor = None
        self._scene_poly = None

        if not VTK_AVAILABLE or VTK_RENDER_DISABLED:
            self.status.setText(
                "VTK disabled for offscreen Qt." if VTK_RENDER_DISABLED
                else f"VTK unavailable (`pip install vtk`).\n{VTK_IMPORT_ERROR}"
            )
            return
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.02, 0.025, 0.03)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vtk_widget.Initialize()

    # -- builders -------------------------------------------------------------
    def _point_polydata(self, points: np.ndarray, colors: np.ndarray) -> Any:
        vpoints = vtk.vtkPoints()
        vpoints.SetData(numpy_to_vtk(np.ascontiguousarray(points.astype(np.float32)), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vpoints)
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(poly)
        glyph.Update()
        out = glyph.GetOutput()
        out.GetPointData().SetScalars(_color_array(colors))
        return out

    def _render(self) -> None:
        if self.vtk_widget is not None:
            self.vtk_widget.GetRenderWindow().Render()

    # -- scene ----------------------------------------------------------------
    def set_scene(self, points: np.ndarray, colors: np.ndarray, point_size: int, *, keep_camera: bool) -> None:
        if self.renderer is None or points.size == 0:
            return
        self.renderer.RemoveAllViewProps()
        self.overlay_actor = self.centroid_actor = self.axes_actor = None
        self._scene_poly = self._point_polydata(points, colors)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._scene_poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        self.renderer.AddActor(actor)
        self.scene_actor = actor
        world_axes = vtk.vtkAxesActor()
        world_axes.SetTotalLength(0.05, 0.05, 0.05)
        self.renderer.AddActor(world_axes)
        if not keep_camera:
            self.renderer.ResetCamera()
        self._render()

    def recolor_scene(self, colors: np.ndarray) -> None:
        if self._scene_poly is None:
            return
        self._scene_poly.GetPointData().SetScalars(_color_array(colors))
        self._scene_poly.Modified()
        self._render()

    # -- overlays -------------------------------------------------------------
    def set_overlay_points(self, points: np.ndarray | None, colors: np.ndarray | None, point_size: int) -> None:
        if self.renderer is None:
            return
        if self.overlay_actor is not None:
            self.renderer.RemoveActor(self.overlay_actor)
            self.overlay_actor = None
        if points is not None and points.shape[0] > 0:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self._point_polydata(points, colors))
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(point_size)
            self.renderer.AddActor(actor)
            self.overlay_actor = actor
        self._render()

    def set_axes(self, starts: np.ndarray | None, ends: np.ndarray | None, colors: np.ndarray | None) -> None:
        if self.renderer is None:
            return
        if self.axes_actor is not None:
            self.renderer.RemoveActor(self.axes_actor)
            self.axes_actor = None
        if starts is not None and starts.shape[0] > 0:
            pts = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            for i in range(starts.shape[0]):
                a = pts.InsertNextPoint(*[float(v) for v in starts[i]])
                b = pts.InsertNextPoint(*[float(v) for v in ends[i]])
                lines.InsertNextCell(2)
                lines.InsertCellPoint(a)
                lines.InsertCellPoint(b)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(lines)
            poly.GetCellData().SetScalars(_color_array(colors))
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetLineWidth(3)
            self.renderer.AddActor(actor)
            self.axes_actor = actor
        self._render()

    def set_centroids(self, centroids: np.ndarray | None, colors: np.ndarray | None, radius: float) -> None:
        if self.renderer is None:
            return
        if self.centroid_actor is not None:
            self.renderer.RemoveActor(self.centroid_actor)
            self.centroid_actor = None
        if centroids is not None and centroids.shape[0] > 0:
            poly = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(np.ascontiguousarray(centroids.astype(np.float32)), deep=True))
            poly.SetPoints(pts)
            poly.GetPointData().SetScalars(_color_array(colors))
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(max(radius, 1e-3))
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
        self._render()

    def set_point_size(self, size: int) -> None:
        if self.scene_actor is not None:
            self.scene_actor.GetProperty().SetPointSize(size)
            self._render()

    def reset_camera(self) -> None:
        if self.renderer is not None:
            self.renderer.ResetCamera()
            self._render()


def _color_array(colors: np.ndarray) -> Any:
    arr = numpy_to_vtk(np.ascontiguousarray(colors.astype(np.uint8)), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetName("colors")
    return arr


def _np_to_vtk(arr: np.ndarray) -> Any:
    return numpy_to_vtk(np.ascontiguousarray(arr), deep=True)


# ------------------------------------------------------------------------- window
class BackendViewer(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.setWindowTitle("Backend Viewer — Instance Segmentation & 6D Pose")
        self.resize(1380, 880)
        self.args = args
        self.segmenter = None
        self.pose_pipeline = None
        self.model_points: np.ndarray | None = None
        self.diameter = 0.1
        self.scene_points: np.ndarray | None = None
        self.scene_features: np.ndarray | None = None
        self.scene_name = ""
        self.forward = None
        self.seg_result = None
        self.pose_result = None
        self.pose_items: list[dict[str, Any]] = []
        self.mode = "seg"  # "seg" or "pose"
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
        s = QVBoxLayout(side)

        self.model_combo = QComboBox()
        try:
            for sku in ModelRegistry(self.args.registry_root).list_skus():
                self.model_combo.addItem(sku)
        except Exception:  # noqa: BLE001
            pass
        self.model_combo.addItem("From checkpoint...")
        self.load_model_btn = QPushButton("Load model")
        self.load_model_btn.clicked.connect(self.on_load_model)
        open_scene_btn = QPushButton("Open scene (.npz)...")
        open_scene_btn.clicked.connect(self.on_open_scene)
        self.scene_label = QLabel("No scene")
        self.scene_label.setWordWrap(True)
        model_form = QFormLayout()
        model_form.addRow("Model", self.model_combo)
        s.addLayout(model_form)
        s.addWidget(self.load_model_btn)
        s.addWidget(open_scene_btn)
        s.addWidget(self.scene_label)

        self.run_seg_btn = QPushButton("Run segmentation")
        self.run_seg_btn.clicked.connect(self.on_run_seg)
        self.run_seg_btn.setEnabled(False)
        self.run_pose_btn = QPushButton("Run pose")
        self.run_pose_btn.clicked.connect(self.on_run_pose)
        self.run_pose_btn.setEnabled(False)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_seg_btn)
        run_row.addWidget(self.run_pose_btn)
        s.addLayout(run_row)

        # Clustering
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
        s.addWidget(QLabel("— Clustering —"))
        s.addLayout(cl_form)
        s.addWidget(self.recluster_btn)

        # View options
        self.point_size = QSpinBox(); self.point_size.setRange(1, 20); self.point_size.setValue(4)
        self.point_size.valueChanged.connect(lambda v: self.vtk.set_point_size(v))
        self.show_bg = QCheckBox("Show background points"); self.show_bg.setChecked(True); self.show_bg.stateChanged.connect(lambda *_: self.refresh_view())
        self.show_centroids = QCheckBox("Show centroids (seg)"); self.show_centroids.setChecked(True); self.show_centroids.stateChanged.connect(lambda *_: self.refresh_view())
        self.show_model = QCheckBox("Show posed model (pose)"); self.show_model.setChecked(True); self.show_model.stateChanged.connect(lambda *_: self.refresh_view())
        self.show_axes = QCheckBox("Show pose axes (pose)"); self.show_axes.setChecked(True); self.show_axes.stateChanged.connect(lambda *_: self.refresh_view())
        view_form = QFormLayout(); view_form.addRow("Point size", self.point_size)
        s.addWidget(QLabel("— View —"))
        s.addLayout(view_form)
        for w in (self.show_bg, self.show_centroids, self.show_model, self.show_axes):
            s.addWidget(w)

        s.addWidget(QLabel("Instances (click to isolate)"))
        self.instance_list = QListWidget()
        self.instance_list.currentItemChanged.connect(self.on_instance_selected)
        s.addWidget(self.instance_list, 1)
        reset_cam = QPushButton("Reset camera")
        reset_cam.clicked.connect(lambda: self.vtk.reset_camera())
        s.addWidget(reset_cam)
        self.details = QPlainTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(130)
        s.addWidget(self.details)

        self.vtk = VtkScenePanel()
        splitter.addWidget(side)
        splitter.addWidget(self.vtk)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 980])
        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Load a model, open a scene, then Run segmentation or Run pose.")

    # -- async ----------------------------------------------------------------
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
        # Bound-method slots -> queued connection -> result handled on the GUI
        # thread (VTK/Qt must never be touched from the worker thread).
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
        if thread in self._threads:
            self._threads.remove(thread)
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()
        thread.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        for w in (self.load_model_btn, self.run_seg_btn, self.run_pose_btn, self.recluster_btn):
            w.setEnabled(not busy)
        if not busy:
            has_scene = self.scene_points is not None
            self.run_seg_btn.setEnabled(self.segmenter is not None and has_scene)
            self.run_pose_btn.setEnabled(self.pose_pipeline is not None and has_scene)
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
        self.segmenter, self.pose_pipeline, self.model_points, self.diameter, model_id = payload
        self.prob_spin.setValue(self.segmenter.clustering.object_probability_threshold)
        self.eps_spin.setValue(self.segmenter.clustering.dbscan_eps_m)
        self.minpts_spin.setValue(int(self.segmenter.clustering.min_cluster_points))
        pose_txt = "pose available" if self.pose_pipeline is not None else "segmentation only (no pose in a raw checkpoint)"
        self.statusBar().showMessage(f"Model loaded: {model_id} (device={self.segmenter.device}; {pose_txt}).")

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
        self._set_busy(False)

    def _clustering_config(self):
        from src.inference.instance_clustering import VotedCenterClusteringConfig

        return VotedCenterClusteringConfig(
            object_probability_threshold=float(self.prob_spin.value()),
            dbscan_eps_m=float(self.eps_spin.value()),
            dbscan_min_samples=int(self.segmenter.clustering.dbscan_min_samples),
            min_cluster_points=int(self.minpts_spin.value()),
        )

    # -- segmentation ---------------------------------------------------------
    def on_run_seg(self) -> None:
        if self.segmenter is None or self.scene_points is None:
            return
        seg, pts, feats, cfg = self.segmenter, self.scene_points, self.scene_features, self._clustering_config()
        self._run_async(lambda: (lambda f: (f, seg.cluster(f, cfg)))(seg.forward(pts, feats)),
                        self._on_segmented, "Running segmentation (model + clustering)...")

    def on_recluster(self) -> None:
        if self.forward is None or self.segmenter is None:
            return
        seg, fwd, cfg = self.segmenter, self.forward, self._clustering_config()
        self._run_async(lambda: (fwd, seg.cluster(fwd, cfg)), self._on_segmented, "Re-clustering...")

    def _on_segmented(self, payload: Any) -> None:
        self.forward, self.seg_result = payload
        self.mode = "seg"
        self.recluster_btn.setEnabled(True)
        self._populate_seg_list()
        self.refresh_view(reset_camera=True)
        t = self.seg_result.timings_ms
        self.statusBar().showMessage(
            f"[seg] {self.seg_result.instance_count} instances | "
            f"{int((self.seg_result.instance_labels > 0).sum())}/{self.seg_result.points_camera.shape[0]} fg | "
            f"inference {t.get('inference_ms', 0):.0f} ms + clustering {t.get('clustering_ms', 0):.0f} ms"
        )

    def _populate_seg_list(self) -> None:
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        item = QListWidgetItem("All instances"); item.setData(Qt.ItemDataRole.UserRole, -1)
        self.instance_list.addItem(item)
        for inst in self.seg_result.instances:
            it = QListWidgetItem(f"#{inst.instance_id}  •  {inst.point_count} pts  •  prob {inst.mean_object_probability:.2f}")
            it.setData(Qt.ItemDataRole.UserRole, inst.instance_id)
            self.instance_list.addItem(it)
        self.instance_list.setCurrentRow(0)
        self.instance_list.blockSignals(False)

    # -- pose -----------------------------------------------------------------
    def on_run_pose(self) -> None:
        if self.pose_pipeline is None or self.scene_points is None:
            return
        pipe, pts, feats, cfg = self.pose_pipeline, self.scene_points, self.scene_features, self._clustering_config()

        def work():
            pipe.clustering = cfg  # keep pose consistent with the tuned clustering
            return pipe.infer_from_points(pts, feats)

        self._run_async(work, self._on_pose_result, "Running pose pipeline (instance + pose)...")

    def _on_pose_result(self, result: Any) -> None:
        self.pose_result = result
        self.mode = "pose"
        axis_len = max(self.diameter, 1e-3) * 0.5
        self.pose_items = []
        for inst in result.instances:
            T = np.asarray(inst.object_to_camera, dtype=np.float64)
            model_pf = _flip_to_positive_forward(self.model_points @ T[:3, :3].T + T[:3, 3]) if self.model_points is not None else None
            origin_pf = _flip_to_positive_forward(T[:3, 3].reshape(1, 3))[0]
            axis_ends = np.stack([_flip_to_positive_forward((T[:3, 3] + T[:3, k] * axis_len).reshape(1, 3))[0] for k in range(3)])
            self.pose_items.append({
                "id": int(inst.instance_id), "model_pf": model_pf, "color": palette_color(inst.instance_id),
                "origin": origin_pf, "axis_ends": axis_ends, "confidence": float(inst.confidence or 0.0),
                "model_fit": float(inst.diagnostics.get("model_fit", float("nan"))), "point_count": int(inst.point_count),
            })
        self._populate_pose_list()
        self.refresh_view(reset_camera=True)
        self.statusBar().showMessage(
            f"[pose] {len(self.pose_items)} instances | top conf "
            f"{self.pose_items[0]['confidence']:.3f}" if self.pose_items else "[pose] no instances"
        )

    def _populate_pose_list(self) -> None:
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        item = QListWidgetItem("All instances"); item.setData(Qt.ItemDataRole.UserRole, -1)
        self.instance_list.addItem(item)
        for it in self.pose_items:
            entry = QListWidgetItem(f"#{it['id']}  •  conf {it['confidence']:.2f}  •  fit {it['model_fit']:.3f}  •  {it['point_count']} pts")
            entry.setData(Qt.ItemDataRole.UserRole, it["id"])
            self.instance_list.addItem(entry)
        self.instance_list.setCurrentRow(0)
        self.instance_list.blockSignals(False)

    # -- rendering ------------------------------------------------------------
    def refresh_view(self, *_args: Any, reset_camera: bool = False) -> None:
        if self.mode == "pose" and self.pose_result is not None:
            self._render_pose(reset_camera)
        elif self.seg_result is not None:
            self._render_seg(reset_camera)

    def _render_seg(self, reset_camera: bool) -> None:
        points = self.seg_result.points_camera
        labels = self.seg_result.instance_labels
        colors = instance_colors(labels)
        if self.show_bg.isChecked():
            shown_pts, shown_cols, self._index_map = points, colors, np.arange(points.shape[0])
        else:
            fg = labels > 0
            shown_pts, shown_cols, self._index_map = points[fg], colors[fg], np.flatnonzero(fg)
        self._shown_labels = labels[self._index_map]
        self.vtk.set_scene(shown_pts, shown_cols, self.point_size.value(), keep_camera=not reset_camera)
        if self.show_centroids.isChecked() and self.seg_result.instances:
            centroids = np.stack([i.centroid_camera for i in self.seg_result.instances])
            ccolors = np.stack([palette_color(i.instance_id) for i in self.seg_result.instances])
            self.vtk.set_centroids(centroids, ccolors, max(self.diameter, 1e-3) * 0.12)
        self.on_instance_selected(self.instance_list.currentItem(), None)

    def _render_pose(self, reset_camera: bool) -> None:
        # Dim grey observed scene; the posed models pop on top.
        pts = self.scene_points
        dim = np.tile(_SCENE_DIM_COLOR, (pts.shape[0], 1))
        self.vtk.set_scene(pts, dim, max(1, self.point_size.value() - 1), keep_camera=not reset_camera)
        self._render_pose_overlay()

    def _render_pose_overlay(self) -> None:
        if self.mode != "pose":
            return
        selected = int(self.instance_list.currentItem().data(Qt.ItemDataRole.UserRole)) if self.instance_list.currentItem() else -1
        items = self.pose_items if selected < 0 else [it for it in self.pose_items if it["id"] == selected]
        if self.show_model.isChecked() and self.model_points is not None and items:
            model_pts = np.concatenate([it["model_pf"] for it in items], axis=0)
            model_cols = np.concatenate([np.tile(it["color"], (it["model_pf"].shape[0], 1)) for it in items], axis=0)
            self.vtk.set_overlay_points(model_pts, model_cols, self.point_size.value() + 2)
        else:
            self.vtk.set_overlay_points(None, None, 0)
        if self.show_axes.isChecked() and items:
            starts = np.concatenate([np.tile(it["origin"], (3, 1)) for it in items], axis=0)
            ends = np.concatenate([it["axis_ends"] for it in items], axis=0)
            cols = np.tile(_AXIS_COLORS, (len(items), 1))
            self.vtk.set_axes(starts, ends, cols)
        else:
            self.vtk.set_axes(None, None, None)

    def on_instance_selected(self, current: QListWidgetItem | None, _prev: Any) -> None:
        if current is None:
            return
        instance_id = int(current.data(Qt.ItemDataRole.UserRole))
        if self.mode == "pose":
            self._render_pose_overlay()
            self.details.setPlainText(self._pose_details(instance_id))
            return
        if self.seg_result is None or self.vtk.scene_actor is None:
            return
        if instance_id < 0:
            self.vtk.recolor_scene(instance_colors(self._shown_labels))
            self.details.setPlainText(self._seg_summary())
        else:
            base = instance_colors(self._shown_labels)
            dimmed = (base.astype(np.float32) * 0.18).astype(np.uint8)
            mask = self._shown_labels == instance_id
            shown = base.copy(); shown[~mask] = dimmed[~mask]
            self.vtk.recolor_scene(shown)
            inst = next((i for i in self.seg_result.instances if i.instance_id == instance_id), None)
            if inst is not None:
                ext = inst.bbox_max_camera - inst.bbox_min_camera
                self.details.setPlainText(
                    f"Instance #{inst.instance_id}\npoints: {inst.point_count}\n"
                    f"mean object prob: {inst.mean_object_probability:.3f}\n"
                    f"centroid (m): ({inst.centroid_camera[0]:.3f}, {inst.centroid_camera[1]:.3f}, {inst.centroid_camera[2]:.3f})\n"
                    f"bbox extent (m): ({ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f})"
                )

    def _pose_details(self, instance_id: int) -> str:
        if instance_id < 0:
            if not self.pose_items:
                return "No pose instances."
            confs = [it["confidence"] for it in self.pose_items]
            return (f"Scene: {self.scene_name}\npose instances: {len(self.pose_items)}\n"
                    f"top confidence: {max(confs):.3f}\nred=x, green=y, blue=z object axes")
        it = next((p for p in self.pose_items if p["id"] == instance_id), None)
        if it is None:
            return ""
        o = it["origin"]
        return (f"Instance #{it['id']}\nconfidence: {it['confidence']:.3f}\nmodel_fit: {it['model_fit']:.3f}\n"
                f"points: {it['point_count']}\norigin (m): ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f})\n"
                f"(lower model_fit = posed model fits the crop better)")

    def _seg_summary(self) -> str:
        counts = [i.point_count for i in self.seg_result.instances]
        return (f"Scene: {self.scene_name}\ninstances: {self.seg_result.instance_count}\n"
                f"foreground: {int((self.seg_result.instance_labels > 0).sum())} / {self.seg_result.points_camera.shape[0]}\n"
                f"points/instance: min {min(counts) if counts else 0}, max {max(counts) if counts else 0}")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        for thread in list(self._threads):
            thread.quit()
            thread.wait(1500)
        super().closeEvent(event)


# ---------------------------------------------------------------- loading helpers
def _load_from_bundle(sku: str, registry_root: str, device: str):
    from src.inference.device import resolve_device
    from src.inference.instance_segmentation import InstanceSegmenter
    from src.inference.pose_pipeline import PosePipeline
    from src.registry.model_registry import ModelRegistry

    registry = ModelRegistry(registry_root)
    loaded = registry.load(registry.resolve(sku), device=resolve_device(device))
    segmenter = InstanceSegmenter.from_loaded_bundle(loaded)
    pose_pipeline = PosePipeline.from_loaded_bundle(loaded)
    model_points = np.asarray(loaded.model_points_object, dtype=np.float64)
    return segmenter, pose_pipeline, model_points, float(loaded.diameter_m), f"{sku}:{loaded.bundle.version}"


def _load_from_checkpoint(path: str, device: str):
    from src.inference.instance_segmentation import InstanceSegmenter

    seg = InstanceSegmenter.from_checkpoint(path, device=device)
    return seg, None, None, 0.1, f"checkpoint:{Path(path).name}"


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
    parser = argparse.ArgumentParser(description="Inspect instance segmentation + 6D pose (PySide6 + VTK)")
    parser.add_argument("--sku", default=None, help="Preselect this registry SKU.")
    parser.add_argument("--scene", default=None, help="Preload this scene .npz.")
    parser.add_argument("--registry-root", default="models")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    viewer = BackendViewer(args)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
