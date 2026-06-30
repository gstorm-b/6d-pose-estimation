"""PySide6 GUI for tuning processed point-cloud augmentation/noise settings."""

from __future__ import annotations

import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import PointCloudAugmentationConfig, augment_point_cloud  # noqa: E402

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


FIELD_LABELS = {
    "enabled": "Enabled",
    "xyz_jitter_std": "XYZ jitter std",
    "xyz_jitter_clip": "XYZ jitter clip",
    "depth_noise_std": "Depth noise std",
    "point_dropout_prob": "Point dropout",
    "outlier_ratio": "Outlier ratio",
    "outlier_std": "Outlier std",
    "normal_jitter_std": "Normal jitter std",
    "random_z_rotation": "Random Z rotation",
    "depth_quadratic_noise": "Depth quadratic",
    "depth_quantization_m": "Depth quantization",
    "incidence_dropout_max_prob": "Incidence dropout",
    "incidence_dropout_power": "Incidence power",
    "edge_dropout_prob": "Edge dropout",
    "edge_dropout_neighbors": "Edge neighbors",
    "edge_dropout_gap_m": "Edge gap",
    "blob_dropout_count": "Blob count",
    "blob_dropout_radius_m": "Blob radius",
    "camera_fallback_standoff_m": "Camera standoff",
}

FIELD_TOOLTIPS = {
    "xyz_jitter_std": "Gaussian jitter added to x/y/z in meters.",
    "xyz_jitter_clip": "Clamp absolute XYZ jitter to this meter value.",
    "depth_noise_std": "Gaussian noise on camera z/depth in meters.",
    "point_dropout_prob": "Random point dropout probability; dropped slots are refilled.",
    "outlier_ratio": "Fraction of points replaced with random outliers.",
    "outlier_std": "Outlier spread around the cloud center in meters.",
    "normal_jitter_std": "Gaussian jitter added to normals before re-normalization.",
    "depth_quadratic_noise": "Along-ray std coefficient: std = coefficient * range^2.",
    "depth_quantization_m": "Quantize camera z/depth to this meter step.",
    "incidence_dropout_max_prob": "Max dropout probability for grazing surfaces.",
    "incidence_dropout_power": "Higher values limit incidence dropout to steeper grazing angles.",
    "edge_dropout_prob": "Dropout probability for points near depth discontinuities.",
    "edge_dropout_neighbors": "Neighbor count used to detect depth discontinuities in XY.",
    "edge_dropout_gap_m": "Z gap threshold in meters for edge dropout.",
    "blob_dropout_count": "Number of local dropout holes per cloud.",
    "blob_dropout_radius_m": "Radius of each local dropout hole in meters.",
    "camera_fallback_standoff_m": "Fallback camera distance when the processed sample has no camera position.",
}


def find_augment_block(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if isinstance(node.get("augment"), dict):
            return node["augment"]
        for value in node.values():
            found = find_augment_block(value)
            if found is not None:
                return found
    return None


def set_augment_block(node: dict[str, Any], augment: dict[str, Any]) -> dict[str, Any]:
    found = find_augment_block(node)
    if found is not None:
        found.clear()
        found.update(augment)
        return node
    dataset = node.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        node["dataset"] = {}
        dataset = node["dataset"]
    dataset["augment"] = augment
    return node


def compact_float(value: float) -> float:
    return float(f"{value:.9g}")


def semantic_colors(semantic: np.ndarray) -> np.ndarray:
    colors = np.full((semantic.shape[0], 3), [132, 132, 132], dtype=np.uint8)
    colors[np.asarray(semantic) > 0] = [40, 190, 85]
    return colors


def changed_colors(
    clean_points: np.ndarray,
    noisy_points: np.ndarray,
    clean_semantic: np.ndarray,
    noisy_semantic: np.ndarray,
) -> np.ndarray:
    colors = semantic_colors(noisy_semantic)
    displacement = np.linalg.norm(noisy_points - clean_points, axis=1)
    changed = (displacement > 1e-6) | (np.asarray(clean_semantic) != np.asarray(noisy_semantic))
    colors[changed] = [235, 65, 45]
    return colors


def project_points(
    points: np.ndarray,
    axes: tuple[int, int],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    lo = bounds_min[list(axes)]
    hi = bounds_max[list(axes)]
    span = np.maximum(hi - lo, 1e-6)
    uv = (points[:, list(axes)] - lo) / span
    x = padding + uv[:, 0] * max(1, width - 2 * padding - 1)
    y = padding + (1.0 - uv[:, 1]) * max(1, height - 2 * padding - 1)
    return x.astype(np.int32), y.astype(np.int32)


def raster_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    axes: tuple[int, int],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    width: int,
    height: int,
    title: str,
) -> QImage:
    image = np.full((height, width, 3), 246, dtype=np.uint8)
    px, py = project_points(points, axes, bounds_min, bounds_max, width, height, padding=14)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid]
    py = py[valid]
    point_colors = colors[valid]
    for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
        xx = np.clip(px + dx, 0, width - 1)
        yy = np.clip(py + dy, 0, height - 1)
        image[yy, xx] = point_colors
    qimage = QImage(image.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    painter = QPainter(qimage)
    painter.fillRect(0, 0, width, 22, Qt.GlobalColor.white)
    painter.drawText(8, 16, title)
    painter.end()
    return qimage


def combine_images(left: QImage, right: QImage, gap: int = 16) -> QImage:
    out = QImage(left.width() + right.width() + gap, max(left.height(), right.height()), QImage.Format.Format_RGB888)
    out.fill(Qt.GlobalColor.white)
    painter = QPainter(out)
    painter.drawImage(0, 0, left)
    painter.drawImage(left.width() + gap, 0, right)
    painter.end()
    return out


class VtkNoisePreviewPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.renderer = None
        self.vtk_widget = None
        self.clean_actor = None
        self.noisy_actor = None
        self.axes_actor = None
        self.label_actors: list[Any] = []

        if not VTK_AVAILABLE or VTK_RENDER_DISABLED:
            if VTK_RENDER_DISABLED:
                self.status_label.setText("VTK rendering is disabled for offscreen Qt sessions.")
            else:
                self.status_label.setText(
                    "VTK is not installed. Install the optional dependency with: pip install vtk\n"
                    f"Import error: {VTK_IMPORT_ERROR}"
                )
            return

        self.status_label.setText("Load a processed sample to preview clean/noisy point clouds in 3D.")
        self.vtk_widget = QVTKRenderWindowInteractor(self)  # type: ignore[misc]
        layout.addWidget(self.vtk_widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.02, 0.025, 0.03)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vtk_widget.Initialize()

    def _point_polydata(self, points: np.ndarray, colors: np.ndarray) -> Any:
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points.astype(np.float32)), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(poly)
        glyph.Update()
        out = glyph.GetOutput()
        color_array = numpy_to_vtk(
            np.ascontiguousarray(colors.astype(np.uint8)),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        color_array.SetName("colors")
        out.GetPointData().SetScalars(color_array)
        return out

    def _make_actor(self, points: np.ndarray, colors: np.ndarray, point_size: int) -> Any:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._point_polydata(points, colors))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        return actor

    def _add_label(self, text: str, position: np.ndarray) -> None:
        if self.renderer is None:
            return
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
        actor.GetTextProperty().SetColor(0.92, 0.92, 0.92)
        actor.GetTextProperty().SetFontSize(16)
        self.renderer.AddActor(actor)
        self.label_actors.append(actor)

    def clear(self, message: str) -> None:
        self.status_label.setText(message)
        if self.renderer is not None and self.vtk_widget is not None:
            self.renderer.RemoveAllViewProps()
            self.clean_actor = None
            self.noisy_actor = None
            self.axes_actor = None
            self.label_actors = []
            self.vtk_widget.GetRenderWindow().Render()

    def set_clouds(
        self,
        clean_points: np.ndarray,
        noisy_points: np.ndarray,
        clean_colors: np.ndarray,
        noisy_colors: np.ndarray,
        *,
        point_size: int,
        mode: str,
        reset_camera: bool,
    ) -> None:
        if not VTK_AVAILABLE or self.renderer is None or self.vtk_widget is None:
            self.status_label.setText("VTK is not available.")
            return
        if clean_points.size == 0 or noisy_points.size == 0:
            self.clear("Point cloud is empty.")
            return

        span = np.maximum(np.ptp(np.vstack([clean_points, noisy_points]), axis=0), 1e-6)
        offset = max(float(span[0]) * 1.35, float(span.max()) * 0.45, 0.04)
        if mode == "side-by-side":
            clean_render = clean_points.copy()
            noisy_render = noisy_points.copy()
            clean_render[:, 0] -= offset * 0.5
            noisy_render[:, 0] += offset * 0.5
            clean_label_pos = clean_render.mean(axis=0) + np.array([0.0, 0.0, float(span[2]) * 0.62 + 0.01])
            noisy_label_pos = noisy_render.mean(axis=0) + np.array([0.0, 0.0, float(span[2]) * 0.62 + 0.01])
        else:
            clean_render = clean_points
            noisy_render = noisy_points
            clean_label_pos = clean_render.mean(axis=0) + np.array([0.0, 0.0, float(span[2]) * 0.62 + 0.01])
            noisy_label_pos = clean_label_pos + np.array([float(span[0]) * 0.08 + 0.01, 0.0, 0.0])

        self.renderer.RemoveAllViewProps()
        self.clean_actor = self._make_actor(clean_render, clean_colors, point_size)
        self.noisy_actor = self._make_actor(noisy_render, noisy_colors, point_size)
        if mode == "overlay":
            self.clean_actor.GetProperty().SetOpacity(0.35)
            self.noisy_actor.GetProperty().SetOpacity(0.92)
        self.renderer.AddActor(self.clean_actor)
        self.renderer.AddActor(self.noisy_actor)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.05, 0.05, 0.05)
        self.renderer.AddActor(axes)
        self.axes_actor = axes
        self.label_actors = []
        self._add_label("clean", clean_label_pos)
        self._add_label("noisy", noisy_label_pos)

        if reset_camera:
            self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.status_label.setText(
            f"3D {mode}: clean + noisy. Controls: left-drag rotate, middle-drag pan, right-drag/wheel zoom."
        )

    def set_point_size(self, point_size: int) -> None:
        if self.vtk_widget is None:
            return
        for actor in (self.clean_actor, self.noisy_actor):
            if actor is not None:
                actor.GetProperty().SetPointSize(point_size)
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        if self.renderer is not None and self.vtk_widget is not None:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()


class NoiseTuningWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Processed Data Noise Tuning")
        self.resize(1480, 920)

        self.default_dataset_root = PROJECT_ROOT / "processed-data" / "pointnet2_semseg_k41144"
        self.sample_path: Path | None = None
        self.config_path: Path | None = PROJECT_ROOT / "configs" / "train" / "pointnet2_instance_k41144.yaml"
        self.raw_points: np.ndarray | None = None
        self.preview_points: np.ndarray | None = None
        self.clean_features: np.ndarray | None = None
        self.clean_semantic: np.ndarray | None = None
        self.clean_instance: np.ndarray | None = None
        self._reset_3d_camera_next = True

        self.controls: dict[str, QCheckBox | QDoubleSpinBox | QSpinBox] = {}
        self._build_ui()
        if self.config_path.exists():
            self.config_edit.setText(str(self.config_path))
            self.load_config(self.config_path)
        if not self.dataset_edit.text() and self.default_dataset_root.exists():
            self.dataset_edit.setText(str(self.default_dataset_root))
        if self.dataset_edit.text() and Path(self.dataset_edit.text()).exists():
            self.refresh_samples()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)
        self.setCentralWidget(root)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        data_group = QGroupBox("Data")
        data_form = QFormLayout(data_group)
        self.dataset_edit = QLineEdit()
        self.config_edit = QLineEdit()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(0)
        self.seed_spin.valueChanged.connect(self.refresh_preview)
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["XY", "XZ", "YZ"])
        self.projection_combo.currentTextChanged.connect(self.refresh_preview)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["semantic", "changed"])
        self.color_combo.currentTextChanged.connect(self.refresh_preview)
        self.view3d_combo = QComboBox()
        self.view3d_combo.addItems(["side-by-side", "overlay"])
        self.view3d_combo.currentTextChanged.connect(self.refresh_preview)
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(4)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)
        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(["scene_center", "none"])
        self.normalize_combo.currentTextChanged.connect(self.refresh_preview)

        dataset_buttons = QHBoxLayout()
        browse_dataset = QPushButton("Browse Dataset")
        browse_dataset.clicked.connect(self.browse_dataset)
        refresh_samples = QPushButton("Refresh")
        refresh_samples.clicked.connect(self.refresh_samples)
        dataset_buttons.addWidget(browse_dataset)
        dataset_buttons.addWidget(refresh_samples)

        config_buttons = QHBoxLayout()
        browse_config = QPushButton("Browse Config")
        browse_config.clicked.connect(self.browse_config)
        reload_config = QPushButton("Reload")
        reload_config.clicked.connect(self.reload_config)
        config_buttons.addWidget(browse_config)
        config_buttons.addWidget(reload_config)

        data_form.addRow("Processed dataset", self.dataset_edit)
        data_form.addRow("", dataset_buttons)
        data_form.addRow("Train config", self.config_edit)
        data_form.addRow("", config_buttons)
        data_form.addRow("Seed", self.seed_spin)
        data_form.addRow("Normalize", self.normalize_combo)
        data_form.addRow("Color", self.color_combo)
        data_form.addRow("3D view", self.view3d_combo)
        data_form.addRow("Point size", self.point_size_spin)
        data_form.addRow("2D projection", self.projection_combo)
        left_layout.addWidget(data_group)

        self.sample_list = QListWidget()
        self.sample_list.currentTextChanged.connect(self.on_sample_selected)
        sample_group = QGroupBox("Samples")
        sample_layout = QVBoxLayout(sample_group)
        sample_layout.addWidget(self.sample_list)
        left_layout.addWidget(sample_group)

        noise_group = QGroupBox("Augmentation")
        noise_form = QFormLayout(noise_group)
        self._add_controls(noise_form)
        left_layout.addWidget(noise_group)

        actions = QGridLayout()
        apply_button = QPushButton("Apply Preview")
        apply_button.clicked.connect(self.refresh_preview)
        reset_button = QPushButton("Reset Off")
        reset_button.clicked.connect(self.reset_off)
        save_preset_button = QPushButton("Save Augment YAML")
        save_preset_button.clicked.connect(self.save_augment_yaml)
        save_config_button = QPushButton("Save Config As")
        save_config_button.clicked.connect(self.save_config_as)
        update_config_button = QPushButton("Update Config")
        update_config_button.clicked.connect(self.update_config_in_place)
        reset_camera_button = QPushButton("Reset 3D Camera")
        reset_camera_button.clicked.connect(self.reset_3d_camera)
        actions.addWidget(apply_button, 0, 0)
        actions.addWidget(reset_button, 0, 1)
        actions.addWidget(save_preset_button, 1, 0)
        actions.addWidget(save_config_button, 1, 1)
        actions.addWidget(update_config_button, 2, 0, 1, 2)
        actions.addWidget(reset_camera_button, 3, 0, 1, 2)
        left_layout.addLayout(actions)
        left_layout.addStretch(1)

        self.vtk_preview = VtkNoisePreviewPanel()
        self.preview_label = QLabel("Load a processed sample to preview noise.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(880, 520)
        self.preview_label.setStyleSheet("background: #f5f5f5; border: 1px solid #cccccc;")
        preview_tabs = QTabWidget()
        preview_tabs.addTab(self.vtk_preview, "3D Point Cloud")
        preview_tabs.addTab(self.preview_label, "2D Projection")
        right_layout.addWidget(preview_tabs, stretch=1)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.yaml_text = QTextEdit()
        self.yaml_text.setReadOnly(True)
        bottom.addWidget(self.stats_text)
        bottom.addWidget(self.yaml_text)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        right_layout.addWidget(bottom, stretch=0)

    def _add_controls(self, form: QFormLayout) -> None:
        default = PointCloudAugmentationConfig(enabled=True)
        for field in fields(PointCloudAugmentationConfig):
            name = field.name
            value = getattr(default, name)
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.stateChanged.connect(self.refresh_preview)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(0, 5000)
                if name == "edge_dropout_neighbors":
                    widget.setRange(1, 256)
                widget.setValue(int(value))
                widget.valueChanged.connect(self.refresh_preview)
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(0.0, 10.0)
                widget.setSingleStep(0.0001)
                if "prob" in name or name == "outlier_ratio":
                    widget.setRange(0.0, 1.0)
                    widget.setSingleStep(0.01)
                elif name.endswith("_power"):
                    widget.setRange(0.0, 20.0)
                    widget.setSingleStep(0.1)
                elif name.endswith("_standoff_m"):
                    widget.setRange(0.0, 5.0)
                    widget.setSingleStep(0.05)
                elif name in {"normal_jitter_std"}:
                    widget.setRange(0.0, 2.0)
                    widget.setSingleStep(0.01)
                widget.setValue(float(value))
                widget.valueChanged.connect(self.refresh_preview)
            widget.setToolTip(FIELD_TOOLTIPS.get(name, ""))
            self.controls[name] = widget
            form.addRow(FIELD_LABELS.get(name, name), widget)

    def browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select processed dataset", self.dataset_edit.text() or str(PROJECT_ROOT))
        if path:
            self.dataset_edit.setText(path)
            self.refresh_samples()

    def browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select train config", self.config_edit.text() or str(PROJECT_ROOT), "YAML files (*.yaml *.yml);;All files (*)")
        if path:
            self.config_edit.setText(path)
            self.load_config(Path(path))

    def reload_config(self) -> None:
        path = Path(self.config_edit.text())
        if path.exists():
            self.load_config(path)

    def refresh_samples(self) -> None:
        root = Path(self.dataset_edit.text())
        self.sample_list.clear()
        if not root.exists():
            return
        samples = []
        for split in ("train", "val", "test"):
            split_dir = root / split
            if split_dir.exists():
                samples.extend(sorted(split_dir.glob("*.npz")))
        if not samples:
            samples.extend(sorted(root.glob("**/*.npz")))
        for sample in samples:
            self.sample_list.addItem(str(sample.relative_to(root)))
        if self.sample_list.count() > 0:
            self.sample_list.setCurrentRow(0)

    def on_sample_selected(self, relative: str) -> None:
        if not relative:
            return
        path = Path(self.dataset_edit.text()) / relative
        self.load_sample(path)

    def load_sample(self, path: Path) -> None:
        try:
            with np.load(path) as data:
                self.raw_points = np.asarray(data["points"], dtype=np.float32)
                self.clean_features = np.asarray(data["features"], dtype=np.float32) if "features" in data.files else None
                self.clean_semantic = np.asarray(data["semantic_labels"], dtype=np.int64)
                self.clean_instance = np.asarray(data["instance_labels"], dtype=np.int64)
            self.sample_path = path
            self._reset_3d_camera_next = True
            self.refresh_preview()
        except Exception as exc:  # noqa: BLE001 - surface sample/schema problems in the GUI.
            QMessageBox.warning(self, "Could not load sample", str(exc))

    def load_config(self, path: Path) -> None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                dataset = data.get("dataset", {})
                if isinstance(dataset, dict):
                    root = dataset.get("root")
                    if root:
                        root_path = Path(str(root))
                        if not root_path.is_absolute():
                            root_path = PROJECT_ROOT / root_path
                        self.dataset_edit.setText(str(root_path))
                    normalize = str(dataset.get("normalize", "")).strip()
                    if normalize in {"scene_center", "none"}:
                        blocked = self.normalize_combo.blockSignals(True)
                        self.normalize_combo.setCurrentText(normalize)
                        self.normalize_combo.blockSignals(blocked)
            augment = find_augment_block(data) or {}
            config = PointCloudAugmentationConfig.from_mapping(augment)
            self.set_config(config)
            self.config_path = path
            if self.dataset_edit.text() and Path(self.dataset_edit.text()).exists():
                self.refresh_samples()
            self.refresh_preview()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not load config", str(exc))

    def set_config(self, config: PointCloudAugmentationConfig) -> None:
        for name, widget in self.controls.items():
            value = getattr(config, name)
            blocked = widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            widget.blockSignals(blocked)

    def current_augment_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, widget in self.controls.items():
            if isinstance(widget, QCheckBox):
                values[name] = bool(widget.isChecked())
            elif isinstance(widget, QSpinBox):
                values[name] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = compact_float(widget.value())
        return values

    def current_config(self) -> PointCloudAugmentationConfig:
        return PointCloudAugmentationConfig.from_mapping(self.current_augment_dict())

    def reset_off(self) -> None:
        self.set_config(PointCloudAugmentationConfig(enabled=False))
        self.refresh_preview()

    def on_point_size_changed(self, value: int) -> None:
        self.vtk_preview.set_point_size(value)
        self.refresh_preview()

    def reset_3d_camera(self) -> None:
        self.vtk_preview.reset_camera()

    def normalized_points_and_camera(self) -> tuple[np.ndarray, np.ndarray | None]:
        assert self.raw_points is not None
        mode = self.normalize_combo.currentText()
        if mode == "none":
            return self.raw_points.astype(np.float32), np.zeros(3, dtype=np.float32)
        if mode == "scene_center":
            center = self.raw_points.mean(axis=0).astype(np.float32)
            return (self.raw_points - center.reshape(1, 3)).astype(np.float32), -center.reshape(3)
        raise ValueError(f"Unknown normalization mode: {mode}")

    def refresh_preview(self) -> None:
        self.yaml_text.setPlainText(yaml.safe_dump({"augment": self.current_augment_dict()}, sort_keys=False))
        if self.raw_points is None or self.clean_semantic is None or self.clean_instance is None:
            return
        config = self.current_config()
        rng = np.random.default_rng(self.seed_spin.value())
        points, camera_position = self.normalized_points_and_camera()
        self.preview_points = points
        noisy_points, _features, noisy_semantic, _instance = augment_point_cloud(
            points,
            self.clean_features,
            self.clean_semantic,
            self.clean_instance,
            config,
            rng,
            camera_position=camera_position,
        )
        self.update_stats(config, points, camera_position, noisy_points, noisy_semantic)
        self.update_vtk_preview(noisy_points, noisy_semantic)
        self.update_preview_image(noisy_points, noisy_semantic)

    def update_stats(
        self,
        config: PointCloudAugmentationConfig,
        clean_points: np.ndarray,
        camera_position: np.ndarray | None,
        noisy_points: np.ndarray,
        noisy_semantic: np.ndarray,
    ) -> None:
        assert self.clean_semantic is not None
        displacement = np.linalg.norm(noisy_points - clean_points, axis=1)
        moved = displacement > 1e-6
        label_changed = np.asarray(noisy_semantic) != np.asarray(self.clean_semantic)
        object_to_background = (np.asarray(self.clean_semantic) > 0) & (np.asarray(noisy_semantic) == 0)
        camera_text = "unknown" if camera_position is None else np.array2string(camera_position, precision=4)
        lines = [
            f"sample: {self.sample_path}",
            f"points: {clean_points.shape[0]}",
            f"normalize: {self.normalize_combo.currentText()}",
            f"camera position in preview frame: {camera_text}",
            f"augmentation enabled: {config.enabled}",
            f"moved points: {int(moved.sum())} ({100.0 * float(moved.mean()):.2f}%)",
            f"mean displacement: {float(displacement[moved].mean()) * 1000.0 if moved.any() else 0.0:.3f} mm",
            f"p95 displacement: {float(np.percentile(displacement, 95)) * 1000.0:.3f} mm",
            f"max displacement: {float(displacement.max()) * 1000.0:.3f} mm",
            f"semantic changed: {int(label_changed.sum())}",
            f"object -> background: {int(object_to_background.sum())}",
            f"clean object points: {int((self.clean_semantic > 0).sum())}",
            f"noisy object points: {int((noisy_semantic > 0).sum())}",
        ]
        self.stats_text.setPlainText("\n".join(lines))

    def preview_colors(self, noisy_points: np.ndarray, noisy_semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.preview_points is not None
        assert self.clean_semantic is not None
        clean_colors = semantic_colors(self.clean_semantic)
        if self.color_combo.currentText() == "changed":
            noisy_colors = changed_colors(self.preview_points, noisy_points, self.clean_semantic, noisy_semantic)
        else:
            noisy_colors = semantic_colors(noisy_semantic)
        return clean_colors, noisy_colors

    def update_vtk_preview(self, noisy_points: np.ndarray, noisy_semantic: np.ndarray) -> None:
        assert self.preview_points is not None
        clean_colors, noisy_colors = self.preview_colors(noisy_points, noisy_semantic)
        reset_camera = self._reset_3d_camera_next
        self._reset_3d_camera_next = False
        self.vtk_preview.set_clouds(
            self.preview_points,
            noisy_points,
            clean_colors,
            noisy_colors,
            point_size=self.point_size_spin.value(),
            mode=self.view3d_combo.currentText(),
            reset_camera=reset_camera,
        )

    def update_preview_image(self, noisy_points: np.ndarray, noisy_semantic: np.ndarray) -> None:
        assert self.preview_points is not None
        assert self.clean_semantic is not None
        projection = self.projection_combo.currentText()
        axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[projection]
        all_points = np.vstack([self.preview_points, noisy_points])
        bounds_min = all_points.min(axis=0)
        bounds_max = all_points.max(axis=0)
        span = np.maximum(bounds_max - bounds_min, 1e-6)
        bounds_min -= span * 0.04
        bounds_max += span * 0.04

        clean_colors, noisy_colors = self.preview_colors(noisy_points, noisy_semantic)
        width = max(360, int((self.preview_label.width() - 32) / 2))
        height = max(320, self.preview_label.height())
        clean_img = raster_cloud(self.preview_points, clean_colors, axes, bounds_min, bounds_max, width, height, f"clean {projection}")
        noisy_img = raster_cloud(noisy_points, noisy_colors, axes, bounds_min, bounds_max, width, height, f"noisy {projection}")
        combined = combine_images(clean_img, noisy_img)
        self.preview_label.setPixmap(QPixmap.fromImage(combined))

    def save_augment_yaml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save augment YAML",
            str(PROJECT_ROOT / "configs" / "noise_preset.yaml"),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump({"augment": self.current_augment_dict()}, sort_keys=False), encoding="utf-8")

    def save_config_as(self) -> None:
        source = Path(self.config_edit.text())
        if not source.exists():
            QMessageBox.warning(self, "Missing config", "Load a train config before saving a full config copy.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save train config as",
            str(source.with_name(source.stem + "_noise_tuned.yaml")),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Invalid config", "Train config must be a YAML mapping.")
            return
        set_augment_block(data, self.current_augment_dict())
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def update_config_in_place(self) -> None:
        source = Path(self.config_edit.text())
        if not source.exists():
            QMessageBox.warning(self, "Missing config", "Load a train config before updating it.")
            return
        answer = QMessageBox.question(
            self,
            "Update config?",
            f"Overwrite the augment block in:\n{source}\n\nComments may not be preserved by YAML serialization.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Invalid config", "Train config must be a YAML mapping.")
            return
        set_augment_block(data, self.current_augment_dict())
        source.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    app = QApplication(sys.argv)
    window = NoiseTuningWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
