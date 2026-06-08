"""PySide6 GUI for generating and inspecting raw synthetic datasets."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPoint, QProcess, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.raw_dataset import RawSample, discover_samples, sample_basic_stats  # noqa: E402
from src.data.validation import DatasetValidationReport, validate_dataset  # noqa: E402

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


BLENDER_DEFAULT = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_synthetic_blender.py"
PREVIEW_FILES = (
    ("RGB", "rgb.png"),
    ("Overlay", "label_overlay.png"),
    ("Depth", "depth_preview.png"),
    ("Instance Mask", "instance_mask.png"),
    ("Normals", "normal_camera.png"),
)


class DatasetLoadWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(int, str, object)
    failed = Signal(int, str)

    def __init__(self, token: int, dataset_root: Path) -> None:
        super().__init__()
        self.token = token
        self.dataset_root = dataset_root

    @Slot()
    def run(self) -> None:
        try:
            rows: list[dict[str, Any]] = []
            samples = discover_samples(self.dataset_root)
            for index, sample in enumerate(samples, start=1):
                if QThread.currentThread().isInterruptionRequested():
                    self.failed.emit(self.token, "Dataset loading was interrupted.")
                    return
                self.progress.emit(self.token, f"Loading {sample.name} ({index}/{len(samples)})")
                try:
                    stats = sample_basic_stats(sample)
                    error = ""
                except Exception as exc:  # noqa: BLE001 - keep loading other samples.
                    stats = {}
                    error = str(exc)
                rows.append(
                    {
                        "name": sample.name,
                        "path": str(sample.path),
                        "stats": stats,
                        "error": error,
                    }
                )
            self.finished.emit(self.token, str(self.dataset_root), rows)
        except Exception as exc:  # noqa: BLE001 - surface worker errors to the GUI.
            self.failed.emit(self.token, str(exc))


class SampleLoadWorker(QObject):
    finished = Signal(int, str, str, object)
    failed = Signal(int, str, str)

    def __init__(self, token: int, sample_path: Path) -> None:
        super().__init__()
        self.token = token
        self.sample_path = sample_path

    @Slot()
    def run(self) -> None:
        sample = RawSample(self.sample_path.name, self.sample_path)
        try:
            stats = sample_basic_stats(sample)
            metadata = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
            details = json.dumps({"stats": stats, "settings": metadata.get("settings", {})}, indent=2)
            images: dict[str, dict[str, Any]] = {}
            for _title, file_name in PREVIEW_FILES:
                path = sample.path / file_name
                if not path.exists():
                    images[file_name] = {"image": None, "error": f"Missing: {path.name}"}
                    continue
                image = QImage(str(path))
                if image.isNull():
                    images[file_name] = {"image": None, "error": f"Could not load: {path.name}"}
                else:
                    images[file_name] = {"image": image, "error": ""}
            self.finished.emit(self.token, sample.name, details, images)
        except Exception as exc:  # noqa: BLE001 - display bad sample details instead of crashing.
            self.failed.emit(self.token, sample.name, str(exc))


class PointCloudLoadWorker(QObject):
    finished = Signal(int, str, object, object)
    failed = Signal(int, str, str)

    def __init__(self, token: int, sample_path: Path, points_key: str) -> None:
        super().__init__()
        self.token = token
        self.sample_path = sample_path
        self.points_key = points_key

    @Slot()
    def run(self) -> None:
        sample = RawSample(self.sample_path.name, self.sample_path)
        try:
            with np.load(sample.sensor_path) as data:
                if self.points_key not in data.files:
                    raise KeyError(f"missing {self.points_key} in sensor_data.npz")
                points = np.asarray(data[self.points_key], dtype=np.float32)
                instance_ids = np.asarray(data["point_instance_ids"], dtype=np.uint16)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"{self.points_key} must have shape (N, 3), got {points.shape}")
            if instance_ids.shape != (points.shape[0],):
                raise ValueError("point_instance_ids length does not match points")
            self.finished.emit(self.token, sample.name, points, instance_ids)
        except Exception as exc:  # noqa: BLE001 - surface point cloud load failure to GUI.
            self.failed.emit(self.token, sample.name, str(exc))


class ValidationWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, token: int, dataset_root: Path, project_root: Path) -> None:
        super().__init__()
        self.token = token
        self.dataset_root = dataset_root
        self.project_root = project_root

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(self.token, f"Validating {self.dataset_root}")
            report = validate_dataset(self.dataset_root, project_root=self.project_root)
            self.finished.emit(self.token, report)
        except Exception as exc:  # noqa: BLE001 - surface validation errors to the GUI.
            self.failed.emit(self.token, str(exc))


class PanZoomImageView(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.image_label = QLabel("No sample selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(True)
        self.setWidget(self.image_label)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 200)
        self.setMouseTracking(True)
        self.original_image = QImage()
        self.zoom_percent = 100
        self._dragging = False
        self._drag_start = QPoint()
        self._h_scroll_start = 0
        self._v_scroll_start = 0

    def set_image(self, image: QImage) -> None:
        self.original_image = image
        self._apply_zoom()

    def clear(self, message: str) -> None:
        self.original_image = QImage()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(message)
        self.image_label.adjustSize()

    def set_zoom_percent(self, value: int) -> None:
        self.zoom_percent = max(5, min(800, value))
        self._apply_zoom()

    def fit_to_view(self) -> int:
        if self.original_image.isNull():
            return self.zoom_percent
        viewport_size = self.viewport().size()
        if self.original_image.width() == 0 or self.original_image.height() == 0:
            return self.zoom_percent
        zoom = int(
            min(
                viewport_size.width() / self.original_image.width(),
                viewport_size.height() / self.original_image.height(),
            )
            * 100
        )
        self.set_zoom_percent(max(5, min(800, zoom)))
        return self.zoom_percent

    def _apply_zoom(self) -> None:
        if self.original_image.isNull():
            return
        pixmap = QPixmap.fromImage(self.original_image)
        width = max(1, int(self.original_image.width() * self.zoom_percent / 100.0))
        height = max(1, int(self.original_image.height() * self.zoom_percent / 100.0))
        self.image_label.setText("")
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(width, height)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._h_scroll_start = self.horizontalScrollBar().value()
            self._v_scroll_start = self.verticalScrollBar().value()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            delta = event.position().toPoint() - self._drag_start
            self.horizontalScrollBar().setValue(self._h_scroll_start - delta.x())
            self.verticalScrollBar().setValue(self._v_scroll_start - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class PreviewImagePanel(QWidget):
    def __init__(self, title: str, file_name: str) -> None:
        super().__init__()
        self.file_name = file_name
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view = PanZoomImageView()
        layout.addWidget(self.title_label)
        layout.addWidget(self.view, 1)

    def set_image(self, image: QImage) -> None:
        self.view.set_image(image)

    def clear(self, message: str) -> None:
        self.view.clear(message)

    def set_zoom_percent(self, value: int) -> None:
        self.view.set_zoom_percent(value)

    def fit_to_view(self) -> int:
        return self.view.fit_to_view()

    def set_view_size(self, value: int) -> None:
        self.view.setMinimumSize(value, value)


class VtkPointCloudPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.renderer = None
        self.vtk_widget = None
        self.point_actor = None
        self.interactor_style = None

        if not VTK_AVAILABLE or VTK_RENDER_DISABLED:
            if VTK_RENDER_DISABLED:
                self.status_label.setText("VTK rendering is disabled for offscreen Qt sessions.")
                return
            self.status_label.setText(
                "VTK is not installed. Install the optional dependency with: pip install vtk\n"
                f"Import error: {VTK_IMPORT_ERROR}"
            )
            return

        self.status_label.setText("No point cloud loaded")
        self.vtk_widget = QVTKRenderWindowInteractor(self)  # type: ignore[misc]
        layout.addWidget(self.vtk_widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.02, 0.025, 0.03)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.set_trackball_camera_style()
        self.vtk_widget.Initialize()

    def set_trackball_camera_style(self) -> None:
        if not VTK_AVAILABLE or self.vtk_widget is None or self.renderer is None:
            return
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor_style.SetDefaultRenderer(self.renderer)
        interactor.SetInteractorStyle(self.interactor_style)
        self.status_label.setText("Interaction: left-drag rotate, middle-drag pan, right-drag/wheel zoom")

    def clear(self, message: str) -> None:
        self.status_label.setText(message)
        if self.renderer is not None:
            self.renderer.RemoveAllViewProps()
            self.vtk_widget.GetRenderWindow().Render()

    def set_point_cloud(self, points: np.ndarray, instance_ids: np.ndarray, point_size: int) -> None:
        if not VTK_AVAILABLE or self.renderer is None:
            self.status_label.setText("VTK is not available.")
            return
        if points.size == 0:
            self.clear("Point cloud is empty")
            return

        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(points.shape[0])
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("instance_colors")

        for idx, point in enumerate(points):
            vtk_points.SetPoint(idx, float(point[0]), float(point[1]), float(point[2]))
            color = self._instance_color(int(instance_ids[idx]))
            colors.InsertNextTuple3(color[0], color[1], color[2])

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        vertices = vtk.vtkCellArray()
        for idx in range(points.shape[0]):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(idx)
        poly_data.SetVerts(vertices)
        poly_data.GetPointData().SetScalars(colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.05, 0.05, 0.05)
        self.renderer.AddActor(axes)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.status_label.setText(
            f"Loaded {points.shape[0]} points, {len(np.unique(instance_ids))} instance ids. "
            "Controls: left-drag rotate, middle-drag pan, right-drag/wheel zoom."
        )

    def set_point_size(self, point_size: int) -> None:
        if not VTK_AVAILABLE or self.renderer is None:
            return
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()
        while actor:
            actor.GetProperty().SetPointSize(point_size)
            actor = actors.GetNextActor()
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        if VTK_AVAILABLE and self.renderer is not None:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

    def _instance_color(self, instance_id: int) -> tuple[int, int, int]:
        if instance_id == 0:
            return (96, 96, 96)
        rng = np.random.default_rng(instance_id * 7919 + 17)
        color = rng.integers(64, 255, size=3)
        return int(color[0]), int(color[1]), int(color[2])


class DatasetGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("K41144 Synthetic Dataset GUI")
        self.resize(1320, 820)
        self.process: QProcess | None = None
        self.samples: list[RawSample] = []
        self.validation_report: DatasetValidationReport | None = None
        self.preview_panels: dict[str, PreviewImagePanel] = {}
        self.worker_threads: list[QThread] = []
        self.worker_objects: list[QObject] = []
        self.dataset_load_token = 0
        self.sample_load_token = 0
        self.point_cloud_token = 0
        self.validation_token = 0
        self.current_sample_name = ""
        self.current_point_cloud: tuple[np.ndarray, np.ndarray] | None = None
        self._build_ui()
        self._set_default_paths()
        self.load_dataset(Path(self.dataset_path_edit.text()))

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_tabs())
        splitter.setSizes([380, 940])
        self.setCentralWidget(splitter)

    def _build_sidebar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout(dataset_group)
        self.dataset_path_edit = QLineEdit()
        dataset_buttons = QHBoxLayout()
        browse_dataset = QPushButton("Browse")
        browse_dataset.clicked.connect(self.browse_dataset)
        refresh_dataset = QPushButton("Refresh")
        refresh_dataset.clicked.connect(lambda: self.load_dataset(Path(self.dataset_path_edit.text())))
        dataset_buttons.addWidget(browse_dataset)
        dataset_buttons.addWidget(refresh_dataset)
        dataset_layout.addRow("Raw dataset", self.dataset_path_edit)
        dataset_layout.addRow("", dataset_buttons)
        layout.addWidget(dataset_group)

        generation_group = QGroupBox("Generation")
        form = QFormLayout(generation_group)

        self.blender_path_edit = QLineEdit(BLENDER_DEFAULT)
        self.model_path_edit = QLineEdit("object-model/K41144.stl")
        self.class_name_edit = QLineEdit("K41144")
        self.class_name_edit.setPlaceholderText("Defaults to STL filename stem")
        self.output_path_edit = QLineEdit()
        form.addRow("Blender", self._path_row(self.blender_path_edit, self.browse_blender))
        form.addRow("Model", self._path_row(self.model_path_edit, self.browse_model))
        form.addRow("Class name", self.class_name_edit)
        form.addRow("Output", self._path_row(self.output_path_edit, self.browse_output))

        self.samples_spin = self._spin(1, 100000, 3)
        self.objects_spin = self._spin(1, 200, 30)
        self.width_spin = self._spin(32, 4096, 320)
        self.height_spin = self._spin(32, 4096, 240)
        self.model_scale_spin = self._double_spin(0.000001, 1000.0, 1.0, 6, 0.001)
        self.model_scale_spin.setToolTip("Scale applied to STL vertices before simulation. Use 0.001 for millimeter STL files.")
        self.depth_camera_location_row, self.depth_camera_location_spins = self._xyz_spins((0.0, -0.05, 0.42))
        self.depth_camera_target_row, self.depth_camera_target_spins = self._xyz_spins((0.0, 0.0, 0.025))
        self.depth_camera_lens_spin = self._double_spin(1.0, 300.0, 35.0, 2, 1.0)
        self.rgb_camera_location_row, self.rgb_camera_location_spins = self._xyz_spins((0.0, -0.05, 0.42))
        self.rgb_camera_target_row, self.rgb_camera_target_spins = self._xyz_spins((0.0, 0.0, 0.025))
        self.rgb_camera_lens_spin = self._double_spin(1.0, 300.0, 35.0, 2, 1.0)
        self.debug_camera_location_row, self.debug_camera_location_spins = self._xyz_spins((0.45, -0.55, 0.45))
        self.debug_camera_target_row, self.debug_camera_target_spins = self._xyz_spins((0.0, 0.0, 0.06))
        self.debug_camera_lens_spin = self._double_spin(1.0, 300.0, 22.0, 2, 1.0)
        self.debug_camera_lens_spin.setToolTip("Camera used only for simulation preview videos.")
        self.light_location_row, self.light_location_spins = self._xyz_spins((0.0, -0.22, 0.45))
        self.light_energy_spin = self._double_spin(0.0, 100000.0, 90.0, 2, 10.0)
        self.light_size_spin = self._double_spin(0.001, 10.0, 0.45, 3, 0.01)
        self.bin_x_spin = self._double_spin(0.01, 5.0, 0.24, 3, 0.01)
        self.bin_y_spin = self._double_spin(0.01, 5.0, 0.24, 3, 0.01)
        self.bin_wall_height_spin = self._double_spin(0.01, 5.0, 0.14, 3, 0.01)
        self.drop_height_min_spin = self._double_spin(0.001, 5.0, 0.12, 3, 0.01)
        self.drop_height_max_spin = self._double_spin(0.001, 5.0, 0.34, 3, 0.01)
        self.objects_per_layer_spin = self._spin(1, 100, 6)
        self.spawn_min_distance_spin = self._double_spin(0.0, 1.0, 0.045, 3, 0.005)
        self.collision_margin_spin = self._double_spin(0.0, 0.1, 0.00002, 6, 0.00001)
        self.object_restitution_spin = self._double_spin(0.0, 1.0, 0.05, 3, 0.01)
        self.object_restitution_spin.setToolTip("Rigid-body bounce/restitution for spawned objects. Lower values reduce rebounds.")
        self.spawn_settle_frames_spin = self._spin(0, 5000, 35)
        self.spawn_settle_frames_spin.setToolTip("If greater than 0, spawn one object, settle it for this many frames, then spawn the next object.")
        self.min_visible_objects_spin = self._spin(0, 200, 12)
        self.min_visible_points_spin = self._spin(0, 10000000, 8000)
        self.max_sample_attempts_spin = self._spin(1, 1000, 12)
        self.settle_frames_spin = self._spin(1, 5000, 260)
        self.record_video_check = QCheckBox("Record simulation video")
        self.record_video_check.setChecked(False)
        self.record_video_check.setToolTip("Write an MP4 preview of the accepted sample's drop/settle animation.")
        self.video_frame_step_spin = self._spin(1, 240, 4)
        self.video_frame_step_spin.setToolTip("Render every Nth simulation frame to keep preview videos fast.")
        self.video_fps_spin = self._spin(1, 120, 24)

        self.spawn_strategy_combo = QComboBox()
        self.spawn_strategy_combo.addItems(["layered", "random"])
        self.collision_shape_combo = QComboBox()
        self.collision_shape_combo.addItems(["CONVEX_HULL", "MESH"])
        self.allow_out_of_bin_filtering_check = QCheckBox("Legacy filter out-of-bin")
        self.allow_out_of_bin_filtering_check.setChecked(False)
        self.allow_out_of_bin_filtering_check.setToolTip(
            "Debug-only mode. Leave unchecked for training data so out-of-bin attempts are rejected."
        )

        form.addRow("Samples", self.samples_spin)
        form.addRow("Objects", self.objects_spin)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Model scale", self.model_scale_spin)
        form.addRow("Depth cam XYZ", self.depth_camera_location_row)
        form.addRow("Depth target XYZ", self.depth_camera_target_row)
        form.addRow("Depth lens", self.depth_camera_lens_spin)
        form.addRow("RGB cam XYZ", self.rgb_camera_location_row)
        form.addRow("RGB target XYZ", self.rgb_camera_target_row)
        form.addRow("RGB lens", self.rgb_camera_lens_spin)
        form.addRow("Debug cam XYZ", self.debug_camera_location_row)
        form.addRow("Debug target XYZ", self.debug_camera_target_row)
        form.addRow("Debug lens", self.debug_camera_lens_spin)
        form.addRow("Light XYZ", self.light_location_row)
        form.addRow("Light energy", self.light_energy_spin)
        form.addRow("Light size", self.light_size_spin)
        form.addRow("Bin X", self.bin_x_spin)
        form.addRow("Bin Y", self.bin_y_spin)
        form.addRow("Wall height", self.bin_wall_height_spin)
        form.addRow("Drop min", self.drop_height_min_spin)
        form.addRow("Drop max", self.drop_height_max_spin)
        form.addRow("Spawn", self.spawn_strategy_combo)
        form.addRow("Objects/layer", self.objects_per_layer_spin)
        form.addRow("Spawn min dist", self.spawn_min_distance_spin)
        form.addRow("Spawn settle frames", self.spawn_settle_frames_spin)
        form.addRow("Collision margin", self.collision_margin_spin)
        form.addRow("Collision shape", self.collision_shape_combo)
        form.addRow("Object bounce", self.object_restitution_spin)
        form.addRow("Min visible objects", self.min_visible_objects_spin)
        form.addRow("Min visible points", self.min_visible_points_spin)
        form.addRow("Max attempts", self.max_sample_attempts_spin)
        form.addRow("Settle frames", self.settle_frames_spin)
        form.addRow("Video", self.record_video_check)
        form.addRow("Video frame step", self.video_frame_step_spin)
        form.addRow("Video FPS", self.video_fps_spin)
        form.addRow("Out-of-bin", self.allow_out_of_bin_filtering_check)

        preset_actions = QHBoxLayout()
        import_preset_button = QPushButton("Import Preset")
        import_preset_button.clicked.connect(self.import_generation_settings)
        export_preset_button = QPushButton("Export Preset")
        export_preset_button.clicked.connect(self.export_generation_settings)
        preset_actions.addWidget(import_preset_button)
        preset_actions.addWidget(export_preset_button)
        form.addRow("Preset", preset_actions)

        self.start_button = QPushButton("Start Generation")
        self.start_button.clicked.connect(self.start_generation)
        self.single_sample_video_button = QPushButton("1 Sample + Video")
        self.single_sample_video_button.clicked.connect(lambda: self.start_generation(single_sample_video=True))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_generation)
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
        actions.addWidget(self.single_sample_video_button)
        actions.addWidget(self.stop_button)
        form.addRow("", actions)
        layout.addWidget(generation_group)

        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        return scroll

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_logs_tab(), "Logs")
        tabs.addTab(self._build_samples_tab(), "Samples")
        tabs.addTab(self._build_preview_tab(), "Preview")
        tabs.addTab(self._build_3d_tab(), "3D View")
        tabs.addTab(self._build_validation_tab(), "Validation")
        return tabs

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.log_text)
        return tab

    def _build_samples_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.sample_table = QTableWidget(0, 6)
        self.sample_table.setHorizontalHeaderLabels(["Sample", "Objects", "Visible", "Points", "Seed", "Policy"])
        self.sample_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.sample_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sample_table.cellClicked.connect(self.on_sample_table_clicked)
        layout.addWidget(self.sample_table, 2)
        self.sample_details = QTextEdit()
        self.sample_details.setReadOnly(True)
        layout.addWidget(self.sample_details, 1)
        return tab

    def _build_preview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.preview_view_count_combo = QComboBox()
        self.preview_view_count_combo.addItems(["1", "2", "4", "5"])
        self.preview_view_count_combo.setCurrentText("5")
        self.preview_view_count_combo.currentTextChanged.connect(self.update_preview_layout)
        self.preview_start_combo = QComboBox()
        for title, file_name in PREVIEW_FILES:
            self.preview_start_combo.addItem(title, file_name)
        self.preview_start_combo.currentIndexChanged.connect(self.update_preview_layout)
        self.preview_size_spin = self._spin(120, 900, 280)
        self.preview_size_spin.valueChanged.connect(self.update_preview_size)
        self.preview_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_zoom_slider.setRange(25, 400)
        self.preview_zoom_slider.setValue(100)
        self.preview_zoom_slider.valueChanged.connect(self.update_preview_zoom)
        self.preview_zoom_label = QLabel("100%")
        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.fit_preview_views)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.preview_zoom_slider.setValue(100))
        controls.addWidget(QLabel("Views"))
        controls.addWidget(self.preview_view_count_combo)
        controls.addWidget(QLabel("Start"))
        controls.addWidget(self.preview_start_combo)
        controls.addWidget(QLabel("Size"))
        controls.addWidget(self.preview_size_spin)
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.preview_zoom_slider, 1)
        controls.addWidget(self.preview_zoom_label)
        controls.addWidget(fit_button)
        controls.addWidget(reset_button)
        layout.addLayout(controls)

        self.preview_grid = QGridLayout()
        layout.addLayout(self.preview_grid, 1)
        for title, file_name in PREVIEW_FILES:
            panel = PreviewImagePanel(title, file_name)
            panel.set_view_size(self.preview_size_spin.value())
            self.preview_panels[file_name] = panel
            self.preview_grid.addWidget(panel, 0, 0)
        self.update_preview_layout()
        return tab

    def _build_3d_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.point_cloud_frame_combo = QComboBox()
        self.point_cloud_frame_combo.addItem("Camera points", "points_camera")
        self.point_cloud_frame_combo.addItem("World points", "points_world")
        self.point_cloud_frame_combo.currentIndexChanged.connect(self.reload_current_point_cloud)
        self.point_size_spin = self._spin(1, 20, 4)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload_current_point_cloud)
        reset_camera_button = QPushButton("Reset Camera")
        reset_camera_button.clicked.connect(lambda: self.vtk_panel.reset_camera())
        reset_interaction_button = QPushButton("Reset Interaction")
        reset_interaction_button.clicked.connect(lambda: self.vtk_panel.set_trackball_camera_style())
        controls.addWidget(QLabel("Frame"))
        controls.addWidget(self.point_cloud_frame_combo)
        controls.addWidget(QLabel("Point size"))
        controls.addWidget(self.point_size_spin)
        controls.addWidget(reload_button)
        controls.addWidget(reset_camera_button)
        controls.addWidget(reset_interaction_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.vtk_panel = VtkPointCloudPanel()
        layout.addWidget(self.vtk_panel, 1)
        return tab

    def _build_validation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        self.validate_button = QPushButton("Validate Dataset")
        self.validate_button.clicked.connect(self.run_validation)
        top.addWidget(self.validate_button)
        top.addStretch(1)
        layout.addLayout(top)
        self.validation_summary = QTextEdit()
        self.validation_summary.setReadOnly(True)
        self.validation_summary.setMaximumHeight(140)
        layout.addWidget(self.validation_summary)
        self.validation_table = QTableWidget(0, 7)
        self.validation_table.setHorizontalHeaderLabels(
            ["Sample", "Status", "Objects", "Visible", "Points", "Out of bin", "Issues"]
        )
        self.validation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.validation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.validation_table.cellClicked.connect(self.on_validation_table_clicked)
        layout.addWidget(self.validation_table)
        return tab

    def update_preview_layout(self) -> None:
        if not hasattr(self, "preview_grid"):
            return
        for panel in self.preview_panels.values():
            self.preview_grid.removeWidget(panel)
            panel.setVisible(False)
        view_count = int(self.preview_view_count_combo.currentText())
        start = self.preview_start_combo.currentIndex()
        ordered_files = [file_name for _title, file_name in PREVIEW_FILES]
        visible_files = [ordered_files[(start + idx) % len(ordered_files)] for idx in range(view_count)]
        columns = 1 if view_count == 1 else 2
        for idx, file_name in enumerate(visible_files):
            panel = self.preview_panels[file_name]
            panel.setVisible(True)
            self.preview_grid.addWidget(panel, idx // columns, idx % columns)

    def update_preview_size(self, value: int) -> None:
        for panel in self.preview_panels.values():
            panel.set_view_size(value)

    def update_preview_zoom(self, value: int) -> None:
        self.preview_zoom_label.setText(f"{value}%")
        for panel in self.preview_panels.values():
            panel.set_zoom_percent(value)

    def fit_preview_views(self) -> None:
        visible_panels = [panel for panel in self.preview_panels.values() if panel.isVisible()]
        zooms = [panel.fit_to_view() for panel in visible_panels]
        if zooms:
            zoom = max(25, min(400, min(zooms)))
            for panel in visible_panels:
                panel.set_zoom_percent(zoom)
            self.preview_zoom_slider.blockSignals(True)
            self.preview_zoom_slider.setValue(zoom)
            self.preview_zoom_label.setText(f"{self.preview_zoom_slider.value()}%")
            self.preview_zoom_slider.blockSignals(False)

    def _set_default_paths(self) -> None:
        existing_dataset = PROJECT_ROOT / "synthetic-data" / "K41144"
        self.dataset_path_edit.setText(str(existing_dataset))
        self.output_path_edit.setText(str(self._new_output_path()))

    def _new_output_path(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "synthetic-data" / f"K41144_gui_{stamp}"

    def _path_row(self, edit: QLineEdit, callback: Any) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QPushButton("...")
        button.setMaximumWidth(36)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _double_spin(self, minimum: float, maximum: float, value: float, decimals: int, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _xyz_spins(
        self,
        values: tuple[float, float, float],
        *,
        minimum: float = -10.0,
        maximum: float = 10.0,
        decimals: int = 3,
        step: float = 0.01,
    ) -> tuple[QWidget, tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        spins: list[QDoubleSpinBox] = []
        for label, value in zip(("X", "Y", "Z"), values, strict=True):
            layout.addWidget(QLabel(label))
            spin = self._double_spin(minimum, maximum, value, decimals, step)
            spin.setMinimumWidth(72)
            layout.addWidget(spin)
            spins.append(spin)
        layout.addStretch(1)
        return row, (spins[0], spins[1], spins[2])

    def _extend_xyz_arg(self, args: list[str], flag: str, spins: tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]) -> None:
        args.append(flag)
        args.extend(str(spin.value()) for spin in spins)

    def _xyz_values(self, spins: tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]) -> list[float]:
        return [spin.value() for spin in spins]

    def _set_xyz_values(self, spins: tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox], values: object) -> None:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("XYZ setting must be a list of 3 numbers")
        for spin, value in zip(spins, values, strict=True):
            spin.setValue(float(value))

    def generation_settings(self) -> dict[str, Any]:
        return {
            "model": self.model_path_edit.text(),
            "class_name": self.class_name_edit.text().strip(),
            "model_scale": self.model_scale_spin.value(),
            "output": self.output_path_edit.text(),
            "samples": self.samples_spin.value(),
            "objects": self.objects_spin.value(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "depth_camera_location": self._xyz_values(self.depth_camera_location_spins),
            "depth_camera_target": self._xyz_values(self.depth_camera_target_spins),
            "depth_camera_lens": self.depth_camera_lens_spin.value(),
            "rgb_camera_location": self._xyz_values(self.rgb_camera_location_spins),
            "rgb_camera_target": self._xyz_values(self.rgb_camera_target_spins),
            "rgb_camera_lens": self.rgb_camera_lens_spin.value(),
            "debug_camera_location": self._xyz_values(self.debug_camera_location_spins),
            "debug_camera_target": self._xyz_values(self.debug_camera_target_spins),
            "debug_camera_lens": self.debug_camera_lens_spin.value(),
            "light_location": self._xyz_values(self.light_location_spins),
            "light_energy": self.light_energy_spin.value(),
            "light_size": self.light_size_spin.value(),
            "bin_x": self.bin_x_spin.value(),
            "bin_y": self.bin_y_spin.value(),
            "bin_wall_height": self.bin_wall_height_spin.value(),
            "drop_height_min": self.drop_height_min_spin.value(),
            "drop_height_max": self.drop_height_max_spin.value(),
            "spawn_strategy": self.spawn_strategy_combo.currentText(),
            "objects_per_layer": self.objects_per_layer_spin.value(),
            "spawn_min_distance": self.spawn_min_distance_spin.value(),
            "spawn_settle_frames": self.spawn_settle_frames_spin.value(),
            "collision_margin": self.collision_margin_spin.value(),
            "collision_shape": self.collision_shape_combo.currentText(),
            "object_restitution": self.object_restitution_spin.value(),
            "min_visible_objects": self.min_visible_objects_spin.value(),
            "min_visible_points": self.min_visible_points_spin.value(),
            "max_sample_attempts": self.max_sample_attempts_spin.value(),
            "settle_frames": self.settle_frames_spin.value(),
            "record_simulation_video": self.record_video_check.isChecked(),
            "simulation_video_frame_step": self.video_frame_step_spin.value(),
            "simulation_video_fps": self.video_fps_spin.value(),
            "allow_out_of_bin_filtering": self.allow_out_of_bin_filtering_check.isChecked(),
        }

    def apply_generation_settings(self, settings: dict[str, Any]) -> None:
        line_edits = {
            "model": self.model_path_edit,
            "class_name": self.class_name_edit,
            "output": self.output_path_edit,
        }
        int_spins = {
            "samples": self.samples_spin,
            "objects": self.objects_spin,
            "width": self.width_spin,
            "height": self.height_spin,
            "objects_per_layer": self.objects_per_layer_spin,
            "spawn_settle_frames": self.spawn_settle_frames_spin,
            "min_visible_objects": self.min_visible_objects_spin,
            "min_visible_points": self.min_visible_points_spin,
            "max_sample_attempts": self.max_sample_attempts_spin,
            "settle_frames": self.settle_frames_spin,
            "simulation_video_frame_step": self.video_frame_step_spin,
            "simulation_video_fps": self.video_fps_spin,
        }
        double_spins = {
            "model_scale": self.model_scale_spin,
            "depth_camera_lens": self.depth_camera_lens_spin,
            "rgb_camera_lens": self.rgb_camera_lens_spin,
            "debug_camera_lens": self.debug_camera_lens_spin,
            "light_energy": self.light_energy_spin,
            "light_size": self.light_size_spin,
            "bin_x": self.bin_x_spin,
            "bin_y": self.bin_y_spin,
            "bin_wall_height": self.bin_wall_height_spin,
            "drop_height_min": self.drop_height_min_spin,
            "drop_height_max": self.drop_height_max_spin,
            "spawn_min_distance": self.spawn_min_distance_spin,
            "collision_margin": self.collision_margin_spin,
            "object_restitution": self.object_restitution_spin,
        }
        xyz_spins = {
            "depth_camera_location": self.depth_camera_location_spins,
            "depth_camera_target": self.depth_camera_target_spins,
            "rgb_camera_location": self.rgb_camera_location_spins,
            "rgb_camera_target": self.rgb_camera_target_spins,
            "debug_camera_location": self.debug_camera_location_spins,
            "debug_camera_target": self.debug_camera_target_spins,
            "light_location": self.light_location_spins,
        }

        for key, edit in line_edits.items():
            if key in settings and settings[key] is not None:
                edit.setText(str(settings[key]))
        for key, spin in int_spins.items():
            if key in settings and settings[key] is not None:
                spin.setValue(int(settings[key]))
        for key, spin in double_spins.items():
            if key in settings and settings[key] is not None:
                spin.setValue(float(settings[key]))
        for key, spins in xyz_spins.items():
            if key in settings and settings[key] is not None:
                self._set_xyz_values(spins, settings[key])
        if "spawn_strategy" in settings:
            self.spawn_strategy_combo.setCurrentText(str(settings["spawn_strategy"]))
        if "collision_shape" in settings:
            self.collision_shape_combo.setCurrentText(str(settings["collision_shape"]))
        if "allow_out_of_bin_filtering" in settings:
            self.allow_out_of_bin_filtering_check.setChecked(bool(settings["allow_out_of_bin_filtering"]))
        if "record_simulation_video" in settings:
            self.record_video_check.setChecked(bool(settings["record_simulation_video"]))

    def import_generation_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import generator preset", str(PROJECT_ROOT), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            settings = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError("Preset must contain a JSON object.")
            self.apply_generation_settings(settings)
            self.log_text.append(
                f"[gui] Imported generator preset: {path} "
                f"(objects={self.objects_spin.value()}, output={self.output_path_edit.text()}, "
                f"collision={self.collision_shape_combo.currentText()})"
            )
        except Exception as exc:  # noqa: BLE001 - show malformed presets in the GUI.
            QMessageBox.warning(self, "Could not import preset", str(exc))

    def export_generation_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export generator preset", str(PROJECT_ROOT / "configs" / "generator_preset.json"), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.generation_settings(), indent=2), encoding="utf-8")
            self.log_text.append(f"[gui] Exported generator preset: {path}")
        except Exception as exc:  # noqa: BLE001 - show filesystem errors in the GUI.
            QMessageBox.warning(self, "Could not export preset", str(exc))

    def browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select raw dataset", self.dataset_path_edit.text())
        if path:
            self.dataset_path_edit.setText(path)
            self.load_dataset(Path(path))

    def browse_blender(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select blender executable", self.blender_path_edit.text())
        if path:
            self.blender_path_edit.setText(path)

    def browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select STL model", str(PROJECT_ROOT / "object-model"), "STL files (*.stl);;All files (*)")
        if path:
            model_path = Path(path)
            self.model_path_edit.setText(self._display_path(model_path))
            current_class_name = self.class_name_edit.text().strip()
            if not current_class_name or current_class_name == "K41144":
                self.class_name_edit.setText(model_path.stem)

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", str(PROJECT_ROOT / "synthetic-data"))
        if path:
            self.output_path_edit.setText(path)

    def start_worker(self, worker: QObject) -> None:
        thread = QThread(self)
        self.worker_threads.append(thread)
        self.worker_objects.append(worker)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        if hasattr(worker, "finished"):
            worker.finished.connect(thread.quit)  # type: ignore[attr-defined]
        if hasattr(worker, "failed"):
            worker.failed.connect(thread.quit)  # type: ignore[attr-defined]
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda thread=thread, worker=worker: self.on_worker_thread_finished(thread, worker))
        thread.start()

    def on_worker_thread_finished(self, thread: QThread, worker: QObject) -> None:
        if thread in self.worker_threads:
            self.worker_threads.remove(thread)
        if worker in self.worker_objects:
            self.worker_objects.remove(worker)
        thread.deleteLater()

    def load_dataset(self, dataset_root: Path) -> None:
        self.dataset_load_token += 1
        token = self.dataset_load_token
        self.samples = []
        self.sample_table.setRowCount(0)
        self.sample_details.setPlainText(f"Loading dataset:\n{dataset_root}")
        self.clear_preview("Loading dataset...")
        self.log_text.append(f"[gui] Loading dataset on worker thread: {dataset_root}")
        worker = DatasetLoadWorker(token, dataset_root)
        worker.progress.connect(self.on_dataset_load_progress)
        worker.finished.connect(self.on_dataset_loaded)
        worker.failed.connect(self.on_dataset_load_failed)
        self.start_worker(worker)

    @Slot(int, str)
    def on_dataset_load_progress(self, token: int, message: str) -> None:
        if token == self.dataset_load_token:
            self.sample_details.setPlainText(message)

    @Slot(int, str, object)
    def on_dataset_loaded(self, token: int, dataset_root: str, rows: object) -> None:
        if token != self.dataset_load_token:
            return
        self.samples = [RawSample(str(row["name"]), Path(str(row["path"]))) for row in rows]
        self.sample_table.setRowCount(0)
        for row_data in rows:
            row = self.sample_table.rowCount()
            self.sample_table.insertRow(row)
            stats = row_data.get("stats", {})
            error = row_data.get("error", "")
            if not error:
                values = [
                    row_data.get("name", ""),
                    stats.get("object_count", ""),
                    stats.get("visible_objects", ""),
                    stats.get("visible_points", ""),
                    stats.get("seed", ""),
                    stats.get("out_of_bin_policy", ""),
                ]
            else:
                values = [row_data.get("name", ""), "", "", "", "", f"error: {error}"]
            for col, value in enumerate(values):
                self.sample_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.sample_table.resizeColumnsToContents()
        if self.samples:
            self.sample_table.selectRow(0)
            self.show_sample(self.samples[0])
        else:
            self.sample_details.setPlainText(f"No sample folders found in:\n{dataset_root}")
            self.clear_preview("No sample selected")
        self.log_text.append(f"[gui] Dataset loaded: {dataset_root} ({len(self.samples)} samples)")

    @Slot(int, str)
    def on_dataset_load_failed(self, token: int, message: str) -> None:
        if token != self.dataset_load_token:
            return
        self.samples = []
        self.sample_table.setRowCount(0)
        self.sample_details.setPlainText(f"Could not load dataset:\n{message}")
        self.clear_preview("No sample selected")
        self.log_text.append(f"[gui] Dataset load failed: {message}")

    def on_sample_table_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self.samples):
            self.show_sample(self.samples[row])

    def on_validation_table_clicked(self, row: int, _column: int) -> None:
        item = self.validation_table.item(row, 0)
        if not item:
            return
        self.select_sample_by_name(item.text())

    def select_sample_by_name(self, sample_name: str) -> None:
        for row, sample in enumerate(self.samples):
            if sample.name == sample_name:
                self.sample_table.selectRow(row)
                self.show_sample(sample)
                return

    def show_sample(self, sample: RawSample) -> None:
        self.sample_load_token += 1
        token = self.sample_load_token
        self.current_sample_name = sample.name
        self.sample_details.setPlainText(f"Loading sample:\n{sample.name}")
        self.clear_preview("Loading preview...")
        self.vtk_panel.clear("Loading point cloud...")
        worker = SampleLoadWorker(token, sample.path)
        worker.finished.connect(self.on_sample_loaded)
        worker.failed.connect(self.on_sample_load_failed)
        self.start_worker(worker)
        self.load_point_cloud(sample)

    @Slot(int, str, str, object)
    def on_sample_loaded(self, token: int, sample_name: str, details: str, images: object) -> None:
        if token != self.sample_load_token or sample_name != self.current_sample_name:
            return
        self.sample_details.setPlainText(details)
        for _title, file_name in PREVIEW_FILES:
            result = images.get(file_name, {})
            panel = self.preview_panels[file_name]
            error = result.get("error", "")
            image = result.get("image")
            if error:
                panel.clear(error)
            elif isinstance(image, QImage) and not image.isNull():
                panel.set_image(image)
                panel.set_zoom_percent(self.preview_zoom_slider.value())
            else:
                panel.clear(f"Could not load: {file_name}")

    @Slot(int, str, str)
    def on_sample_load_failed(self, token: int, sample_name: str, message: str) -> None:
        if token != self.sample_load_token or sample_name != self.current_sample_name:
            return
        self.sample_details.setPlainText(f"Could not load sample:\n{message}")
        self.clear_preview("No sample selected")

    def clear_preview(self, message: str = "No sample selected") -> None:
        for panel in self.preview_panels.values():
            panel.clear(message)

    def load_point_cloud(self, sample: RawSample) -> None:
        self.point_cloud_token += 1
        token = self.point_cloud_token
        if not VTK_AVAILABLE:
            self.vtk_panel.clear(
                "VTK is not installed. Install the optional dependency with: pip install vtk\n"
                f"Import error: {VTK_IMPORT_ERROR}"
            )
            return
        points_key = str(self.point_cloud_frame_combo.currentData())
        worker = PointCloudLoadWorker(token, sample.path, points_key)
        worker.finished.connect(self.on_point_cloud_loaded)
        worker.failed.connect(self.on_point_cloud_failed)
        self.start_worker(worker)

    def reload_current_point_cloud(self) -> None:
        if not self.current_sample_name:
            return
        for sample in self.samples:
            if sample.name == self.current_sample_name:
                self.vtk_panel.clear("Loading point cloud...")
                self.load_point_cloud(sample)
                return

    @Slot(int, str, object, object)
    def on_point_cloud_loaded(self, token: int, sample_name: str, points: object, instance_ids: object) -> None:
        if token != self.point_cloud_token or sample_name != self.current_sample_name:
            return
        self.current_point_cloud = (points, instance_ids)
        self.vtk_panel.set_point_cloud(points, instance_ids, self.point_size_spin.value())

    @Slot(int, str, str)
    def on_point_cloud_failed(self, token: int, sample_name: str, message: str) -> None:
        if token != self.point_cloud_token or sample_name != self.current_sample_name:
            return
        self.current_point_cloud = None
        self.vtk_panel.clear(f"Could not load point cloud:\n{message}")

    def on_point_size_changed(self, value: int) -> None:
        self.vtk_panel.set_point_size(value)

    def run_validation(self) -> None:
        dataset_root = Path(self.dataset_path_edit.text())
        self.validation_token += 1
        token = self.validation_token
        self.validation_summary.setPlainText(f"Validating dataset:\n{dataset_root}")
        self.validation_table.setRowCount(0)
        self.validate_button.setEnabled(False)
        self.log_text.append(f"[gui] Validating dataset on worker thread: {dataset_root}")
        worker = ValidationWorker(token, dataset_root, PROJECT_ROOT)
        worker.progress.connect(self.on_validation_progress)
        worker.finished.connect(self.on_validation_finished)
        worker.failed.connect(self.on_validation_failed)
        self.start_worker(worker)

    @Slot(int, str)
    def on_validation_progress(self, token: int, message: str) -> None:
        if token == self.validation_token:
            self.validation_summary.setPlainText(message)

    @Slot(int, object)
    def on_validation_finished(self, token: int, report: object) -> None:
        if token != self.validation_token:
            return
        self.validate_button.setEnabled(True)
        self.validation_report = report
        self.populate_validation(report)
        self.log_text.append(f"[gui] Validation finished: {report.sample_count} samples, {report.status_counts}")

    @Slot(int, str)
    def on_validation_failed(self, token: int, message: str) -> None:
        if token != self.validation_token:
            return
        self.validate_button.setEnabled(True)
        self.validation_summary.setPlainText(f"Validation failed:\n{message}")
        self.log_text.append(f"[gui] Validation failed: {message}")

    def populate_validation(self, report: DatasetValidationReport) -> None:
        self.validation_summary.setPlainText(json.dumps({"status_counts": report.status_counts, **report.summary}, indent=2))
        self.validation_table.setRowCount(0)
        for result in report.samples:
            row = self.validation_table.rowCount()
            self.validation_table.insertRow(row)
            values = [
                result.sample,
                result.status,
                result.object_count,
                result.visible_objects,
                result.visible_points,
                result.out_of_bin_count,
                "; ".join(result.issues),
            ]
            for col, value in enumerate(values):
                self.validation_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.validation_table.resizeColumnsToContents()

    def start_generation(self, single_sample_video: bool = False) -> None:
        blender_path = Path(self.blender_path_edit.text())
        output_path = Path(self.output_path_edit.text())
        if not blender_path.exists():
            QMessageBox.warning(self, "Blender not found", f"Blender executable does not exist:\n{blender_path}")
            return
        if output_path.exists() and any(output_path.iterdir()):
            QMessageBox.warning(
                self,
                "Output folder is not empty",
                "Choose a new empty output folder. Raw datasets should not be overwritten from the GUI.",
            )
            return
        if self.allow_out_of_bin_filtering_check.isChecked():
            answer = QMessageBox.question(
                self,
                "Legacy filtering enabled",
                "Legacy filtering can hide out-of-bin objects and accept the remaining scene. "
                "Use strict reject mode for training datasets.\n\nContinue with legacy filtering?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        command_args = self.build_generator_args(
            output_path,
            override_samples=1 if single_sample_video else None,
            force_record_video=single_sample_video,
            override_video_frame_step=1 if single_sample_video else None,
        )
        policy = "filter" if self.allow_out_of_bin_filtering_check.isChecked() else "strict reject"
        self.log_text.append(f"[gui] Out-of-bin policy: {policy}")
        if single_sample_video:
            self.log_text.append("[gui] Single-sample video mode: forcing --samples 1, --record-simulation-video, and video frame step 1.")
        self.log_text.append(
            "[gui] Simulation estimate: "
            f"{self.estimated_simulation_frames()} frames per attempt, "
            f"{self.objects_spin.value()} objects, "
            f"{self.width_spin.value()}x{self.height_spin.value()} render."
        )
        if self.collision_shape_combo.currentText() == "MESH":
            self.log_text.append("[gui] Warning: MESH collision is much slower than CONVEX_HULL for bending_pipe.")
        if self.estimated_simulation_frames() > 800 or self.objects_spin.value() > 20:
            self.log_text.append("[gui] Long run expected: reduce Objects/Settle frames for smoke tests before generating a full dataset.")
        self.log_text.append(f"> {blender_path} {' '.join(command_args)}")
        self.log_text.append("")
        self.process = QProcess(self)
        self.process.setProgram(str(blender_path))
        self.process.setArguments(command_args)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_process_output)
        self.process.finished.connect(self.on_process_finished)
        self.start_button.setEnabled(False)
        self.single_sample_video_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.process.start()

    def estimated_simulation_frames(self) -> int:
        spawn_settle_frames = self.spawn_settle_frames_spin.value()
        if spawn_settle_frames <= 0:
            return self.settle_frames_spin.value()
        return 1 + self.settle_frames_spin.value() + max(0, self.objects_spin.value() - 1) * spawn_settle_frames

    def build_generator_args(
        self,
        output_path: Path,
        *,
        override_samples: int | None = None,
        force_record_video: bool = False,
        override_video_frame_step: int | None = None,
    ) -> list[str]:
        args = [
            "--background",
            "--python",
            str(GENERATOR_SCRIPT),
            "--",
            "--model",
            self.model_path_edit.text(),
            "--model-scale",
            str(self.model_scale_spin.value()),
            "--output",
            self._display_path(output_path),
            "--samples",
            str(override_samples if override_samples is not None else self.samples_spin.value()),
            "--objects",
            str(self.objects_spin.value()),
            "--width",
            str(self.width_spin.value()),
            "--height",
            str(self.height_spin.value()),
            "--bin-x",
            str(self.bin_x_spin.value()),
            "--bin-y",
            str(self.bin_y_spin.value()),
            "--bin-wall-height",
            str(self.bin_wall_height_spin.value()),
            "--drop-height-min",
            str(self.drop_height_min_spin.value()),
            "--drop-height-max",
            str(self.drop_height_max_spin.value()),
            "--spawn-strategy",
            self.spawn_strategy_combo.currentText(),
            "--objects-per-layer",
            str(self.objects_per_layer_spin.value()),
            "--spawn-min-distance",
            str(self.spawn_min_distance_spin.value()),
            "--spawn-settle-frames",
            str(self.spawn_settle_frames_spin.value()),
            "--collision-margin",
            str(self.collision_margin_spin.value()),
            "--collision-shape",
            self.collision_shape_combo.currentText(),
            "--object-restitution",
            str(self.object_restitution_spin.value()),
            "--min-visible-objects",
            str(self.min_visible_objects_spin.value()),
            "--min-visible-points",
            str(self.min_visible_points_spin.value()),
            "--max-sample-attempts",
            str(self.max_sample_attempts_spin.value()),
            "--settle-frames",
            str(self.settle_frames_spin.value()),
        ]
        self._extend_xyz_arg(args, "--depth-camera-location", self.depth_camera_location_spins)
        self._extend_xyz_arg(args, "--depth-camera-target", self.depth_camera_target_spins)
        args.extend(["--depth-camera-lens", str(self.depth_camera_lens_spin.value())])
        self._extend_xyz_arg(args, "--rgb-camera-location", self.rgb_camera_location_spins)
        self._extend_xyz_arg(args, "--rgb-camera-target", self.rgb_camera_target_spins)
        args.extend(["--rgb-camera-lens", str(self.rgb_camera_lens_spin.value())])
        self._extend_xyz_arg(args, "--debug-camera-location", self.debug_camera_location_spins)
        self._extend_xyz_arg(args, "--debug-camera-target", self.debug_camera_target_spins)
        args.extend(["--debug-camera-lens", str(self.debug_camera_lens_spin.value())])
        self._extend_xyz_arg(args, "--light-location", self.light_location_spins)
        args.extend(["--light-energy", str(self.light_energy_spin.value())])
        args.extend(["--light-size", str(self.light_size_spin.value())])
        class_name = self.class_name_edit.text().strip()
        if class_name:
            args.extend(["--class-name", class_name])
        if self.record_video_check.isChecked() or force_record_video:
            args.append("--record-simulation-video")
            video_frame_step = override_video_frame_step if override_video_frame_step is not None else self.video_frame_step_spin.value()
            args.extend(["--simulation-video-frame-step", str(video_frame_step)])
            args.extend(["--simulation-video-fps", str(self.video_fps_spin.value())])
        if self.allow_out_of_bin_filtering_check.isChecked():
            args.append("--allow-out-of-bin-filtering")
        return args

    def on_process_output(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log_text.moveCursor(QTextCursor.MoveOperation.End)
            self.log_text.insertPlainText(data)
            self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.log_text.append(f"\n[gui] Blender finished with exit_code={exit_code}, exit_status={exit_status.name}")
        self.start_button.setEnabled(True)
        self.single_sample_video_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process = None
        output_path = Path(self.output_path_edit.text())
        self.dataset_path_edit.setText(str(output_path))
        self.load_dataset(output_path)

    def stop_generation(self) -> None:
        if not self.process:
            return
        self.log_text.append("\n[gui] Terminating Blender process...")
        self.process.terminate()
        QTimer.singleShot(3000, self.kill_generation_if_running)

    def kill_generation_if_running(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.log_text.append("[gui] Killing Blender process...")
            self.process.kill()

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)


def main() -> int:
    app = QApplication(sys.argv)
    window = DatasetGui()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
