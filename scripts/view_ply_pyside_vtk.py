"""PySide6 + VTK viewer for colored PLY point clouds."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import vtkmodules.all as vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.util.numpy_support import vtk_to_numpy

    VTK_AVAILABLE = True
    VTK_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - VTK is an optional GUI dependency.
    vtk = None
    QVTKRenderWindowInteractor = None
    vtk_to_numpy = None
    VTK_AVAILABLE = False
    VTK_IMPORT_ERROR = str(exc)


VTK_RENDER_DISABLED = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"


@dataclass(frozen=True)
class LoadedPly:
    path: Path
    points: np.ndarray
    colors: np.ndarray | None
    stats: dict[str, Any]


class PlyLoadWorker(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str, str)

    def __init__(self, token: int, path: Path) -> None:
        super().__init__()
        self.token = token
        self.path = path

    @Slot()
    def run(self) -> None:
        try:
            loaded = load_ply_as_arrays(self.path)
            self.finished.emit(self.token, loaded)
        except Exception as exc:  # noqa: BLE001 - show load failure in GUI.
            self.failed.emit(self.token, str(self.path), str(exc))


class VtkPointCloudPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.renderer = None
        self.vtk_widget = None
        self.point_actor = None
        self.axes_actor = None
        self.interactor_style = None

        if not VTK_AVAILABLE or VTK_RENDER_DISABLED:
            if VTK_RENDER_DISABLED:
                self.status_label.setText("VTK rendering is disabled for offscreen Qt sessions.")
            else:
                self.status_label.setText(
                    "VTK is not installed. Install the optional dependency with: pip install vtk\n"
                    f"Import error: {VTK_IMPORT_ERROR}"
                )
            return

        self.status_label.setText("No PLY loaded")
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
        if self.renderer is not None and self.vtk_widget is not None:
            self.renderer.RemoveAllViewProps()
            self.point_actor = None
            self.axes_actor = None
            self.vtk_widget.GetRenderWindow().Render()

    def set_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray | None,
        *,
        point_size: int,
        title: str,
    ) -> None:
        if not VTK_AVAILABLE or self.renderer is None or self.vtk_widget is None:
            self.status_label.setText("VTK is not available.")
            return
        if points.size == 0:
            self.clear("Point cloud is empty")
            return

        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(points.shape[0])
        for idx, point in enumerate(points):
            vtk_points.SetPoint(idx, float(point[0]), float(point[1]), float(point[2]))

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        vertices = vtk.vtkCellArray()
        for idx in range(points.shape[0]):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(idx)
        poly_data.SetVerts(vertices)

        if colors is not None:
            color_array = vtk.vtkUnsignedCharArray()
            color_array.SetNumberOfComponents(3)
            color_array.SetName("colors")
            for color in colors:
                color_array.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))
            poly_data.GetPointData().SetScalars(color_array)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        if colors is None:
            actor.GetProperty().SetColor(0.35, 0.65, 1.0)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.05, 0.05, 0.05)

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.AddActor(axes)
        self.point_actor = actor
        self.axes_actor = axes
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.status_label.setText(
            f"Loaded {points.shape[0]} points from {title}. "
            "Controls: left-drag rotate, middle-drag pan, right-drag/wheel zoom."
        )

    def set_point_size(self, point_size: int) -> None:
        if self.point_actor is None or self.vtk_widget is None:
            return
        self.point_actor.GetProperty().SetPointSize(point_size)
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        if VTK_AVAILABLE and self.renderer is not None and self.vtk_widget is not None:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()


class PlyViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PLY Point Cloud Viewer")
        self.resize(1280, 820)
        self.load_token = 0
        self.current_file: Path | None = None
        self.current_cloud: LoadedPly | None = None
        self._threads: list[QThread] = []

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.open_list_item)
        self.file_list.currentItemChanged.connect(self.preview_selected_item)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.vtk_panel = VtkPointCloudPanel()
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 30)
        self.point_size_spin.setValue(4)
        self.point_size_spin.valueChanged.connect(self.vtk_panel.set_point_size)

        self._build_ui()
        self._build_menu()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        controls = QHBoxLayout()
        open_file_button = QPushButton("Open PLY")
        open_file_button.clicked.connect(self.open_file_dialog)
        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(self.open_folder_dialog)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload_current_file)
        reset_camera_button = QPushButton("Reset Camera")
        reset_camera_button.clicked.connect(self.vtk_panel.reset_camera)
        reset_interaction_button = QPushButton("Reset Interaction")
        reset_interaction_button.clicked.connect(self.vtk_panel.set_trackball_camera_style)
        controls.addWidget(open_file_button)
        controls.addWidget(open_folder_button)
        controls.addWidget(reload_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Point size"))
        controls.addWidget(self.point_size_spin)
        controls.addWidget(reset_camera_button)
        controls.addWidget(reset_interaction_button)
        controls.addStretch(1)
        root_layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("PLY files"))
        left_layout.addWidget(self.file_list, 2)
        left_layout.addWidget(QLabel("Details"))
        left_layout.addWidget(self.details, 1)
        splitter.addWidget(left)
        splitter.addWidget(self.vtk_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 940])
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)
        self.details.setPlainText("Open a .ply file or a folder containing .ply files.")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_file_action = QAction("Open PLY...", self)
        open_file_action.triggered.connect(self.open_file_dialog)
        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.triggered.connect(self.open_folder_dialog)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_file_action)
        file_menu.addAction(open_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def open_file_dialog(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Open PLY", str(Path.cwd()), "PLY files (*.ply)")
        if path:
            self.set_file_list([Path(path)])
            self.file_list.setCurrentRow(0)

    def open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open folder", str(Path.cwd()))
        if folder:
            paths = sorted(Path(folder).glob("*.ply"))
            if not paths:
                QMessageBox.information(self, "No PLY files", f"No .ply files found in:\n{folder}")
                return
            self.set_file_list(paths)
            self.file_list.setCurrentRow(0)

    def set_file_list(self, paths: list[Path]) -> None:
        self.file_list.clear()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.file_list.addItem(item)

    def open_list_item(self, item: QListWidgetItem) -> None:
        path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        self.load_ply(path)

    def preview_selected_item(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        path = Path(str(current.data(Qt.ItemDataRole.UserRole)))
        if path != self.current_file:
            self.load_ply(path)

    def reload_current_file(self) -> None:
        if self.current_file is not None:
            self.load_ply(self.current_file)

    def load_ply(self, path: Path) -> None:
        self.load_token += 1
        token = self.load_token
        self.current_file = path
        self.current_cloud = None
        self.vtk_panel.clear(f"Loading {path.name}...")
        self.details.setPlainText(f"Loading:\n{path}")
        worker = PlyLoadWorker(token, path)
        worker.finished.connect(self.on_ply_loaded)
        worker.failed.connect(self.on_ply_failed)
        self.start_worker(worker)

    def start_worker(self, worker: QObject) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        worker.finished.connect(thread.quit)  # type: ignore[attr-defined]
        worker.failed.connect(thread.quit)  # type: ignore[attr-defined]
        worker.finished.connect(worker.deleteLater)  # type: ignore[attr-defined]
        worker.failed.connect(worker.deleteLater)  # type: ignore[attr-defined]
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._forget_thread(thread))
        self._threads.append(thread)
        thread.start()

    def _forget_thread(self, thread: QThread) -> None:
        if thread in self._threads:
            self._threads.remove(thread)

    @Slot(int, object)
    def on_ply_loaded(self, token: int, loaded: object) -> None:
        if token != self.load_token or not isinstance(loaded, LoadedPly):
            return
        self.current_cloud = loaded
        self.vtk_panel.set_point_cloud(
            loaded.points,
            loaded.colors,
            point_size=self.point_size_spin.value(),
            title=loaded.path.name,
        )
        self.details.setPlainText(format_stats(loaded))

    @Slot(int, str, str)
    def on_ply_failed(self, token: int, path: str, message: str) -> None:
        if token != self.load_token:
            return
        self.current_cloud = None
        self.vtk_panel.clear(f"Could not load PLY:\n{message}")
        self.details.setPlainText(f"Could not load:\n{path}\n\n{message}")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override.
        for thread in list(self._threads):
            thread.quit()
            thread.wait(1000)
        super().closeEvent(event)


def load_ply_as_arrays(path: Path) -> LoadedPly:
    if not VTK_AVAILABLE:
        raise RuntimeError(f"VTK is not available: {VTK_IMPORT_ERROR}")
    if not path.exists():
        raise FileNotFoundError(path)

    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly_data = reader.GetOutput()
    if poly_data is None or poly_data.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")

    vtk_points = poly_data.GetPoints()
    points = vtk_to_numpy(vtk_points.GetData()).astype(np.float32, copy=True)
    colors = extract_colors(poly_data)
    stats = build_stats(path, points, colors)
    return LoadedPly(path=path, points=points, colors=colors, stats=stats)


def extract_colors(poly_data: Any) -> np.ndarray | None:
    point_data = poly_data.GetPointData()
    scalars = point_data.GetScalars() if point_data is not None else None
    if scalars is None:
        return None
    array = vtk_to_numpy(scalars)
    if array.ndim == 1:
        return None
    if array.shape[1] < 3:
        return None
    colors = array[:, :3]
    if np.issubdtype(colors.dtype, np.floating):
        max_value = float(np.nanmax(colors)) if colors.size else 1.0
        if max_value <= 1.0:
            colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8, copy=True)


def build_stats(path: Path, points: np.ndarray, colors: np.ndarray | None) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = points.mean(axis=0)
    extent = maxs - mins
    stats: dict[str, Any] = {
        "path": str(path),
        "points": int(points.shape[0]),
        "has_colors": colors is not None,
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
    }
    if colors is not None:
        unique_colors = np.unique(colors.reshape(-1, 3), axis=0)
        stats["unique_colors"] = int(unique_colors.shape[0])
    return stats


def format_stats(loaded: LoadedPly) -> str:
    stats = loaded.stats
    lines = [
        f"File: {loaded.path}",
        f"Points: {stats['points']}",
        f"Colors: {'yes' if stats['has_colors'] else 'no'}",
    ]
    if "unique_colors" in stats:
        lines.append(f"Unique colors: {stats['unique_colors']}")
    lines.extend(
        [
            "",
            "Bounds:",
            f"  x: {stats['min'][0]:.6f} .. {stats['max'][0]:.6f}",
            f"  y: {stats['min'][1]:.6f} .. {stats['max'][1]:.6f}",
            f"  z: {stats['min'][2]:.6f} .. {stats['max'][2]:.6f}",
            "",
            "Center:",
            f"  ({stats['center'][0]:.6f}, {stats['center'][1]:.6f}, {stats['center'][2]:.6f})",
            "",
            "Extent:",
            f"  ({stats['extent'][0]:.6f}, {stats['extent'][1]:.6f}, {stats['extent'][2]:.6f})",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    app = QApplication(sys.argv)
    viewer = PlyViewer()
    if len(sys.argv) > 1:
        paths = [Path(arg) for arg in sys.argv[1:] if Path(arg).suffix.lower() == ".ply"]
        if paths:
            viewer.set_file_list(paths)
            viewer.file_list.setCurrentRow(0)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
