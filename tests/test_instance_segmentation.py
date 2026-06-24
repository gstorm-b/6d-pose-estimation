"""Tests for the standalone InstanceSegmenter. Runs under pytest or:

    python tests/test_instance_segmentation.py

The contract test uses a fake model (no checkpoint/GPU needed); the real test is
guarded on the packaged bending_pipe bundle + a processed sample.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.instance_clustering import VotedCenterClusteringConfig  # noqa: E402
from src.inference.instance_segmentation import InstanceSegmenter  # noqa: E402

MODELS_ROOT = PROJECT_ROOT / "models"
PROCESSED = PROJECT_ROOT / "processed-data" / "pointnet2_semseg_bending_pipe_wave2" / "test"


class _TwoBlobModel:
    """Fake instance model: all foreground, votes toward two x-separated centers."""

    def to(self, device):  # noqa: D401 - mimic nn.Module
        return self

    def eval(self):
        return self

    def __call__(self, tensor):
        import torch

        xyz = tensor.squeeze(0).detach().cpu().numpy()[:, :3]
        center_a = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        center_b = np.array([-0.1, 0.0, 0.0], dtype=np.float32)
        center = np.where(xyz[:, :1] > 0, center_a, center_b)
        offsets = (center - xyz).astype(np.float32)
        logits = np.tile(np.array([-10.0, 10.0], dtype=np.float32), (xyz.shape[0], 1))
        return {
            "semantic_logits": torch.from_numpy(logits).unsqueeze(0),
            "offsets": torch.from_numpy(offsets).unsqueeze(0),
        }


def _two_blob_scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.1, 0.0, 0.5], 0.01, size=(300, 3))
    b = rng.normal([-0.1, 0.0, 0.5], 0.01, size=(300, 3))
    return np.vstack([a, b]).astype(np.float32)


def test_contract_with_fake_model() -> None:
    seg = InstanceSegmenter(
        instance_model=_TwoBlobModel(),
        clustering=VotedCenterClusteringConfig(
            object_probability_threshold=0.5, dbscan_eps_m=0.02, dbscan_min_samples=5, min_cluster_points=10
        ),
        num_classes=2,
        device="cpu",
    )
    points = _two_blob_scene()
    result = seg.segment_points(points)
    assert result.instance_count == 2, f"expected 2 instances, got {result.instance_count}"
    assert result.instance_labels.shape[0] == points.shape[0]
    assert result.object_probabilities.min() > 0.5  # all foreground
    # labels partition the foreground; per-instance summaries are consistent.
    covered = 0
    for inst in result.instances:
        assert inst.point_count == inst.point_indices.shape[0] > 0
        assert np.all(result.instance_labels[inst.point_indices] == inst.instance_id)
        assert np.all(np.isfinite(inst.centroid_camera))
        covered += inst.point_count
    assert covered == int((result.instance_labels > 0).sum())
    # The two clusters sit on opposite sides in x.
    xs = sorted(float(inst.centroid_camera[0]) for inst in result.instances)
    assert xs[0] < 0 < xs[1], f"clusters not x-separated: {xs}"


def test_subsampling_caps_points() -> None:
    seg = InstanceSegmenter(instance_model=_TwoBlobModel(), clustering=VotedCenterClusteringConfig(),
                            num_classes=2, scene_num_points=200, device="cpu")
    points = _two_blob_scene()  # 600 points
    result = seg.segment_points(points)
    assert result.points_camera.shape[0] == 200
    assert result.source_point_count == 600


def test_real_bundle_segmentation() -> None:
    if not (MODELS_ROOT / "bending_pipe").exists() or not glob.glob(str(PROCESSED / "*.npz")):
        print("  (skip: missing bending_pipe bundle or processed sample)")
        return
    from src.inference.device import resolve_device
    from src.registry.model_registry import ModelRegistry

    registry = ModelRegistry(MODELS_ROOT)
    loaded = registry.load(registry.resolve("bending_pipe"), device=resolve_device("cuda"))
    seg = InstanceSegmenter.from_loaded_bundle(loaded)

    data = np.load(sorted(glob.glob(str(PROCESSED / "*.npz")))[0])
    result = seg.segment_points(data["points"].astype(np.float32), data["features"].astype(np.float32))
    print(f"  real bundle: {result.instance_count} instances, "
          f"{int((result.instance_labels > 0).sum())}/{result.points_camera.shape[0]} fg points")
    assert result.instance_count >= 1
    assert result.instance_labels.shape[0] == result.points_camera.shape[0]
    assert all(np.all(np.isfinite(inst.centroid_camera)) for inst in result.instances)


def test_load_scene_pcd_convention() -> None:
    """The .pcd loader must produce positive-forward points + away-from-camera
    normals (the synthetic convention the instance model fires on)."""
    pcd = PROJECT_ROOT / "real-data" / "scene_001.pcd"
    if not pcd.exists():
        print("  (skip: no real-data/scene_001.pcd)")
        return
    try:
        import open3d  # noqa: F401
    except Exception:  # noqa: BLE001
        print("  (skip: open3d not installed)")
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("_vis", PROJECT_ROOT / "scripts" / "view_instance_segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    points, normals, info = module._load_scene_pcd(pcd, target_points=16384)
    assert points.shape[0] <= 16384 and points.shape[1] == 3
    assert normals.shape == points.shape
    assert np.all(np.isfinite(points)) and np.all(np.isfinite(normals))
    assert np.median(points[:, 2]) > 0, "points must be positive-forward (z > 0)"
    assert np.median(normals[:, 2]) > 0, "normals must point away from the camera (z > 0)"
    print(f"  pcd loader: {info['raw_points']}->{points.shape[0]} pts, in_mm={info['in_mm']}, flip_z={info['flip_z']}")


def _run_all() -> int:
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
