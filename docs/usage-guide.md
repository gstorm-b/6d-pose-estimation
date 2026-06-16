# Pose Backend Usage Guide

How to use the trained 6D pose backend: serve it, call it as a library, run
instance segmentation on its own, inspect the model registry, and evaluate on new
data. Companion docs: [backend-api.md](backend-api.md) (HTTP contract),
[real-data-capture-guide.md](real-data-capture-guide.md) (real-data eval),
[dense-pile-pose-gap.md](dense-pile-pose-gap.md) (confidence/model-fit behavior).

## Conventions

- **Units: metres** everywhere. Convert at the boundary (mm depth -> `/1000`).
- **Frames are explicit and directional**: poses are `object_to_camera` (4x4
  rigid). Scene points enter in the **positive-forward camera** convention
  (`z` positive in front); poses are returned in the **Blender metadata camera**
  frame. Raw synthetic `metadata.json` bakes model scale into the rotation -
  rigidify with `rigid_object_to_camera_from_scaled` before comparing.
- **Device**: GPU by default, CPU fallback (`src/inference/device.resolve_device`).

## Components

| Component | What it is | Entry point |
|-----------|-----------|-------------|
| Registry + bundles | Per-SKU versioned model bundles under `models/<sku>/<version>/` | `src/registry/model_registry.py` |
| Pose library | In-memory scene -> ranked poses | `src/inference/pose_pipeline.py` |
| Instance segmentation | Scene -> per-point instance labels (no pose) | `src/inference/instance_segmentation.py` |
| HTTP service | Stdlib server over the library | `scripts/run_backend_service.py` |
| Eval harness | Pipeline metrics on a frame folder | `scripts/prepare_real_eval_set.py` |
| Packaging | Build a bundle from checkpoints | `scripts/package_model_bundle.py` |

Bundles are gitignored (large checkpoints); rebuild with the packaging script.
List what is installed:

```python
from src.registry.model_registry import ModelRegistry
r = ModelRegistry("models")
print(r.list_skus())                 # ['K41144', 'bending_pipe']
print(r.versions("bending_pipe"))    # ['v1', 'v2']  (latest = v2)
```

## 1. Run the HTTP service

```bash
python scripts/run_backend_service.py                 # reads configs/backend/service.yaml
python scripts/run_backend_service.py --host 0.0.0.0 --port 8080 --warmup
```

```bash
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/skus
# POST a scene (points or a base64 depth npz) -> ranked object_to_camera; see backend-api.md
```

Poses come back ranked by confidence; for bin-picking, take the top one (top-1
pick success is ~1.0 thanks to the model-fit confidence term). No auth - bind to a
trusted network only.

## 2. Use the pose library in Python

```python
import numpy as np
from src.inference.device import resolve_device
from src.inference.pose_pipeline import PosePipeline
from src.registry.model_registry import ModelRegistry

registry = ModelRegistry("models")
loaded = registry.load(registry.resolve("bending_pipe"), device=resolve_device("cuda"))
pipeline = PosePipeline.from_loaded_bundle(loaded)

# scene points in the positive-forward camera frame, (N, 3) float32 (+ optional normals)
result = pipeline.infer_from_points(points_camera, features=normals)
# or straight from a depth frame:
# result = pipeline.infer_scene(depth_m, intrinsics, normal_camera)

for inst in result.instances:            # already ranked, highest confidence first
    T = inst.object_to_camera            # 4x4 object->camera
    print(inst.instance_id, inst.confidence, inst.diagnostics["model_fit"], T[:3, 3])
```

## 3. Run instance segmentation standalone

Segment a bin into object instances (labels + clusters + centroids) without the
pose stage - for visualization, debugging, or a different downstream stage. Loads
from a bundle (`--sku`) or directly from a training checkpoint (`--checkpoint`).

```bash
# from a packaged bundle, a processed sample, with a colored point cloud
python scripts/run_instance_segmentation.py --sku bending_pipe \
    --points processed-data/pointnet2_semseg_bending_pipe_wave2/test/<sample>.npz \
    --out out/seg --save-cloud

# from a raw checkpoint, a depth frame, with a 2D instance mask
python scripts/run_instance_segmentation.py \
    --checkpoint experiments/<run>/checkpoints/best.pt \
    --depth frame/depth.npy --intrinsics frame/intrinsics.json --normals frame/normal.npy \
    --out out/seg --save-cloud --save-mask
```

Outputs in `--out`: `instance_labels.npy` (per-point int labels, 0 = background),
`instances.json` (count + per-instance point_count / centroid / bbox / object
probability + timings), and optionally `segmentation.ply` (points colored by
instance) and `instance_mask.png` (depth input only). Tune clustering inline with
`--object-probability-threshold / --dbscan-eps-m / --dbscan-min-samples /
--min-cluster-points`.

In Python:

```python
from src.inference.instance_segmentation import InstanceSegmenter
seg = InstanceSegmenter.from_loaded_bundle(loaded)          # or from_checkpoint(path)
res = seg.segment_points(points_camera, features=normals)   # or seg.segment_depth(depth, intr)
print(res.instance_count, [i.point_count for i in res.instances])
# Re-cluster a cached forward pass cheaply (no second model run):
fwd = seg.forward(points_camera, features=normals)
res = seg.cluster(fwd)                                       # or cluster(fwd, custom_config)
```

### Interactive viewer (PySide6 + VTK) — segmentation AND pose

One app inspects both stages in 3D:

```bash
python scripts/view_instance_segmentation.py --sku bending_pipe \
    --scene processed-data/pointnet2_semseg_bending_pipe_wave2/test/<sample>.npz
```

Pick the model (registry SKU, or a raw instance checkpoint for segmentation only)
and open any scene `.npz` (points/points_camera + optional features), then:

- **Run segmentation** — cloud colored per instance, click an instance to isolate,
  show centroids, and re-tune object-prob / dbscan-eps / min-cluster-points live
  (re-clustering reuses the cached forward pass).
- **Run pose** — overlays each object's posed model points + pose axes (red=x,
  green=y, blue=z) on a dimmed scene, ranked by confidence; click an instance to
  isolate its pose. This is the standard 6D-pose eyeball: does the posed model
  snap onto the observed points? Lower `model_fit` = better fit.

To just view a saved colored cloud without re-running the model, open
`segmentation.ply` (from the CLI above) in the generic
`scripts/view_ply_pyside_vtk.py`.

## 4. Evaluate on a frame folder (synthetic or real)

```bash
# turn raw synthetic samples into the camera-agnostic eval format
python scripts/prepare_real_eval_set.py export-synthetic \
    --raw synthetic-data/bending_pipe_4 --out real-data --set demo --sku bending_pipe --limit 8 --save-normal
# run the pipeline + metrics (ADD / ADD_0.1d / recall / top-1,3 pick success)
python scripts/prepare_real_eval_set.py evaluate --set real-data/demo --sku bending_pipe
```

For real captures and semi-automatic ICP labeling, see
[real-data-capture-guide.md](real-data-capture-guide.md).

## 5. Package a new bundle

After training instance + pose models, bundle them for the registry:

```bash
python scripts/package_model_bundle.py --sku bending_pipe --version v3 \
    --instance-checkpoint experiments/<instance_run>/checkpoints/best.pt \
    --pose-checkpoint experiments/<pose_run>/checkpoints/best_add.pt \
    --object-probability-threshold 0.5 --dbscan-eps-m 0.006 --dbscan-min-samples 8 \
    --min-cluster-points 32 --num-points 16384 --overwrite
```

The newest numeric version resolves as `latest`. The bundle vendors the STL +
sampled model points/keypoints + diameter + symmetry, so it is self-contained.

## Troubleshooting

- **Slow first call**: the bundle loads on first use (instance + pose models to
  GPU). `--warmup` the service to pre-load. The instance stage dominates latency.
- **Wrong-looking ADD with huge rotation, tiny translation**: you compared against
  a raw `metadata.json` pose - rigidify it first (`rigid_object_to_camera_from_scaled`).
- **GPU non-determinism (weak GPUs)**: the instance model can give slightly
  different clusters across separate calls; do not join per-point labels across
  two calls - use one call's output.
- **Tests** are standalone (`python tests/test_*.py`) since pytest is not
  installed; they are pytest-compatible.
