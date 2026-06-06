# Session Summary: Synthetic Data For 6D Pose / Bin Picking

This file summarizes the current project context so another Codex session can continue without re-discovering the same decisions.

## Project Goal

Build a 6D pose estimation / robotic bin-picking pipeline from scratch for the industrial object:

```text
object-model/K41144.stl
```

Initial direction:

- Generate synthetic raw data with Blender.
- Use point cloud as the main signal, following the paper in `arcticle/`.
- Keep raw data rich enough to support several later training paths:
  - PointNet / PointNet++ segmentation.
  - PPRNet++-style point-wise pose regression.
  - YOLO segmentation from RGB + instance masks.
  - Later pose estimation and ICP/refinement experiments.

## Session Bootstrap

For future Codex sessions, first read:

```text
docs/codex-session-bootstrap.md
```

That file lists the minimum project context, required engineering rules, relevant files by task type, and the expected senior-engineer workflow for this repository.

## Paper Notes

The article in `arcticle/` was read and summarized here:

```text
arcticle/instance-segmentation-based-6d-pose-estimation-for-bin-picking.md
```

Main idea from the paper:

- Use point cloud instead of RGB for low-texture industrial objects.
- Split the task into instance segmentation and 6D pose estimation.
- Generate synthetic scenes with physics simulation to avoid hand-labeling real point clouds.
- Use depth/normal features and ICP refinement.

## Object Model

The object model is:

```text
object-model/K41144.stl
```

Observed STL bounding box:

```text
approx size: 30 mm x 93.6 mm x 20 mm
```

The STL is assumed to be in real metric scale.

Important issue found:

- The STL origin is near one end of the part, not at the bbox center.
- Blender rigid body uses the object origin as the rigid body's center of mass.
- Without correction, the object behaves like a pendulum and tends to settle upright.

Generator fix:

- Recenter mesh vertices around bbox center before rigid-body simulation.
- Store transform metadata so the exported `object_to_world` and `object_to_camera` still refer to the original STL coordinate frame.

## Current Scripts

### Blender Raw Data Generator

Main script:

```text
scripts/generate_synthetic_blender.py
```

Purpose:

- Load `K41144.stl`.
- Recenter mesh for correct physics.
- Create a bin/container.
- Spawn many copies of the object.
- Run Blender rigid-body simulation.
- Render RGB visual reference.
- Raycast from the camera to export depth, masks, normals, point cloud and labels.
- Export object pose ground truth.
- Reject/retry bad samples if any object leaves the bin or too few objects/points are visible.
- Keep the static Blender scene loaded once per generator run; per attempt only dynamic object copies are spawned and deleted.

Recommended generation command:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --samples 100 --objects 30 --width 320 --height 240 --bin-wall-height 0.14 --drop-height-min 0.12 --drop-height-max 0.34 --spawn-strategy layered --objects-per-layer 6 --spawn-min-distance 0.045 --collision-margin 0.00002 --min-visible-objects 12 --min-visible-points 8000 --max-sample-attempts 12 --settle-frames 260
```

Important generator options:

- `--samples`: number of accepted samples to write.
- `--objects`: number of object copies to spawn before filtering.
- `--spawn-strategy layered`: spreads initial objects into height bands to avoid explosive overlaps.
- `--objects-per-layer`: number of objects per initial height band.
- `--drop-height-min`, `--drop-height-max`: random spawn height range.
- `--collision-margin`: rigid-body margin in meters. Current recommended value is `0.00002`, or 0.02 mm.
- `--out-of-bin-tolerance`: tolerance around bin x/y bounds for post-physics world-bbox checks.
- `--allow-out-of-bin-filtering`: optional legacy/debug behavior that hides out-of-bin objects and accepts the remaining scene if visibility thresholds pass. Do not use it for training datasets. Default behavior is stricter: reject the whole attempt.
- `--min-visible-objects`: reject samples with fewer visible instances.
- `--min-visible-points`: reject samples with too few object points.
- `--max-sample-attempts`: retries before failing a sample.

### Raw Dataset Validator

CLI script:

```text
scripts/validate_raw_dataset.py
```

Purpose:

- Validate required raw sample files and `sensor_data.npz` schema.
- Check instance ids across metadata, masks, and point labels.
- Check sparse samples and out-of-bin objects using metadata poses and STL bbox.
- Fail datasets generated with legacy out-of-bin filtering unless `--allow-legacy-filtering` is passed.

Recommended command:

```powershell
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\K41144_out_of_bin_reject_test2
```

### Open3D Point Cloud Viewer

Viewer script:

```text
scripts/view_point_cloud_open3d.py
```

Purpose:

- Read `sensor_data.npz`.
- Display `points_camera` or `points_world`.
- Color points by `point_instance_ids`.
- Optionally export a colored `.ply`.

Example:

```powershell
python .\scripts\view_point_cloud_open3d.py .\synthetic-data\K41144\sample_000001
```

Useful options:

```powershell
--world
--voxel-size 0.001
--save-ply output.ply
--no-window
```

### PySide6 Dataset GUI

GUI script:

```text
scripts/dataset_gui.py
```

Purpose:

- Run the Blender generator from a local desktop UI.
- Stream Blender generation logs while the process runs.
- Open any raw dataset folder and list `sample_xxxxxx` samples.
- Preview `rgb.png`, `label_overlay.png`, `depth_preview.png`, `instance_mask.png`, and `normal_camera.png`.
- Adjust preview layout with 1, 2, 4, or 5 simultaneous image views, configurable view size, zoom, and drag-to-pan image navigation.
- View object point clouds in a VTK-backed `3D View` tab when the optional `vtk` dependency is installed. The VTK widget explicitly uses Trackball Camera interaction: left-drag rotate, middle-drag pan, right-drag or wheel zoom.
- Show basic metadata stats per sample.
- Run built-in raw dataset validation for required files, NPZ schema, instance ids, sparse samples, and out-of-bin object checks.
- Keep the UI responsive by running dataset loading, sample preview loading, point-cloud loading, and validation in `QThread` workers. Blender generation uses `QProcess` signals.

Run command:

```powershell
python .\scripts\dataset_gui.py
```

GUI dependencies are listed in:

```text
requirements-gui.txt
```

The GUI does not manually edit raw samples. It creates new output folders through the existing generator CLI and reads existing datasets for inspection and validation.

## Raw Dataset Structure

Current raw dataset root:

```text
synthetic-data/K41144/
```

Each sample folder has:

```text
sample_xxxxxx/
  sensor_data.npz
  metadata.json
  rgb.png
  label_overlay.png
  depth_preview.png
  instance_mask.png
  normal_camera.png
  depth_preview.pgm
  instance_mask.ppm
  normal_camera.ppm
```

`rgb.png` is visual reference only. The current material uses black metallic K41144 objects.

`label_overlay.png` is for quick visual QA.

## `sensor_data.npz`

Each sample contains:

```text
depth_m              (H, W)       float32
instance_mask        (H, W)       uint16
normal_world         (H, W, 3)    float32
normal_camera        (H, W, 3)    float32
points_camera        (N, 3)       float32
points_world         (N, 3)       float32
point_instance_ids   (N,)         uint16
point_semantic_ids   (N,)         uint16
point_pixels         (N, 2)       uint16
```

Conventions:

- `instance_mask == 0` means background.
- `point_instance_ids` are visible object instance ids.
- `point_semantic_ids == 1` means class `K41144`.
- Camera point coordinates are in a positive-forward convention stored by the generator.

## `metadata.json`

Top-level metadata contains:

```text
model_path
class_name
class_id
camera_intrinsics
camera_to_world
world_to_camera
settings
object_count
visible_object_point_count
instances[]
files
```

Each instance contains:

```text
instance_id
name
class_id
collision_shape
collision_margin_m
spawn_layer
spawn_height
original_model_center_offset
original_model_to_body
body_to_world
object_to_world
object_to_camera
```

Pose convention:

- `body_to_world`: pose of the recentered physics body.
- `original_model_to_body`: transform from the original STL frame to the recentered body frame.
- `object_to_world`: pose of the original STL coordinate frame in world coordinates.
- `object_to_camera`: pose of the original STL coordinate frame in camera coordinates.

## Issues Found And Fixed

### Object Settled Upright

Symptom:

- Objects tended to stand upright after falling.

Cause:

- STL origin was not near the center of mass.
- Blender rigid body used that origin as the rigid body's center of mass.

Fix:

- Recenter mesh for physics.
- Preserve original STL frame in metadata.

### Unrealistic Object Gap

Symptom:

- Objects could appear to have a large collision gap.

Cause:

- Blender rigid body default `collision_margin` is about `0.04`.

Fix:

- Set `collision_margin = 0.00002`.

### Too Many Objects Left The Bin

Symptom:

- A 100-sample dataset generated with 30 spawned objects had many sparse samples.
- Median final visible object count was about 10.
- Some samples had only 2-4 visible objects.

Cause:

- 30 objects were spawned in a way that caused strong overlap/collision impulses.
- The old inside-bin filter was too permissive.

Fix:

- Add `--spawn-strategy layered`.
- Add stricter inside-bin filtering using `bin_x / 2 + out_of_bin_tolerance`.
- Add reject/retry thresholds.
- Later update: replace silent filtering with post-physics world-space bbox rejection. If any object leaves the bin, reject the whole attempt by default.
- Keep bin/camera/light/STL template loaded once per generator run; only dynamic K41144 copies are deleted and regenerated per attempt.

Validation output after the fix:

```text
synthetic-data/K41144_generator_validation/
```

Observed validation samples:

```text
sample_000000: 30 objects, 31051 object points
sample_000001: 29 objects, 29973 object points
sample_000002: 30 objects, 30539 object points
```

Validation output after strict out-of-bin rejection:

```text
synthetic-data/K41144_out_of_bin_reject_test2/
```

Observed validation samples:

```text
sample_000000: 30 objects, 30 visible instances, 30979 object points
sample_000001: 30 objects, 30 visible instances, 31552 object points
sample_000002: 30 objects, 30 visible instances, 30993 object points
```

Observed rejections:

```text
2 rejected attempts, both reason=out_of_bin with one object outside the bin.
```

### RGB Was White / Overexposed

Symptom:

- `rgb.png` was nearly all white.

Fix:

- K41144 material changed to black metallic.
- Use Filmic color management.
- Reduce exposure and light intensity.

Validation output:

```text
synthetic-data/K41144_rgb_material_test/
```

Observed test:

```text
12 / 12 instances visible
14038 object points
RGB shows black metallic objects
```

## Dataset Status

The older `synthetic-data/K41144/` 100-sample dataset was audited:

- File/schema consistency is clean.
- Metadata and point labels match.
- Collision margin is correctly set.
- Main issue: many sparse scenes due to old spawn strategy.

Use the old dataset only for smoke tests.

For real raw-data generation, regenerate with the latest generator command in this file.

## Training Dataset Direction

Raw data should remain in `synthetic-data/`.

Detailed PointNet++ implementation plan:

```text
docs/pointnetplusplus-implementation-plan.md
```

Processed training-ready data should be generated later into a separate folder, for example:

```text
processed-data/
  pointnet_k41144/
  pprnet_k41144/
  yolo_k41144/
```

### For PointNet / PointNet++

PointNet-style training data should contain fixed-size sampled point clouds:

```text
points              (num_points, 3 or 6)
semantic_labels     (num_points,)
instance_labels     (num_points,)
```

Recommended MVP:

```text
points = points_camera
semantic_labels = point_semantic_ids
instance_labels = point_instance_ids
```

Implemented MVP converter:

```text
scripts/prepare_pointnet2_semseg_dataset.py
src/data/depth_unprojection.py
src/data/pointnet2_processing.py
src/data/augmentation.py
src/data/pointnet2_dataset.py
configs/train/pointnet2_semseg_k41144.yaml
```

Recommended conversion command:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\K41144 --out .\processed-data\pointnet2_semseg_k41144 --config .\configs\train\pointnet2_semseg_k41144.yaml
```

Current processed dataset:

```text
processed-data/pointnet2_semseg_k41144/
```

Observed stats:

```text
77 samples
train/val/test = 62/8/7
16384 sampled points per sample
10650 object points and 5734 background points per sample
```

Important correction:

- Semantic segmentation should reconstruct full-scene points from `depth_m` and label them with `instance_mask`.
- Do not train PointNet++ semantic segmentation directly from raw `points_camera`, because those raw arrays contain object-only points.

To include normals:

- Use `point_pixels` to index into `normal_camera[v, u]`.
- Concatenate `points_camera` and per-point normals to get `(N, 6)`.

In the processed PointNet++ MVP, normals are stored as a separate `features` array and the dataset loader exposes `model_input = xyz + features` when normals are enabled.

### PointNet++ Semantic Training MVP

Implemented files:

```text
scripts/train_pointnet2_semseg.py
src/models/pointnet2_ops.py
src/models/pointnet2_semseg.py
src/training/config.py
src/training/metrics.py
src/training/checkpoint.py
src/training/seed.py
```

Current environment:

```text
torch 2.12.0+cu126
pytorch3d 0.7.9
pyyaml 6.0.3
tqdm 4.67.3
scikit-learn 1.8.0
matplotlib already installed
```

CUDA is available in the installed PyTorch build:

```text
GPU: NVIDIA GeForce MX150
compute capability: (6, 1)
torch CUDA runtime: 12.6
torch arch list includes: sm_61
```

The MX150 has only 2GB VRAM, so start with `batch_size=1` and reduced/debug point counts when changing the model. Full 16384-point training fits with the current `pytorch3d` ops backend.

Runtime smoke tests passed:

- PyTorch import.
- `PointNet2SemSegDataset` loads processed smoke data.
- PointNet++ forward pass returns `(B, N, 2)` logits on CUDA.
- One-epoch CPU smoke training on `processed-data/pointnet2_semseg_k41144_smoke_config`.
- One-epoch CUDA smoke training on `processed-data/pointnet2_semseg_k41144_smoke_config`.

CPU smoke experiment:

```text
experiments/pointnet2_semseg_k41144_20260530_151203/
```

CUDA smoke experiment:

```text
experiments/pointnet2_semseg_k41144_20260530_153222/
```

Observed CUDA smoke metrics:

```text
train_loss=0.7583
val_loss=0.6951
val_miou=0.1750
val_object_recall=0.0000
```

This only verifies the end-to-end training path; it is not a quality baseline.

Full synthetic baseline:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_180212
dataset: processed-data/pointnet2_semseg_k41144
points per sample: 16384
epochs: 20
batch_size: 1
device: cuda
runtime: about 21.6 minutes on NVIDIA GeForce MX150
```

Best validation epoch:

```text
epoch=19
val_loss=0.00499
val_mean_iou=0.9961
val_object_iou=0.9973
val_object_recall=0.9984
```

Final epoch:

```text
epoch=20
val_loss=0.00518
val_mean_iou=0.9948
val_object_iou=0.9964
val_object_recall=0.9969
```

Validation previews:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
output: experiments/pointnet2_semseg_k41144_20260606_180212/previews/val
sample_count: 8
overall_accuracy=0.9982
mean_iou=0.9960
object_iou=0.9972
object_precision=0.9990
object_recall=0.9982
confusion_matrix=[[45789,83],[155,85045]]
```

This is the first meaningful PointNet++ semantic segmentation baseline. It passes the synthetic validation target by a wide margin, but PLY previews still need human visual inspection and test evaluation before sign-off.

PointNet++ ops optimization:

```text
default ops backend: pytorch3d
fallback ops backend: pure_torch
config: configs/train/pointnet2_semseg_k41144.yaml
profile output: experiments/pointnet2_profile_pytorch3d_backend/
```

PyTorch3D backend check:

```text
pytorch3d._C available
CUDA forward/backward with pytorch3d backend: pass
train smoke with config backend=pytorch3d: pass
```

Profiler result on `processed-data/pointnet2_semseg_k41144`, 16384 points, batch size 1:

```text
pure_torch mean_seconds_per_batch=0.8340
pytorch3d mean_seconds_per_batch=0.1143
speedup: about 7.3x for the profiled training batch
```

PyTorch3D batch-size probe:

```text
batch_size=2 mean_seconds_per_batch=0.1872, max_reserved_mib=288.0
batch_size=4 mean_seconds_per_batch=0.4654, max_reserved_mib=572.0
```

Interpretation:

- Use `pytorch3d` for normal training on the current machine.
- Batch size 2 or 4 now fits on MX150 for the profiled config.
- Keep `pure_torch` for debugging and machines without PyTorch3D.
- Do not build a custom extension yet; tune batch size, DataLoader, and metric synchronization first.

### For PPRNet++-Style Training

Do not overwrite raw format. Create a converter later.

Potential training fields:

```text
points
semantic_labels
instance_labels
centroid_offsets
rotation_labels
visibility_labels
object_poses
instance_ids
```

The current raw dataset contains enough information to derive these fields.

### For YOLO Segmentation

The current raw data can be converted to YOLO segmentation format:

- Use `rgb.png` as input image.
- Use `instance_mask` from `sensor_data.npz` to create polygon masks.
- Since there is only one class, YOLO class id should be `0`.

Possible converter:

```text
scripts/export_yolo_segmentation.py
```

## Recommended Next Todo

1. Add `scripts/eval_pointnet2_semseg.py`.
2. Run held-out test evaluation for `experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt`.
3. Export optional test previews.
4. Add inference script/API for processed samples and later raw depth input.
5. After semantic segmentation sign-off, start the instance segmentation or PPRNet++-style pose regression branch.
