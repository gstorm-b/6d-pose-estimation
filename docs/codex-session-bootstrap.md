# Codex Session Bootstrap

Use this file at the beginning of a new Codex session to recover the minimum project context and work like a senior engineer on this repository.

## First Files To Read

Read these files in this order:

```text
docs/engineering-rules.md
docs/session-summary-synthetic-data.md
docs/pointnetplusplus-implementation-plan.md
docs/pointnet2-instance-segmentation-plan.md
docs/pointnet2-pose-estimation-plan.md
docs/dataset-build-and-test-guide.md
```

Then inspect the current user request, usually under:

```text
docs/request/
```

If the task touches synthetic generation or dataset quality, inspect:

```text
scripts/generate_synthetic_blender.py
scripts/dataset_gui.py
src/data/raw_dataset.py
src/data/validation.py
```

If the task touches visualization, inspect:

```text
scripts/view_point_cloud_open3d.py
scripts/dataset_gui.py
scripts/view_ply_pyside_vtk.py
```

If the task touches PointNet++ or training, inspect:

```text
docs/pointnetplusplus-implementation-plan.md
docs/pointnet2-instance-segmentation-plan.md
docs/pointnet2-pose-estimation-plan.md
docs/dataset-build-and-test-guide.md
```

and then look for current `configs/`, `src/`, `scripts/`, `processed-data/`, and `experiments/` folders before assuming they exist.

## Non-Negotiable Project Rules

- Raw generated data under `synthetic-data/` is immutable. Do not manually edit samples.
- Do not train from raw Blender output directly. Convert raw data into `processed-data/`.
- Do not train until the dataset passes validation.
- Keep coordinate-frame direction explicit in names and docs, for example `object_to_camera`, `camera_to_world`, `body_to_world`.
- Use meters internally for geometry, point clouds, transforms, camera settings, and simulation settings.
- Keep scripts thin and move reusable logic into `src/`.
- Keep config outside reusable code where practical.
- When changing behavior, schema, CLI flags, generation settings, validation policy, or workflow, update related Markdown in the same task.
- Confirm with the user before non-trivial implementation or structural changes unless the user has already explicitly approved the plan.

## Current Project Shape

Project goal:

- Build a 6D pose estimation and robotic bin-picking pipeline for `object-model/K41144.stl`.
- Generate synthetic raw data with Blender.
- Use point cloud as the main training signal.
- Preserve enough raw signals to support PointNet++, PPRNet++-style pose regression, YOLO segmentation, and later pose refinement.

Important object facts:

- Object: `object-model/K41144.stl`.
- Approximate size: `30 mm x 93.6 mm x 20 mm`.
- STL origin is not centered. Generator recenters mesh vertices for physics and stores transforms so exported poses still refer to the original STL coordinate frame.
- Secondary design/test object: `object-model/bending_pipe.stl`.
- `bending_pipe.stl` is authored in millimeters and should be generated with `--model-scale 0.001`.
- Test K41144 and bending_pipe as separate single-class datasets unless a future multi-class experiment explicitly requires mixing them.

Important raw sample files:

```text
metadata.json
sensor_data.npz
rgb.png
label_overlay.png
depth_preview.png
instance_mask.png
normal_camera.png
```

Important `sensor_data.npz` keys:

```text
depth_m
instance_mask
normal_world
normal_camera
points_camera
points_world
point_instance_ids
point_semantic_ids
point_pixels
```

Important metadata pose fields:

```text
original_model_to_body
body_to_world
object_to_world
object_to_camera
camera_to_world
world_to_camera
object_to_depth_camera
object_to_rgb_camera
depth_camera_to_world
rgb_camera_to_world
```

Legacy camera fields refer to the depth camera. RGB render camera fields are stored separately for `rgb.png`.

## Senior Engineer Workflow

Before editing:

- Read the relevant files instead of guessing.
- Check whether the repo has existing helpers, schemas, or conventions.
- Keep changes scoped to the request.
- Do not refactor unrelated code while implementing a feature.
- Treat generated datasets and prior experiment outputs as historical artifacts.

During implementation:

- Prefer explicit data structures and type hints.
- Make CLI inputs and outputs clear.
- Fail loudly when data is invalid.
- Record rejection reasons and data-quality decisions.
- Use reusable modules for logic and keep GUI/scripts as orchestration layers.
- For GUI work, keep heavy file I/O and validation off the UI thread. Use Qt worker threads and signals; keep rendering/UI updates on the UI thread.
- For VTK/Qt work, keep VTK rendering in the UI thread and load point arrays in worker threads.

After editing:

- Run targeted compile/import checks.
- Run validators or smoke tests on existing datasets when possible.
- If a command would generate a new raw dataset, be explicit about the output folder and avoid overwriting existing data.
- Summarize what changed, what was verified, and what was not run.

## Dataset And Validation Expectations

Raw dataset validation should check at minimum:

- Required files exist.
- `metadata.json` is readable.
- `sensor_data.npz` has expected keys.
- Array shapes are consistent.
- Instance ids match between metadata, masks, and point labels.
- Semantic ids are valid.
- Visible point and object counts meet configured thresholds.
- Pose matrices are finite.
- Objects are not out of bin, using `object_to_world` and model geometry or bbox.

Dataset generation should:

- Use the Blender generator as the source of truth.
- Generate into a new output folder unless overwrite is explicitly requested.
- Reject attempts with out-of-bin objects by default.
- Record generator settings and rejection stats.

## GUI Notes

The PySide6 GUI entrypoint is:

```powershell
python .\scripts\dataset_gui.py
```

GUI responsibilities:

- Run the Blender generator via `QProcess`.
- Expose model path, class name, model scale, depth-camera, RGB-camera, and light generation parameters.
- Expose spawn settle frames, object restitution, and JSON generator preset import/export in the Generation group.
- Stream logs without blocking the UI.
- Browse raw datasets and preview samples.
- Load sample previews and point clouds in worker threads.
- Render VTK content on the UI thread.
- Run raw dataset validation in a worker thread.

The GUI must not manually edit raw samples.

## PointNet++ Direction

Do not start training from current raw point arrays without reading:

```text
docs/pointnetplusplus-implementation-plan.md
```

Important limitation:

- Current `points_camera` and `points_world` arrays are object-only.
- Semantic segmentation needs background/bin points too.
- A PointNet++ converter should reconstruct full-scene points from `depth_m` and `camera_intrinsics`, then use `instance_mask` for labels.

Processed datasets should go under:

```text
processed-data/
```

Experiments should go under:

```text
experiments/
```

## Current Pose Direction

Instance segmentation Phase 0-10 is complete. The pose branch has Phase 11-13 implemented:

```text
src/data/pose_crop_dataset.py
src/training/pose_geometry.py
src/training/pose_metrics.py
src/training/pose_losses.py
src/models/pointnet2_pose.py
scripts/train_pointnet2_pose.py
scripts/eval_pointnet2_pose.py
configs/train/pointnet2_pose_k41144.yaml
configs/train/pointnet2_pose_bending_pipe.yaml
```

Pose training consumes crop exports from:

```text
scripts/export_pose_instance_crops.py
```

Frame and scale details:

- Pose targets use generator metadata camera coordinates; `PoseCropDataset` flips crop point-cloud `z` from positive-forward depth convention into metadata camera convention.
- K41144 symmetry set is `{I, R_y(pi)}`.
- bending_pipe symmetry set is `{I}` and uses `model_scale: 0.001`.
- Pose model uses crop-centered xyz+normal input plus 18 aux features: normalized crop centroid, normalized bbox extents relative to centroid, and xyz central moment features.

Tiny GT-crop overfit smoke results from Phase 13:

```text
K41144 2 crops: best ADD 6.627 mm, best translation 0.045 mm, ADD_0.1d success 1.0.
bending_pipe 2 crops: best ADD 0.218 mm, best translation 0.014 mm, ADD_0.1d success 1.0.
```

Phase 14 full GT-crop evaluation has been run, but the direct crop-level PointNet++ pose MVP did not meet target:

```text
K41144:
  crops: experiments/phase14_pose_crops_k41144_gt_*_xyznormal
  experiment: experiments/pointnet2_pose_k41144_20260607_230547
  test: ADD 27.719 mm, translation 30.225 mm, ADD_0.1d success 0.123

bending_pipe:
  crops: experiments/phase14_pose_crops_bending_pipe_gt_*_xyznormal
  experiment: experiments/pointnet2_pose_bending_pipe_20260607_230546
  test: ADD 80.936 mm, translation 71.057 mm, ADD_0.1d success 0.000
```

Phase 15-18 pose update:

```text
Primary pose path is now PS6D-style keypoint/center voting.

Implemented:
  src/models/pointnet2_pose_voting.py
  configs/train/pointnet2_pose_voting_k41144.yaml
  configs/train/pointnet2_pose_voting_bending_pipe.yaml

Training script additions:
  checkpoints/best_add.pt
  --resume
  --resume-new-experiment
```

GT-crop test results with keypoint voting:

```text
K41144:
  experiment: experiments/pointnet2_pose_k41144_20260608_023605
  checkpoint: checkpoints/best_add.pt
  test: ADD 5.607 mm, translation 6.021 mm, ADD_0.1d success 0.897

bending_pipe:
  experiment: experiments/pointnet2_pose_bending_pipe_20260608_021309
  checkpoint: checkpoints/best_add.pt
  test: ADD 9.899 mm, translation 9.529 mm, ADD_0.1d success 0.833
```

Predicted-crop test results:

```text
K41144 predicted crops:
  crops: experiments/phase17_pose_crops_k41144_pred_test_xyznormal
  matched_gt_iou >= 0.5: ADD 10.719 mm, translation 11.087 mm, ADD_0.1d success 0.738
  all predicted crops:    ADD 21.932 mm, translation 22.831 mm, ADD_0.1d success 0.540

bending_pipe predicted crops:
  crops: experiments/phase17_pose_crops_bending_pipe_pred_test_xyznormal
  matched_gt_iou >= 0.5: ADD 12.146 mm, translation 13.470 mm, ADD_0.1d success 0.923
  all predicted crops:    ADD 18.456 mm, translation 18.612 mm, ADD_0.1d success 0.714
```

Remaining gap:

- ADD_0.1d target is met on GT-crop test for both objects.
- Strict translation `< 5 mm` is still not fully met.
- Next improvement should add optional ICP/refinement or stronger translation supervision and report refined pose separately from raw keypoint-voting pose.

## Useful Starting Commands

Inspect repository shape:

```powershell
Get-ChildItem -Force
Get-ChildItem scripts -File
Get-ChildItem docs -Recurse -File
```

Run GUI import smoke test:

```powershell
python -c "import PySide6; import numpy; import PIL; print('gui imports ok')"
```

Run Python compile checks after script/module edits:

```powershell
python -m py_compile scripts\dataset_gui.py src\data\raw_dataset.py src\data\validation.py
```

Run the Open3D point-cloud viewer:

```powershell
python .\scripts\view_point_cloud_open3d.py .\synthetic-data\K41144\sample_000001
```

Run the PySide6/VTK PLY viewer:

```powershell
python .\scripts\view_ply_pyside_vtk.py .\experiments\pointnet2_semseg_k41144_20260606_180212\inference\processed_sample_000011\prediction.ply
```

Run the Blender generator manually:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --samples 3 --objects 30 --width 320 --height 240 --bin-wall-height 0.14 --drop-height-min 0.12 --drop-height-max 0.34 --spawn-strategy layered --objects-per-layer 6 --spawn-min-distance 0.045 --collision-margin 0.00002 --min-visible-objects 12 --min-visible-points 8000 --max-sample-attempts 12 --settle-frames 260 --output synthetic-data/K41144_test_new
```

## Handoff Checklist

At the end of a session, report:

- Files changed.
- Behavior changed.
- Commands/tests run.
- Any generated output folders.
- Any known risks or follow-up tasks.
