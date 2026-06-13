# Dataset Build And Test Guide

Date: 2026-06-08

Status: current through pose Phase 23.

This guide is the operational runbook for rebuilding datasets and rerunning tests for the current single-class bin-picking pipeline.

The current pipeline is:

```text
raw Blender synthetic dataset
  -> processed PointNet++ semantic/instance dataset
  -> instance segmentation checkpoint
  -> pose crop export from GT or predicted instances
  -> PointNet++ PS6D-style keypoint/center voting pose checkpoint
  -> GT-crop and predicted-crop pose evaluation
  -> optional raw-vs-refined pose evaluation
```

## Ground Rules

- Do not overwrite existing raw datasets in `synthetic-data/`.
- Generate each object as a separate single-class dataset. Do not mix K41144 and bending_pipe unless a future multi-class phase explicitly requires it.
- Use `object-model/K41144.stl` with `model_scale=1.0`.
- Use `object-model/bending_pipe.stl` with `model_scale=0.001`.
- Raw point clouds use positive-forward depth camera coordinates. Pose crops convert points/normals into Blender metadata camera coordinates by flipping `z`.
- K41144 pose metrics use symmetry `{I, R_y(pi)}`.
- bending_pipe pose metrics use symmetry `{I}`.

Recommended output naming:

```text
synthetic-data/<object>_<date_or_run_name>
processed-data/pointnet2_semseg_<object>_<date_or_run_name>
experiments/phaseXX_<description>_<object>_<split>
```

## Environment Checks

From project root:

```powershell
python --version
python -m py_compile scripts\generate_synthetic_blender.py scripts\prepare_pointnet2_semseg_dataset.py scripts\validate_raw_dataset.py
python -m py_compile scripts\train_pointnet2_instance_seg.py scripts\eval_pointnet2_instance_seg.py scripts\export_pose_instance_crops.py
python -m py_compile scripts\train_pointnet2_pose.py scripts\eval_pointnet2_pose.py
```

Expected:

```text
No output from py_compile means the syntax check passed.
```

## Build Raw Synthetic Dataset

### Option A: GUI

Run:

```powershell
python .\scripts\dataset_gui.py
```

In the Generation group:

- Select `object-model/K41144.stl` or `object-model/bending_pipe.stl`.
- Set class name to `K41144` or `bending_pipe`.
- Set model scale to `1.0` for K41144 or `0.001` for bending_pipe.
- Set depth camera, RGB camera, debug video camera, light, spawn, restitution, and settle-frame parameters.
- Use `1 Sample + Video` to generate a single accepted sample with `spawn_simulation.mp4`. This mode forces video frame step 1 so object falling and settling are easier to inspect before running a full dataset.
- Use Import Preset / Export Preset for repeatable generator settings.
- Choose a new empty output folder.

### Option B: CLI Preset For bending_pipe

The current recommended bending_pipe preset uses progressive spawn mode and scales to 30 objects:

```text
configs/generator/bending_pipe_progressive.json
```

Run Blender:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --settings-file .\configs\generator\bending_pipe_progressive.json --output .\synthetic-data\bending_pipe_new
```

The older `configs/generator/bending_pipe_active_spawn_stable.json` is kept as a delayed-mode reference. Progressive mode drops one object at a time onto the pile, so 30 long parts no longer roll out of the bin. Add `--record-failed-video` while tuning to capture an MP4 of any rejected attempt under `<output>/rejected/`.

### Option C: CLI Example For K41144

Run Blender:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --model .\object-model\K41144.stl --class-name K41144 --model-scale 1.0 --output .\synthetic-data\K41144_new --samples 80 --objects 30 --width 640 --height 480 --bin-wall-height 0.14 --drop-height-min 0.12 --drop-height-max 0.34 --spawn-strategy layered --objects-per-layer 6 --spawn-min-distance 0.045 --spawn-settle-frames 35 --collision-margin 0.0005 --object-restitution 0.01 --min-visible-objects 12 --min-visible-points 8000 --max-sample-attempts 12 --settle-frames 260
```

Generation notes:

- `--spawn-mode progressive` (default) drops one object at a time from `pile_top + bounding_radius + drop_clearance`, settles it, freezes it as a passive collider, then drops the next, followed by a final all-active relaxation. This is the PyBullet-style mode and produces the fewest out-of-bin ejections. Tune with `--drop-clearance-min/--drop-clearance-max`, `--progressive-settle-frames`, `--final-relax-frames`.
- If objects look like they freeze before settling and end up floating, raise `--progressive-settle-frames`, or use `--no-progressive-freeze` to keep all objects active so the pile keeps re-settling each stage (slower but no mid-fall freezing).
- `--spawn-mode delayed` is the legacy parked delayed-activation path (uses `--drop-height-min/max`, `--spawn-settle-frames`). `--spawn-mode batch` activates all objects at frame 1. Both enforce a hard 3D spawn separation of `2 * bounding_radius + max(collision_margin, 0.0005)`.
- Out-of-bin is center-based (`world_center_xy`): a long part leaning on the wall with its center inside the bin is accepted; only a center outside the bin or a part dropping through the floor is rejected.
- A physics explosion watchdog rejects an attempt early with `reason="physics_explosion"` when any activated object exceeds the speed limit (`--explosion-speed-limit`, automatic by default, `0` disables) or leaves the vertical sanity band.
- `--record-failed-video` renders an MP4 of every rejected attempt into `<output>/rejected/sample_XXXXXX_attempt_YY_<reason>.mp4` for failure investigation.
- Physics defaults since 2026-06-10: `collision_margin 0.0005`, `physics_substeps 60`, `object_linear_damping 0.05`, `object_angular_damping 0.15`. See `docs/generator-physics-and-production-readiness-review.md` for the verified root cause behind these values.
- Simulation preview videos use the debug camera parameters: `--debug-camera-location`, `--debug-camera-target`, and `--debug-camera-lens`.
- For parameter tuning, add `--samples 1 --record-simulation-video --simulation-video-frame-step 1` and inspect `sample_000000/spawn_simulation.mp4`. The preview uses a bright temporary material and hides bin walls only during video render, after raw RGB/depth/mask/metadata have already been written.

Raw sample folders should contain:

```text
depth.png
rgb.png
mask.png
normal.png
point_cloud.ply
metadata.json
```

## Validate Raw Dataset

Run after every generation:

```powershell
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\K41144_new --json-output .\experiments\validation_K41144_new.json
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\bending_pipe_new --json-output .\experiments\validation_bending_pipe_new.json
```

Expected:

```text
Validation: OK
OK equals Samples
```

Current reference raw validation:

```text
synthetic-data/K41144:       77/77 OK
synthetic-data/bending_pipe: 30/30 OK
```

If validation fails:

- Check out-of-bin counts first.
- Check `rejection_stats` in `dataset_summary.json` for `physics_explosion` entries; they include the object, frame, and measured speed. Frequent explosions usually mean the bin is too crowded for the object size or the collision margin was lowered again.
- Increase `spawn_min_distance`, `spawn_settle_frames`, or `settle_frames`.
- Reduce `object_restitution`.
- Do not use `--allow-out-of-bin-filtering` for training datasets unless doing a legacy-data audit.

## Build Processed PointNet++ Dataset

Processed datasets are derived from raw datasets and are safe to regenerate into a new folder.

K41144:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\K41144 --out .\processed-data\pointnet2_semseg_k41144_new --num-points 16384 --use-normals --object-fraction-target 0.65 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 7
```

bending_pipe:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\bending_pipe --out .\processed-data\pointnet2_semseg_bending_pipe_new --num-points 16384 --use-normals --object-fraction-target 0.65 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 7
```

Expected processed structure:

```text
processed-data/pointnet2_semseg_<object>_new/
  conversion_config.json
  dataset_stats.json
  schema.md
  train/*.npz
  val/*.npz
  test/*.npz
```

Each processed `.npz` should include:

```text
points
features
semantic_labels
instance_labels
point_pixels
raw_sample
```

## Instance Segmentation Smoke Test

Use the smoke test before full training to verify the processed dataset, model, loss, and GPU path.

K41144:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --epochs 2 --batch-size 1 --device cuda --limit-train-samples 2 --limit-val-samples 2 --disable-augment
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe --epochs 2 --batch-size 1 --device cuda --limit-train-samples 2 --limit-val-samples 2 --disable-augment
```

Expected:

```text
[train] epoch=1 ...
[train] epoch=2 ...
wrote experiment: experiments/pointnet2_instance_<object>_<timestamp>
checkpoints/best.pt exists
```

## Train Instance Segmentation

K41144:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --device cuda
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe --device cuda
```

Current reference checkpoints:

```text
K41144:
  experiments/pointnet2_instance_k41144_20260607_165729/checkpoints/best.pt

bending_pipe:
  experiments/pointnet2_instance_bending_pipe_20260607_171218/checkpoints/best.pt
```

## Evaluate Instance Segmentation

K41144:

```powershell
python .\scripts\eval_pointnet2_instance_seg.py --checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_k41144 --split test --out .\experiments\pointnet2_instance_k41144_20260607_165729\test_metrics.json --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.004 --min-cluster-points 96 --preview-samples 3
```

bending_pipe:

```powershell
python .\scripts\eval_pointnet2_instance_seg.py --checkpoint .\experiments\pointnet2_instance_bending_pipe_20260607_171218\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_bending_pipe --split test --out .\experiments\pointnet2_instance_bending_pipe_20260607_171218\test_metrics.json --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.006 --min-cluster-points 32 --preview-samples 3
```

Inspect:

```text
overall_semantic_metrics.object_iou
mean_instance_metrics.instance_recall
mean_instance_metrics.instance_precision
mean_instance_metrics.instance_mean_iou
mean_instance_metrics.merge_count
mean_instance_metrics.split_count
```

Use preview PLY files to debug merge/split errors.

## Export GT Pose Crops

Pose training does not read raw Blender folders directly. Export pose crops from processed datasets.

K41144:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split train --source gt --raw-root .\synthetic-data\K41144 --out .\experiments\pose_crops_k41144_gt_train_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split val --source gt --raw-root .\synthetic-data\K41144 --out .\experiments\pose_crops_k41144_gt_val_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split test --source gt --raw-root .\synthetic-data\K41144 --out .\experiments\pose_crops_k41144_gt_test_xyznormal
```

bending_pipe:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split train --source gt --raw-root .\synthetic-data\bending_pipe --out .\experiments\pose_crops_bending_pipe_gt_train_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split val --source gt --raw-root .\synthetic-data\bending_pipe --out .\experiments\pose_crops_bending_pipe_gt_val_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split test --source gt --raw-root .\synthetic-data\bending_pipe --out .\experiments\pose_crops_bending_pipe_gt_test_xyznormal
```

Current reference GT crop exports:

```text
K41144:
  train: experiments/phase14_pose_crops_k41144_gt_train_xyznormal   (1406 crops)
  val:   experiments/phase14_pose_crops_k41144_gt_val_xyznormal     (185 crops)
  test:  experiments/phase14_pose_crops_k41144_gt_test_xyznormal    (146 crops)

bending_pipe:
  train: experiments/phase14_pose_crops_bending_pipe_gt_train_xyznormal (142 crops)
  val:   experiments/phase14_pose_crops_bending_pipe_gt_val_xyznormal   (18 crops)
  test:  experiments/phase14_pose_crops_bending_pipe_gt_test_xyznormal  (18 crops)
```

## Pose Dataset Identity Test

Run this before training or after any frame/scale change.

K41144:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --out .\experiments\pose_identity_k41144_val.json --identity-baseline --device cuda
```

bending_pipe:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --out .\experiments\pose_identity_bending_pipe_val.json --identity-baseline --device cuda
```

Expected:

```text
ADD near 0 mm
translation near 0 mm
pose_success_add_0.1d = 1.0
```

Current bending_pipe identity reference:

```text
ADD 0.000004 mm
translation 0.000001 mm
ADD_0.1d success 1.0
```

## Pose Tiny Overfit Smoke Test

This catches architecture, loss, Kabsch, symmetry, and frame bugs quickly.

K41144:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --epochs 80 --batch-size 2 --device cuda --limit-train-samples 2 --limit-val-samples 2 --dropout 0.0 --learning-rate 0.001
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --epochs 120 --batch-size 2 --device cuda --limit-train-samples 2 --limit-val-samples 2 --dropout 0.0 --learning-rate 0.001
```

Expected pass:

```text
ADD_0.1d success reaches 1.0 on the 2-crop validation set.
ADD drops well below 0.1 * object diameter.
```

Current bending_pipe keypoint-voting tiny overfit reference:

```text
experiment: experiments/pointnet2_pose_bending_pipe_20260608_015437
best ADD roughly 3-4 mm
ADD_0.1d success 1.0
```

## Train Pose Keypoint Voting

K41144:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --epochs 30 --batch-size 32 --device cuda --learning-rate 0.001 --dropout 0.05
```

Optional K41144 fine-tune from `best_add.pt`:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --epochs 40 --batch-size 32 --device cuda --learning-rate 0.0005 --dropout 0.05 --resume .\experiments\pointnet2_pose_k41144_20260608_021636\checkpoints\best_add.pt --resume-new-experiment
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --epochs 90 --batch-size 16 --device cuda --learning-rate 0.001 --dropout 0.05
```

Optional bending_pipe fine-tune:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --epochs 150 --batch-size 16 --device cuda --learning-rate 0.0002 --dropout 0.05 --resume .\experiments\pointnet2_pose_bending_pipe_20260608_015529\checkpoints\best_add.pt --resume-new-experiment
```

Important checkpoint:

```text
checkpoints/best_add.pt
```

Use `best_add.pt` for pose evaluation because it is selected by validation ADD, not only total loss.

## Evaluate GT-Crop Pose

K41144:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\pointnet2_pose_k41144_20260608_023605\test_best_add_eval.json --device cuda
```

bending_pipe:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\pointnet2_pose_bending_pipe_20260608_021309\test_best_add_eval.json --device cuda
```

Current Phase 18 GT-crop test reference:

```text
K41144:
  sample_count: 146
  ADD 5.607 mm
  translation 6.021 mm
  ADD_0.1d success 0.897

bending_pipe:
  sample_count: 18
  ADD 9.899 mm
  translation 9.529 mm
  ADD_0.1d success 0.833
```

## Export Predicted Pose Crops

K41144:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split test --source predicted --checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --raw-root .\synthetic-data\K41144 --out .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.004 --min-cluster-points 96
```

bending_pipe:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split test --source predicted --checkpoint .\experiments\pointnet2_instance_bending_pipe_20260607_171218\checkpoints\best.pt --raw-root .\synthetic-data\bending_pipe --out .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.006 --min-cluster-points 32
```

Current predicted crop reference:

```text
K41144:
  total_crops=150
  crops_with_object_to_camera=150

bending_pipe:
  total_crops=14
  crops_with_object_to_camera=14
```

## Evaluate Predicted-Crop Pose

Always run both filtered and unfiltered evaluations.

K41144, matched crops:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --out .\experiments\pointnet2_pose_k41144_20260608_023605\pred_test_iou05_eval.json --min-matched-gt-iou 0.5 --device cuda
```

K41144, all predicted crops:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --out .\experiments\pointnet2_pose_k41144_20260608_023605\pred_test_all_eval.json --device cuda
```

bending_pipe, matched crops:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --out .\experiments\pointnet2_pose_bending_pipe_20260608_021309\pred_test_iou05_eval.json --min-matched-gt-iou 0.5 --device cuda
```

bending_pipe, all predicted crops:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --out .\experiments\pointnet2_pose_bending_pipe_20260608_021309\pred_test_all_eval.json --device cuda
```

Current Phase 18 predicted-crop reference:

```text
K41144, matched_gt_iou >= 0.5:
  sample_count 103
  mean matched IoU 0.805
  ADD 10.719 mm
  translation 11.087 mm
  ADD_0.1d success 0.738

K41144, all predicted crops:
  sample_count 150
  mean matched IoU 0.661
  ADD 21.932 mm
  translation 22.831 mm
  ADD_0.1d success 0.540

bending_pipe, matched_gt_iou >= 0.5:
  sample_count 13
  mean matched IoU 0.958
  ADD 12.146 mm
  translation 13.470 mm
  ADD_0.1d success 0.923

bending_pipe, all predicted crops:
  sample_count 14
  mean matched IoU 0.909
  ADD 18.456 mm
  translation 18.612 mm
  ADD_0.1d success 0.714
```

## Phase 18-20 Pose Audit, Ablation, And Refinement

The current raw keypoint-voting pose model meets ADD_0.1d target on GT-crop test for both objects:

```text
K41144 GT-crop test ADD_0.1d:       0.897
bending_pipe GT-crop test ADD_0.1d: 0.833
```

Remaining known gap:

```text
translation_error_mm is still above the strict 5 mm target:
  K41144 test:       6.021 mm
  bending_pipe test: 9.529 mm
```

Phase 20 adds optional hybrid ICP refinement and reports refined pose separately from raw keypoint-voting pose.

Phase 18 audit commands:

```powershell
python .\scripts\analyze_pose_errors.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\phase18_pose_error_audit_k41144_gt_test.json --export-worst-ply 5 --device cuda

python .\scripts\analyze_pose_errors.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\phase18_pose_error_audit_bending_gt_test.json --export-worst-ply 5 --device cuda
```

Current audit finding:

```text
K41144 bucket counts:
  large_translation_large_rotation: 22
  large_translation_small_rotation: 25
  small_translation_large_rotation: 12
  ok_or_mild_error: 87

bending_pipe bucket counts:
  large_translation_large_rotation: 1
  large_translation_small_rotation: 9
  small_translation_large_rotation: 1
  ok_or_mild_error: 7
```

Phase 19 center-vote aggregation flags:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\phase19_pose_ablation_k41144_gt_test_median.json --center-vote-aggregation median --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\phase19_pose_ablation_bending_gt_test_geometric_median.json --center-vote-aggregation geometric_median --device cuda
```

Keep `mean` as the raw default. Robust aggregation improved translation only mildly and not consistently across ADD and both objects.

Phase 20 recommended optional refinement preset:

```text
configs/eval/pose_refinement_defaults.json
```

GT-crop refined evaluation:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\phase20_refine_hybrid_wide16_t20_r20_k41144_gt.json --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\phase20_refine_hybrid_wide16_t20_r20_bending_gt.json --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

Current Phase 20 GT-crop refined reference:

```text
K41144:
  raw     ADD 5.607 mm, translation 6.021 mm, ADD_0.1d 0.897
  refined ADD 4.896 mm, translation 5.311 mm, ADD_0.1d 0.925
  accepted 128/146, catastrophic ADD worse > 0.05d: 0

bending_pipe:
  raw     ADD 9.899 mm, translation 9.529 mm, ADD_0.1d 0.833
  refined ADD 5.870 mm, translation 5.282 mm, ADD_0.1d 0.944
  accepted 16/18, catastrophic ADD worse > 0.05d: 0
```

Current Phase 20 predicted-crop refined reference:

```text
K41144, matched_gt_iou >= 0.5:
  raw     ADD 10.719 mm, translation 11.087 mm, ADD_0.1d 0.738
  refined ADD 10.049 mm, translation 10.793 mm, ADD_0.1d 0.777

K41144, all predicted:
  raw     ADD 21.932 mm, translation 22.831 mm, ADD_0.1d 0.540
  refined ADD 21.476 mm, translation 23.001 mm, ADD_0.1d 0.567

bending_pipe, matched_gt_iou >= 0.5:
  raw     ADD 12.146 mm, translation 13.470 mm, ADD_0.1d 0.923
  refined ADD 10.928 mm, translation 12.038 mm, ADD_0.1d 0.923

bending_pipe, all predicted:
  raw     ADD 18.456 mm, translation 18.612 mm, ADD_0.1d 0.714
  refined ADD 16.840 mm, translation 16.682 mm, ADD_0.1d 0.786
```

Known caveat:

```text
K41144 predicted-all translation worsens slightly after refinement even though ADD and ADD_0.1d improve.
Do not treat refinement as a replacement for Phase 21 predicted-crop quality retuning.
```

## Phase 21 Predicted-Crop Sweep

Run the full K41144 sweep:

```powershell
python .\scripts\sweep_pose_crop_export_params.py --instance-checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --instance-data .\processed-data\pointnet2_semseg_k41144 --raw-root .\synthetic-data\K41144 --pose-config .\configs\train\pointnet2_pose_voting_k41144.yaml --pose-checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --out .\experiments\phase21_k41144_pred_crop_sweep.json --device cuda --refine-pose
```

Smoke test:

```powershell
python .\scripts\sweep_pose_crop_export_params.py --instance-checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --instance-data .\processed-data\pointnet2_semseg_k41144 --raw-root .\synthetic-data\K41144 --pose-config .\configs\train\pointnet2_pose_voting_k41144.yaml --pose-checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --out .\experiments\phase21_smoke_k41144_pred_crop_sweep.json --object-probability-thresholds 0.50 --dbscan-eps-m 0.004 --min-cluster-points 96 --limit-samples 1 --device cuda --refine-pose
```

Smoke reference:

```text
prob_0p5_eps_0p004_min_96:
  crops 28
  matched IoU>=0.5 crops 17
  all refined ADD_0.1d 0.464 on the one-sample smoke slice
```

## Phase 22 Learned Translation Refiner

Train K41144 refiner:

```powershell
python .\scripts\train_pointnet2_pose_refiner.py --config .\configs\train\pointnet2_pose_refiner_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --device cuda
```

Train bending_pipe refiner:

```powershell
python .\scripts\train_pointnet2_pose_refiner.py --config .\configs\train\pointnet2_pose_refiner_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --device cuda
```

Evaluate with a learned refiner checkpoint:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\k41144_gt_with_translation_refiner.json --translation-refiner-checkpoint .\experiments\<refiner_run>\checkpoints\best_add.pt --device cuda
```

Optional geometry after learned refiner:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\k41144_gt_with_translation_refiner_and_icp.json --translation-refiner-checkpoint .\experiments\<refiner_run>\checkpoints\best_add.pt --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

Smoke reference:

```text
experiments/pointnet2_pose_refiner_k41144_20260608_215456
4-crop validation slice: raw ADD 6.964 mm -> refined ADD 6.743 mm at epoch 2
```

Keep learned refinement experimental until it beats Phase 20 hybrid refinement in the full Phase 23 suite.

## Phase 23 Pose Evaluation Suite

Run the full raw/refined suite:

```powershell
python .\scripts\run_pose_evaluation_suite.py --device cuda --refine-pose
```

Run a quick smoke suite:

```powershell
python .\scripts\run_pose_evaluation_suite.py --out-dir .\experiments\phase23_smoke_pose_eval_suite --device cuda --limit-samples 2 --refine-pose
```

Outputs:

```text
summary.json
summary.md
per-case evaluation JSON files
```

## Quick Regression Checklist

Run this after pose/data code changes:

```powershell
python -m py_compile src\data\pose_crop_dataset.py src\models\pointnet2_pose.py src\models\pointnet2_pose_voting.py src\models\pointnet2_pose_refiner.py src\training\pose_losses.py src\training\pose_metrics.py src\training\pose_refiner_losses.py src\inference\pose_refinement.py scripts\train_pointnet2_pose.py scripts\train_pointnet2_pose_refiner.py scripts\eval_pointnet2_pose.py scripts\export_pose_instance_crops.py scripts\analyze_pose_errors.py scripts\sweep_pose_crop_export_params.py scripts\run_pose_evaluation_suite.py

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --out .\experiments\smoke_pose_identity_k41144_val.json --identity-baseline --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --out .\experiments\smoke_pose_identity_bending_val.json --identity-baseline --device cuda
```

Pass criteria:

```text
py_compile passes
identity ADD near 0 mm
identity ADD_0.1d success = 1.0
```
