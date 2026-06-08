# Pose Estimation Completion And Accuracy Plan

Date: 2026-06-08

Status: Phase 18-23 implemented. Phase 22 learned translation refinement is available but remains an experimental optional path until trained on larger datasets and accepted by the Phase 23 suite.

This plan is written for engineers implementing the next pose-estimation improvements. It evaluates the current state against `docs/pointnet2-pose-estimation-plan.md`, then defines the next implementation phases with concrete deliverables, files, commands, acceptance gates, and rollback criteria.

## Executive Summary

The pose branch has moved from direct crop-level regression to a PS6D-style keypoint/center voting model. That change solved the biggest architecture gap from Phase 14.

Current state:

```text
Phase 11: Pose crop dataset and model-point utilities        Done
Phase 12: Pose metrics and evaluation skeleton               Done
Phase 13: Direct pose MVP                                    Done, baseline only
Phase 14: Full GT-crop direct-pose training                  Done, target not met
Phase 15: PS6D-style keypoint/center voting                  Done
Phase 16: Full GT-crop keypoint-voting evaluation            Done
Phase 17: Predicted-crop pose evaluation                     Done
Phase 18: Raw pose error audit                               Done
Phase 19: Center-vote aggregation ablation                   Done
Phase 20: Conservative optional refinement                   Done
Phase 21: K41144 predicted-crop quality retune tools         Done
Phase 22: Learned translation refiner                        Done, experimental
Phase 23: Automated pose evaluation suite                    Done
```

Current Phase 18 baseline:

```text
K41144, GT-crop test:
  ADD 5.607 mm
  translation 6.021 mm
  ADD_0.1d success 0.897

bending_pipe, GT-crop test:
  ADD 9.899 mm
  translation 9.529 mm
  ADD_0.1d success 0.833

K41144, predicted crops, matched_gt_iou >= 0.5:
  ADD 10.719 mm
  translation 11.087 mm
  ADD_0.1d success 0.738

K41144, predicted crops, all:
  ADD 21.932 mm
  translation 22.831 mm
  ADD_0.1d success 0.540

bending_pipe, predicted crops, matched_gt_iou >= 0.5:
  ADD 12.146 mm
  translation 13.470 mm
  ADD_0.1d success 0.923

bending_pipe, predicted crops, all:
  ADD 18.456 mm
  translation 18.612 mm
  ADD_0.1d success 0.714
```

Phase 20 verified optional refinement, using `configs/eval/pose_refinement_defaults.json`:

```text
K41144, GT-crop test:
  raw     ADD 5.607 mm, translation 6.021 mm, ADD_0.1d success 0.897
  refined ADD 4.896 mm, translation 5.311 mm, ADD_0.1d success 0.925

bending_pipe, GT-crop test:
  raw     ADD 9.899 mm, translation 9.529 mm, ADD_0.1d success 0.833
  refined ADD 5.870 mm, translation 5.282 mm, ADD_0.1d success 0.944

K41144, predicted crops, matched_gt_iou >= 0.5:
  raw     ADD 10.719 mm, translation 11.087 mm, ADD_0.1d success 0.738
  refined ADD 10.049 mm, translation 10.793 mm, ADD_0.1d success 0.777

K41144, predicted crops, all:
  raw     ADD 21.932 mm, translation 22.831 mm, ADD_0.1d success 0.540
  refined ADD 21.476 mm, translation 23.001 mm, ADD_0.1d success 0.567

bending_pipe, predicted crops, matched_gt_iou >= 0.5:
  raw     ADD 12.146 mm, translation 13.470 mm, ADD_0.1d success 0.923
  refined ADD 10.928 mm, translation 12.038 mm, ADD_0.1d success 0.923

bending_pipe, predicted crops, all:
  raw     ADD 18.456 mm, translation 18.612 mm, ADD_0.1d success 0.714
  refined ADD 16.840 mm, translation 16.682 mm, ADD_0.1d success 0.786
```

Conclusion:

- The learned raw pose model is usable and meets ADD_0.1d target on GT-crop test for both objects.
- The strict translation target `< 5 mm` is not met yet.
- K41144 predicted-crop performance is now limited by instance crop quality.
- bending_pipe predicted matched-crop performance is strong, but all-predicted performance still falls when bad crops are included.

The next work should prioritize accuracy in this order:

1. K41144 predicted-crop quality and clustering retune.
2. Automated evaluation suite so each accuracy change is measured consistently.
3. Optional learned translation/confidence head if non-training refinement plateaus.

## Current System Contract

Do not break these contracts:

- Pose training and evaluation read pose crop exports, not raw Blender folders directly.
- Raw synthetic data remains immutable.
- K41144 and bending_pipe are separate single-class datasets.
- `object_to_camera` remains the transform direction for all pose outputs.
- Raw keypoint-voting pose must remain available even after refinement is added.
- Any refinement must be reported separately from raw network pose.
- K41144 symmetry remains `{I, R_y(pi)}`.
- bending_pipe symmetry remains `{I}`.

## Accuracy Targets

Use these as the next engineering gates.

### Gate A: Raw GT-Crop Pose

Minimum:

```text
K41144:
  ADD_0.1d success >= 0.90
  translation_error_mm <= 5.5

bending_pipe:
  ADD_0.1d success >= 0.85
  translation_error_mm <= 8.0
```

Stretch:

```text
K41144:
  ADD_0.1d success >= 0.92
  translation_error_mm <= 5.0

bending_pipe:
  ADD_0.1d success >= 0.90
  translation_error_mm <= 5.0
```

### Gate B: Refined GT-Crop Pose

Minimum:

```text
K41144:
  refined translation_error_mm <= 5.0
  refined ADD_0.1d success >= raw ADD_0.1d success

bending_pipe:
  refined translation_error_mm <= 7.0
  refined ADD_0.1d success >= raw ADD_0.1d success
```

Stretch:

```text
Both objects:
  refined translation_error_mm <= 5.0
  refined ADD_0.1d success >= 0.90
```

### Gate C: Predicted-Crop Pose

Minimum:

```text
K41144, matched_gt_iou >= 0.5:
  ADD_0.1d success >= 0.80

K41144, all predicted:
  ADD_0.1d success >= 0.60

bending_pipe, matched_gt_iou >= 0.5:
  ADD_0.1d success >= 0.92

bending_pipe, all predicted:
  ADD_0.1d success >= 0.75
```

Stretch:

```text
K41144, matched_gt_iou >= 0.5:
  ADD_0.1d success >= 0.85

K41144, all predicted:
  ADD_0.1d success >= 0.70

bending_pipe, all predicted:
  ADD_0.1d success >= 0.80
```

## Phase 18: Raw Pose Error Audit

Status: implemented.

Purpose:

Before changing model or refinement logic, produce per-crop diagnostics that explain where the current errors come from.

Deliverables:

```text
scripts/analyze_pose_errors.py
experiments/pose_error_audit_<object>_<split>.json
experiments/pose_error_audit_<object>_<split>/worst_*.ply
```

Implementation details:

- Load a pose checkpoint and pose crop dataset.
- Run raw pose inference.
- Save per-crop metrics:
  - `translation_error_mm`
  - `rotation_error_deg_symmetry_aware`
  - `add_mm`
  - `pose_success_add_0.1d`
  - `matched_gt_iou`
  - `crop_point_count`
  - `sample_name`
  - `crop_index`
  - predicted origin
  - GT origin
  - origin delta vector in camera frame
- Bucket failures by:
  - large translation, small rotation error
  - small translation, large rotation error
  - both large
  - low matched IoU
  - tiny crop point count
- Export optional PLY for worst cases:
  - crop points
  - transformed predicted model points
  - transformed GT model points if GT is available
  - origin markers

Primary files:

```text
scripts/analyze_pose_errors.py
src/training/pose_metrics.py
src/training/pose_geometry.py
```

Commands:

```powershell
python .\scripts\analyze_pose_errors.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\pose_error_audit_k41144_gt_test.json --export-worst-ply 20 --device cuda

python .\scripts\analyze_pose_errors.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\pose_error_audit_bending_pipe_gt_test.json --export-worst-ply 20 --device cuda
```

Pass criteria:

```text
JSON includes every evaluated crop.
Worst-case PLYs open and show predicted/GT model alignment.
Audit confirms whether translation or rotation dominates each object's remaining error.
```

Expected finding:

- K41144 should have moderate translation-dominant residuals.
- bending_pipe may have split-dependent translation error due to limited training data and partial visible geometry.

## Phase 19: Center-Vote Aggregation Ablation

Status: implemented; raw default remains `mean`.

Purpose:

Improve translation without retraining by replacing the current mean of per-point center votes with robust aggregation.

Why this is first:

- Current translation comes from `mean(camera_point + predicted_origin_offset)`.
- Mean is sensitive to bad visible points and bad votes.
- A robust aggregation can reduce translation error with low implementation risk.

Deliverables:

```text
src/training/pose_metrics.py
scripts/eval_pointnet2_pose.py
configs/eval/pose_refinement_defaults.json
```

Add evaluator flags:

```text
--center-vote-aggregation mean|median|trimmed_mean|weighted_trimmed_mean|geometric_median
--center-vote-trim-fraction 0.1
--center-vote-max-residual-fraction 0.25
```

Implementation details:

Current:

```python
origin = origin_votes.mean(dim=1)
```

Implement:

```text
mean:
  current behavior

median:
  coordinate-wise median of origin votes

trimmed_mean:
  compute median origin
  compute distance of each vote to median
  keep closest 70-90 percent
  average kept votes

weighted_trimmed_mean:
  same as trimmed_mean
  weight votes by inverse residual to median

geometric_median:
  Weiszfeld iterations over origin votes
  cap iterations to keep eval fast
```

Important:

- Keep default as `mean` until ablation proves improvement.
- Report selected aggregation in JSON.
- Do not use GT pose to choose aggregation per crop.

Evaluation matrix:

```text
Objects:
  K41144
  bending_pipe

Datasets:
  GT test
  predicted test with IoU >= 0.5
  all predicted test

Aggregations:
  mean
  median
  trimmed_mean trim=0.1
  trimmed_mean trim=0.2
  weighted_trimmed_mean trim=0.1
  geometric_median
```

Commands:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\pose_ablation_k41144_gt_test_trimmed.json --center-vote-aggregation trimmed_mean --center-vote-trim-fraction 0.1 --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\pose_ablation_bending_gt_test_trimmed.json --center-vote-aggregation trimmed_mean --center-vote-trim-fraction 0.1 --device cuda
```

Pass criteria:

```text
No regression in ADD_0.1d success.
At least one robust aggregation improves translation on GT test for either object.
If improvement is consistent on both objects, update default config/eval guide.
```

Rollback criteria:

```text
If all robust aggregation modes reduce ADD_0.1d or increase translation, keep mean as default and move to Phase 20.
```

## Phase 20: Conservative ICP / Pose Refinement

Status: implemented as optional raw-vs-refined evaluator path.

Purpose:

Add optional post-network pose refinement, reported separately from raw pose.

Design principle:

The crop point cloud is partial visible geometry. Full-mesh ICP can incorrectly pull the object origin toward visible surfaces. Therefore refinement must be conservative and gated.

Deliverables:

```text
src/inference/pose_refinement.py
scripts/eval_pointnet2_pose.py
docs/dataset-build-and-test-guide.md
docs/pointnet2-pose-estimation-plan.md
```

Evaluator flags:

```text
--refine-pose
--refinement-method translation_only_icp|point_to_point_icp|hybrid
--refinement-max-iterations 10
--refinement-distance-threshold-fraction 0.08
--refinement-max-translation-delta-fraction 0.10
--refinement-max-rotation-delta-deg 10
--refinement-accept-if-improved
```

Refinement candidates:

### Candidate A: Translation-Only ICP

Keep raw rotation fixed.

Algorithm:

```text
1. Transform model points with raw rotation/origin.
2. Find nearest model point for each crop point.
3. Keep inliers below distance threshold.
4. Estimate translation delta as robust median/trimmed mean of:
   crop_point - nearest_model_point
5. Clamp translation delta to max_translation_delta_fraction * diameter.
6. Accept only if one-sided crop-to-model Chamfer improves.
```

Why:

- Low risk to rotation.
- Directly targets the current translation gap.
- Suitable as first refinement gate.

### Candidate B: Point-To-Point ICP

Update rotation and translation.

Algorithm:

```text
1. Transform model points with current pose.
2. Establish nearest-neighbor correspondences between crop points and transformed model points.
3. Keep inliers below threshold.
4. Solve Kabsch update.
5. Apply small pose update.
6. Reject if rotation delta exceeds max_rotation_delta_deg.
7. Reject if translation delta exceeds max_translation_delta_fraction * diameter.
8. Accept only if self-alignment score improves.
```

Why:

- Can help bending_pipe rotations if raw keypoints are slightly off.
- Higher risk than translation-only.

### Candidate C: Hybrid

Run translation-only first, then one or two point-to-point ICP steps with stricter gates.

Default recommendation:

```text
Start with translation_only_icp.
Only enable point_to_point_icp after audit shows rotation-dominant failures.
```

Implementation notes:

- Use PyTorch for nearest-neighbor distances if tensors are already on GPU.
- Use chunking for memory safety if model/crop point counts grow.
- `model_points_object` from the dataset is enough for initial implementation.
- Do not require Open3D for the default path.

JSON output additions:

```text
raw_metrics
refined_metrics
refinement:
  enabled
  method
  accepted_count
  rejected_count
  mean_alignment_before
  mean_alignment_after
  mean_translation_delta_mm
  mean_rotation_delta_deg
```

Pass criteria:

```text
Refined GT-crop ADD_0.1d success >= raw GT-crop ADD_0.1d success.
Refined translation improves on at least one object.
No catastrophic outliers where refined ADD worsens by > 0.05d.
```

Rollback criteria:

```text
If refinement improves self-alignment but worsens GT metrics, do not enable as recommended setting.
Keep code behind --refine-pose for future real-data experiments.
```

## Phase 21: Predicted-Crop Quality Retune For K41144

Status: implemented.

Purpose:

Improve full-pipeline K41144 pose by reducing bad predicted crops. Current K41144 all-predicted ADD_0.1d is `0.540`, while matched-crop ADD_0.1d is `0.738`, indicating instance crop quality is now a major limiter.

Deliverables:

```text
scripts/sweep_pose_crop_export_params.py
experiments/phase21_k41144_pred_crop_sweep.json
```

Implemented command smoke:

```powershell
python .\scripts\sweep_pose_crop_export_params.py --instance-checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --instance-data .\processed-data\pointnet2_semseg_k41144 --raw-root .\synthetic-data\K41144 --pose-config .\configs\train\pointnet2_pose_voting_k41144.yaml --pose-checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --out .\experiments\phase21_smoke_k41144_pred_crop_sweep.json --object-probability-thresholds 0.50 --dbscan-eps-m 0.004 --min-cluster-points 96 --limit-samples 1 --device cuda --refine-pose
```

Full sweep command:

```powershell
python .\scripts\sweep_pose_crop_export_params.py --instance-checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --instance-data .\processed-data\pointnet2_semseg_k41144 --raw-root .\synthetic-data\K41144 --pose-config .\configs\train\pointnet2_pose_voting_k41144.yaml --pose-checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --out .\experiments\phase21_k41144_pred_crop_sweep.json --device cuda --refine-pose
```

Sweep parameters:

```text
object_probability_threshold:
  0.45, 0.50, 0.55, 0.60

dbscan_eps_m:
  0.003, 0.004, 0.005, 0.006, 0.007

min_cluster_points:
  64, 96, 128, 160
```

Metrics to collect for each sweep row:

```text
total_predicted_crops
crops_with_pose
matched_gt_iou_mean
matched_gt_iou_at_least_0.5
predicted-crop ADD_0.1d, IoU>=0.5
predicted-crop ADD_0.1d, all
merge_count
split_count
```

Important:

- Do not tune only on all-predicted pose success. That can hide low crop recall.
- Keep separate diagnosis:
  - pose model quality on matched crops
  - full-pipeline quality on all crops
  - instance recall/precision

Pass criteria:

```text
Find a K41144 export config with:
  all-predicted ADD_0.1d >= 0.60
  matched_gt_iou >= 0.5 crop count not reduced by more than 10 percent
```

Stretch:

```text
all-predicted ADD_0.1d >= 0.70
```

## Phase 22: Learned Translation Refinement Head

Status: implemented as an optional experimental refiner.

Purpose:

If robust aggregation and conservative ICP fail to meet translation targets, add a small learned residual head focused only on origin correction.

Do not start this phase before Phase 19 and Phase 20 results are known.

Deliverables:

```text
src/models/pointnet2_pose_refiner.py
src/training/pose_refiner_losses.py
scripts/train_pointnet2_pose_refiner.py
scripts/eval_pointnet2_pose.py integration
configs/train/pointnet2_pose_refiner_k41144.yaml
configs/train/pointnet2_pose_refiner_bending_pipe.yaml
```

Inputs:

```text
crop points in camera frame
crop normals
raw predicted pose
raw origin vote statistics:
  mean
  median
  trimmed_mean
  vote covariance
  vote residual quantiles
raw keypoint vote statistics
crop aux features
```

Outputs:

```text
origin_delta_camera_normalized (3)
```

Loss:

```text
translation residual Smooth L1
+ ADD loss with raw rotation fixed
```

Implemented files:

```text
src/models/pointnet2_pose_refiner.py
src/training/pose_refiner_losses.py
scripts/train_pointnet2_pose_refiner.py
configs/train/pointnet2_pose_refiner_k41144.yaml
configs/train/pointnet2_pose_refiner_bending_pipe.yaml
scripts/eval_pointnet2_pose.py --translation-refiner-checkpoint
```

Smoke command:

```powershell
python .\scripts\train_pointnet2_pose_refiner.py --config .\configs\train\pointnet2_pose_refiner_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --epochs 2 --batch-size 2 --limit-train-samples 4 --limit-val-samples 4 --device cuda
```

Smoke result:

```text
experiments/pointnet2_pose_refiner_k41144_20260608_215456
epoch 2 val raw ADD 6.964 mm -> refined ADD 6.743 mm on the 4-crop smoke validation slice
```

Training:

- Train on GT crops first.
- Then optionally train with predicted crops filtered by `matched_gt_iou >= 0.5`.
- Keep raw pose model frozen.

Pass criteria:

```text
GT-crop translation improves below Phase 20 result.
ADD_0.1d success does not regress.
Predicted-crop matched performance improves or stays flat.
```

Risk:

- A learned refiner can overfit small bending_pipe data.
- Require tiny-overfit, train/val/test, and predicted-crop tests before adoption.

## Phase 23: Evaluation Suite And Release Gate

Status: implemented.

Purpose:

Make the pose pipeline reproducible with a single command so future accuracy changes are safe.

Deliverables:

```text
scripts/run_pose_evaluation_suite.py
experiments/pose_eval_suite_<timestamp>/summary.json
experiments/pose_eval_suite_<timestamp>/summary.md
```

The suite should run:

```text
1. py_compile for pose scripts/modules
2. identity baseline on K41144 val
3. identity baseline on bending_pipe val
4. raw GT-crop test K41144
5. raw GT-crop test bending_pipe
6. refined GT-crop test K41144 if --refine-pose
7. refined GT-crop test bending_pipe if --refine-pose
8. predicted-crop IoU>=0.5 K41144
9. predicted-crop all K41144
10. predicted-crop IoU>=0.5 bending_pipe
11. predicted-crop all bending_pipe
```

Summary table columns:

```text
object
dataset
sample_count
matched_crop_iou
raw_add_mm
raw_translation_mm
raw_add_0.1d
refined_add_mm
refined_translation_mm
refined_add_0.1d
delta_add_mm
delta_translation_mm
```

Pass criteria:

```text
Evaluation suite runs without manual command editing.
Suite summary includes all metrics required for go/no-go.
All result JSONs are written under one timestamped folder.
```

Implemented command:

```powershell
python .\scripts\run_pose_evaluation_suite.py --device cuda --refine-pose
```

Smoke command:

```powershell
python .\scripts\run_pose_evaluation_suite.py --out-dir .\experiments\phase23_smoke_pose_eval_suite --device cuda --limit-samples 2 --refine-pose
```

Smoke output:

```text
experiments/phase23_smoke_pose_eval_suite/summary.json
experiments/phase23_smoke_pose_eval_suite/summary.md
```

## Recommended Implementation Order

Implementation status:

```text
Phase 18: Raw pose error audit                         Done
Phase 19: Center-vote aggregation ablation             Done
Phase 20: Conservative refinement                      Done
Phase 21: K41144 predicted-crop quality retune tools   Done
Phase 22: Learned translation refiner                  Done, experimental
Phase 23: Evaluation suite                             Done
```

Reason:

- Audit first prevents blind optimization.
- Robust aggregation is low-risk and may solve part of translation gap without retraining.
- Conservative refinement directly targets the current residual.
- K41144 predicted-crop retune addresses the largest full-pipeline gap.
- Evaluation suite is now the release gate before adopting future accuracy changes.
- Learned refinement is powerful but higher risk, especially on small bending_pipe data, so keep it optional until full-data training proves it.

## Engineer Task Breakdown

### Engineer A: Metrics And Audit

Owns:

```text
scripts/analyze_pose_errors.py
per-crop JSON
worst-case PLY export
failure buckets
```

Done when:

```text
K41144 and bending_pipe GT test audits exist.
Predicted-crop audits exist for IoU>=0.5 and all predicted.
Worst 20 cases can be visually inspected.
```

### Engineer B: Translation Aggregation

Owns:

```text
center-vote aggregation implementations
eval flags
ablation result table
```

Done when:

```text
mean/median/trimmed/weighted/geometric modes all evaluate.
Best mode is chosen from validation/test consistency.
Default remains mean unless another mode wins clearly.
```

### Engineer C: Refinement

Owns:

```text
src/inference/pose_refinement.py
--refine-pose evaluator integration
raw-vs-refined metrics
accept/reject gates
```

Done when:

```text
translation_only_icp runs on both objects.
Refinement cannot silently replace raw metrics.
Refinement either improves metrics or is documented as disabled-by-default.
```

### Engineer D: Predicted-Crop Sweep

Owns:

```text
K41144 clustering/export parameter sweep
full-pipeline predicted-crop summary
recommended inference config update if found
```

Done when:

```text
Sweep identifies whether K41144 all-predicted ADD_0.1d can exceed 0.60.
If not, failure modes are documented.
```

### Engineer E: Evaluation Suite And Docs

Owns:

```text
scripts/run_pose_evaluation_suite.py
docs/dataset-build-and-test-guide.md updates
docs/pointnet2-pose-estimation-plan.md updates
```

Done when:

```text
One command produces current raw/refined summary.
Docs list latest accepted commands and metrics.
```

## Testing Matrix

Every implementation phase must run:

```powershell
python -m py_compile src\data\pose_crop_dataset.py src\models\pointnet2_pose.py src\models\pointnet2_pose_voting.py src\training\pose_losses.py src\training\pose_metrics.py scripts\train_pointnet2_pose.py scripts\eval_pointnet2_pose.py scripts\export_pose_instance_crops.py
```

After geometry/refinement changes:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --out .\experiments\smoke_pose_identity_k41144_val.json --identity-baseline --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --out .\experiments\smoke_pose_identity_bending_val.json --identity-baseline --device cuda
```

Core raw pose tests:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\current_k41144_gt_test.json --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\current_bending_gt_test.json --device cuda
```

Predicted-crop tests:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --out .\experiments\current_k41144_pred_iou05.json --min-matched-gt-iou 0.5 --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --out .\experiments\current_k41144_pred_all.json --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --out .\experiments\current_bending_pred_iou05.json --min-matched-gt-iou 0.5 --device cuda

python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --out .\experiments\current_bending_pred_all.json --device cuda
```

## Documentation Updates Required After Implementation

Update these files after each accepted phase:

```text
docs/pointnet2-pose-estimation-plan.md
docs/dataset-build-and-test-guide.md
docs/codex-session-bootstrap.md
```

Record:

```text
changed files
new commands
new metrics
accepted/rejected refinements
remaining risks
next phase
```

## Decision Rules

Adopt an accuracy improvement only if:

```text
1. It improves at least one target metric.
2. It does not regress ADD_0.1d beyond 0.02 absolute on either object.
3. It is evaluated on both K41144 and bending_pipe.
4. It is reported separately if it is a refinement stage.
5. It can be reproduced from documented commands.
```

Reject or keep experimental if:

```text
1. It improves self-alignment but worsens GT ADD.
2. It helps GT crops but hurts predicted crops substantially.
3. It depends on GT labels at inference time.
4. It hides bad instance crops instead of reporting them.
```

## Recommended Next Task For Codex

Continue after Phase 23 with validation and data scaling:

```text
1. Run the full Phase 21 K41144 predicted-crop sweep without --limit-samples.
2. Train pose/refiner models on the next larger K41144 and bending_pipe datasets.
3. Run scripts/run_pose_evaluation_suite.py without --limit-samples.
4. Adopt learned translation refinement only if the suite proves it beats Phase 20 hybrid refinement on both objects.
```

The implementation plan is complete; remaining work is experimental validation on larger data.
