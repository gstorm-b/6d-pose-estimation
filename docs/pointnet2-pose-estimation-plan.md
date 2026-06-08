# PointNet++ Pose Estimation Plan

Date: 2026-06-07

Status: Phase 11-23 implemented. The raw PS6D-style keypoint/center voting model remains the primary learned pose output; optional hybrid ICP and learned translation-refiner paths are available and reported separately. The remaining work is full-data validation, not additional plan scaffolding.

Detailed next implementation plan:

```text
docs/pose-estimation-completion-and-accuracy-plan.md
```

## Goal

Implement the first 6D pose-estimation stage on top of the completed instance segmentation pipeline.

The pose stage should consume the same crop interface for:

```text
GT instance crops
predicted instance crops
```

This separation is important because later evaluation must report:

```text
pose quality with GT instances
pose quality with predicted instances
full pipeline quality
```

Do not couple pose training directly to raw Blender folders. Raw synthetic data remains immutable. Pose training should read processed crop data exported by the pose bridge.

## Current Inputs

Completed bridge artifacts:

```text
src/inference/pose_bridge.py
scripts/export_pose_instance_crops.py
docs/k41144-pose-symmetry-audit.md
```

Current per-sample crop NPZ schema:

```text
crop_points_camera           (sum_crop_points, 3) float32
crop_point_indices           (sum_crop_points,) int64
crop_offsets                 (num_crops, 2) int64
crop_instance_ids            (num_crops,) int64
crop_point_counts            (num_crops,) int64
crop_centroids_camera        (num_crops, 3) float32
crop_bbox_min_camera         (num_crops, 3) float32
crop_bbox_max_camera         (num_crops, 3) float32
crop_matched_gt_instance_ids (num_crops,) int64
crop_matched_gt_ious         (num_crops,) float32
crop_object_to_camera        (num_crops, 4, 4) float32
crop_features                (sum_crop_points, 3) float32, optional normal_camera features
sample_name                  string
source                       string: gt or predicted
```

Existing exported test crop sets:

```text
experiments/pose_bridge_k41144_gt_test
experiments/pose_bridge_k41144_pred_test
experiments/pose_bridge_bending_pipe_gt_test
experiments/pose_bridge_bending_pipe_pred_test
```

## Coordinate Frames

Keep frame directions explicit in every tensor and JSON key.

The GT pose target is:

```text
object_to_camera
```

It maps points from the original STL object frame into the generator metadata camera frame.

Important frame detail:

- Bridge crop point clouds store the project depth point convention where visible points have positive `z`.
- Blender/generator metadata uses camera-local coordinates where visible points have negative `z`.
- `PoseCropDataset` converts crop points and centroids into metadata camera frame by flipping `z` before pose targets, losses, and metrics.

For a crop:

```text
crop_centroid_camera = mean(crop_points_camera)
object_origin_camera = object_to_camera[:3, 3]
object_origin_offset_from_crop_centroid_camera = object_origin_camera - crop_centroid_camera
```

The pose model should predict:

```text
rotation_object_to_camera
object_origin_offset_from_crop_centroid_camera
```

Then recover:

```text
object_origin_camera_pred = crop_centroid_camera + predicted_offset
object_to_camera_pred = [rotation_object_to_camera_pred, object_origin_camera_pred]
```

Do not train translation as visible centroid. Visible centroid is a segmentation/crop property, not the original STL object origin.

## Normalization Strategy

For the MVP pose model, use crop-centered camera coordinates:

```text
points_crop_centered_camera = crop_points_camera - crop_centroid_camera
```

Optional scale normalization:

```text
points_pose_input = points_crop_centered_camera / model_diameter_m
target_offset_normalized = object_origin_offset_from_crop_centroid_camera / model_diameter_m
```

The MVP pose model also receives crop aux features:

```text
crop_aux_features =
  [crop_centroid_camera / model_diameter_m,
   (crop_bbox_min_camera - crop_centroid_camera) / model_diameter_m,
   (crop_bbox_max_camera - crop_centroid_camera) / model_diameter_m,
   second central moment xyz,
   covariance cross terms xy/xz/yz,
   third central moment xyz]
```

The absolute crop centroid is available at inference time and was required for stable bending_pipe tiny-overfit because crop-centered local shape alone can make different bent-pipe poses look too similar. Phase 14 adds crop normals and moment features for a stronger full-split signal.

Recommendation:

- Use `model_diameter_m` normalization in the pose dataset.
- Store the scale in sample metadata.
- Convert predictions back to meters before metrics.

This keeps K41144 and bending_pipe numerically comparable while preserving meter outputs.

## Symmetry

K41144 symmetry audit:

```text
S_k41144 = {I, R_y(pi)}
```

Use finite symmetry only in loss and metrics. Keep raw predicted `object_to_camera` in outputs.

Bending pipe:

```text
S_bending_pipe = {I}
```

Do not use continuous axial symmetry for K41144. The mesh is close enough to fool some aggregate distances, but 90-degree rotations around local `Y` have local errors that are too large for exact equivalence.

## Pose Representation

Recommended MVP output:

```text
rotation_6d_object_to_camera  (6,)
origin_offset_from_crop_centroid_camera_normalized (3,)
```

Why 6D rotation for the MVP:

- Avoids quaternion sign ambiguity during regression.
- Converts directly to a valid rotation matrix with Gram-Schmidt.
- Symmetry-aware loss can operate on rotation matrices.

Keep quaternion conversion utilities optional for later PS6D-style voting or robot interface needs.

## Loss Design

Initial total loss:

```text
total_loss =
  translation_weight * smooth_l1(origin_offset_pred, origin_offset_gt)
  + model_points_weight * symmetry_aware_model_point_loss
  + rotation_weight * symmetry_aware_rotation_loss
```

Recommended MVP:

- Use translation Smooth L1 in normalized units.
- Use symmetry-aware model-point ADD loss with GT translation during training so rotation and translation signals do not fight on tiny crops.
- Use symmetry-aware rotation loss as Frobenius matrix loss plus 6D representation loss. The 6D term made bending_pipe rotation overfit stable.

Symmetry-aware model-point loss:

```text
loss_pose = min_{S in symmetry_set}
  mean || R_pred * model_points + t_pred - R_gt * S * model_points - t_gt ||
```

For K41144:

```text
symmetry_set = [I, R_y(pi)]
```

For bending_pipe:

```text
symmetry_set = [I]
```

Sample model points from the original STL model frame. Use meters.

## Metrics

Track these on validation and test:

```text
translation_error_m
translation_error_mm
rotation_error_deg_symmetry_aware
add_m
add_mm
add_auc
pose_success_add_0.1d
pose_success_add_0.05d
matched_crop_iou
crop_source: gt or predicted
```

`0.1d` means 10 percent of object diameter.

Report metrics separately for:

```text
GT crops
predicted crops with matched_gt_iou >= 0.5
all predicted crops
```

This prevents poor instance segmentation crops from hiding whether the pose model itself is working.

## Data Modules

Create:

```text
src/data/pose_crop_dataset.py
```

Responsibilities:

- Load a pose bridge `manifest.json`.
- Load per-sample crop NPZ files.
- Expand variable-length crops into per-crop samples.
- Filter crops with missing `object_to_camera`.
- Optionally filter predicted crops by `matched_gt_iou`.
- Sample fixed `num_points` per crop with replacement when needed.
- Return crop-centered points and target pose.

Dataset item draft:

```python
{
    "points_crop_centered_camera": Tensor[num_points, 3],
    "points_pose_input": Tensor[num_points, C],
    "crop_centroid_camera": Tensor[3],
    "object_to_camera": Tensor[4, 4],
    "rotation_object_to_camera": Tensor[3, 3],
    "object_origin_offset_from_crop_centroid_camera": Tensor[3],
    "object_origin_offset_from_crop_centroid_camera_normalized": Tensor[3],
    "model_diameter_m": Tensor[],
    "symmetry_class": str,
    "sample_name": str,
    "crop_index": int,
    "matched_gt_iou": float,
}
```

Implementation note:

- Crop NPZ files now store optional `crop_features` copied from processed PointNet++ normal features.
- `PoseCropDataset` remains backward-compatible with old xyz-only crop exports.
- Normal vectors are converted into the same metadata camera frame as crop points by flipping `z`.

## Model Modules

Create:

```text
src/models/pointnet2_pose.py
```

MVP architecture:

```text
crop points -> PointNet++ encoder -> global feature
  -> translation residual head
  -> rotation 6D head
```

This is intentionally simpler than scene-level PointNet++:

- Pose input is one crop, not the whole bin.
- The model needs a robust global shape feature.
- Instance segmentation remains responsible for object separation.

Initial config:

```yaml
model:
  name: pointnet2_pose
  input_features: xyz_normal
  num_points: 1024
  ops_backend: pytorch3d
  sa_npoints: [256, 64, 16]
  sa_nsamples: [32, 32, 32]
  global_feature_dim: 256
  aux_channels: 18
  dropout: 0.2
```

## Training Scripts

Create:

```text
configs/train/pointnet2_pose_k41144.yaml
configs/train/pointnet2_pose_bending_pipe.yaml
scripts/train_pointnet2_pose.py
scripts/eval_pointnet2_pose.py
```

Training should start on GT crops only:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_k41144.yaml --data .\experiments\pose_bridge_k41144_gt_test --epochs 100 --batch-size 16 --device cuda
```

After GT-crop overfit passes, train on train/val pose crop exports, then evaluate on:

```text
GT test crops
predicted test crops with matched_gt_iou >= 0.5
all predicted test crops
```

## Proposed Phases

### Phase 11: Pose Crop Dataset And Model-Point Utilities

Status: Done.

Deliverables:

```text
src/data/pose_crop_dataset.py
src/training/pose_geometry.py
configs/train/pointnet2_pose_k41144.yaml
configs/train/pointnet2_pose_bending_pipe.yaml
```

Pass criteria:

```text
Pose crop dataset loads GT and predicted crop manifests.
Dataset returns frame-explicit tensors.
Model diameter and sampled model points are in meters.
K41144 symmetry set is {I, R_y(pi)}.
Bending pipe symmetry set is {I}.
Unit tests or smoke scripts verify rotation conversions and ADD under symmetry.
```

Implementation notes:

- `src/data/pose_crop_dataset.py` loads exported crop roots and expands variable-length crops into fixed-size pose samples.
- `src/training/pose_geometry.py` parses STL vertices, samples model points, computes bbox diameter, converts 6D rotations, and applies finite symmetry sets.
- For millimeter-authored STL assets such as `bending_pipe.stl`, dataset config uses `model_scale: 0.001`; the metadata `object_to_camera` rotation block is normalized back to SO(3).
- Pose crop tensors are converted from positive-forward depth camera coordinates to metadata/Blender camera coordinates before target construction.

### Phase 12: Pose Metrics And Evaluation Skeleton

Status: Done.

Deliverables:

```text
src/training/pose_metrics.py
scripts/eval_pointnet2_pose.py
```

Pass criteria:

```text
Identity prediction against GT returns near-zero errors.
K41144 GT @ R_y(pi) is accepted as equivalent under symmetry-aware ADD.
K41144 GT @ R_y(pi/2) is not accepted as equivalent.
Evaluation JSON separates GT crops, predicted crops with matched_gt_iou >= 0.5, and all predicted crops.
```

Implementation notes:

- `src/training/pose_metrics.py` reports translation, rotation, ADD, ADD success at `0.05d` and `0.1d`, matched crop IoU, and ADD AUC helper.
- `scripts/eval_pointnet2_pose.py` evaluates either a checkpoint or `--identity-baseline`.
- Separation is done by running the evaluator on a GT crop root, a predicted crop root with `--min-matched-gt-iou 0.5`, and the predicted crop root without filtering.
- Identity baseline smoke checks on four GT crops:
  - K41144: ADD `0.0000012 mm`, translation `0.00000047 mm`, ADD_0.1d success `1.0`.
  - bending_pipe: ADD `0.0000035 mm`, translation `0.00000047 mm`, ADD_0.1d success `1.0`.
- K41144 symmetry smoke confirmed `R_y(pi)` is accepted and `R_y(pi/2)` is not treated as an exact symmetry.

### Phase 13: Pose Model MVP

Status: Done.

Deliverables:

```text
src/models/pointnet2_pose.py
src/training/pose_losses.py
scripts/train_pointnet2_pose.py
```

Acceptance target on tiny GT-crop overfit:

```text
translation_error_mm < 2.0
ADD_0.1d_success >= 0.95
loss decreases smoothly
```

Implementation notes:

- `src/models/pointnet2_pose.py` reuses the PointNet++ backbone, global max/mean pools crop features, concatenates 18 aux channels, then predicts translation offset and 6D rotation.
- `src/training/pose_losses.py` combines normalized translation Smooth L1, symmetry-aware rotation loss, and symmetry-aware model-point loss.
- `scripts/train_pointnet2_pose.py` supports CLI overrides for learning rate, dropout, translation/rotation/model-point weights, sample limits, and frozen BatchNorm.

Tiny GT-crop overfit results:

```text
K41144, 2 crops:
  experiment: experiments/pointnet2_pose_k41144_20260607_222708
  best ADD: 6.627 mm at epoch 120
  best translation: 0.045 mm at epoch 117
  ADD_0.1d success: 1.0

bending_pipe, 2 crops:
  experiment: experiments/pointnet2_pose_bending_pipe_20260607_222609
  best ADD: 0.218 mm at epoch 144
  best translation: 0.014 mm at epoch 146
  ADD_0.1d success: 1.0
```

Key design finding:

- bending_pipe failed to overfit two crops when aux only contained relative bbox extents. Adding absolute crop centroid made the two-crop pose problem identifiable and stable.

### Phase 14: Full GT-Crop Pose Training

Status: Implemented, target not met.

Train on GT crops from train/val splits.

Initial target:

```text
validation translation_error_mm < 5.0
validation ADD_0.1d_success >= 0.80
```

New Phase 14 GT crop datasets:

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

Raw validation before crop export:

```text
synthetic-data/K41144:       77/77 OK
synthetic-data/bending_pipe: 30/30 OK
```

Training/eval runs:

```text
K41144:
  experiment: experiments/pointnet2_pose_k41144_20260607_230547
  val:  ADD 24.456 mm, translation 27.884 mm, ADD_0.1d success 0.189
  test: ADD 27.719 mm, translation 30.225 mm, ADD_0.1d success 0.123

bending_pipe:
  experiment: experiments/pointnet2_pose_bending_pipe_20260607_230546
  val:  ADD 76.739 mm, translation 66.322 mm, ADD_0.1d success 0.000
  test: ADD 80.936 mm, translation 71.057 mm, ADD_0.1d success 0.000
```

Diagnostic findings:

- Identity baseline on the new val crop datasets is near zero for both objects, so frame conversion, scale handling, and pose labels are correct.
- K41144 improves with xyz+normal input and 18 aux features, but still misses the Phase 14 target by a wide margin.
- bending_pipe does not generalize from 24 training scenes to 3 validation scenes with this global crop-regression MVP.
- The current MVP can memorize tiny crops, but full split results show a real architecture/objective limitation rather than a dataset schema bug.

Recommended next phase:

- Add a PS6D-style per-point pose voting head or dense keypoint/center voting before direct global pose regression.
- Keep direct crop-level PointNet++ pose as a baseline, not the primary full-pipeline pose model.

### Phase 15: PS6D-Style Keypoint/Center Voting

Status: Done.

Phase 14 showed that direct global crop regression was too weak. Phase 15 replaced it as the primary pose path with a PS6D-style voting model.

Implemented files:

```text
src/models/pointnet2_pose_voting.py
configs/train/pointnet2_pose_voting_k41144.yaml
configs/train/pointnet2_pose_voting_bending_pipe.yaml
```

Dataset additions in `src/data/pose_crop_dataset.py`:

```text
sampled_points_object_normalized
sampled_origin_offsets_camera_normalized
keypoints_object
keypoints_camera
sampled_keypoint_offsets_camera_normalized
```

Model outputs:

```text
predicted_points_object_normalized
predicted_origin_offsets_camera_normalized
predicted_keypoint_offsets_camera_normalized
rotation_6d
```

Pose reconstruction:

- Rotation is reconstructed from voted mesh keypoints with Kabsch.
- Translation uses the mean object-center vote because keypoint-derived origin was less stable on partial crops.
- Direct `rotation_6d` remains available as an auxiliary baseline, but keypoint voting is the primary rotation path for metrics.

Loss:

```text
total_loss =
  object_coord_weight * object_coord_loss
  + center_vote_weight * center_vote_loss
  + keypoint_vote_weight * keypoint_vote_loss
  + rotation_weight * symmetry_aware_rotation_loss
  + model_points_weight * symmetry_aware_model_point_loss
```

Training script additions:

- `scripts/train_pointnet2_pose.py` writes `checkpoints/best_add.pt` in addition to loss-selected `best.pt`.
- `--resume` supports continuing from an existing checkpoint.
- `--resume-new-experiment` loads a checkpoint but writes metrics/checkpoints to a new experiment directory.

### Phase 16: Full GT-Crop Keypoint-Voting Evaluation

Status: Done.

K41144:

```text
experiment: experiments/pointnet2_pose_k41144_20260608_023605
checkpoint: checkpoints/best_add.pt
val:  ADD 5.637 mm, translation 5.558 mm, ADD_0.1d success 0.892
test: ADD 5.607 mm, translation 6.021 mm, ADD_0.1d success 0.897
```

bending_pipe:

```text
experiment: experiments/pointnet2_pose_bending_pipe_20260608_021309
checkpoint: checkpoints/best_add.pt
val:  ADD 14.481 mm, translation 13.616 mm, ADD_0.1d success 0.722
test: ADD 9.899 mm, translation 9.529 mm, ADD_0.1d success 0.833
```

Result versus Phase 14:

- K41144 improved from test ADD `27.719 mm` / ADD_0.1d `0.123` to ADD `5.607 mm` / ADD_0.1d `0.897`.
- bending_pipe improved from test ADD `80.936 mm` / ADD_0.1d `0.000` to ADD `9.899 mm` / ADD_0.1d `0.833`.
- ADD success target is met on K41144 and bending_pipe test GT crops.
- The strict translation target `< 5 mm` is still not fully met, especially for bending_pipe.

### Phase 17: Predicted-Crop Pose Evaluation

Status: Done.

Evaluate the same pose checkpoint on predicted instance crops.

Report:

```text
pose metrics for matched_gt_iou >= 0.5
pose metrics for all predicted crops
crop recall inherited from instance segmentation
full-pipeline pose success
```

Exported predicted test crops:

```text
K41144:
  crops: experiments/phase17_pose_crops_k41144_pred_test_xyznormal
  total_crops: 150
  crops_with_object_to_camera: 150

bending_pipe:
  crops: experiments/phase17_pose_crops_bending_pipe_pred_test_xyznormal
  total_crops: 14
  crops_with_object_to_camera: 14
```

Predicted-crop metrics:

```text
K41144, matched_gt_iou >= 0.5:
  ADD 10.719 mm, translation 11.087 mm, ADD_0.1d success 0.738, mean IoU 0.805

K41144, all predicted crops:
  ADD 21.932 mm, translation 22.831 mm, ADD_0.1d success 0.540, mean IoU 0.661

bending_pipe, matched_gt_iou >= 0.5:
  ADD 12.146 mm, translation 13.470 mm, ADD_0.1d success 0.923, mean IoU 0.958

bending_pipe, all predicted crops:
  ADD 18.456 mm, translation 18.612 mm, ADD_0.1d success 0.714, mean IoU 0.909
```

Interpretation:

- Predicted K41144 pose quality is now limited significantly by instance crop quality.
- bending_pipe predicted crops are cleaner, and the matched-crop pose result exceeds the ADD_0.1d target.

### Phase 18-20: Audit, Center-Vote Ablation, And Optional Refinement

Status: Implemented; refinement remains optional and is reported separately from raw pose.

The learned pose is stable enough for a refinement stage, but raw translation error remains above the strict `< 5 mm` target on both objects. Phase 18-20 added:

```text
scripts/analyze_pose_errors.py
center-vote aggregation flags in scripts/eval_pointnet2_pose.py
src/inference/pose_refinement.py
configs/eval/pose_refinement_defaults.json
raw-vs-refined evaluator JSON summaries
```

The refinement path is:

```text
predicted object_to_camera -> ICP init
model points -> scene crop points
refined object_to_camera
```

Verified Phase 20 recommended preset:

```text
method: hybrid
distance_threshold_fraction: 0.16
max_translation_delta_fraction: 0.20
max_rotation_delta_deg: 20
accept_if_improved: true
```

GT-crop test result with the preset:

```text
K41144:
  raw     ADD 5.607 mm, translation 6.021 mm, ADD_0.1d 0.897
  refined ADD 4.896 mm, translation 5.311 mm, ADD_0.1d 0.925

bending_pipe:
  raw     ADD 9.899 mm, translation 9.529 mm, ADD_0.1d 0.833
  refined ADD 5.870 mm, translation 5.282 mm, ADD_0.1d 0.944
```

Operational constraints:

- Keep raw keypoint-voting pose as the primary learned output.
- Add ICP as an optional evaluator/inference flag, not as a silent replacement.
- Report GT-crop raw, GT-crop refined, predicted-crop raw, and predicted-crop refined separately.
- Use conservative distance thresholds because full-mesh-to-visible-crop ICP can pull the object origin toward visible partial surfaces.
- Keep center-vote default as `mean`; robust aggregation did not improve both objects consistently.

## Risks

### Crop Quality Can Dominate Pose Metrics

Predicted K41144 crops have lower matched IoU than bending_pipe crops.

Mitigation:

- Always report GT-crop pose and predicted-crop pose separately.
- Filter predicted crops by `matched_gt_iou >= 0.5` for pose-model diagnosis.
- Report all-predicted-crop metrics for full pipeline quality.

### Original STL Origin Is Not Visible Centroid

K41144 STL origin is not centered.

Mitigation:

- Predict `object_origin_offset_from_crop_centroid_camera`.
- Never use visible centroid as the final object translation target.

### Symmetry Can Hide Real Grasp Differences

K41144 has a strong 180-degree local-Y symmetry candidate, but not continuous axial symmetry.

Mitigation:

- Use only `{I, R_y(pi)}` for K41144 in loss/metrics.
- Keep raw `object_to_camera` in predictions and logs.

### Crop-Level Global Regression Is Too Weak

Phase 14 shows that even xyz+normal crop-level global regression does not meet held-out pose targets.

Mitigation:

- Move to per-point voting/keypoint supervision inspired by PS6D.
- Use direct crop pose regression as a baseline and ablation.
- Keep normals and moment features in the crop schema for future ablations.

## Immediate Next Task

Follow the completion and accuracy plan:

```text
docs/pose-estimation-completion-and-accuracy-plan.md
```

Recommended next validation sequence:

```text
1. Run the full K41144 predicted-crop sweep.
2. Train/evaluate on larger K41144 and bending_pipe datasets.
3. Run the full pose evaluation suite without --limit-samples.
4. Promote learned translation refinement only if it beats Phase 20 hybrid refinement across both objects.
```
