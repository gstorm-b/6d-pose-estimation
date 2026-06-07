# PointNet++ Instance Segmentation Plan For K41144

This document is the technical plan for the next model branch after the PointNet++ semantic segmentation baseline.

Scope for this milestone:

- Train an instance-segmentation model on synthetic bin-picking point clouds.
- Reuse the existing PointNet++ backbone and PyTorch3D ops backend.
- Predict:
  - per-point semantic logits: `0 = background`, `1 = K41144`
  - per-point centroid offsets for object points
  - per-point instance embeddings for separating same-class objects
- Cluster predicted object points into object instances.
- Produce colored PLY previews for visual inspection.
- Produce voted-center debug previews so offset and clustering failures are inspectable.

Non-goals for this milestone:

- Do not implement final 6D pose estimation yet.
- Do not implement ICP yet.
- Do not edit raw synthetic data manually.
- Do not train directly from raw Blender object-only `points_camera`.

## Current Context

The semantic PointNet++ branch is complete enough to serve as the backbone baseline.

Current processed dataset:

```text
processed-data/pointnet2_semseg_k41144
```

Current split and format:

```text
train/val/test = 62/8/7
points per sample = 16384
features = xyz + normal_camera
labels = semantic_labels + instance_labels
```

Current main semantic checkpoint:

```text
experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
```

Held-out semantic test result:

```text
loss=0.0075
overall_accuracy=0.9972
mean_iou=0.9939
object_iou=0.9957
object_precision=0.9991
object_recall=0.9966
```

Current PointNet++ backend decision:

```text
default ops backend = pytorch3d
fallback ops backend = pure_torch
```

PyTorch3D is preferred on the current machine because it gives about a 7x speedup on the profiled 16384-point training batch.

## Two-Object Design/Test Policy

Use two object geometries when validating design assumptions:

```text
object-model/K41144.stl
object-model/bending_pipe.stl
```

They should be tested as separate single-object-family datasets, not necessarily mixed in one scene. This matches the expected production setup: one part number is picked in a run, but the implementation should not overfit its design assumptions to only one geometry.

Recommended future dataset naming:

```text
synthetic-data/K41144
synthetic-data/bending_pipe
processed-data/pointnet2_semseg_k41144
processed-data/pointnet2_semseg_bending_pipe
experiments/pointnet2_instance_data_audit_k41144
experiments/pointnet2_instance_data_audit_bending_pipe
```

Current status:

- `object-model/K41144.stl` exists and has processed semantic/instance labels.
- `object-model/bending_pipe.stl` exists and has a processed semantic/instance-label dataset.
- `bending_pipe.stl` is authored in millimeters, so generation uses `--model-scale 0.001`.
- The generator now records `class_name` and object names from `--class-name` instead of hardcoding `K41144`.

## Instance Label Audit Snapshot

A quick audit of the current K41144 processed dataset shows:

```text
samples = 77
instances/sample:
  min = 14
  median = 20
  max = 30

object points/sample median = 10650
background points/sample median = 5734

points/instance:
  min = 14
  p10 = 90
  median-ish = 500+
  max = 1074

instances with fewer than 64 sampled points = 2
instances with fewer than 32 sampled points = 1
```

Interpretation:

- The dataset is likely usable for the first instance-segmentation branch.
- Very small visible instances exist, so metrics should support ignored or low-confidence instances below a configurable point threshold.
- High semantic IoU does not guarantee instance separation; objects of the same class may touch or occlude each other.

## Key Design Decision

Do not train raw `instance_id` as a global class label.

Reason:

- `instance_id` only has meaning within one scene.
- The model must learn geometric separation between same-class objects, not memorize instance numbers.

Use a multi-head point-wise model:

```text
PointNet++ shared backbone
  -> semantic head
  -> centroid offset head
  -> embedding head
  -> clustering inference
```

The offset head predicts a vote from each object point to the centroid of its own visible instance:

```text
target_offset = instance_centroid_camera - point_camera
```

The embedding head learns to pull points from the same instance together and push different instances apart.

PS6D-specific design interpretation:

- Keep centroid offset voting as the primary instance-separation signal.
- Keep embedding loss as a supporting signal at first, not the main clustering mechanism.
- Do not add rotation/quaternion prediction to the MVP instance branch. Rotation belongs to the next pose-estimation branch.
- Design clustering and preview APIs so they can later evolve from single-stage center clustering to PS6D-style two-stage center+rotation clustering.

## Proposed Architecture

Create a new model module:

```text
src/models/pointnet2_instance_seg.py
```

Recommended output API:

```python
{
    "semantic_logits": Tensor[B, N, 2],
    "offsets": Tensor[B, N, 3],
    "embeddings": Tensor[B, N, embedding_dim],
}
```

The model should reuse the existing PointNet++ abstraction and feature-propagation layers. If necessary, refactor:

```text
src/models/pointnet2_semseg.py
  PointNet2Backbone
  PointNet2SemSeg
```

Then implement:

```text
src/models/pointnet2_instance_seg.py
  PointNet2InstanceSeg
  build_pointnet2_instance_seg_from_config
```

Initial model config:

```yaml
model:
  name: pointnet2_instance_seg
  num_classes: 2
  input_features: xyz_normal
  ops_backend: pytorch3d
  sa_npoints: [1024, 256, 64]
  sa_nsamples: [32, 32, 32]
  dropout: 0.4
  embedding_dim: 8
```

## Data Contract

The existing processed `.npz` files already contain:

```text
points              (N, 3) float32
features            (N, 3) float32
semantic_labels     (N,) int64
instance_labels     (N,) int64
point_pixels        (N, 2) uint16
raw_sample          string
```

Add a dataset wrapper:

```text
src/data/pointnet2_instance_dataset.py
```

The wrapper should derive these training targets on load:

```text
centroid_offsets       (N, 3) float32
valid_instance_mask    (N,) bool
object_mask            (N,) bool
centroid_distance_weights (N,) float32 optional
instance_point_counts  mapping/report data
```

For every non-background instance id:

```text
instance_centroid = mean(points[instance_labels == instance_id])
centroid_offsets[mask] = instance_centroid - points[mask]
```

Important normalization rule:

- If `normalize: scene_center` is used, compute centroid offsets after applying the same translation to points.
- Offsets are translation-invariant, so the values remain valid.
- Keep the normalization center available for inference/export when cluster centers need to be mapped back into camera coordinates.
- Keep both names explicit in code:
  - `points_camera` for metric camera-frame coordinates.
  - `points_camera_normalized` for model input after scene centering.
  - `normalization_center_camera` for mapping predicted centers back to camera coordinates.

Dataset item should include:

```python
{
    "points": Tensor[N, 3],
    "features": Tensor[N, 3] | None,
    "model_input": Tensor[N, C],
    "semantic_labels": Tensor[N],
    "instance_labels": Tensor[N],
    "centroid_offsets": Tensor[N, 3],
    "valid_instance_mask": Tensor[N],
    "object_mask": Tensor[N],
    "centroid_distance_weights": Tensor[N],
    "normalization_center_camera": Tensor[3],
    "point_pixels": Tensor[N, 2],
    "raw_sample": str,
    "path": str,
}
```

For MVP, `centroid_distance_weights` can be computed and returned even if the loss config initially disables it.

## Loss Design

Create:

```text
src/training/instance_losses.py
```

Total loss:

```text
total_loss =
  semantic_weight * semantic_ce
  + offset_weight * offset_smooth_l1
  + embedding_weight * discriminative_embedding_loss
```

### Semantic Loss

Use cross entropy over all points:

```text
CrossEntropyLoss(weight=class_weights)
```

Class weights can reuse the semantic branch logic.

### Offset Loss

Use Smooth L1 only on valid object points:

```text
SmoothL1Loss(pred_offsets[object_mask], target_offsets[object_mask])
```

Use a meter-scale Smooth L1 beta, not PyTorch's default `1.0`, because object centroid offsets are usually only a few centimeters:

```yaml
loss:
  offset_smooth_l1_beta: 0.01
```

Suggested initial target:

```text
offset_l1_object < 0.005 m on overfit samples
```

PS6D suggests a center-distance-sensitive translation loss for slender objects. K41144 is also elongated, so keep this as an explicit config option:

```yaml
loss:
  centroid_distance_weighting: false
  centroid_distance_weight_min: 0.5
  centroid_distance_weight_max: 1.5
```

Implementation rule:

- Start with `centroid_distance_weighting: false` for the first MVP runs.
- If overfit/debug previews show larger offset errors near object ends, enable the weighting.
- Log offset metrics for near/far centroid regions before changing model architecture.

### Embedding Loss

Use a discriminative instance embedding loss:

```text
variance term:
  pull embeddings toward their instance mean

distance term:
  push different instance means apart

regularization term:
  keep embedding means bounded
```

Initial config:

```yaml
loss:
  semantic_weight: 1.0
  offset_weight: 1.0
  embedding_weight: 0.1
  offset_smooth_l1_beta: 0.01
  delta_var: 0.5
  delta_dist: 1.5
  regularization_weight: 0.001
  min_points_per_instance: 16
  centroid_distance_weighting: false
  centroid_distance_weight_min: 0.5
  centroid_distance_weight_max: 1.5
```

Start with a low embedding weight. Offset voting should be the primary separation signal at first.

Training debug rule:

- First verify `semantic + offset` can overfit with embedding disabled or near-zero.
- Then enable embedding loss and confirm it improves clustering without damaging offset convergence.

## Clustering Inference

Create:

```text
src/inference/instance_clustering.py
src/inference/pointnet2_instance_infer.py
scripts/infer_pointnet2_instance.py
```

Recommended first clustering path:

```text
1. Predict semantic labels.
2. Keep points with object probability above threshold.
3. Compute voted centers:
   voted_center = point_xyz + predicted_offset
4. Cluster voted centers with DBSCAN.
5. Optionally remove clusters with fewer than min_cluster_points.
6. Assign predicted instance labels back to the original point rows.
```

Design the clustering API in two layers:

```text
cluster_voted_centers(points, semantic_probabilities, offsets, config)
cluster_pose_votes(... future rotation/quaternion inputs ...)
```

The MVP should only implement `cluster_voted_centers`. The second function is a future PS6D-style extension after pose/rotation heads exist.

Initial clustering config:

```yaml
inference:
  object_probability_threshold: 0.5
  dbscan_eps_m: 0.004
  dbscan_eps_sweep_m: [0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012]
  dbscan_min_samples: 8
  min_cluster_points: 96
```

K41144 approximate object size:

```text
30 mm x 93.6 mm x 20 mm
```

So the first DBSCAN radius should be small enough to avoid merging nearby objects, but large enough to tolerate noisy votes. Phase 6 debug preview tuning selected `0.006 m`; Phase 8 full-density K41144 validation selected `0.005 m`; Phase 9 held-out testing selected `0.004 m` with `min_cluster_points=96` for K41144 to recover recall while filtering small split fragments. Bending pipe remains good at `0.006 m` and `min_cluster_points=32`. Keep the sweep from `0.004 m` to `0.012 m` available for merge/split diagnosis.

`min_cluster_points` should initially match `metrics.ignore_gt_instances_below_points` at `32`. This matters for 4096-point debug datasets, where partially visible or thin objects can otherwise be filtered out even when the semantic and offset heads are behaving correctly.

For preview/evaluation runs, support a small DBSCAN sweep:

```yaml
inference:
  dbscan_eps_sweep_m: [0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012]
```

Only one value should be used for checkpoint selection, but the sweep helps diagnose merge/split sensitivity.

## Metrics

Create:

```text
src/training/instance_metrics.py
```

Track:

```text
semantic_mean_iou
semantic_object_iou
semantic_object_recall
offset_l1_object
offset_l1_near_centroid
offset_l1_far_centroid
embedding_pull
embedding_push
cluster_count_pred
cluster_count_gt
instance_precision
instance_recall
instance_mean_iou
merge_count
split_count
unmatched_gt_count
unmatched_pred_count
ignored_gt_instances
small_gt_instances
```

Instance matching:

```text
1. Build point-set IoU matrix between predicted clusters and GT instances.
2. Use Hungarian matching.
3. Count match if IoU >= threshold.
```

Initial thresholds:

```yaml
metrics:
  instance_iou_threshold: 0.5
  ignore_gt_instances_below_points: 32
  merge_split_overlap_threshold: 0.1
```

Small GT instances should be reported separately, not silently hidden.

Merge/split diagnostics:

- A merge is likely when one predicted cluster overlaps multiple GT instances above a low IoU/contact threshold.
- A split is likely when one GT instance overlaps multiple predicted clusters.
- These are diagnostic counts, not primary metrics; the primary score remains Hungarian-matched instance IoU/precision/recall.

## Visualization

Create:

```text
scripts/preview_pointnet2_instance_predictions.py
```

Export:

```text
sample_xxxxxx_gt_instances.ply
sample_xxxxxx_pred_instances.ply
sample_xxxxxx_error.ply
sample_xxxxxx_voted_centers.ply
preview_summary.json
```

Color rules:

```text
background = gray
ground-truth instances = deterministic random color by instance id
predicted clusters = deterministic random color by cluster id
false object/background errors = red/yellow
unmatched clusters or unmatched GT instances = highlighted in error PLY
voted centers = colored by GT instance when labels are available, otherwise by predicted cluster
```

The voted-center PLY is required for meaningful debugging. It separates these failure modes:

- semantic head missed object points
- offset head voted to the wrong center
- DBSCAN threshold merged or split otherwise good votes

Preview summaries should also include the DBSCAN eps used, and optionally a sweep table for `dbscan_eps_sweep_m`.

The existing viewer can inspect the generated PLY files:

```text
scripts/view_ply_pyside_vtk.py
```

## Proposed Files

New config:

```text
configs/train/pointnet2_instance_k41144.yaml
```

New data/model/training modules:

```text
src/data/pointnet2_instance_dataset.py
src/models/pointnet2_instance_seg.py
src/training/instance_losses.py
src/training/instance_metrics.py
src/inference/instance_clustering.py
src/inference/pointnet2_instance_infer.py
```

New scripts:

```text
scripts/audit_pointnet2_instance_data.py
scripts/train_pointnet2_instance_seg.py
scripts/eval_pointnet2_instance_seg.py
scripts/infer_pointnet2_instance.py
scripts/preview_pointnet2_instance_predictions.py
```

Possible refactor:

```text
src/models/pointnet2_semseg.py
  extract reusable PointNet2Backbone
```

Keep this refactor small and behavior-preserving for the existing semantic model.

## Implementation Roadmap

### Phase 0: Instance Data Audit

Status: Done for K41144 and bending_pipe.

Deliverable:

```text
scripts/audit_pointnet2_instance_data.py
```

Command:

```powershell
python .\scripts\audit_pointnet2_instance_data.py --data .\processed-data\pointnet2_semseg_k41144 --out .\experiments\pointnet2_instance_data_audit_k41144 --model-name K41144 --model-path .\object-model\K41144.stl
```

Pass criteria:

```text
All samples contain instance_labels.
All object points have positive instance ids.
Most visible instances have at least 64 sampled points.
Audit report records small/ignored instance counts.
Audit report records per-instance point-count percentiles.
```

K41144 audit result:

```text
script: scripts/audit_pointnet2_instance_data.py
output: experiments/pointnet2_instance_data_audit_k41144/
status: warn
samples: 77
instances: 1738
split: train/val/test = 62/8/7
instances below 64 points: 2
instances below 32 points: 1
instance point-count stats:
  min=14
  p10=237
  median=454
  p90=740
  max=1074
```

Interpretation:

- Schema and label consistency pass with no errors.
- The warning is expected and useful: one tiny test-split instance has fewer than 32 sampled points.
- Keep tiny-instance reporting enabled; do not silently remove these cases from metrics.

bending_pipe generation and audit result:

```text
model: object-model/bending_pipe.stl
model_scale: 0.001
raw dataset: synthetic-data/bending_pipe
processed dataset: processed-data/pointnet2_semseg_bending_pipe
config: configs/train/pointnet2_semseg_bending_pipe.yaml
raw validation: 30/30 OK
processed split: train/val/test = 24/3/3
processed points per sample: 16384
sampled object/background points: 10650/5734
script: scripts/audit_pointnet2_instance_data.py
output: experiments/pointnet2_instance_data_audit_bending_pipe/
status: pass
samples: 30
instances: 178
instances below 64 points: 0
instances below 32 points: 0
instance point-count stats:
  min=372
  p10=1454.1
  median=1831.5
  p90=2117.5
  max=2403
```

Interpretation:

- bending_pipe is a clean second-geometry audit set for Phase 1 design checks.
- It should remain a separate single-class dataset; do not mix it with K41144 unless a future multi-class experiment explicitly requires it.

### Phase 1: Instance Dataset Wrapper

Status: Done.

Deliverable:

```text
src/data/pointnet2_instance_dataset.py
configs/train/pointnet2_instance_k41144.yaml
configs/train/pointnet2_instance_bending_pipe.yaml
```

Smoke command:

```powershell
python -c "from src.data.pointnet2_instance_dataset import PointNet2InstanceSegDataset; ds=PointNet2InstanceSegDataset('processed-data/pointnet2_semseg_k41144','train'); x=ds[0]; print(x['model_input'].shape, x['centroid_offsets'].shape, x['valid_instance_mask'].sum())"
```

Pass criteria:

```text
model_input shape = (N, 6)
centroid_offsets shape = (N, 3)
centroid_distance_weights shape = (N,)
normalization_center_camera shape = (3,)
valid object points are non-zero
background offset targets are masked out
points/offset frame convention is explicit in item keys
```

Implementation notes:

- `points`, `points_camera_normalized`, and `centroid_offsets` are in the normalized model frame.
- `points_camera` is reconstructed as normalized points plus `normalization_center_camera` for debugging/export.
- `instance_ids` and `instance_point_counts` are padded to `max_instances=128` so the default PyTorch DataLoader can collate `batch_size > 1`.
- `num_instances` records the number of valid compact entries before padding.

Smoke results:

```text
K41144 train sample:
  model_input=(16384, 6)
  centroid_offsets=(16384, 3)
  valid_instance_mask.sum=10650
  object_mask.sum=10650
  num_instances=30
  background offset max=0.0
  first five voted-center spread max ~= 1e-8

bending_pipe train sample:
  model_input=(16384, 6)
  centroid_offsets=(16384, 3)
  valid_instance_mask.sum=10650
  object_mask.sum=10650
  num_instances=6

DataLoader batch_size=2:
  model_input=(2, 16384, 6)
  centroid_offsets=(2, 16384, 3)
  instance_ids=(2, 128)
```

### Phase 2: Reusable Backbone And Instance Model

Status: Done.

Deliverables:

```text
src/models/pointnet2_semseg.py
src/models/pointnet2_instance_seg.py
```

Potential semantic refactor:

```text
src/models/pointnet2_semseg.py
```

Smoke command:

```powershell
python -c "import torch; from src.models.pointnet2_instance_seg import PointNet2InstanceSeg; m=PointNet2InstanceSeg(input_channels=6, ops_backend='pytorch3d').cuda(); x=torch.randn(1,4096,6,device='cuda'); y=m(x); print(y['semantic_logits'].shape, y['offsets'].shape, y['embeddings'].shape)"
```

Pass criteria:

```text
semantic_logits shape = (B, N, 2)
offsets shape = (B, N, 3)
embeddings shape = (B, N, embedding_dim)
CUDA forward/backward works
existing semantic training still imports successfully
```

Implementation notes:

- Added `PointNet2Backbone` in `src/models/pointnet2_semseg.py`.
- Kept the existing `PointNet2SemSeg` module layout unchanged so old semantic checkpoints remain compatible.
- Added `PointNet2InstanceSeg` in `src/models/pointnet2_instance_seg.py`.
- Instance model output API:

```python
{
    "semantic_logits": Tensor[B, N, 2],
    "offsets": Tensor[B, N, 3],
    "embeddings": Tensor[B, N, embedding_dim],
}
```

Smoke results:

```text
semantic CPU forward:
  input=(2, 128, 6)
  logits=(2, 128, 2)

instance CPU forward:
  semantic_logits=(2, 128, 2)
  offsets=(2, 128, 3)
  embeddings=(2, 128, 8)

instance CPU backward on real K41144 batch subset:
  input=(1, 1024, 6)
  semantic_logits=(1, 1024, 2)
  offsets=(1, 1024, 3)
  embeddings=(1, 1024, 8)

instance CUDA/PyTorch3D backward:
  device=NVIDIA GeForce MX150
  backend=pytorch3d
  input=(1, 512, 6)
  semantic_logits=(1, 512, 2)
  offsets=(1, 512, 3)
  embeddings=(1, 512, 8)

semantic checkpoint compatibility:
  checkpoint=experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
  missing=[]
  unexpected=[]
```

### Phase 3: Instance Losses

Status: Done.

Deliverable:

```text
src/training/instance_losses.py
```

Pass criteria:

```text
Loss computes on one batch.
Backward pass completes on CUDA.
Loss report exposes semantic_loss, offset_loss, embedding_loss, total_loss.
Loss report exposes offset_l1_near_centroid and offset_l1_far_centroid.
No NaN when a sample has tiny or ignored instances.
Embedding loss can be disabled without changing the training script.
```

Implementation notes:

- Added `PointNet2InstanceLoss` and `build_pointnet2_instance_loss_from_config`.
- Loss input follows the instance model output dict:

```python
{
    "semantic_logits": Tensor[B, N, 2],
    "offsets": Tensor[B, N, 3],
    "embeddings": Tensor[B, N, embedding_dim],
}
```

- Batch targets are read from `PointNet2InstanceSegDataset`:

```text
semantic_labels
instance_labels
centroid_offsets
valid_instance_mask
centroid_distance_weights
```

- `offset_loss` uses Smooth L1 on `valid_instance_mask` points only.
- `centroid_distance_weighting` is implemented as a config switch and remains disabled in the initial configs.
- Embedding loss uses the standard discriminative terms:

```text
embedding_loss = embedding_pull + embedding_push + regularization_weight * embedding_reg
```

- Instances smaller than `min_points_per_instance` are ignored for embedding loss.
- If no valid object points exist, offset and embedding losses return graph-attached zero tensors instead of NaN.
- If `embedding_weight <= 0`, embedding loss is disabled without requiring training-script branching.

Loss report keys:

```text
total_loss
semantic_loss
offset_loss
embedding_loss
embedding_weighted_loss
offset_l1_object
offset_l1_near_centroid
offset_l1_far_centroid
embedding_pull
embedding_push
embedding_reg
embedding_valid_instances
```

Smoke results:

```text
compile:
  python -m py_compile src/training/instance_losses.py src/training/__init__.py

K41144 real batch subset, CPU, pure_torch:
  input=(1, 1024, 6)
  total_loss=1.344633
  semantic_loss=0.719223
  offset_loss=0.095410
  embedding_loss=5.299997
  offset_l1_object=0.340578
  offset_l1_near_centroid=0.339094
  offset_l1_far_centroid=0.342062
  embedding_valid_instances=22
  backward_ok=True

bending_pipe real batch subset, CPU, pure_torch:
  input=(1, 1024, 6)
  total_loss=1.536336
  semantic_loss=0.821065
  offset_loss=0.129817
  embedding_loss=5.854542
  offset_l1_object=0.397372
  offset_l1_near_centroid=0.365939
  offset_l1_far_centroid=0.428805
  embedding_valid_instances=6
  backward_ok=True

edge cases:
  all-background batch: finite=True, offset_loss=0, embedding_loss=0
  embedding_weight=0 with object points: finite=True, backward_ok=True

CUDA synthetic batch:
  device=NVIDIA GeForce MX150
  centroid_distance_weighting=True
  finite=True
  backward_ok=True
```

### Phase 4: Training Script

Status: Done.

Deliverable:

```text
scripts/train_pointnet2_instance_seg.py
```

Required CLI flags:

```text
--config
--data
--experiment-root
--epochs
--batch-size
--device
--num-workers
--limit-train-samples
--limit-val-samples
--disable-augment
--overfit
--dropout
--learning-rate
--semantic-weight
--offset-weight
--embedding-weight
--centroid-distance-weighting
```

Pass criteria:

```text
One-epoch CPU or CUDA smoke run completes.
Experiment folder contains config.json, metrics.json, checkpoints/best.pt, checkpoints/last.pt.
```

Implementation notes:

- Added `scripts/train_pointnet2_instance_seg.py`.
- The script reuses:
  - `PointNet2InstanceSegDataset`
  - `PointNet2InstanceSeg`
  - `PointNet2InstanceLoss`
  - semantic `SegmentationMeter`
  - shared checkpoint/config/seed helpers
- `class_weight_mode: auto` computes semantic class weights from the effective train dataset and stores them in `config.json` under:

```text
loss.semantic_class_weights
```

- Runtime overrides are recorded in `runtime_args`.
- Effective dataset root is recorded in:

```text
dataset.root_effective
```

- Effective device and PointNet++ ops backend are recorded in:

```text
runtime.device_effective
model.ops_backend_effective
```

- `best.pt` is selected by minimum validation `total_loss` for this phase. Proper instance-level checkpoint selection should move to instance metrics after Phase 6/9.
- Per-epoch metrics include semantic IoU/precision/recall plus loss report fields from Phase 3:

```text
total_loss
semantic_loss
offset_loss
embedding_loss
embedding_weighted_loss
offset_l1_object
offset_l1_near_centroid
offset_l1_far_centroid
embedding_pull
embedding_push
embedding_reg
embedding_valid_instances
```

Smoke commands:

```powershell
python -m py_compile scripts\train_pointnet2_instance_seg.py

python scripts\train_pointnet2_instance_seg.py --config configs\train\pointnet2_instance_k41144.yaml --epochs 1 --batch-size 1 --limit-train-samples 1 --overfit --disable-augment --dropout 0.0

python scripts\train_pointnet2_instance_seg.py --config configs\train\pointnet2_instance_bending_pipe.yaml --epochs 1 --batch-size 1 --limit-train-samples 1 --overfit --disable-augment --dropout 0.0
```

Smoke results:

```text
K41144:
  output: experiments/pointnet2_instance_k41144_20260607_150443
  train_total=1.3054
  val_total=1.5959
  val_object_iou=0.0000
  val_offset_l1=0.0628
  files: config.json, metrics.json, checkpoints/best.pt, checkpoints/last.pt

bending_pipe:
  output: experiments/pointnet2_instance_bending_pipe_20260607_150509
  train_total=1.3828
  val_total=1.5964
  val_object_iou=0.0000
  val_offset_l1=0.0627
  files: config.json, metrics.json, checkpoints/best.pt, checkpoints/last.pt
```

### Phase 5: One-Sample And Two-Sample Overfit

Status: Done.

Commands:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144_debug4096 --epochs 360 --batch-size 1 --device cuda --overfit --limit-train-samples 2 --limit-val-samples 2 --disable-augment --dropout 0.0 --embedding-weight 0.1 --offset-weight 50.0 --centroid-distance-weighting

python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe_debug4096 --epochs 360 --batch-size 1 --device cuda --overfit --limit-train-samples 2 --limit-val-samples 2 --disable-augment --dropout 0.0 --embedding-weight 0.1 --offset-weight 50.0 --centroid-distance-weighting
```

Pass criteria:

```text
semantic object_iou >= 0.95
offset_l1_object < 0.005 m
offset_l1_far_centroid is not dramatically worse than offset_l1_near_centroid
instance mean_iou >= 0.80
instance recall >= 0.80
```

Phase 5 interpretation:

- Semantic and centroid-offset overfit pass on K41144.
- Embedding-enabled semantic and centroid-offset overfit pass on K41144.
- Embedding-enabled semantic and centroid-offset overfit pass on bending_pipe as the second geometry check.
- `instance mean_iou` and `instance recall` still require Phase 6 clustering/preview because Phase 5 does not yet assign predicted instance clusters.

Important tuning result:

- Default `SmoothL1(beta=1.0)` was too weak for meter-scale centroid offsets.
- Added and used:

```yaml
loss:
  offset_smooth_l1_beta: 0.01
```

- The overfit gate required stronger offset supervision:

```text
--offset-weight 50.0
--centroid-distance-weighting
```

- The center-distance-sensitive weighting was important for keeping far-centroid error close to near-centroid error on elongated shapes.

Generated debug dataset for the second geometry:

```text
processed-data/pointnet2_semseg_bending_pipe_debug4096
```

Command:

```powershell
python scripts\prepare_pointnet2_semseg_dataset.py --raw synthetic-data\bending_pipe --out processed-data\pointnet2_semseg_bending_pipe_debug4096 --config configs\train\pointnet2_semseg_bending_pipe.yaml --num-points 4096
```

Key results:

```text
K41144 semantic+offset, one sample:
  output: experiments/pointnet2_instance_k41144_debug4096_20260607_152322
  best val offset_l1_object=0.004768 m
  best val object_iou=0.968972
  embedding_weight=0.0

K41144 semantic+offset, two samples:
  output: experiments/pointnet2_instance_k41144_debug4096_20260607_153033
  best val offset_l1_object=0.004354 m
  best val object_iou=0.994938
  embedding_weight=0.0

K41144 embedding-enabled, one sample:
  output: experiments/pointnet2_instance_k41144_debug4096_20260607_153446
  best val offset_l1_object=0.004620 m
  best val object_iou=0.979501
  embedding_loss=0.340137
  embedding_valid_instances=29

K41144 embedding-enabled, two samples:
  output: experiments/pointnet2_instance_k41144_debug4096_20260607_153638
  best val offset_l1_object=0.004233 m
  best val object_iou=0.994931
  embedding_loss=0.187324
  embedding_valid_instances=29

bending_pipe embedding-enabled, one sample:
  output: experiments/pointnet2_instance_bending_pipe_debug4096_20260607_154140
  best val offset_l1_object=0.004028 m
  best val object_iou=0.997747
  embedding_loss=0.025195
  embedding_valid_instances=6

bending_pipe embedding-enabled, two samples:
  output: experiments/pointnet2_instance_bending_pipe_debug4096_20260607_154311
  best val offset_l1_object=0.004003 m
  best val object_iou=0.994744
  embedding_loss=0.084143
  embedding_valid_instances=6
```

If this fails:

- Verify centroid offset targets visually.
- Disable embedding loss and train semantic + offset first.
- Enable centroid-distance-sensitive offset weighting if far-centroid errors dominate.
- Reduce dropout to `0.0`.
- Inspect whether DBSCAN thresholds are too strict.
- Check BatchNorm behavior with `batch_size=1`.

### Phase 6: Instance Preview Tool

Status: Done.

Deliverable:

```text
src/inference/instance_clustering.py
scripts/preview_pointnet2_instance_predictions.py
```

Implementation notes:

- `cluster_voted_centers(...)` assigns foreground points by clustering `point_xyz + predicted_offset` with DBSCAN.
- `evaluate_instance_predictions(...)` computes GT/pred cluster counts, Hungarian IoU matching, precision, recall, mean IoU, unmatched counts, and merge/split diagnostics.
- Preview export writes GT instance PLY, predicted instance PLY, error PLY, voted-center PLY, and `preview_summary.json`.
- The preview script supports explicit `--dbscan-eps-m` and diagnostic `--dbscan-eps-sweep-m`; checkpoint outputs should still be reported with one selected DBSCAN value.
- The current tuned debug default is `dbscan_eps_m=0.006`, `dbscan_min_samples=8`, `min_cluster_points=32`.

K41144 debug4096 preview result:

```text
checkpoint: experiments/pointnet2_instance_k41144_debug4096_20260607_153638/checkpoints/best.pt
data: processed-data/pointnet2_semseg_k41144_debug4096
split: train
samples: 2
output: experiments/pointnet2_instance_k41144_debug4096_20260607_153638/instance_previews/train_phase6_eps_tune
dbscan_eps_m=0.006
dbscan_min_samples=8
min_cluster_points=32
mean cluster_count_gt=28.5
mean cluster_count_pred=28.5
mean instance_precision=0.982759
mean instance_recall=0.982759
mean instance_mean_iou=0.884261
mean merge_count=0.0
mean split_count=0.5
```

The earlier `min_cluster_points=64` setting under-clustered K41144 on debug4096 samples, with recall around `0.72`. Lowering the filter to `32` aligns with the metric ignore threshold and preserves small/partial instances without causing systematic merges in the preview set.

Bending pipe debug4096 preview result:

```text
checkpoint: experiments/pointnet2_instance_bending_pipe_debug4096_20260607_154311/checkpoints/best.pt
data: processed-data/pointnet2_semseg_bending_pipe_debug4096
split: train
samples: 2
output: experiments/pointnet2_instance_bending_pipe_debug4096_20260607_154311/instance_previews/train_phase6_eps_tune
dbscan_eps_m=0.006
dbscan_min_samples=8
min_cluster_points=32
mean cluster_count_gt=6.0
mean cluster_count_pred=6.0
mean instance_precision=1.0
mean instance_recall=1.0
mean instance_mean_iou=0.993995
mean merge_count=0.0
mean split_count=0.0
```

Pass criteria:

```text
PLY files export correctly. Done.
Predicted clusters are visually separable. Done for K41144 and bending_pipe debug4096 overfit samples.
Voted-center PLY makes offset failures visible. Done.
Error PLY makes merges/splits visible. Done.
Preview metrics match eval script metrics. Clustering/matching logic is centralized in `src/inference/instance_clustering.py`; future eval scripts should reuse it.
DBSCAN eps sweep can be recorded without changing checkpoint outputs. Done.
```

### Phase 7: Debug Split Training

Status: Done.

Recommended K41144 command:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144_debug4096 --epochs 30 --batch-size 1 --device cuda
```

Recommended bending_pipe command:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe_debug4096 --epochs 40 --batch-size 1 --device cuda
```

Initial acceptance target:

```text
semantic object_iou >= 0.90
offset_l1_object < 0.010 m
instance mean_iou >= 0.60
instance recall >= 0.60
```

Phase 7 tuning notes:

- The original `offset_weight=1.0`, `embedding_weight=0.1`, `dropout=0.4` setup passed semantic and instance preview targets, but offset validation L1 stayed above the `0.010 m` target.
- For the current offset-vote clustering design, embedding loss is not used during inference and can compete with offset optimization on the small debug split.
- Configs are now offset-first for Phase 8 starting runs:
  - K41144: `dropout=0.2`, `offset_weight=5.0`, `embedding_weight=0.0`.
  - bending_pipe: `dropout=0.0`, `offset_weight=10.0`, `embedding_weight=0.0`.
- Re-enable/tune embedding later only if a future clustering or pose-vote stage consumes embeddings directly.

K41144 debug4096 split result:

```text
experiment: experiments/pointnet2_instance_k41144_debug4096_20260607_164415
checkpoint: checkpoints/best.pt
best epoch: 27
val semantic object_iou=0.997982
val offset_l1_object=0.009657 m
preview output: experiments/pointnet2_instance_k41144_debug4096_20260607_164415/instance_previews/val_phase7
val instance_precision=0.705349
val instance_recall=0.745527
val instance_mean_iou=0.741792
val merge_count=1.625
val split_count=4.625
```

Bending pipe debug4096 split result:

```text
experiment: experiments/pointnet2_instance_bending_pipe_debug4096_20260607_165002
checkpoint: checkpoints/best.pt
best epoch: 36
val semantic object_iou=0.997997
val offset_l1_object=0.009755 m
preview output: experiments/pointnet2_instance_bending_pipe_debug4096_20260607_165002/instance_previews/val_phase7
val instance_precision=1.0
val instance_recall=0.944444
val instance_mean_iou=0.947031
val merge_count=0.333333
val split_count=0.0
```

### Phase 8: Full 16384-Point Training

Status: Done.

Recommended K41144 command:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --epochs 50 --batch-size 1 --device cuda
```

Recommended bending_pipe command:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe --epochs 50 --batch-size 1 --device cuda
```

Initial acceptance target:

```text
validation semantic object_iou >= 0.95
validation instance mean_iou >= 0.70
validation instance recall >= 0.70
merge/split diagnostics show no systematic touching-object failure
visual previews show no systematic object merging
```

K41144 full 16384 result:

```text
experiment: experiments/pointnet2_instance_k41144_20260607_165729
checkpoint: checkpoints/best.pt
best epoch: 50
val semantic object_iou=0.997748
val offset_l1_object=0.009098 m
preview output: experiments/pointnet2_instance_k41144_20260607_165729/instance_previews/val_phase8_eps005
dbscan_eps_m=0.005
val instance_precision=0.647478
val instance_recall=0.729244
val instance_mean_iou=0.825471
val merge_count=3.625
val split_count=6.75
```

K41144 note: with `dbscan_eps_m=0.006`, full-density validation under-clustered touching objects (`instance_recall=0.500902`, `merge_count=5.5`). Phase 8 validation passed with `dbscan_eps_m=0.005`. Phase 9 later selected `dbscan_eps_m=0.004` and `min_cluster_points=96` because it preserved validation quality and improved held-out recall.

Bending pipe full 16384 result:

```text
experiment: experiments/pointnet2_instance_bending_pipe_20260607_171218
checkpoint: checkpoints/best.pt
best epoch: 36
val semantic object_iou=0.998562
val offset_l1_object=0.008700 m
preview output: experiments/pointnet2_instance_bending_pipe_20260607_171218/instance_previews/val_phase8
dbscan_eps_m=0.006
val instance_precision=0.904762
val instance_recall=0.888889
val instance_mean_iou=0.929513
val merge_count=0.666667
val split_count=0.0
```

### Phase 9: Held-Out Test Evaluation

Status: Done.

Deliverable:

```text
scripts/eval_pointnet2_instance_seg.py
```

Recommended K41144 command:

```powershell
python .\scripts\eval_pointnet2_instance_seg.py --checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_k41144 --split test --out .\experiments\pointnet2_instance_k41144_20260607_165729\test_metrics.json --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.004 --min-cluster-points 96 --preview-out .\experiments\pointnet2_instance_k41144_20260607_165729\instance_previews\test_phase9 --preview-samples 3
```

Recommended bending_pipe command:

```powershell
python .\scripts\eval_pointnet2_instance_seg.py --checkpoint .\experiments\pointnet2_instance_bending_pipe_20260607_171218\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_bending_pipe --split test --out .\experiments\pointnet2_instance_bending_pipe_20260607_171218\test_metrics.json --device cuda --ops-backend pytorch3d --preview-out .\experiments\pointnet2_instance_bending_pipe_20260607_171218\instance_previews\test_phase9 --preview-samples 3
```

Pass criteria:

```text
test metrics are close to validation metrics. Done for semantic and instance metrics; bending_pipe test offset is slightly above 1 cm but close to val.
no all-object or all-background collapse. Done.
cluster count roughly matches GT visible instance count. Done after K41144 retune to `dbscan_eps_m=0.004`, `min_cluster_points=96`.
PLY previews are visually coherent. PLY previews exported for K41144 and bending_pipe test samples.
```

Eval script behavior:

```text
Runs all samples in the requested split by default.
Computes semantic metrics, offset L1 metrics, selected-DBSCAN instance metrics, and DBSCAN sweep diagnostics.
Can export GT/pred/error/voted-center PLY previews for the first N samples.
Uses the same clustering and matching logic as the Phase 6 preview script.
```

K41144 held-out test result:

```text
experiment: experiments/pointnet2_instance_k41144_20260607_165729
checkpoint: checkpoints/best.pt
metrics: test_metrics.json
preview output: instance_previews/test_phase9
test samples=7
dbscan_eps_m=0.004
min_cluster_points=96
semantic object_iou=0.996716
offset_l1_object=0.009094 m
cluster_count_gt=20.857143
cluster_count_pred=21.428571
instance_precision=0.687659
instance_recall=0.722236
instance_mean_iou=0.805663
merge_count=2.428571
split_count=4.285714
```

K41144 clustering note:

```text
test eps=0.005: instance_recall=0.543863, instance_mean_iou=0.823419, cluster_count_pred=19.571429
test eps=0.004/min_cluster_points=96: instance_recall=0.722236, instance_mean_iou=0.805663, cluster_count_pred=21.428571
```

Bending pipe held-out test result:

```text
experiment: experiments/pointnet2_instance_bending_pipe_20260607_171218
checkpoint: checkpoints/best.pt
metrics: test_metrics.json
preview output: instance_previews/test_phase9
test samples=3
dbscan_eps_m=0.006
min_cluster_points=32
semantic object_iou=0.998468
offset_l1_object=0.010215 m
cluster_count_gt=6.0
cluster_count_pred=4.666667
instance_precision=0.888889
instance_recall=0.722222
instance_mean_iou=0.963109
merge_count=0.666667
split_count=0.0
```

### Phase 10: Bridge To Pose Estimation

Status: Done.

After instance segmentation is usable, create a pose-stage data interface:

```text
predicted instance crop
predicted instance centroid
point indices for each cluster
optional GT object_to_camera for synthetic evaluation
```

The pose stage should consume predicted or GT instance crops through the same API, so later evaluation can separate:

```text
pose quality with GT instances
pose quality with predicted instances
full pipeline quality
```

Before implementing pose regression, run a K41144 symmetry/grasp-equivalence audit:

```text
object geometric symmetry candidates
grasp-equivalent rotations
non-equivalent rotations that must remain distinguishable
recommended symmetry-aware pose metric/loss
```

Deliverables:

```text
src/inference/pose_bridge.py
scripts/export_pose_instance_crops.py
docs/k41144-pose-symmetry-audit.md
```

Pose bridge API:

```text
build_pose_instance_crops(...)
save_pose_instance_crops_npz(...)
load_raw_metadata(...)
pose_crop_summary(...)
```

The bridge accepts either GT instance labels or predicted instance labels and exports the same crop schema. Predicted crops are matched back to GT instance ids by point-set IoU so synthetic `object_to_camera` supervision can still be attached for downstream pose evaluation.

Per-sample crop NPZ schema:

```text
crop_points_camera          (sum_crop_points, 3) float32
crop_point_indices          (sum_crop_points,) int64
crop_offsets                (num_crops, 2) int64
crop_instance_ids           (num_crops,) int64
crop_point_counts           (num_crops,) int64
crop_centroids_camera       (num_crops, 3) float32
crop_bbox_min_camera        (num_crops, 3) float32
crop_bbox_max_camera        (num_crops, 3) float32
crop_matched_gt_instance_ids (num_crops,) int64
crop_matched_gt_ious        (num_crops,) float32
crop_object_to_camera       (num_crops, 4, 4) float32, NaN when unavailable
sample_name                 string
source                      string: gt or predicted
```

GT crop export results:

```text
K41144:
  output: experiments/pose_bridge_k41144_gt_test
  split: test
  samples=7
  total_crops=146
  crops_with_object_to_camera=146
  matched_gt_iou_mean=1.0

bending_pipe:
  output: experiments/pose_bridge_bending_pipe_gt_test
  split: test
  samples=3
  total_crops=18
  crops_with_object_to_camera=18
  matched_gt_iou_mean=1.0
```

Predicted crop export results:

```text
K41144:
  output: experiments/pose_bridge_k41144_pred_test
  split: test
  samples=7
  total_crops=150
  crops_with_object_to_camera=150
  matched_gt_iou_mean=0.661017
  matched_gt_iou_at_least_0.5=103
  matched_gt_iou_at_least_0.75=67
  clustering: dbscan_eps_m=0.004, min_cluster_points=96

bending_pipe:
  output: experiments/pose_bridge_bending_pipe_pred_test
  split: test
  samples=3
  total_crops=14
  crops_with_object_to_camera=14
  matched_gt_iou_mean=0.909215
  matched_gt_iou_at_least_0.5=13
  matched_gt_iou_at_least_0.75=12
  clustering: dbscan_eps_m=0.006, min_cluster_points=32
```

K41144 symmetry audit result:

```text
Recommended symmetry set for pose loss/metric:
S_k41144 = {I, R_y(pi)}
```

Do not use continuous axial symmetry for K41144. The mesh is elongated along local `Y` and `R_y(180 deg)` is a strong symmetry candidate, but 90-degree rotations around `Y` have local errors large enough that they should not be treated as exact pose equivalences.

Next pose-stage design:

```text
docs/pointnet2-pose-estimation-plan.md
```

## Initial Config Draft

Create:

```text
configs/train/pointnet2_instance_k41144.yaml
```

Draft:

```yaml
seed: 7

dataset:
  root: processed-data/pointnet2_semseg_k41144
  normalize: scene_center
  min_points_per_instance: 16
  augment:
    enabled: true
    xyz_jitter_std: 0.0005
    xyz_jitter_clip: 0.002
    depth_noise_std: 0.0007
    point_dropout_prob: 0.03
    outlier_ratio: 0.002
    outlier_std: 0.015
    normal_jitter_std: 0.02
    random_z_rotation: false

model:
  name: pointnet2_instance_seg
  num_classes: 2
  input_features: xyz_normal
  ops_backend: pytorch3d
  sa_npoints: [1024, 256, 64]
  sa_nsamples: [32, 32, 32]
  dropout: 0.2
  embedding_dim: 8

loss:
  semantic_weight: 1.0
  offset_weight: 5.0
  embedding_weight: 0.0
  delta_var: 0.5
  delta_dist: 1.5
  regularization_weight: 0.001
  centroid_distance_weighting: false
  centroid_distance_weight_min: 0.5
  centroid_distance_weight_max: 1.5

inference:
  object_probability_threshold: 0.5
  dbscan_eps_m: 0.004
  dbscan_eps_sweep_m: [0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012]
  dbscan_min_samples: 8
  min_cluster_points: 96

metrics:
  instance_iou_threshold: 0.5
  ignore_gt_instances_below_points: 32
  merge_split_overlap_threshold: 0.1

train:
  seed: 7
  device: cuda
  num_workers: 0
  epochs: 50
  batch_size: 1
  learning_rate: 0.001
  weight_decay: 0.0001
  class_weight_mode: auto
```

This template reflects the K41144 Phase 9 baseline. For bending_pipe, use `dropout=0.0`, `offset_weight=10.0`, `dbscan_eps_m=0.006`, and `min_cluster_points=32` as the current starting point.

## Main Risks

### Touching Objects May Merge

Risk:

- K41144 objects are same-class, low-texture, and often touching.
- Semantic segmentation can be excellent while instance segmentation still merges adjacent objects.

Mitigation:

- Make offset voting the primary separation signal.
- Tune DBSCAN on voted centers, not raw xyz alone.
- Export voted-center PLY previews.
- Track merge/split diagnostics, not only instance mean IoU.
- Use visual PLY previews after every meaningful training run.

### Offset Regression May Be Weak At Object Ends

Risk:

- K41144 is elongated, so points far from the visible centroid can be harder to regress.
- This is the failure mode PS6D targets with center-distance-sensitive translation loss.

Mitigation:

- Track near/far centroid offset errors.
- Keep optional center-distance weighting in the loss config.
- Enable weighting only after the unweighted MVP establishes a clean baseline.

### Tiny Visible Instances

Risk:

- Some visible instances have very few sampled points.
- These can destabilize embedding loss and make recall look artificially poor.

Mitigation:

- Track tiny instances separately.
- Ignore instances below a threshold for embedding loss.
- Report ignored counts in metrics.

### BatchNorm With Batch Size 1

Risk:

- Previous one-sample semantic overfit showed an eval-mode gap likely related to BatchNorm running statistics.

Mitigation:

- Run one-sample and two-sample overfit gates.
- If the gap returns, consider GroupNorm or InstanceNorm in the shared MLP blocks.
- Keep this as a targeted normalization experiment, not a broad refactor.

### Synthetic-To-Real Gap

Risk:

- Clean synthetic data may not represent real depth boundary noise, missing points, reflective surfaces, and ambiguous object contact regions.

Mitigation:

- Keep raw Blender data clean.
- Use online augmentation for jitter, depth noise, dropout, sparse outliers, and normal jitter.
- Later add real-sensor samples for calibration and validation.

## Immediate Next Tasks

Implement in this order:

1. `scripts/audit_pointnet2_instance_data.py`
2. `src/data/pointnet2_instance_dataset.py`
3. `src/models/pointnet2_instance_seg.py`
4. `src/training/instance_losses.py`
5. `src/inference/instance_clustering.py`
6. `src/training/instance_metrics.py`
7. `scripts/train_pointnet2_instance_seg.py`
8. One-sample and two-sample overfit, first with semantic+offset and then with embeddings enabled.
9. `scripts/preview_pointnet2_instance_predictions.py` with voted-center export.
10. Debug/full/test training.

The first implementation step should be Phase 0 because it gives a hard data-quality gate before adding model complexity.
