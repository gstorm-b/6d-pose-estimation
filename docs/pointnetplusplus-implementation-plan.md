# PointNet++ Implementation Plan For K41144 Semantic Segmentation

This document is the technical plan for implementing the first trainable point-cloud model in this project.

Scope for the first milestone:

- Train PointNet++ semantic segmentation on synthetic bin-picking scenes.
- Predict per-point labels:
  - `0 = background / bin / non-target geometry`
  - `1 = K41144`
- Keep dataset and model design compatible with a later instance-segmentation branch.

Non-goals for the first milestone:

- Do not implement pose estimation yet.
- Do not implement ICP yet.
- Do not train directly from raw Blender folders.
- Do not require hard-to-install CUDA point-cloud extensions for the MVP.

## Current Status Snapshot

Status as of this plan update:

- Raw dataset validation exists and `synthetic-data/K41144` passes validation: `77/77 OK`.
- Processed PointNet++ semantic dataset exists at `processed-data/pointnet2_semseg_k41144`.
- Processed split: `train/val/test = 62/8/7`.
- Processed sample size: `16384` points.
- Processed sampling balance: `10650` object points and `5734` background points per sample.
- Data conversion, unprojection, augmentation hooks, and PyTorch dataset wrapper are implemented.
- Pure-PyTorch PointNet++ semantic segmentation MVP is implemented.
- Training script, metrics, checkpoint, seed, and config helpers are implemented.
- PyTorch GPU is installed and verified:
  - `torch 2.12.0+cu126`
  - `torch.cuda.is_available() == True`
  - GPU: `NVIDIA GeForce MX150`
  - compute capability: `(6, 1)`
  - PyTorch arch list includes `sm_61`
- CUDA forward pass works.
- CPU and CUDA one-epoch smoke training on the small smoke dataset both completed.

Current blocker before claiming a working model:

- The model has not yet demonstrated that it can overfit 1-2 samples.
- Full processed dataset training has not yet been run.
- Prediction preview/evaluation scripts are not implemented yet.
- The model has not yet passed a quality gate on validation/test data.

## References

Project article note:

```text
arcticle/instance-segmentation-based-6d-pose-estimation-for-bin-picking.md
```

Relevant article interpretation:

- The full pipeline has two main stages:
  1. Instance segmentation on scene point cloud.
  2. 6D pose estimation per segmented instance.
- Semantic segmentation is a branch inside the instance-segmentation network.
- PointNet++ is the proposed backbone for extracting point features.
- Instance segmentation later requires both semantic logits and instance embeddings.

External package references:

- PyTorch official install selector: https://pytorch.org/get-started/locally/
- PyTorch Geometric official install notes: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Package Decision

### Primary Stack For MVP

Use:

```text
Python
PyTorch
NumPy
tqdm
PyYAML
scikit-learn
matplotlib
```

Rationale:

- PyTorch is the stable training foundation.
- A pure-PyTorch PointNet++ MVP is easier to debug on Windows.
- It avoids blocking the project on CUDA extension compatibility.
- Dataset and training code stay readable while the data format is still evolving.

### Optional Stack For Later Optimization

Consider PyTorch Geometric later, not as an MVP dependency.

Reason:

- PyG provides useful point-cloud and graph primitives, but optional extension packages must match the installed PyTorch and CUDA versions.
- This is powerful but adds installation friction, especially on Windows.
- If pure-PyTorch sampling/grouping becomes too slow, revisit PyG or custom CUDA ops after the dataset and training target are stable.

### Do Not Use For MVP

Avoid making these mandatory at first:

```text
pytorch3d
torch-points-kernels
third-party pointnet2 CUDA forks
```

Reason:

- They can be useful later, but they increase install/debug risk before we have a stable processed dataset and baseline metrics.

## Environment Plan

Create an isolated environment when starting from a clean machine. Exact PyTorch command should be selected from the official PyTorch install page for the machine's CUDA/runtime.

For the current project machine, PyTorch CUDA has already been installed with the official CUDA 12.6 wheel:

```text
torch 2.12.0+cu126
```

The project-level install hint is stored in:

```text
requirements-train.txt
```

After install, always verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Minimum expected packages:

```text
torch
numpy
tqdm
pyyaml
scikit-learn
matplotlib
```

Recommended project files:

```text
requirements-train.txt
configs/train/pointnet2_semseg_k41144.yaml
```

Do not add package pins until the actual environment is confirmed.

Current hardware caution:

- GPU is `NVIDIA GeForce MX150` with 2GB VRAM.
- Prefer `--batch-size 1` for CUDA experiments.
- Full `16384`-point samples may run out of memory with the current pure-PyTorch grouping implementation.
- If CUDA OOM occurs, create a reduced processed dataset with `4096` or `8192` points and keep the full dataset for final baselines.

## Data Design

### Important Current Raw Dataset Limitation

Current raw `sensor_data.npz` stores:

```text
points_camera
points_world
point_instance_ids
point_semantic_ids
```

But those point arrays are object-only. Background/bin points are present in:

```text
depth_m
instance_mask
```

Therefore semantic training must not use current `points_camera` directly as the full input. The processed dataset converter should reconstruct full-scene points from `depth_m`.

### Raw To Processed Converter

Create:

```text
scripts/prepare_pointnet2_semseg_dataset.py
```

Input:

```text
--raw synthetic-data/K41144
--out processed-data/pointnet2_semseg_k41144
--config configs/train/pointnet2_semseg_k41144.yaml
```

Current implementation status:

- `scripts/prepare_pointnet2_semseg_dataset.py` exists.
- `src/data/depth_unprojection.py` reconstructs full-scene camera points from `depth_m`.
- `src/data/pointnet2_processing.py` performs balanced object/background sampling and writes fixed-size `.npz` samples.
- `src/data/augmentation.py` contains conservative NumPy sensor-noise augmentations.
- `src/data/pointnet2_dataset.py` provides a PyTorch dataset wrapper with optional online augmentation.
- `configs/train/pointnet2_semseg_k41144.yaml` stores the default conversion and augmentation settings.

Converter responsibilities:

1. Read each raw sample:
   - `sensor_data.npz`
   - `metadata.json`
2. Use `depth_m` and `camera_intrinsics` to unproject all valid depth pixels.
3. Use `instance_mask` to assign labels:
   - `instance_mask == 0 -> semantic_label = 0`
   - `instance_mask > 0 -> semantic_label = 1`
4. Store instance labels for later:
   - `instance_mask == 0 -> instance_label = 0`
   - `instance_mask > 0 -> instance_label = instance_id`
5. Optionally attach normal features:
   - `normal_camera[v, u]`
6. Sample or pad to fixed `num_points`.
7. Write processed `.npz` files and stats.

### Processed Dataset Structure

Recommended structure:

```text
processed-data/
  pointnet2_semseg_k41144/
    train/
      sample_000000.npz
    val/
      sample_000001.npz
    test/
      sample_000002.npz
    dataset_stats.json
    conversion_config.json
    schema.md
```

Each processed sample:

```text
points              (num_points, 3) float32
features            (num_points, C) float32 optional
semantic_labels     (num_points,) int64
instance_labels     (num_points,) int64
point_pixels        (num_points, 2) uint16
raw_sample          string or metadata field
```

For MVP:

```text
features = normals only, or omit features entirely
model input = xyz, optionally xyz + normals
```

Recommended first config:

```text
num_points: 16384
use_normals: true
background_label: 0
object_label: 1
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
```

### Sampling Policy

Semantic segmentation needs both background and object points.

Recommended MVP sampling:

```text
object_fraction_target: 0.65
background_fraction_target: 0.35
```

If a sample does not have enough background or object points:

- Use all available points from that group.
- Fill the remainder from the other group.
- If still short, sample with replacement.

Reason:

- Raw scenes may be object-heavy because the camera looks into the bin.
- Balanced-ish sampling avoids a model that learns only the majority class.
- Keep the original full point distribution stats in `dataset_stats.json`.

## Model Design

### MVP Model

Create reusable module:

```text
src/models/pointnet2_semseg.py
```

Architecture:

```text
Input xyz/features
  -> Set Abstraction 1
  -> Set Abstraction 2
  -> Set Abstraction 3
  -> Feature Propagation 3
  -> Feature Propagation 2
  -> Feature Propagation 1
  -> per-point MLP classifier
  -> logits (num_points, num_classes)
```

Initial model scale:

```text
num_classes: 2
input_channels: 3 if xyz only
input_channels: 6 if xyz + normals
sa_npoints: [4096, 1024, 256]
sa_radii: [0.015, 0.035, 0.075]
sa_nsamples: [32, 32, 32]
feature_dims: [64, 128, 256]
dropout: 0.4
```

The radii are initial guesses in meters and should be validated against the K41144 size:

```text
approx object size: 0.030 m x 0.0936 m x 0.020 m
```

### Implementation Choice For PointNet++ Ops

For MVP, implement grouping in PyTorch:

- Farthest point sampling.
- Ball query or k-nearest-neighbor grouping.
- Local PointNet MLP over grouped points.
- Three-nearest interpolation for feature propagation.

Implementation notes:

- Pure PyTorch will be slower than CUDA kernels.
- Keep `num_points=16384` as target, but allow smaller debug configs like `4096`.
- Add small tests for tensor shapes before long training.

If pure PyTorch is too slow:

1. Try PyG with matching Torch/CUDA wheels.
2. Replace only sampling/grouping ops.
3. Keep dataset, model API, training loop unchanged.

## Training Design

Create:

```text
scripts/train_pointnet2_semseg.py
src/data/pointnet2_dataset.py
src/training/metrics.py
src/training/checkpoint.py
```

Training input:

```text
processed-data/pointnet2_semseg_k41144
```

Training output:

```text
experiments/
  pointnet2_semseg_k41144_YYYY-MM-DD_001/
    config.yaml
    metrics.json
    train.log
    checkpoints/
      best.pt
      last.pt
    previews/
```

Loss:

```text
CrossEntropyLoss
```

Use class weights if background/object imbalance is high.

Metrics:

```text
overall_accuracy
mean_class_accuracy
background_iou
k41144_iou
mean_iou
precision_object
recall_object
confusion_matrix
```

Recommended first training config:

```yaml
seed: 7
device: cuda
num_workers: 0

dataset:
  root: processed-data/pointnet2_semseg_k41144
  num_points: 16384
  use_normals: true
  normalize: scene_center
  augment: true

model:
  name: pointnet2_semseg
  num_classes: 2
  input_features: xyz_normal
  sa_npoints: [4096, 1024, 256]
  sa_radii: [0.015, 0.035, 0.075]
  sa_nsamples: [32, 32, 32]
  dropout: 0.4

train:
  epochs: 100
  batch_size: 4
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  class_weight_mode: auto
  save_every_epochs: 10
  validate_every_epochs: 1
```

Debug config:

```yaml
dataset:
  num_points: 4096
model:
  sa_npoints: [1024, 256, 64]
train:
  epochs: 2
  batch_size: 2
```

## Augmentation Plan

Use conservative augmentations first:

```text
random z rotation around camera/world vertical axis if frame convention is correct
small xyz jitter
small random point dropout
small depth-like noise along camera ray
```

Do not use arbitrary 3D rotations for bin scenes by default because gravity and camera viewpoint matter.

Record augment config in each experiment folder.

Implementation note:

- Keep raw Blender data clean.
- Apply noise in processed-data creation or online dataset loading.
- Current online augmentation supports xyz jitter, depth-axis noise, point dropout, sparse outliers, normal jitter, and optional z rotation.

## Validation Gates

Before training:

- Processed dataset exists.
- Each `.npz` has expected keys.
- Shapes and dtypes match schema.
- Both classes appear in train and val splits.
- Per-sample object/background point counts are reasonable.
- Sample visualization confirms labels align with geometry.

Before trusting a model:

- Overfit one or two samples.
- Train debug config for 2 epochs.
- Confirm loss decreases.
- Confirm object recall is non-zero.
- Save prediction previews.

Acceptance criteria for MVP:

```text
debug overfit: near-perfect IoU on 1-2 samples
small validation run: object recall clearly above naive baseline
full synthetic validation: report mean IoU and object recall
```

## Repository Plan

Recommended files to add:

```text
configs/
  train/
    pointnet2_semseg_k41144.yaml

scripts/
  prepare_pointnet2_semseg_dataset.py
  train_pointnet2_semseg.py
  eval_pointnet2_semseg.py
  preview_pointnet2_semseg_labels.py

src/
  data/
    pointnet2_dataset.py
    depth_unprojection.py
  models/
    pointnet2_semseg.py
    pointnet2_ops.py
  training/
    metrics.py
    checkpoint.py
    seed.py
```

Keep scripts thin. Put reusable code in `src/`.

## Detailed Roadmap To A Completed PointNet++ Model

This section is the execution plan from the current project state to a completed semantic PointNet++ baseline. Do not skip the gates: each one catches a different class of failure.

### Phase 0: Baseline State Audit

Status: Done.

Artifacts:

- `scripts/validate_raw_dataset.py`
- `processed-data/pointnet2_semseg_k41144`
- `configs/train/pointnet2_semseg_k41144.yaml`

Commands:

```powershell
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\K41144
python -c "import json; from pathlib import Path; s=json.loads(Path('processed-data/pointnet2_semseg_k41144/dataset_stats.json').read_text()); print(s['splits']); print(s['sampled_object_points']); print(s['sampled_background_points'])"
```

Pass criteria:

- Raw dataset has no validation failures.
- Processed dataset has non-empty train/val/test splits.
- Both background and object classes appear in processed train and val samples.

Current result:

```text
raw: 77/77 OK
processed split: train/val/test = 62/8/7
sampled points: 16384
object/background: 10650/5734
```

### Phase 1: Debug-Size Processed Dataset

Status: Done.

Reason:

- The MX150 has only 2GB VRAM.
- Full `16384`-point pure-PyTorch PointNet++ may OOM or be too slow.
- Overfit/debug gates should run on `4096` or `8192` points first.

Deliverables:

- `processed-data/pointnet2_semseg_k41144_debug4096`
- `processed-data/pointnet2_semseg_k41144_debug4096/dataset_stats.json`
- Optional config:
  - `configs/train/pointnet2_semseg_k41144_debug.yaml`

Command:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\K41144 --out .\processed-data\pointnet2_semseg_k41144_debug4096 --config .\configs\train\pointnet2_semseg_k41144.yaml --num-points 4096
```

Pass criteria:

- Converter finishes without raw validation failures.
- Output has train/val/test split.
- Each sample has:
  - `points.shape == (4096, 3)`
  - `features.shape == (4096, 3)` when normals are enabled
  - `semantic_labels.shape == (4096,)`
  - both labels `0` and `1`

If this fails:

- Fix converter/schema before touching the model.

Current result:

```text
output: processed-data/pointnet2_semseg_k41144_debug4096
sample_count: 77
train/val/test: 62/8/7
points per sample: 4096
object/background per sample: 2662/1434
features: normal_camera, shape (4096, 3)
labels: both 0 and 1 present in checked train/val/test samples
```

### Phase 2: Dataset And Model Smoke Tests

Status: Done.

Deliverables:

- A small script or command sequence that validates:
  - dataset item shapes
  - model forward pass
  - loss computation
  - backward pass

Commands:

```powershell
python -c "from src.data.pointnet2_dataset import PointNet2SemSegDataset; ds=PointNet2SemSegDataset('processed-data/pointnet2_semseg_k41144_debug4096','train',use_normals=True,augment={'enabled': True},seed=7); item=ds[0]; print(item['model_input'].shape, item['semantic_labels'].shape)"
python -c "import torch; from src.models.pointnet2_semseg import PointNet2SemSeg; device='cuda'; model=PointNet2SemSeg(input_channels=6, sa_npoints=(512,128,32), sa_nsamples=(16,16,16)).to(device); x=torch.randn(1,4096,6,device=device); y=model(x); loss=torch.nn.functional.cross_entropy(y.transpose(1,2), torch.zeros(1,4096,dtype=torch.long,device=device)); loss.backward(); print(y.shape, float(loss.detach().cpu()))"
```

Pass criteria:

- Forward output is `(B, N, 2)`.
- Backward completes on CUDA.
- No CUDA OOM.

If this fails:

- If OOM: reduce `sa_npoints`, `sa_nsamples`, or `num_points`.
- If tensor shape fails: fix model API before training.

Current result:

```text
dataset: processed-data/pointnet2_semseg_k41144_debug4096
DataLoader train length: 62
model_input shape: (4096, 6)
semantic_labels shape: (4096,)
labels present: 0 and 1
CUDA forward/backward: pass
logits shape: (1, 4096, 2)
device: cuda:0
```

### Phase 3: One-Sample Overfit

Status: Done, with eval-mode caveat.

Reason:

- A model that cannot overfit one sample has a bug in data, labels, loss, optimization, or architecture.

Deliverables:

- A small overfit dataset or train flag.
- Recommended script addition:
  - `--limit-train-samples`
  - `--limit-val-samples`
  - `--disable-augment`
  - `--overfit`

Implementation changes:

- Update `scripts/train_pointnet2_semseg.py` to support limiting train/val samples.
- Disable augmentation for overfit runs.
- Save metrics every epoch.

Recommended command after adding flags:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144_debug4096 --epochs 50 --batch-size 1 --device cuda --overfit --limit-train-samples 1 --limit-val-samples 1 --disable-augment
```

Pass criteria:

- Training loss drops strongly.
- One-sample train object IoU approaches near-perfect.
- Train object recall becomes high and non-zero early.

Target:

```text
train object_iou >= 0.95 on the overfit sample
train object_recall >= 0.95
```

If this fails:

- Check that `semantic_labels` align with geometry.
- Disable class weights temporarily.
- Reduce model dropout to `0.0`.
- Lower learning rate if loss is unstable.
- Verify logits/label shape passed to `CrossEntropyLoss`.
- Try xyz-only input to isolate bad normal features.

Current result:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_153430
dataset: processed-data/pointnet2_semseg_k41144_debug4096
epochs: 50
batch_size: 1
device: cuda
overfit: true
limit_train_samples: 1
limit_val_samples: 1
augmentation: disabled
dropout: 0.0
learning_rate: 0.001
```

Final train-mode metrics:

```text
train_loss=0.0604
train_mean_iou=0.9588
train_object_iou=0.9706
train_object_recall=0.9790
```

Final eval-mode metrics on the same sample:

```text
val_loss=0.3633
val_mean_iou=0.7002
val_object_iou=0.7429
val_object_recall=0.7652
```

Interpretation:

- The model can memorize one sample in train mode, so the basic data/loss/backward path works.
- Eval-mode is weaker on the same sample, likely due to BatchNorm running statistics with `batch_size=1` and a tiny overfit set.
- Do not ignore this. Phase 4 should check whether two-sample overfit and/or longer training improves eval-mode behavior.
- If eval-mode remains poor, replace BatchNorm with a batch-size-1-friendly normalization strategy or tune BatchNorm momentum.

### Phase 4: Two-Sample Overfit

Status: Done.

Reason:

- One-sample overfit can hide data-order or BatchNorm issues.
- Two-sample overfit checks repeated batching and minimal generalization inside tiny data.

Command:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144_debug4096 --epochs 80 --batch-size 1 --device cuda --overfit --limit-train-samples 2 --limit-val-samples 2 --disable-augment
```

Pass criteria:

```text
train mean_iou >= 0.90
train object_recall >= 0.90
loss decreases consistently
```

If this fails:

- Keep debugging before running full dataset training.

Current result:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_153959
dataset: processed-data/pointnet2_semseg_k41144_debug4096
epochs: 80
batch_size: 1
device: cuda
overfit: true
limit_train_samples: 2
limit_val_samples: 2
augmentation: disabled
dropout: 0.0
learning_rate: 0.001
```

Best eval-mode epoch:

```text
epoch=78
train_loss=0.0115
train_mean_iou=0.9939
train_object_iou=0.9957
train_object_recall=0.9976
val_loss=0.0121
val_mean_iou=0.9933
val_object_iou=0.9953
val_object_recall=0.9970
```

Final epoch:

```text
epoch=80
train_loss=0.0098
train_mean_iou=0.9939
train_object_iou=0.9957
train_object_recall=0.9966
val_loss=0.0350
val_mean_iou=0.9663
val_object_iou=0.9758
val_object_recall=0.9771
```

Interpretation:

- Two-sample overfit passes strongly.
- The eval-mode gap observed in Phase 3 is not a blocker for this MVP; with two samples and enough epochs, eval-mode also reaches near-perfect segmentation.
- Continue to visual prediction previews before trusting full-split metrics.

### Phase 5: Prediction Preview Tool

Status: Done.

Reason:

- Metrics alone are not enough for segmentation.
- Need to visually confirm object/background labels on point clouds.

Deliverables:

- `scripts/preview_pointnet2_semseg_predictions.py`
- Output folder under each experiment:

```text
experiments/<run>/previews/
  sample_XXXXXX_gt.ply
  sample_XXXXXX_pred.ply
  sample_XXXXXX_error.ply
```

Preview colors:

- Background ground truth: gray.
- Object ground truth: green.
- Predicted object: blue.
- False positive: red.
- False negative: yellow.

Command:

```powershell
python .\scripts\preview_pointnet2_semseg_predictions.py --checkpoint .\experiments\<run>\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_k41144_debug4096 --split val --out .\experiments\<run>\previews --max-samples 8 --device cuda
```

Pass criteria:

- PLY files open in Open3D/MeshLab.
- Visual labels align with object geometry.
- Failures are interpretable.

If this fails:

- Fix preview/data transform first; do not trust metrics until preview is coherent.

Current result:

```text
script: scripts/preview_pointnet2_semseg_predictions.py
checkpoint: experiments/pointnet2_semseg_k41144_20260606_153959/checkpoints/best.pt
data: processed-data/pointnet2_semseg_k41144_debug4096
split: train
max_samples: 2
device: cuda
output: experiments/pointnet2_semseg_k41144_20260606_153959/previews/train
```

Generated files:

```text
preview_summary.json
sample_000000_gt.ply
sample_000000_pred.ply
sample_000000_error.ply
sample_000001_gt.ply
sample_000001_pred.ply
sample_000001_error.ply
```

Preview export checks:

```text
PLY header: valid ASCII PLY
vertices per PLY: 4096
sample_count: 2
overall_accuracy: 0.9972
mean_iou: 0.9938
object_iou: 0.9957
object_recall: 0.9981
confusion_matrix: [[2855, 13], [10, 5314]]
```

Interpretation:

- Preview export works and matches the two-sample overfit checkpoint metrics.
- The files still need human visual inspection in a PLY viewer before using preview quality as a sign-off.

### Phase 6: Debug Dataset Training

Status: Done.

Reason:

- After overfit passes, train on the full debug4096 split to test real validation behavior within MX150 memory limits.

Recommended config:

```yaml
conversion:
  num_points: 4096
model:
  sa_npoints: [512, 128, 32]
  sa_nsamples: [16, 16, 16]
  dropout: 0.2
train:
  epochs: 20
  batch_size: 1
  learning_rate: 0.001
```

Command:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144_debug4096 --epochs 20 --batch-size 1 --device cuda
```

Pass criteria:

- Validation object recall is non-zero.
- Validation object IoU is clearly above the trivial all-background baseline.
- Loss decreases over the first few epochs.
- Prediction previews show recognizable object regions.

Minimum debug target:

```text
val object_recall > 0.50
val object_iou > 0.30
```

If this fails:

- Compare augmented vs non-augmented training.
- Inspect class weights.
- Inspect labels and previews.
- Consider using xyz-only or normal-only ablations.

Current result:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_155444
dataset: processed-data/pointnet2_semseg_k41144_debug4096
epochs: 20
batch_size: 1
device: cuda
train/val split: 62/8
runtime: about 16.2 minutes on NVIDIA GeForce MX150
```

Best validation epoch:

```text
epoch=19
train_loss=0.0054
train_mean_iou=0.9959
train_object_iou=0.9971
train_object_recall=0.9979
val_loss=0.0066
val_mean_iou=0.9937
val_object_iou=0.9956
val_object_recall=0.9961
```

Final epoch:

```text
epoch=20
train_loss=0.0067
train_mean_iou=0.9949
train_object_iou=0.9964
train_object_recall=0.9973
val_loss=0.0180
val_mean_iou=0.9865
val_object_iou=0.9905
val_object_recall=0.9911
```

Validation preview export:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_155444/checkpoints/best.pt
output: experiments/pointnet2_semseg_k41144_20260606_155444/previews/val
sample_count: 8
files: 8 gt PLY + 8 pred PLY + 8 error PLY + preview_summary.json
```

Preview overall metrics:

```text
overall_accuracy=0.9969
mean_iou=0.9931
object_iou=0.9952
object_precision=0.9997
object_recall=0.9954
confusion_matrix=[[11466, 6], [97, 21199]]
```

Interpretation:

- Debug4096 full split training passes the initial Phase 6 target by a wide margin.
- The validation preview export is coherent numerically, but PLY files should still be visually inspected before moving to a final sign-off.

### Phase 7: Memory And Speed Profiling

Status: Done.

Reason:

- Need to know whether MX150 can train `8192` or `16384` point variants.

Deliverables:

- Short profiling notes in this Markdown or a run summary.
- Recommended optional script:
  - `scripts/profile_pointnet2_memory.py`

Test matrix:

```text
num_points: 4096, 8192, 16384
batch_size: 1
sa_npoints:
  small: [512, 128, 32]
  medium: [1024, 256, 64]
  full: [4096, 1024, 256]
```

Pass criteria:

- Identify the largest config that fits GPU memory.
- Record epoch time or samples/second.
- Pick one debug config and one baseline config.

Decision rule:

- If `16384` OOMs, do not force it on MX150.
- Use `4096` or `8192` for iterative model work.
- Keep `16384` processed data for later stronger GPU or CPU overnight baseline.

Current result:

```text
script: scripts/profile_pointnet2_memory.py
profile output: experiments/pointnet2_profile_20260606/
summary: experiments/pointnet2_profile_20260606/profile_summary.json
gpu: NVIDIA GeForce MX150
gpu_total_mib: 2047.875
```

Additional processed dataset created for profiling:

```text
processed-data/pointnet2_semseg_k41144_debug8192
sample_count: 77
train/val/test: 62/8/7
points per sample: 8192
object/background per sample: 5325/2867
```

Profile matrix:

```text
4096 points, sa_npoints=[1024,256,64], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=1.0731
  max_allocated_mib=130.1
  max_reserved_mib=148.0

8192 points, sa_npoints=[1024,256,64], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=0.9815
  max_allocated_mib=153.5
  max_reserved_mib=170.0

16384 points, sa_npoints=[1024,256,64], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=2.1114
  max_allocated_mib=199.9
  max_reserved_mib=228.0

16384 points, sa_npoints=[4096,1024,256], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=5.7413
  max_allocated_mib=669.3
  max_reserved_mib=804.0
```

Interpretation:

- MX150 can fit all profiled configs with `batch_size=1`.
- The current config `sa_npoints=[1024,256,64]` is safe for 16384-point baseline training.
- The full-style config `sa_npoints=[4096,1024,256]` also fits, but it is much slower and should not be the first long baseline on this GPU.
- Use `4096` for rapid iteration, `8192` as a middle ground, and `16384` with current config for the first stronger synthetic baseline.

### Phase 8: Full Synthetic Baseline

Status: Next.

Reason:

- This is the first meaningful model baseline after overfit/debug gates pass.

Recommended next command, using the full 16384-point processed dataset with the current safe config:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --epochs 100 --batch-size 1 --device cuda
```

Shorter checkpoint run before committing to 100 epochs:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --epochs 20 --batch-size 1 --device cuda
```

Fallback command, using the `4096` debug dataset:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144_debug4096 --epochs 100 --batch-size 1 --device cuda
```

Deliverables:

- Experiment folder:

```text
experiments/pointnet2_semseg_k41144_<timestamp>/
  config.json
  metrics.json
  checkpoints/best.pt
  checkpoints/last.pt
  previews/
```

Pass criteria:

```text
val object_recall >= 0.80
val object_iou >= 0.60
mean_iou improves over debug baseline
previews show coherent object masks
```

These targets are initial synthetic-only thresholds. Tighten them after seeing real validation behavior.

### Phase 9: Test Evaluation

Status: Pending.

Reason:

- Validation split guides development.
- Test split gives the first held-out synthetic estimate.

Deliverables:

- `scripts/eval_pointnet2_semseg.py`
- `experiments/<run>/test_metrics.json`
- Optional test previews.

Command:

```powershell
python .\scripts\eval_pointnet2_semseg.py --checkpoint .\experiments\<run>\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_k41144_debug4096 --split test --device cuda
```

Pass criteria:

- Test metrics are close to validation metrics.
- No class collapses to all-background or all-object.
- Confusion matrix is saved.

### Phase 10: Inference API

Status: Pending.

Reason:

- The model should be usable outside the training script.
- Later robot/bin-picking code should call one inference path.

Deliverables:

- `src/inference/pointnet2_semseg_infer.py`
- `scripts/infer_pointnet2_semseg.py`
- Input:
  - processed `.npz`, or
  - raw depth/intrinsics converted to points with the same preprocessing path.
- Output:
  - per-point semantic logits/probabilities
  - predicted labels
  - optional colored `.ply`

Command:

```powershell
python .\scripts\infer_pointnet2_semseg.py --checkpoint .\experiments\<run>\checkpoints\best.pt --sample .\processed-data\pointnet2_semseg_k41144_debug4096\test\sample_000001.npz --out .\experiments\<run>\inference_preview
```

Pass criteria:

- Inference runs without training-only dependencies.
- Output labels are aligned with input points.
- Runtime and memory usage are recorded for MX150.

### Phase 11: Model Completion Criteria

The PointNet++ semantic model is considered complete for the first milestone only when all criteria below are met:

- Raw dataset passes validation.
- Processed dataset has documented schema and stats.
- CUDA environment is verified.
- Model forward/backward passes.
- One-sample and two-sample overfit gates pass.
- Debug dataset training produces non-zero object recall and coherent previews.
- Full synthetic baseline is trained and checkpointed.
- Test evaluation is saved.
- Inference script can load `best.pt` and produce point labels/previews.
- Documentation records:
  - dataset path
  - config path
  - experiment path
  - best validation metrics
  - test metrics
  - known failure cases

## Immediate Next Tasks

Implement these in order:

1. Create `processed-data/pointnet2_semseg_k41144_debug4096`.
2. Add train script flags:
   - `--limit-train-samples`
   - `--limit-val-samples`
   - `--disable-augment`
   - `--overfit`
3. Run one-sample overfit on CUDA.
4. Run two-sample overfit on CUDA.
5. Add prediction preview PLY export.
6. Train debug4096 full split for 20 epochs.
7. Profile whether `8192` or `16384` points fit MX150.
8. Run the first real synthetic baseline.
9. Add test evaluation script.
10. Add inference script.

## Resolved Decisions

```text
device: CUDA by default, CPU fallback
gpu: NVIDIA GeForce MX150, 2GB VRAM
torch: 2.12.0+cu126
dataset: synthetic-data/K41144 validated raw data
processed baseline: processed-data/pointnet2_semseg_k41144
debug_num_points: 4096
baseline_num_points: 16384 if memory allows, otherwise 4096 or 8192
input_features: xyz + normal_camera
first task target: semantic segmentation, not pose estimation
```

## Key Risks

### MX150 Memory Limit

Risk:

- Full `16384` point training may OOM.

Mitigation:

- Use `4096` for overfit/debug.
- Profile `8192`.
- Keep `batch_size=1`.
- Reduce `sa_npoints` and `sa_nsamples`.

### Pure PyTorch PointNet++ May Be Slow

Risk:

- FPS/kNN/grouping may be slow compared with CUDA extensions.

Mitigation:

- Keep current implementation for correctness.
- Profile after the model learns.
- Swap only `pointnet2_ops.py` later if needed.

### Model May Collapse To Background

Risk:

- Smoke validation already predicted all-background on one epoch.

Mitigation:

- Do not judge after one epoch.
- Run overfit gates.
- Check class weights.
- Track object IoU and object recall, not only overall accuracy.
- Inspect previews.

### Synthetic-To-Real Gap

Risk:

- Clean synthetic training may not survive real depth noise.

Mitigation:

- Keep raw data clean.
- Use online augmentation for jitter, depth noise, dropout, outliers, and normal jitter.
- Later tune augmentation with real sensor samples.

### Label Or Coordinate Mistakes

Risk:

- Model can train on wrongly aligned points/labels and still produce plausible metrics.

Mitigation:

- Preview ground truth and predictions as colored point clouds.
- Keep coordinate-frame names explicit.
- Verify `depth_m -> points_camera` convention against raw `points_camera` samples.
