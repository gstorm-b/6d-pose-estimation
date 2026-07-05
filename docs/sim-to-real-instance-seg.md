# Sim-to-Real Instance Segmentation: Gap, Fixes, Workflows

Status 2026-07-05. This documents why the wave-2 instance model failed on real
camera data, what was changed (Phase 0/1/2), and how to run the new workflows.

## The measured gap

Running bundle `bending_pipe:v2` on the real Basler stereo captures
(`real-data/bending`, 1224x1024, mm units, 36-70% valid pixels):

- The semantic head predicted **65-98% of the scene as "object"** - the bin
  floor itself is classified as foreground and DBSCAN partitions it into fake
  instances. Even with a workspace crop to the bin interior the head stays
  saturated (see `real-data/bending_real/seg_report_model.json`).
- On synthetic test data the same bundle is near perfect (object IoU 0.9996).

Root causes, ranked:

1. **Balanced sampling artifact.** Processed datasets forced 65% object points
   per cloud (`object_fraction_target`), oversampling objects with replacement.
   The model learned a "2/3 of points are object" prior plus a density
   signature (objects denser than background) that no real cloud has - and the
   synthetic eval used the same balanced clouds, so the gap was invisible.
2. **Scene content gap.** Every synthetic scene was a dense pile in one bin
   geometry; the first real deployment scene is a few parts lying flat in a
   different tote surrounded by rig clutter.
3. **Sensor noise never shipped.** The P8 structured-light augmentation existed
   in config but the packaged v2 model trained without it; real stereo drops
   30-60% of pixels exactly where it hurts.
4. **Normal-channel mismatch.** v2 trained on renderer normals (whose sign is
   also regionally inconsistent - measured ~53/47 z-sign split); the real path
   feeds open3d-estimated normals.

## What changed

### Phase 0 - inference-side (no retrain)

- `src/data/pcd_loading.py` - single place that converts real camera `.pcd`
  (mm -> m, flip-z, estimated camera-oriented normals, organized-grid pixels).
  The viewer and CLIs share it.
- `src/inference/workspace_crop.py` - bin ROI crop: `bounds` (calibrated box,
  the production path) and `auto_plane` (RANSAC bin floor + dominant floor
  component + eroded hull).
- `src/inference/classical_segmentation.py` - no-learning baseline: floor-plane
  removal, wall erosion, Euclidean clustering, **crest splitting** (touching
  lying parts separate at their crests), instances ranked topmost-first.
  On the sparse real frames it finds all 8 pipes cleanly; on jumbled piles it
  isolates the topmost parts and merges the buried ones (expected - that is the
  learned model's job).
- `scripts/run_instance_segmentation.py` grew `--method classical`,
  `--points scene.pcd`, `--crop auto|bounds`.

### Phase 1 - data + retrain

- `PointNet2ConversionConfig.sampling_mode`: `balanced` (legacy) | `natural`
  (uniform, true class prior) | `voxel` (density-normalized to `voxel_size_m`;
  transfers between camera resolutions). `normals_source: estimated` computes
  training normals from depth exactly like the real path.
- `InstanceSegmenter` gained the matching `scene_subsample: voxel` mode
  (bundle manifest / train-config `inference` keys `scene_subsample`,
  `scene_voxel_size_m`).
- Four sparse/randomized generator presets
  (`configs/generator/bending_pipe_sparse_{grid_flat,grid_yaw,drop5,drop10}.json`):
  flat rows like the real capture, rotated rows, 5-part random drops at 0.75 m
  standoff, 10-part light pile in a bigger shallower bin.
- Wave-3 train config
  `configs/train/pointnet2_instance_bending_pipe_wave3_sim2real.yaml`:
  voxel/estimated dataset + P8 noise ON + random z-rotation.

### Phase 2 - real evaluation harness

- `prepare_real_eval_set.py import-pcd` - organized `.pcd` folder -> real-eval
  frames: `depth.npy` (metres, positive-forward), fitted pinhole
  `intrinsics.json` (round-trip error ~0), optional estimated `normal.npy`,
  `preview.png` from intensity.
- `prepare_real_eval_set.py eval-seg` - segmentation-only metrics per frame
  (foreground fraction, instance count/sizes; semantic IoU + instance P/R when
  a `seg_labels.npy` GT grid exists), `--method model|classical`, `--crop auto`.
  Reports land in `<set>/seg_report_<method>.json`.

## Workflows

```bash
# Import a folder of camera captures once:
python scripts/prepare_real_eval_set.py import-pcd --pcd real-data/bending \
    --pattern "*points_intensity.pcd" --set bending_real --save-normal

# Classical baseline on real frames (no model needed):
python scripts/prepare_real_eval_set.py eval-seg --set real-data/bending_real --method classical

# Learned model on real frames:
python scripts/prepare_real_eval_set.py --device cuda eval-seg \
    --set real-data/bending_real --method model --sku bending_pipe --crop auto

# One-off segmentation of a single capture:
python scripts/run_instance_segmentation.py --method classical \
    --points real-data/bending/<frame>-points_intensity.pcd --crop auto \
    --out out/seg --save-mask

# Rebuild the processed dataset (wave 3):
python scripts/prepare_pointnet2_semseg_dataset.py \
    --raw synthetic-data/bending_pipe_4 ... synthetic-data/bending_pipe_sparse_drop10 \
    --out processed-data/pointnet2_semseg_bending_pipe_wave3 \
    --sampling-mode voxel --voxel-size-m 0.002 --normals-source estimated
```

## Reference numbers (real-data/bending_real, 6 frames)

| method | mean foreground | mean instances | notes |
|---|---|---|---|
| v2 model + auto crop | ~77% | 11 | saturated semantic head - the "before" |
| wave-3 model + auto crop | ~61% | 3-15 | better, still saturated on the tote floor |
| classical baseline | ~17% | 8 | 8/8 pipes on sparse frames; merges piles |
| true object fraction | ~5-15% | 8-? | pipes only |

The wave-3 acceptance bar: model foreground fraction in the true range on these
frames, instance count matching visible parts on the sparse frames, and no
regression on the synthetic natural-density test split.

## Wave-3 result and what it taught us

Wave-3 (voxel sampling, estimated normals, sparse scenes, P8 noise) hit
val_object_iou 0.982 on the honest synthetic split but **still saturated on the
real tote floor**: pipes get probability ~1.0 (recall 100%) yet the floor sits
at p50 0.998 (AUC pipe-vs-rest 0.76), and the center-voting head also merges
real instances that the classical crest split separates. The residual gap is
sensor structure the white-noise model cannot mimic: the tote's molded floor
ribs and the stereo camera's *correlated* ripple, both at part-like scale.

Two responses, both implemented:

1. **Pseudo-label fine-tuning (wave-3b)** - `scripts/build_real_pseudolabel_dataset.py`
   converts classical-labeled sparse real frames into processed samples
   (voxel-sampled copies, estimated normals, `label_preview.png` for review),
   mixed ~200:1166 with wave-3 synthetic; fine-tune with
   `train_pointnet2_instance_seg.py --init-checkpoint <wave3 best>` and
   `configs/train/pointnet2_instance_bending_pipe_wave3b_finetune.yaml`.
   Pile frames 0003/0004 stay held out to measure honestly.
2. **Hybrid gating at inference** - the geometric floor filter already removes
   the surface the semantic head cannot reject; combine the workspace crop /
   floor clearance with the model's votes when running the learned pipeline on
   real scenes.

Longer-term generator work (not yet done): molded rib geometry on the bin
floor and low-frequency correlated depth ripple in the P8 model.

## Wave-3b result (pseudo-label fine-tune) and current recommendation

Fine-tuning (best at epoch 3 of 12; later epochs overfit) taught the semantic
head to reject the real tote floor: on frame 0000 the floor probability fell
from p50 0.998 to **p50 0.001** while pipes stayed at p50 0.978 (AUC 0.76 ->
0.85). Remaining false positives sit on walls / rim / rib highlights, which the
geometric gate removes anyway. The **center-voting head did not transfer**: on
real frames its votes still merge neighbouring pipes that classical crest
splitting separates cleanly (offsets are accurate on synthetic - val_offset_l1
~0.0067 - but real correlated ripple defeats the vote clustering).

**Recommended production pipeline today (bending_pipe on the Basler cell):**

1. Workspace crop (`bounds` calibrated per cell, or `auto_plane`).
2. Geometric candidate mask: floor clearance + eroded floor hull.
3. Instance splitting: classical crest-split clustering (`--method classical`),
   ranked topmost-first -> pickable candidates for the pose stage.
4. Optional model gate: wave-3b semantic probability as an extra filter on the
   candidates (rejects cables/debris that are not on the trained SKU's shape).

The learned voting path stays the goal for dense piles; next iteration needs
(a) ribbed/molded bin floors + correlated low-frequency depth noise in the
generator, (b) more real captures pseudo-labeled with the classical stage
(sparse scenes are self-labeling), and (c) possibly a higher offset-loss weight
on real samples so the vote head adapts, not just the semantic head.
