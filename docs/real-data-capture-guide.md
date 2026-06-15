# Real-Data Capture & Evaluation Guide (P8)

Everything trained so far is perfect-raycast synthetic. Real structured-light /
stereo depth cameras fail exactly where the metallic K41144 / bending_pipe parts
are hardest: specular dropout, grazing-angle noise, depth quantization, and edge
shadowing. This guide is how we *measure* that sim-to-real gap and feed real
frames through the same pipeline and metrics as synthetic data.

It pairs with two pieces of code:
- `src/data/augmentation.py` - structured-light noise added to training (narrows the gap).
- `scripts/prepare_real_eval_set.py` - the capture/label/evaluate harness (measures the gap).

## Folder format

A real-eval *set* is a folder of frames, camera-agnostic:

```text
<set>/<frame>/depth.npy        # (H, W) float32 depth in metres, 0 = invalid
<set>/<frame>/intrinsics.json  # {fx, fy, cx, cy, width, height}
<set>/<frame>/normal.npy       # optional (H, W, 3) camera-frame normals
<set>/<frame>/labels.json      # GT poses (from `label` or synthetic export)
<set>/<frame>/init_labels.json # operator rough inits (input to `label`)
```

`labels.json` / `init_labels.json`:

```json
{ "sku": "bending_pipe", "source": "icp_refined_init",
  "instances": [ { "instance_id": 0,
                   "object_to_camera": [[r,r,r,tx],[r,r,r,ty],[r,r,r,tz],[0,0,0,1]],
                   "quality": 0.84 } ] }
```

`object_to_camera` is a 4x4 rigid object->camera transform in **metres** (unit
rotation - not the model-scaled matrix found in raw synthetic metadata; the
synthetic exporter rigidifies it for you).

## Capturing real frames

1. Mount the depth camera over the bin at the production standoff (~0.5-0.8 m).
2. Save each frame as `depth.npy` (metres; convert from mm by `/1000`) plus
   `intrinsics.json`. If the SDK gives per-pixel normals, save `normal.npy`;
   otherwise the pipeline runs xyz-only (slightly weaker - normals help rotation).
3. Capture variety: fill level, lighting, part finish, and a few near-empty bins.
   20-50 frames is enough for a first gap measurement.

## Labeling (semi-automatic)

For each instance an operator gives a rough init pose (eyeball, or copy a similar
frame), stored in `init_labels.json`. Then ICP refines it against the visible
crop and writes `labels.json` with a quality score:

```bash
python scripts/prepare_real_eval_set.py label --set real-data/<set> --sku bending_pipe
```

`quality = 1 / (1 + alignment_m / diameter)` (1.0 = perfect surface alignment).
Drop or re-init any instance with quality below ~0.6 before trusting its metrics.

## Evaluating

```bash
python scripts/prepare_real_eval_set.py evaluate --set real-data/<set> --sku bending_pipe
python scripts/prepare_real_eval_set.py evaluate --set real-data/<set> --sku bending_pipe --refine
```

Runs the registry pipeline on every frame, optimally matches predictions to GT
(Hungarian, gated at 0.3 diameter), and reports symmetry-aware ADD, translation,
ADD_0.1d, and recall to `<set>/eval_report.json` plus per-frame `predictions.json`.
`--refine` adds an ICP pass against each crop.

## Validate the harness before a camera exists

Export synthetic raw samples into the exact same format (labels = visible GT,
rigidified) and evaluate them - this exercises the whole harness end to end:

```bash
python scripts/prepare_real_eval_set.py export-synthetic \
    --raw synthetic-data/bending_pipe_4 --out real-data --set bending_synth \
    --sku bending_pipe --limit 10 --save-normal
python scripts/prepare_real_eval_set.py evaluate --set real-data/bending_synth --sku bending_pipe
```

### Known gap: GT crops vs the full predicted pipeline

On dense ~40-object piles the **full predicted path** (predicted instance
segmentation + clustering, then pose) measures markedly worse than the GT-crop
pose eval that gates releases: ADD ~29 mm / ADD_0.1d ~0.60 / recall ~0.66 vs the
GT-crop ADD 4.6 mm / ADD_0.1d 0.96. The pose model is sound (GT crops reproduce
4 mm in-memory); the loss is segmentation quality on dense piles (occluded /
merged parts). Track this end-to-end number - not just the GT-crop gate - as the
production-relevant metric, and improve it via clustering, the learned pose
refiner, or pick-the-top-confidence-only picking rather than every instance.

## Narrowing the gap with sim-to-real noise

`configs/train/pointnet2_instance_bending_pipe_sim2real.yaml` switches on the
structured-light augmentation (along-ray range^2 noise + quantization, grazing
dropout, edge dropout, specular blobs). Retrain with it and compare to the clean
model on the **clean** synthetic gates first:

```bash
python scripts/train_pointnet2_instance_seg.py --config \
    configs/train/pointnet2_instance_bending_pipe_sim2real.yaml --out experiments/<run>
```

Pass criterion: clean-gate ADD_0.1d must not regress by more than 0.02. Then keep
whichever model wins on the held-out **real** set. Until a camera exists the noise
parameters are physically-motivated guesses; recalibrate them against real frames
(match the dropout fraction and depth-noise std to what the camera actually does).
