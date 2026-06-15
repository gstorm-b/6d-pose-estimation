# Dense-Pile Predicted-Path Pose Gap: Diagnosis & Fix

Date: 2026-06-15

## Symptom

The GT-crop pose eval gates releases at ADD ~4.6 mm / ADD_0.1d 0.96, but the full
**predicted** path (predicted instance segmentation + clustering, then pose) on
dense ~40-object `bending_pipe` piles measured far worse: ADD ~29 mm / ADD_0.1d
~0.60 / recall ~0.66 (per Hungarian-matched instance).

## Diagnosis

Reproducible via `scripts/diagnose_dense_pile_gap.py`.

1. **Not segmentation contamination.** 96.9% of predicted clusters are pure
   (>=0.9 of points from one GT object). Merge/contamination is not the cause.
2. **Not a partial-crop / completeness problem alone.** Even pure **and** complete
   crops (>=0.8 of the object's points) only reach ADD_0.1d ~0.53; clean complete
   crops still flip on a large fraction.
3. **Not a symmetry the config missed.** The mesh audit shows `bending_pipe` is
   genuinely asymmetric (nearest pseudo-symmetry x180 is 12.6 mm Chamfer, well
   above tol). An x180-flip oracle does **not** recover the wrong poses, so they
   are not a clean end-for-end flip.
4. **Root cause: train/inference crop-distribution mismatch.** The pose voting
   model was trained on **GT crops** (clean, full object membership) and reaches
   ADD_0.1d 0.96 there, but at inference it sees **predicted-cluster crops** whose
   point distribution differs enough that the orientation is resolved correctly
   only ~60% of the time. An in-memory GT crop reproduces ~4 mm, confirming the
   model and decode are correct; the gap is purely the crop the model is fed.

### The operational reframing

Bin-picking does not need every one of 40 poses correct - the robot picks the
single most accessible/confident object, picks it, and re-images. So the metric
that matters is **top-K pick success**, not mean ADD over all instances.

We measured which per-instance signal separates correct from wrong poses
(no GT needed at inference):

| signal | AUC (correct vs wrong) |
|--------|------------------------|
| **model-fit** (posed model -> crop chamfer / diameter) | **~0.99** |
| point_count | 0.86 |
| confidence (pre-fix) | 0.87 |
| cluster_isolation | 0.78 |
| vote dispersions | ~0.79 |
| purity / completeness | not usable at inference (need GT) |

Model-fit is near-perfect: a wrong pose does not land on its own observed crop
(correct ~0.026 diameter, wrong ~0.81 diameter). It is a pure geometric
self-consistency check - object-agnostic, no GT, no retrain.

## Fix (inference-time, no retrain)

`src/inference/pose_pipeline.py` computes `model_fit` per instance (mean distance
from each crop point to the posed model surface, normalized by diameter, on GPU)
and adds it to the instance diagnostics. `src/inference/confidence.py` adds a
`model_fit` term that **dominates** the confidence (weight 6.0 vs ~3 for the rest;
`model_fit_scale` 0.05), so ranking by confidence ~ ranking by fit. An optional
`InferenceOptions.max_model_fit` gate rejects poses whose model does not fit.

This changes ranking/gating only - never the poses - so it cannot regress the
GT-crop gates, and the per-bundle confidence config can tune the weight/scale.

### Result (rigorous, Hungarian-matched, `prepare_real_eval_set.py evaluate`)

| metric | before | after |
|--------|--------|-------|
| **top-1 pick success** | ~0.70 | **1.0** |
| **top-3 pick success** | ~0.90 | **1.0** |
| per-instance ADD_0.1d | 0.59 | 0.59 (unchanged by design) |

The fit gate (`--max-model-fit 0.05`) additionally turns the raw output into a
high-precision set (precision ~0.88, keeping ~99% of correct poses) for callers
that want every returned pose to be trustworthy, not just the top pick.

## Tried: retrain the pose model on predicted crops (negative result)

To also raise per-instance ADD_0.1d (~0.59) we exported predicted-cluster crops
for train/val/test (`scripts/export_pose_instance_crops.py --source predicted`,
17343/2219/2057 crops) and trained the pose voting model **from scratch** on them
(`experiments/pointnet2_pose_bending_pipe_20260616_003256`). It plateaued at
val ADD_0.1d ~0.54 on predicted crops (best epoch 14). Packaged as a candidate v3
and compared to v2 on the dense-pile diagnostic:

| metric | v2 (GT-trained) | v3 (predicted-trained) |
|--------|-----------------|------------------------|
| raw per-instance precision | **0.58** | 0.42 (worse) |
| fit-gate<0.05 precision / recall | 0.67 / 1.0 | 0.62 / 0.93 (worse) |
| top-1 / top-3 pick success | 1.0 / 1.0 | 1.0 / 1.0 |

**From-scratch training on predicted crops did not help - it hurt.** Predicted
clusters include many split/partial crops whose pose is genuinely ambiguous; as a
training signal they are noisy, and from scratch the model never reaches the
orientation discrimination that clean GT crops teach. v3 was rolled back; **v2 +
the model-fit confidence stays the production config** (it already gives top-1
pick success 1.0). The predicted crops and the diagnostic are kept for reuse.

If raising per-instance precision is revisited, try **finetuning v2** on predicted
crops (preserve the GT orientation prior) or a **GT+predicted mix** rather than
from scratch - both are uncertain and multi-hour on the shared GPU. The model-fit
fix is object-agnostic and already solves the operational metric.
```
