# Pose Estimation Backend Implementation Plan

Date: 2026-06-11

Status: implementation started 2026-06-14 (Wave 1). P0 done. Phases P0-P3 can start before or while data generates; P4-P5 need the large dataset.

Wave 1 implementation log:

```text
P0 Contracts And Scaffolding: DONE (2026-06-14)
  src/registry/bundle_schema.py, model_registry.py, __init__.py
  src/service/schemas.py, __init__.py
  configs/backend/service.yaml
  scripts/package_model_bundle.py
  tests/test_bundle_schema.py (6/6), tests/test_model_registry.py (5/5, loads on cpu + LRU)
  Bundles packaged: models/K41144/v1, models/bending_pipe/v1 (gitignored; rebuild via package script)
  Deviations from plan, intentional:
    - Service schemas use dataclasses, not pydantic (no new dependency until the P7 FastAPI layer).
    - Tests are standalone-runnable (python tests/test_*.py) since pytest is not installed; still pytest-compatible.
    - Each checkpoint already embeds its train config, so the bundle loads config from the checkpoint
      instead of storing a separate yaml; the bundle also vendors the STL + sampled model_points/keypoints
      so it is self-contained (no dependency on object-model/ at serve time).
P1 Symmetry productization: DONE (2026-06-14)
  configs/symmetry/k41144.json, bending_pipe.json
  src/training/pose_geometry.py: axis_angle_to_matrix, SymmetryDefinition, load_symmetry,
    resolve_symmetry_matrices, continuous-axis expansion (dense finite approximation)
  src/data/pose_crop_dataset.py: builds symmetry via resolve_symmetry_matrices (config-driven, name alias kept)
  scripts/audit_object_symmetry.py, tests/test_symmetry.py (6/6)
  Gate met: JSON matrices match built-in within 1e-6; K41144 identity baseline ADD 1.8e-6 mm,
    success 1.0, rotation_error 0.0. Audit reproduces docs/k41144-pose-symmetry-audit.md
    (y180 mean 0.133 / p95 0.317 / max 0.847 mm; no continuous axis).
P2 End-to-end inference library: DONE (2026-06-14)
  src/data/pose_crop_dataset.py: extracted build_pose_model_features (single source for the
    pose model input + 18-d aux), __getitem__ refactored to use it (identity baseline unchanged).
  src/inference/pose_pipeline.py: PosePipeline.from_loaded_bundle; infer_from_points (in-memory,
    GT-label or predicted-clustering path) and infer_scene (depth -> unproject -> poses); reuses
    instance model, cluster_voted_centers, build_pose_instance_crops, build_pose_model_features,
    z-flip, and pose_predictions_to_object_to_camera. No file I/O in the call path.
  tests/test_pose_pipeline.py (2/2).
  Verified: K41144 GT-label path on a processed sample yields 30/30 instances, ADD mean 6.78 mm /
    median 3.59 mm (offline GT-crop reference ~5.6 mm) -> golden-consistent; depth->poses predicted
    path runs end to end with finite poses. Note: per-stage timings are returned; CPU instance stage
    is slow (~14 s) as expected, GPU is the deployment target.
P3 Confidence scoring + ranking + top-K metric: DONE (2026-06-14)
  src/inference/confidence.py: ConfidenceConfig (per-bundle tunable), compute_instance_confidence
    (v1 = weighted geometric mean of bounded sub-scores from center-vote dispersion, keypoint
    dispersion, point count, cluster isolation, optional object prob / refinement inliers),
    cluster_isolation, top_k_pick_success (top1 / topk).
  src/inference/pose_pipeline.py: decode returns vote-dispersion diagnostics; infer computes
    isolation + confidence per instance and ranks highest-first; honors options.min_confidence /
    max_instances.
  tests/test_confidence.py (5/5).
  Verified on a K41144 scene: confidence is strongly informative -- ADD of the top-confidence half
    2.93 mm vs bottom half 10.52 mm; top-1 and top-3 pick success = 1 at the 0.1d (10 mm) threshold.
  Follow-up (not blocking): wire --report-topk into eval_pointnet2_pose.py and the suite (the
    metric function exists; pipeline ranking already proves the value).

Wave 1 (P0-P3) COMPLETE. Default device is GPU with CPU fallback.
```

Wave 2 implementation log (started 2026-06-14):

```text
Data prep: DONE
  scripts/prepare_pointnet2_semseg_dataset.py --raw now accepts multiple roots (merged,
  sample names prefixed by root). Merged bending_pipe_4..18 (skipping 12=1 sample, 14=missing)
  = 1218 samples -> processed-data/pointnet2_semseg_bending_pipe_wave2 (train 974/val 122/test 122,
  16384 points). Used --skip-raw-validation (datasets flagged SPARSE = few visible objects, not corrupt).
P4 two-stage clustering: CODE DONE
  src/inference/instance_clustering.py: TwoStageClusteringConfig + refine_instance_labels_two_stage
    (split via 2-means on voted centers > split_separation_fraction*diameter; merge via union-find on
    centroid proximity + point contiguity). Disabled by default; thresholds are diameter fractions.
  src/inference/pose_pipeline.py predicted path applies it when bundle.inference.two_stage_clustering.enabled.
  tests/test_instance_clustering_two_stage.py 4/4 (merge 2->1, split 1->2, disabled, config).
  Instance retrained on the merged set (40 epochs): val_object_iou 0.9997; test semantic object_iou
    0.9997, instance_recall 0.967, instance_precision 0.919, instance_mean_iou 0.959, split_count 1.78
    (mild over-segmentation of long parts), merge_count 0.26.
    Checkpoint: experiments/pointnet2_instance_bending_pipe_wave2_20260614_154448/checkpoints/best.pt.
  Two-stage integrated into eval (scripts/eval_pointnet2_instance_seg.py --two-stage --object-diameter-m).
  FINDING (rollback per plan): on this dense bending_pipe data with the strong retrained model,
    stage-2 merge does not beat tuned stage-1. Loose merge (0.5/0.25) over-merges the dense 26-object
    scenes (clusters 28.6->4.0, recall 0.94->0.25); tight merge (0.15/0.04) is a near no-op
    (split_count 3.9->3.75, recall preserved). So two-stage stays DISABLED by default for bending_pipe;
    the code is kept and config-gated for objects/scenes where stage-1 under/over-segments more.
P5 pose retrain: GT crops exported
  src/inference/pose_bridge.load_raw_metadata resolves merged prefixed names.
  GT crops in experiments/wave2_pose_crops_bending_gt_{train,val,test} (974/122/122 crop files, all with GT pose).
  Pose voting retraining PENDING (waits for the GPU to free after instance training).

Remaining to finish Wave 2 (run after instance training frees the GPU):
  1. Eval instance checkpoint on the test split; run the Phase 21 sweep incl. stage-2 thresholds.
  2. Train pose voting: train_pointnet2_pose.py --config configs/train/pointnet2_pose_voting_bending_pipe.yaml
       --data experiments/wave2_pose_crops_bending_gt_train --val-data .../gt_val --epochs 90 --batch-size 16 --device cuda
  3. Eval pose (GT + predicted crops) against Gate A/B/C; repackage models/bending_pipe/v2 via package_model_bundle.py.
```

This plan turns the current research pipeline into a commercial bin-picking pose-estimation backend. It is grounded in three sources:

1. The current repository state (instance segmentation Phase 0-10, pose Phase 11-23, generator physics fix 2026-06-10).
2. The two reference papers in `arcticle/` (instance-segmentation-based 6D pose estimation; PS6D symmetry-aware voting).
3. The production-readiness review in `docs/generator-physics-and-production-readiness-review.md`.

Read those documents before implementing any phase here.

## Confirmed Architecture Decisions

Decided with the project owner on 2026-06-11:

```text
D1. Delivery form:   Python core library + thin service layer (FastAPI/gRPC).
D2. Output scope:    Ranked poses only: object_to_camera 4x4 + confidence per instance.
                     No hand-eye transform application, no grasp planning.
D3. Multi-SKU:       Per-SKU model bundles managed by a versioned model registry.
                     API carries an explicit sku field so a future multi-object
                     model can be registered as just another bundle.
D4. Retrain flow:    One-command CLI onboarding pipeline (STL -> dataset -> train
                     -> eval -> registry bundle), resumable per stage.
```

Non-goals for this plan:

- Grasp pose generation, gripper modeling, robot communication.
- Multi-object mixed-bin models (PS6D normalized workpiece space). The registry and API leave room for it, but no phase here builds it.
- Cloud training service / API-triggered retraining. CLI only.
- Camera vendor drivers. The backend consumes depth + intrinsics; capture is the caller's job.

## Target Architecture

Runtime flow (one request):

```text
client request {sku, depth_m or points, intrinsics, options}
  -> registry: resolve sku -> model bundle (cached)
  -> unproject depth to scene point cloud (meters, camera frame)
  -> PointNet++ semantic segmentation -> object points
  -> instance clustering (voted centers + merge/split post-processing)
  -> per-instance crops (in memory, no files)
  -> PS6D-style keypoint/center voting pose per crop
  -> optional conservative refinement (hybrid ICP, gated)
  -> confidence scoring + ranking
  -> response {instances: [{object_to_camera, confidence, point_count, diagnostics}], timings}
```

Repository layout additions:

```text
src/
  registry/
    bundle_schema.py        # manifest dataclasses + validation
    model_registry.py       # discover/load/cache bundles
  inference/
    pose_pipeline.py        # end-to-end scene -> ranked poses (library entrypoint)
    confidence.py           # per-instance scoring + ranking
  service/
    schemas.py              # pydantic request/response models
    runtime.py              # bundle cache, device management, GPU work queue
    app.py                  # FastAPI application
scripts/
  audit_object_symmetry.py  # CAD symmetry audit -> symmetry.json proposal
  onboard_sku.py            # one-command SKU pipeline (D4)
  run_backend_service.py    # service launcher
  benchmark_pose_pipeline.py
  prepare_real_eval_set.py  # sim-to-real harness (P8)
configs/
  backend/service.yaml
  sku/<sku>.yaml            # onboarding recipe per SKU
  symmetry/<sku>.json       # finite/continuous symmetry definition
models/                     # registry root (large binaries; not committed)
```

Model bundle layout (registry contract):

```text
models/<sku>/<version>/
  manifest.json             # schema_version, sku, version, created, git_commit,
                            # model_scale, diameter_m, symmetry ref, file list, metric gates
  symmetry.json             # finite rotation list and/or continuous axis
  model_points.npy          # sampled model points in object frame (meters)
  instance_seg.pt           # checkpoint + its train config
  instance_seg.yaml
  pose_voting.pt
  pose_voting.yaml
  refinement.json           # accepted refinement preset (from Phase 20 defaults)
  inference.yaml            # clustering thresholds, num_points, prob threshold
  eval_metrics.json         # gate evidence from the release suite
```

Frames and units (unchanged project contract):

- Meters everywhere; `object_to_camera` is the output transform direction.
- Pipeline input depth is positive-forward camera convention as produced by `src/data/depth_unprojection.py`; pose stage flips z into metadata camera convention exactly as `PoseCropDataset` does today. The pipeline must reuse that single conversion, not reimplement it.

## Phase P0: Contracts And Scaffolding

Needs large dataset: no. Blocks: everything else.

Purpose: freeze the data contracts so later phases do not churn each other.

Deliverables:

```text
src/registry/bundle_schema.py
src/registry/model_registry.py
src/service/schemas.py
configs/backend/service.yaml
tests/test_bundle_schema.py
tests/test_model_registry.py
docs update: this plan marked started; registry contract documented
```

Implementation details:

- `bundle_schema.py`: typed manifest with explicit fields, `schema_version=1`, strict validation that referenced files exist and hashes match (hash check optional flag).
- `model_registry.py`: `list_skus()`, `resolve(sku, version=None) -> Bundle`, `load(bundle, device) -> LoadedBundle` with LRU cache keyed by (sku, version); thread-safe; VRAM budget parameter that evicts least-recently-used bundles.
- `schemas.py`: request = sku, one of {depth_m (base64 npz) + intrinsics, points_camera}, options (max_instances, refine, min_confidence); response = instances list, per-stage timings_ms, model_version, warnings. Error taxonomy: `unknown_sku`, `invalid_input`, `no_object_points`, `no_instances`, `internal`.
- Package the two existing trained bundles (K41144, bending_pipe current checkpoints) into `models/` by hand to make the contract real from day one.

Pass criteria:

```text
Unit tests pass.
Both existing checkpoints load through the registry on cpu and cuda.
Manifest validation rejects a bundle with a missing file and a wrong schema_version.
```

## Phase P1: Symmetry Productization

Needs large dataset: no. Blocks: P5 (pose retraining), P6 (onboarding).

Purpose: remove the last hardcoded per-object logic. New SKU onboarding must not require code edits.

Current state: `symmetry_matrices_for_name` in `src/training/pose_geometry.py` knows only `k41144` and `none`. Symmetry must become data.

Deliverables:

```text
configs/symmetry/k41144.json
configs/symmetry/bending_pipe.json
src/training/pose_geometry.py: load_symmetry(path_or_dict) -> SymmetryDefinition
src/data/pose_crop_dataset.py: accept symmetry definition object
scripts/audit_object_symmetry.py
tests/test_symmetry.py
```

Symmetry definition schema:

```json
{
  "schema_version": 1,
  "finite_rotations": [
    {"axis": [0, 1, 0], "angle_deg": 180}
  ],
  "continuous_axes": [],
  "notes": "K41144 audit 2026-06-07: R_y(180) mean err 0.128 mm"
}
```

Implementation details:

- Finite rotations expand to the matrix list used by existing symmetry-aware ADD/rotation losses and metrics. Identity is always implicit.
- Continuous axes: implement axis-aware rotation error (project rotation difference onto the component orthogonal to the symmetry axis; ADD-S-style distance for model points). Gate behind config so current SKUs are unaffected. This is required before onboarding any revolute part (washer, plain pipe); it is cheap to implement now and expensive to retrofit later.
- Keep `symmetry_name: k41144` working as a deprecated alias that resolves to the JSON file, so existing configs and checkpoints stay valid.
- `audit_object_symmetry.py`: productize the method from `docs/k41144-pose-symmetry-audit.md` - sample STL vertices, test candidate rotations (180-deg around x/y/z, 90/120/60-deg steps, axis detection for near-continuous), report chamfer distances, and write a proposed `symmetry.json` with a `requires_human_confirmation: true` flag. The onboarding pipeline (P6) stops and asks the operator to confirm.

Pass criteria:

```text
Identity baselines unchanged for both SKUs (ADD ~0, success 1.0).
Existing eval suite numbers reproduce within noise using JSON-driven symmetry.
Unit tests cover: finite expansion, continuous-axis rotation error, alias loading.
Audit script on K41144 reproduces the documented R_y(180) finding.
```

## Phase P2: End-To-End Inference Library

Needs large dataset: no (develops against existing checkpoints + synthetic samples). Blocks: P3, P7.

Purpose: the core product. One in-memory call from a depth frame to ranked poses. Today this path only exists as offline file artifacts (`export_pose_instance_crops.py` -> NPZ -> `eval_pointnet2_pose.py`).

Deliverables:

```text
src/inference/pose_pipeline.py
scripts/benchmark_pose_pipeline.py
tests/test_pose_pipeline_golden.py
```

API sketch:

```python
pipeline = PosePipeline.from_registry(registry, sku="K41144", device="cuda")
result = pipeline.infer_scene(
    depth_m=depth,                  # (H, W) float32 meters, or points_camera=(N, 3)
    intrinsics=intrinsics,          # fx, fy, cx, cy, width, height
    options=InferenceOptions(refine=True, max_instances=10),
)
# result.instances: sorted by confidence desc
# result.instances[0].object_to_camera: (4, 4) float64
# result.timings_ms: {"unproject": ..., "semseg": ..., "cluster": ..., "pose": ..., "refine": ...}
```

Implementation details:

- Reuse, do not rewrite: `depth_unprojection.py` (scene points + normals), `pointnet2_semseg_infer.py` (semantic model), `instance_clustering.py` (DBSCAN on voted centers), `pose_refinement.py` (hybrid ICP). Refactor `pose_bridge.py` so crop construction takes arrays, with the file-based path becoming a thin wrapper - the existing export scripts must keep working unchanged.
- Pose-stage frame conversion (z flip) and aux feature computation must be imported from `pose_crop_dataset.py` helpers, not duplicated. If helpers are private, extract them to module functions first.
- All models loaded once at pipeline construction; explicit `warmup()`; no GT or dataset access anywhere in the call path.
- Determinism: fixed seeds for sampling; document that FPS sampling makes results deterministic per (input, seed).
- Golden consistency test: for 3 raw synthetic samples per SKU, pipeline poses (with GT-free path) must match the offline export+eval path within tolerance (ADD difference < 0.5 mm per matched instance, same instance count plus/minus documented clustering nondeterminism). This test is the proof that the backend equals the validated research path.

Pass criteria:

```text
Golden consistency test passes for K41144 and bending_pipe.
benchmark_pose_pipeline.py reports per-stage latency on dev GPU (MX150 baseline numbers recorded; no target enforced yet).
Zero file I/O in the inference call path (verified by test monkeypatching open/np.load).
```

## Phase P3: Confidence Scoring And Ranking

Needs large dataset: no (tunes on existing eval data; recalibrate later on large data). Blocks: P7 release.

Purpose: commercial bin-picking needs the chosen pick to be correct, not the average crop. Current all-predicted ADD_0.1d (K41144 0.540) looks bad, but matched-crop quality (0.738-0.897) means good ranking can ship a usable system.

Deliverables:

```text
src/inference/confidence.py
scripts/eval_pointnet2_pose.py: --report-topk flag
scripts/run_pose_evaluation_suite.py: top-K columns in summary
tests/test_confidence.py
```

Implementation details:

- Per-instance features, all already computable in the pipeline:
  - center-vote dispersion (std of origin votes, normalized by diameter)
  - keypoint vote residual statistics
  - crop point count and semseg mean object probability
  - cluster isolation (distance to nearest other cluster center / diameter)
  - refinement diagnostics when enabled: inlier fraction, alignment residual, accepted flag
- Score v1: hand-tuned monotonic combination (weighted product of bounded terms), fully explainable, defined in `inference.yaml` per bundle so it is tunable without code.
- Score v2 (optional, after large data): logistic calibration fitted on eval crops labeled by ADD_0.1d correctness. Keep v1 as fallback; adopt v2 only if it wins on the suite.
- New metric implemented in the evaluator: per scene, rank instances by confidence and report `top1_pick_success` and `topk_pick_success@3` (highest-confidence instance pose within ADD 0.1d).

Pass criteria:

```text
On current test scenes: top1_pick_success > all-predicted ADD_0.1d by a clear margin
(quantified target: K41144 top-1 >= 0.85 with current models; >= 0.95 after P4/P5 retrain).
Confidence is monotonically informative: decile analysis shows higher confidence -> higher success.
```

## Phase P4: Instance Segmentation Upgrade (Two-Stage Clustering)

Needs large dataset: yes (retraining and threshold sweeps). Algorithm prototyping can start on existing data.

Purpose: instance crop quality is the proven full-pipeline bottleneck (Phase 21 finding: matched 0.738 vs all 0.540 on K41144). PS6D's ablation shows two-stage clustering matters most for elongated parts - both current SKUs are elongated.

Deliverables:

```text
src/inference/instance_clustering.py: stage-2 merge/split post-processing
scripts/sweep_pose_crop_export_params.py: extended to sweep stage-2 thresholds
experiments/phase21_full_<sku>_sweep.json on the large dataset
retrained instance checkpoints per SKU
```

Implementation details:

- Stage 1 stays DBSCAN on voted centers (optionally + embedding distance, already available from the instance model).
- Stage 2 (PS6D-inspired, adapted to our two-network design):
  - merge: clusters whose center distance < alpha * diameter and whose point sets are contiguous (cross-cluster nearest-neighbor distance below threshold) merge; alpha swept.
  - split: clusters whose center-vote distribution is strongly bimodal (k=2 means separation > beta * diameter with both halves above min points) split; beta swept.
- All thresholds expressed relative to object diameter and stored in `inference.yaml` per bundle - no absolute magic numbers, so they transfer across SKUs.
- Retrain instance segmentation on the large dataset with existing scripts and configs; then run the full Phase 21 sweep (no --limit-samples) including stage-2 parameters.

Pass criteria (large-data test split):

```text
K41144: all-predicted crop pose ADD_0.1d >= 0.70 (stretch 0.80); merge+split counts reduced vs stage-1-only baseline at equal recall.
bending_pipe: all-predicted >= 0.80.
Instance recall does not drop more than 0.02 vs stage-1-only.
```

Rollback: if stage 2 cannot beat tuned stage-1 on the sweep, keep it disabled in `inference.yaml` and document; P3 ranking then carries more weight.

## Phase P5: Pose Model Retraining On Large Dataset

Needs large dataset: yes.

Purpose: re-establish pose accuracy with statistically meaningful test sets (current bending_pipe test n=18 is inside noise).

Deliverables:

```text
retrained pose voting checkpoints per SKU (configs unchanged except symmetry now JSON-driven)
updated GT-crop and predicted-crop evaluations
refinement decision re-run (Phase 20 preset vs Phase 22 learned refiner on full data)
```

Gates (test split of the large dataset):

```text
Gate A raw GT-crop:    K41144 ADD_0.1d >= 0.92, translation <= 5.5 mm
                       bending_pipe ADD_0.1d >= 0.90, translation <= 6.0 mm
Gate B refined GT-crop: translation <= 5.0 mm both SKUs, ADD_0.1d not below raw
Gate C full pipeline:  per P4 targets, evaluated through pose_pipeline (not file path)
Decision rule: learned translation refiner is adopted only if it beats the hybrid
ICP preset on both SKUs in the full suite; otherwise it stays experimental.
```

Training discipline unchanged: identity baseline -> tiny overfit -> full train -> suite. All runs into `experiments/` with configs saved.

## Phase P6: SKU Onboarding Pipeline (One Command)

Needs large dataset: no, but should be validated by re-onboarding an existing SKU with the large-data recipe. Depends on: P0, P1, P5 recipes.

Purpose: D4 - "thay doi object, train lai mau, tuning cau hinh" as one auditable command.

Deliverables:

```text
scripts/onboard_sku.py
configs/sku/_template.yaml
docs/sku-onboarding-runbook.md
```

Stage sequence (each stage idempotent, `--resume-from <stage>`, `--stop-after <stage>`):

```text
 1. validate_model     STL loads; unit/scale heuristic (bbox in plausible meter range)
                       with explicit model_scale required in config; report dims/diameter.
 2. symmetry_audit     run audit_object_symmetry.py; STOP for human confirmation of
                       symmetry.json unless --accept-symmetry given.
 3. generator_preset   instantiate generator preset from template: bin sized from object
                       diameter, drop heights, spawn counts; write configs/generator/<sku>.json.
 4. physics_smoke      1 sample + simulation video, explosion watchdog on; STOP for human
                       video check unless --skip-video-confirm.
 5. generate_dataset   N samples (config), strict out-of-bin rejection.
 6. validate_raw       scripts/validate_raw_dataset.py must pass 100%.
 7. prepare_processed  semseg conversion with standard ratios.
 8. train_instance     + eval; gate on instance metrics from config.
 9. export_crops       GT train/val/test + predicted test.
10. train_pose         identity test, tiny overfit smoke, full train, best_add selection.
11. eval_suite         GT + predicted + refined + top-K through pose_pipeline.
12. package_bundle     write models/<sku>/<version>/ with manifest + eval evidence;
                       refuse to package if any gate failed (override flag exists but
                       writes "gates_overridden": true into the manifest).
13. report             experiments/onboard_<sku>_<date>/report.md with all artifacts.
```

Per-SKU config template captures every tunable so "tuning cau hinh" is one file:

```yaml
sku: my_part
model_path: object-model/my_part.stl
model_scale: 0.001
dataset: {samples: 400, objects_per_scene: 25, ...generator overrides}
instance: {epochs, dbscan_eps_fraction, min_cluster_points, ...}
pose: {epochs, batch_size, keypoint_count, ...}
gates: {instance_recall: 0.95, pose_add01d_gt: 0.90, top1_pick: 0.95, ...}
```

Pass criteria:

```text
Re-onboarding bending_pipe from scratch with the large-data recipe completes with at
most two human confirmations (symmetry, physics video) and zero manual file edits.
Failed gate at any stage stops the pipeline with an actionable message.
```

## Phase P7: Service Layer

Needs large dataset: no. Depends on: P0, P2, P3.

Deliverables:

```text
src/service/app.py, runtime.py
scripts/run_backend_service.py
configs/backend/service.yaml
tests/test_service_contract.py
docs/backend-api.md (request/response examples, error codes, client snippet)
```

Implementation details:

- FastAPI v1 endpoints:
  - `POST /v1/poses` - body per P0 schema; depth transported as compressed npz (base64) or raw float32 buffer; response includes model_version used.
  - `GET /v1/skus` - registry listing with versions and gate status.
  - `GET /v1/health` - device, VRAM, loaded bundles, git/build version.
- GPU concurrency: single worker queue serializing GPU inference (one GPU assumption); request timeout; backpressure with 429 when queue depth exceeded. Async I/O for transport, sync GPU section.
- Bundle cache from P0 registry with VRAM budget from `service.yaml`; `POST /v1/poses` for an uncached SKU triggers load (first-call latency documented).
- Observability: structured JSON logs per request (sku, timings, instance count, top confidence), Prometheus-style metrics endpoint (request count, latency histogram, queue depth, per-stage timings).
- Security explicitly out of scope for v1 beyond bind-address config; note for deployment that it must sit on a trusted cell network.

Pass criteria:

```text
Contract tests pass (schema round-trip, every error code reachable).
Sequential load smoke: 100 requests on dev GPU without leak (VRAM stable).
Two SKUs alternating requests work with cache eviction at a 1-bundle budget.
```

## Phase P8: Sim-To-Real Readiness

Needs large dataset: retrain step yes; harness no. Depends on: P2. Can run parallel to P4-P7.

Purpose: everything is currently perfect-raycast synthetic; the K41144 black-metallic material is exactly where real depth cameras fail. This phase prepares the gap measurement and narrows it - it cannot close it without a real camera (open question Q2).

Deliverables:

```text
src/data/augmentation.py: structured-light-style noise extensions
scripts/prepare_real_eval_set.py
docs/real-data-capture-guide.md
noise-augmented retrain comparison in the suite
```

Implementation details:

- Training-time augmentation (processed/dataloader only; raw stays clean per project rules):
  - depth-dependent Gaussian noise and quantization
  - normal-incidence-dependent dropout (grazing surfaces lose points)
  - edge dropout near depth discontinuities (shadowing)
  - random blob dropout (specular holes)
  - parameters in train configs, off by default until validated
- Real eval harness: folder format `real-data/<set>/<frame>/depth.npy + intrinsics.json (+ optional rgb)`; semi-automatic labeling = operator gives a rough init per instance, refined by full-model ICP, stored as `object_to_camera_label` with a quality score; eval reuses existing metrics and the pipeline.
- Generator additions (separate small task): camera pose jitter and bin pose jitter ranges in the preset for domain randomization of the large dataset v2.

Pass criteria:

```text
Noise-augmented training does not regress clean synthetic gates by more than 0.02 ADD_0.1d.
Harness validated end-to-end on at least one captured set when a camera is available;
until then, validated on synthetic frames exported through the same folder format.
```

## Phase P9: Performance And Packaging

Needs large dataset: no. Depends on: P2, P7. Final tuning depends on production GPU decision (open question Q1).

Deliverables:

```text
benchmark report on target GPU (experiments/backend_benchmark_<gpu>.json)
batched crop inference in pose_pipeline (all crops of a scene in one forward)
optional half-precision path with accuracy check
deployment notes (docs/backend-deployment.md): pinned environment, GPU requirements,
  Windows service / Linux container instructions
```

Notes:

- PyTorch3D custom ops make ONNX/TensorRT export non-trivial; do not budget for it in v1. TorchScript or eager PyTorch with batching and fp16 is the realistic v1 optimization path. Revisit only if the latency budget fails on target hardware.
- Provisional budget until Q1 is answered: < 2.0 s/scene end-to-end on an RTX-3060-class GPU at 640x480, 16384 scene points, <= 30 instances. MX150 numbers are recorded as baseline only.

Pass criteria:

```text
Budget met on target hardware, or a written gap analysis with the next optimization step.
fp16/batching produce ADD within 0.2 mm of fp32 single-crop path on the suite.
```

## Phase P10: Release Gate And Regression Suite

Needs large dataset: yes (golden sets come from it). Depends on: all previous.

Deliverables:

```text
scripts/run_backend_release_suite.py (extends run_pose_evaluation_suite.py)
golden datasets: frozen 20-scene slices per SKU under processed-data/golden_<sku>/
docs/backend-release-checklist.md
per-bundle model card written into the registry (auto-generated from eval evidence)
```

Suite contents:

```text
1. py_compile + unit tests
2. identity baselines per SKU
3. GT-crop and predicted-crop pose eval (file path)
4. pipeline-level eval on golden scenes (library path) incl. top-K pick success
5. golden consistency check (file path vs library path)
6. service contract tests against a live local instance
7. performance smoke vs budget
8. summary.md gate table -> release yes/no
```

Release rule: a bundle ships only if the suite passes with its exact manifest; the suite output is copied into the bundle as `eval_metrics.json`.

## Implementation Order And Dependencies

```text
Wave 1 (start now, while large dataset generates):
  P0 -> P1 -> P2 -> P3        (P1 and P2 can overlap after P0)

Wave 2 (when large dataset is validated):
  P4 -> P5                    (P4 prototype can start earlier on small data)

Wave 3 (after Wave 1, parallel to Wave 2):
  P7 (service)  P8 (sim-to-real harness)

Wave 4 (after Waves 2-3):
  P6 (onboarding, validated by re-onboarding bending_pipe)
  P9 (performance on target GPU)
  P10 (release suite, golden sets, first tagged bundles)
```

Rough effort, single engineer, excluding GPU training wall-time:

```text
P0 2-3d   P1 3-5d   P2 5-8d   P3 3-5d   P4 5-10d  P5 3-5d
P6 5-8d   P7 5-8d   P8 5-10d  P9 3-6d   P10 3-5d
Total: roughly 9-13 working weeks.
```

## Testing Matrix (run on every phase)

```text
python -m py_compile <touched files>
pytest tests/ (grows per phase)
identity baselines (both SKUs) after any geometry/symmetry/pipeline change
golden consistency test after any pipeline change
full release suite before adopting any accuracy-affecting change (Decision Rules
from docs/pose-estimation-completion-and-accuracy-plan.md still apply)
```

## Risks And Mitigations

```text
R1 Instance quality stays the bottleneck after P4
   -> P3 ranking ships value anyway; escalate to embedding-aware stage-1 clustering;
      consider pose-aware clustering (PS6D full design) as a research follow-up.
R2 Sim-to-real gap large on black metallic parts
   -> P8 noise augmentation; capture real set early once camera exists; expect a
      per-deployment refinement-threshold tuning step in the runbook.
R3 Symmetry refactor silently changes training targets
   -> P1 gate: byte-identical symmetry matrices for existing SKUs + identity tests.
R4 VRAM pressure with multiple resident bundles
   -> registry LRU + budget; sequential loading documented for small GPUs.
R5 Windows-only assumptions leak into service code
   -> path handling via pathlib, no shell calls in library/service; CI note to test
      on Linux when a Linux box is available.
R6 Statistical noise in gates (small test splits)
   -> all gates evaluated on the large dataset only; report sample counts next to
      every metric; bending_pipe gates invalid until its test split >= 100 crops.
```

## Open Questions (answer before the marked phase)

```text
Q1 Production GPU target and latency budget        -> needed by P9 (provisional: RTX-3060-class, < 2 s/scene)
Q2 Real depth camera model and availability date    -> needed by P8 capture step
Q3 Deployment OS for the service (Windows service vs Linux container) -> needed by P9 packaging
Q4 Max number of SKUs resident in VRAM simultaneously -> needed by P7 cache budget default
```

These do not block Waves 1-2; defaults are stated and will be revisited.
