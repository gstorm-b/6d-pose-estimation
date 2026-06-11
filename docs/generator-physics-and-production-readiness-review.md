# Generator Physics Diagnosis And Production Readiness Review

Date: 2026-06-10

Status: physics root cause verified by headless Blender probe; fix package A+B+C+D approved and implemented in the same session (see Implementation Status at the end). Production review is current as of pose Phase 23.

This document records two reviews requested by the project owner:

1. Diagnosis of the drop-simulation bug in `scripts/generate_synthetic_blender.py` where spawned objects sometimes shoot away at high speed in a non-gravity direction.
2. A production-readiness review of the implemented plan and models, treating the project as a commercial bin-picking backend for symmetric and asymmetric objects rather than an experiment.

Future sessions should read this file together with `docs/codex-session-bootstrap.md` before touching the generator or planning production work.

## Part 1: Drop-Simulation Physics Diagnosis

### Symptom

When dropping objects into the bin, most drops look physically correct, but some objects suddenly move at very high speed in a direction unrelated to gravity and exit the bin area. The behavior is random across attempts.

### Verified Root Cause

The root cause chain was verified with a headless Blender 4.5 probe (`%TEMP%\blender_physics_probe.py`):

1. **Pending objects are invisible static colliders.** The delayed-activation mechanism keyframes `rigid_body.enabled=False / kinematic=True` before each object's `spawn_frame`. In Blender/Bullet a disabled active body is not removed from the collision world; it becomes a kinematic zero-mass body, i.e. an immovable collider. Probe test 1: a falling ball came to rest on top of a disabled hovering cube (`ball_z=1.699`) instead of falling through.
2. **All objects were placed at their drop positions at frame 1.** The drop volume was therefore full of invisible static obstacles for the whole simulation.
3. **Spawn spacing could not prevent overlap.** Measured bounding radii from the recentered bbox center: K41144 `0.0479 m`, bending_pipe `0.0687 m`. Two randomly rotated copies need up to `2r` separation (`~0.096 m` / `~0.137 m`), while `spawn_min_distance` defaulted to `0.045 m`, was only enforced within one spawn layer, and silently fell back to a best-effort candidate when unsatisfiable. Cross-layer placements had no XY constraint at all, with vertical layer separation well below `2r`. Without `--drop-height-min/max` the CLI default put every layer at the same height.
4. **Interpenetration at activation causes ballistic ejection.** When an object switches to dynamic while overlapping a pending static collider, Bullet's penetration recovery applies a huge impulse along the contact normal. Probe test 2: an active body spawned overlapping a disabled body was ejected at `38.16 m/s`. Whether a given attempt explodes depends on random rotations and positions, which matches the intermittent symptom.
5. **Amplifiers.** `collision_margin=0.00002 m` (0.02 mm) is far below what Bullet handles robustly: at impact speed ~2.6 m/s an object moves ~5.4 mm per substep (20 substeps at 24 fps), so contacts were detected deep and ejected objects could tunnel through the 12 mm bin walls. `linear_damping=0.35` and `angular_damping=0.45` also made normal falls unrealistically slow and floaty.

A visible side artifact of the same cause: falling objects could rest in mid-air on an invisible pending body until that body activated.

### Fix Design (implemented)

- **A. Park pending objects outside the play volume.** Pending objects wait at a parking slot far outside the bin. They are keyframe-teleported to their sampled drop position `PARK_STATIC_LEAD_FRAMES=3` frames before activation and held static, so the kinematic-to-dynamic handoff velocity is zero and no pending collider ever sits inside the drop volume. Location keyframes use CONSTANT interpolation so the parked body never glides across the scene.
- **B. Hard non-overlap placement for coexisting spawns.** Whenever objects activate together (batch mode `spawn_settle_frames=0`, or the first objects whose spawn frame is too early to park), drop positions enforce a hard 3D separation of `2 * bounding_radius + max(collision_margin, 0.0005)` against all coexisting drop positions. If the bin footprint cannot satisfy the separation, the candidate is lifted upward in steps instead of silently accepting an overlapping position. Lift events are recorded per object (`spawn_z_lift_steps`) and per sample.
- **C. Realistic solver parameters.** New defaults: `collision_margin 0.0005 m` (0.5 mm; invisible at dataset resolution, stable for Bullet), `substeps_per_frame 60`, `solver_iterations 30`, `linear_damping 0.05`, `angular_damping 0.15`. New CLI flags: `--object-mass`, `--object-linear-damping`, `--object-angular-damping`, `--physics-substeps`, `--physics-solver-iterations`.
- **D. Explosion watchdog.** Every simulated frame, each activated object's speed is estimated from its position delta. If speed exceeds the limit (`--explosion-speed-limit`; default auto = `max(4.0, 1.5 * sqrt(2 g h_max))`, `0` disables) or the object leaves the vertical sanity band, the attempt is aborted immediately with `reason="physics_explosion"` plus the object name, frame, speed, and location in the rejection stats. This turns silent late `out_of_bin` failures into early, named rejections.
- **E. (Deferred) Collision-shape fidelity.** CONVEX_HULL fills concave regions (significant for bending_pipe), biasing pile poses. Future option: offline convex decomposition (V-HACD) into compound shapes. Not an explosion cause; intentionally not part of this fix.

Edge notes preserved from the review: `object_spawn_edge_margin` still caps at `0.45 * min(bin)` so extremely long parts could in principle clip a wall; the watchdog now converts that case into a named early rejection instead of a silent corrupt sample.

### Validation Plan

- `--samples 1 --record-simulation-video --simulation-video-frame-step 1` per object to inspect drops visually.
- Small smoke datasets in delayed mode and batch mode; expect zero `physics_explosion` and zero `out_of_bin` rejections, then `scripts/validate_raw_dataset.py` must pass.

### Follow-up: Progressive (PyBullet-Style) Spawn Mode (2026-06-11)

After the A+B+C+D fix, a 30-object `bending_pipe` run (drop 0.2-0.8 m, bin 0.42x0.32x0.2) still failed repeatedly with `out_of_bin` (no `physics_explosion`): objects rolled over the shallow wall at sub-watchdog speed because every pending object was teleported to a fixed 0.2-0.8 m height regardless of pile height, and the strict any-object-out reject made 8/8 attempts fail.

The user asked whether Blender can spawn objects one at a time during simulation like PyBullet. Blender has no `stepSimulation()`+`addBody()`, but a headless probe (`%TEMP%\blender_min_probe.py`) verified the equivalent **progressive baking**: settle one active object, freeze it as a PASSIVE collider at its settled pose (re-bake drift measured 0.000 mm), drop the next object just above the pile top, re-bake; a final all-active relaxation lets the pile self-adjust.

Implemented as `--spawn-mode progressive` (now the default; `delayed` and `batch` keep the legacy behavior):

- Each object is dropped from `pile_top + random(drop_clearance_min, drop_clearance_max)`, so impact energy stays low and objects rarely roll out. New flags `--drop-clearance-min/--drop-clearance-max` (default 0.03/0.12 m), `--progressive-settle-frames` (45), `--final-relax-frames` (60).
- Only one active rigid body per stage, so there are no mid-air collisions and the dynamics stay cheap; settled objects become static colliders.
- The explosion watchdog runs per stage; its speed limit derives from the drop clearance, not the absolute pile height.
- Critical implementation detail: baked rigid-body results live on the **evaluated** object, not the basis `matrix_world`. Freezing, the watchdog, the settled-pose log, and the video recorder all read `obj.evaluated_get(depsgraph).matrix_world`; reading the basis returns the spawn pose and the object appears not to fall.
- Drop height must include the bounding radius: the mesh is recentered on its bbox center, so the origin is dropped at `pile_top + bounding_radius + clearance`; otherwise a large part (bending_pipe radius 68.7 mm) spawns interpenetrating the floor and the watchdog ejects it.

Failure investigation: `--record-failed-video` writes an MP4 of every rejected attempt into `<output>/rejected/sample_XXXXXX_attempt_YY_<reason>.mp4`, so out-of-bin/explosion causes can be inspected visually.

### Center-Based Out-Of-Bin Policy (2026-06-11)

With progressive baking, 30-object bending_pipe physics was correct (all 30 fall, settle, no explosion, no ejection), but the sample was still rejected. Diagnostics (`out_of_bin_detail` in the rejection stats) showed the rejected objects had their **center inside the bin** while their bbox merely poked past the wall by 1.9-20.2 mm: a long part lying at an angle and leaning on the wall. The old bbox-based check (in both `find_out_of_bin_objects` and the validator `_check_out_of_bin`) treated that as out of bin.

Policy changed to **center-based** (`out_of_bin_check: world_center_xy`): an object is out of bin only when its recentered bbox center leaves the bin footprint (plus `out_of_bin_tolerance`), or its lowest point drops through the floor. A long part poking past the wall while centered inside is a valid bin pose for bin-picking. The generator and validator use the same rule, so existing K41144 (77/77) and bending_pipe (30/30) datasets still validate, and the verified 30-object progressive bending_pipe sample passes with 0 out-of-bin.

### Progressive Mode Validation (2026-06-11)

```text
30-object bending_pipe, bin 0.42x0.32x0.2, progressive, drop-clearance 0.02-0.06,
progressive-settle 35, final-relax 60:
  accepted attempt 1, 0 rejected, 30/30 objects in bin, 14063 visible points, validation OK
6-object K41144 progressive + success video: continuous 251-frame timeline, video written
6-object K41144 legacy delayed mode: still works, validation OK
Existing datasets re-validated with center-based rule: K41144 77/77 OK, bending_pipe 30/30 OK
```

Verified preset: `configs/generator/bending_pipe_progressive.json`. The older `bending_pipe_active_spawn_stable.json` is kept as a delayed-mode reference.

## Part 2: Production Readiness Review

### Verdict

The architecture and methodology are correct for both symmetric and asymmetric objects: point-cloud-only input, instance segmentation followed by per-instance PS6D-style keypoint/center voting, symmetry applied only in loss/metrics while raw `object_to_camera` is preserved for grasping. Experiment hygiene (immutable raw data, identity baselines, tiny-overfit gates, raw-vs-refined separation, evaluation suite) is strong and transfers to production. However, the current state is a validated research pipeline, not yet a deployable backend. The gaps are production layers that do not exist yet, not a wrong direction.

### What Is Already Right

- Point-cloud-only, two-stage pipeline matches both reference papers for low-texture industrial parts.
- The pivot from direct pose regression to keypoint/center voting after Phase 14 (`ADD 27.7 mm -> 5.6 mm` on K41144 GT crops) was evidence-driven and correct.
- Symmetry contract is production-correct: raw pose outputs untouched, symmetry only in loss/metric (`K41144 {I, R_y(pi)}`, `bending_pipe {I}`), so downstream grasp planning keeps full information.
- Conservative gated ICP refinement reported separately from raw network pose.

### Gaps Toward A Commercial Backend (by severity)

1. **No real-data path.** Training and evaluation are fully synthetic with perfect raycast depth. The K41144 material is black metallic, which is exactly where real structured-light/ToF depth fails (missing data, edge fattening, interreflections). Needed: sensor-noise/dropout augmentation, a small real test set (50-100 scenes, semi-automatic ICP labeling), and a calibration story (hand-eye, real intrinsics).
2. **No end-to-end scene-to-poses service.** Pose currently runs only through offline artifacts: processed dataset -> `export_pose_instance_crops.py` NPZ -> `eval_pointnet2_pose.py`. `src/inference/pose_bridge.py` is dataset-coupled. A backend needs one module: depth frame + intrinsics -> ranked poses (semseg -> clustering -> crops -> voting -> optional refinement, in memory, models loaded once), with measured latency on target hardware (MX150 2GB is a dev GPU, not a deployment spec) and an ONNX/TensorRT path.
3. **No confidence score or candidate ranking.** Current metrics optimize mean ADD over all crops (all-predicted ADD_0.1d: K41144 0.540, bending_pipe 0.714 - far below a commercial >= 0.95 pick bar). Real bin-picking needs the chosen pick to be correct: per-instance confidence (center-vote dispersion, cluster size, refinement residual, visibility proxy) and a top-1/top-K pick-success metric. With matched-crop quality already at 0.738/0.923, good ranking is likely the fastest route to a usable system.
4. **Instance segmentation is the proven bottleneck** (Phase 21: matched 0.738 vs all 0.540 on K41144). PS6D two-stage clustering (stage 1 center+rotation votes, stage 2 merge), which the paper shows matters most for elongated parts like both current SKUs, is planned but not implemented. This is the critical path together with re-running the full Phase 21 sweep on larger data.
5. **Symmetry is hardcoded.** `symmetry_matrices_for_name` knows only `k41144` and `none`; new SKU onboarding is a code change. Needed: config-driven finite symmetry sets, continuous/infinite symmetry support (revolute parts; PS6D handles via axis-aware loss), and a productized CAD symmetry audit (generalize `docs/k41144-pose-symmetry-audit.md`). Per-SKU models are commercially acceptable; treat them as a model-registry concern.
6. **Dataset scale and statistical power.** 77 + 30 raw scenes; bending_pipe test has 18 crops (one sample = 5.6% success swing), so current gate conclusions sit inside noise. Production needs thousands of scenes with domain randomization. The generator infrastructure exists - which is why the physics fix in Part 1 is a prerequisite to scaling data.
7. **Translation target.** Raw translation misses the strict 5 mm target (6.0/9.5 mm); refined sits at ~5.3 mm. Re-derive the requirement from the actual gripper/cell spec instead of chasing a generic 5 mm; many cells tolerate 5-8 mm with ICP.

### Recommended Roadmap (in order)

1. Fix generator physics (done in this session) -> scale synthetic data 10-50x with sensor-noise model and domain randomization.
2. Retrain instance segmentation + pose on the larger data; implement PS6D two-stage clustering; run full Phase 21/23 without `--limit-samples`.
3. Build `src/inference/pose_pipeline.py` end to end (frame -> ranked poses) with confidence scoring and a top-pick success metric; measure latency.
4. First real-data milestone (any industrial depth camera, 50-100 scenes) to measure the sim-to-real gap and tune augmentation.
5. Productize symmetry (config-driven + continuous symmetry + automated CAD audit) when SKU number 3 arrives.

## Implementation Status

Implemented on 2026-06-10 in `scripts/generate_synthetic_blender.py` (fix package A+B+C+D), with matching updates to `scripts/dataset_gui.py`, `configs/generator/bending_pipe_active_spawn_stable.json`, and the generator documentation.

Behavior changes:

- Pending delayed-activation objects now wait parked outside the bin and teleport to their drop position 3 frames before activation (CONSTANT-interpolated location keyframes; zero handoff velocity).
- Coexisting spawns (batch mode, or spawn frames too early to park) enforce a hard 3D separation of `2 * bounding_radius + max(collision_margin, 0.0005)` with upward lift fallback instead of silent overlap.
- New defaults: `collision_margin 0.0005` (was `0.00002`), `substeps_per_frame 60` (was 20), `linear_damping 0.05` (was 0.35), `angular_damping 0.15` (was 0.45).
- New CLI flags: `--object-mass`, `--object-linear-damping`, `--object-angular-damping`, `--physics-substeps`, `--physics-solver-iterations`, `--explosion-speed-limit`.
- New rejection reason `physics_explosion` with object/frame/speed details; new per-instance metadata fields `spawn_parked`, `spawn_z_lift_steps`; new settings fields recording the effective physics configuration.
- GUI exposes linear/angular damping and uses the new collision-margin default; preset JSON updated accordingly.

Validation results are recorded at the end of this section.

### Smoke Validation (2026-06-10)

Delayed mode, `synthetic-data/K41144_physics_fix_smoke_delayed` (10 objects, spawn_settle_frames=12, 2 samples):

```text
accepted_samples=2, rejected_attempts=0
parked_spawns=9 per attempt, auto speed_limit=4.00 m/s
raw validation: 2/2 OK, visible_objects min/median/max = 10/10/10, out_of_bin_samples=0
recorded settings: collision_margin=0.0005, object_linear_damping=0.05, physics_substeps=60
```

Batch mode, `synthetic-data/K41144_physics_fix_smoke_batch` (10 objects, spawn_settle_frames=0, 2 samples):

```text
accepted_samples=2, rejected_attempts=0
separation sampling active: spawn_z_lift_steps up to 2 on crowded spawns, auto speed_limit adjusted to 4.25/4.08 m/s
raw validation: 2/2 OK, visible_objects min/median/max = 10/10/10, out_of_bin_samples=0
```

Watchdog negative test (artificial `--explosion-speed-limit 0.5`, below normal fall speed):

```text
attempt rejected at frame 3 with reason=physics_explosion
diagnostics included object name, frame, speed 0.616 m/s vs limit 0.5, location, z limits
after max attempts the run failed loudly with RuntimeError as designed
```

Both smoke datasets are throwaway artifacts for this fix and must not be used for training.

### Known Pre-Existing Log Artifact

Blender prints `rigidbody_get_shape_convexhull_from_mesh: no vertices to define Convex Hull collision shape with` twice per sample. Verified by re-running the pre-fix generator: the message appears there too, so it is unrelated to this change. A dedicated probe confirmed hidden+disabled rigid bodies still receive working collision shapes (a falling body rests on them), and all generated samples pass raw validation, so the message is cosmetic for this pipeline. Worth a deeper look only if future Blender versions change pending-body behavior.
