# Session Summary: Synthetic Data For 6D Pose / Bin Picking

This file summarizes the current project context so another Codex session can continue without re-discovering the same decisions.

## Project Goal

Build a 6D pose estimation / robotic bin-picking pipeline from scratch for the industrial object:

```text
object-model/K41144.stl
```

Initial direction:

- Generate synthetic raw data with Blender.
- Use point cloud as the main signal, following the paper in `arcticle/`.
- Keep raw data rich enough to support several later training paths:
  - PointNet / PointNet++ segmentation.
  - PPRNet++-style point-wise pose regression.
  - YOLO segmentation from RGB + instance masks.
  - Later pose estimation and ICP/refinement experiments.

## Session Bootstrap

For future Codex sessions, first read:

```text
docs/codex-session-bootstrap.md
```

That file lists the minimum project context, required engineering rules, relevant files by task type, and the expected senior-engineer workflow for this repository.

## Paper Notes

The article in `arcticle/` was read and summarized here:

```text
arcticle/instance-segmentation-based-6d-pose-estimation-for-bin-picking.md
```

Main idea from the paper:

- Use point cloud instead of RGB for low-texture industrial objects.
- Split the task into instance segmentation and 6D pose estimation.
- Generate synthetic scenes with physics simulation to avoid hand-labeling real point clouds.
- Use depth/normal features and ICP refinement.

## Object Model

The primary object model is:

```text
object-model/K41144.stl
```

A second design/test model is also available:

```text
object-model/bending_pipe.stl
```

Observed STL bounding box:

```text
approx size: 30 mm x 93.6 mm x 20 mm
```

`K41144.stl` is in real metric scale. `bending_pipe.stl` is authored in millimeters and must be generated with:

```text
--model-scale 0.001
```

Important issue found:

- The STL origin is near one end of the part, not at the bbox center.
- Blender rigid body uses the object origin as the rigid body's center of mass.
- Without correction, the object behaves like a pendulum and tends to settle upright.

Generator fix:

- Recenter mesh vertices around bbox center before rigid-body simulation.
- Store transform metadata so the exported `object_to_world` and `object_to_camera` still refer to the original STL coordinate frame.

## Current Scripts

### Blender Raw Data Generator

Main script:

```text
scripts/generate_synthetic_blender.py
```

Purpose:

- Load the requested STL model.
- Use `--class-name` for semantic metadata and Blender object names. If omitted, class name defaults to the STL filename stem.
- Use `--model-scale` to convert STL vertex units before simulation while preserving original-model pose metadata.
- Recenter mesh for correct physics.
- Create a bin/container.
- Spawn many copies of the object.
- Run Blender rigid-body simulation. `--spawn-mode` selects how objects enter the scene: `progressive` (default), `grid`, `delayed`, or `batch`.
- `grid` (tidy rows/columns): place objects at a fixed orientation in a row-major grid sized to the bin footprint, then settle per layer. Objects beyond one layer's capacity stack into further layers (each layer placed just above the settled pile below it). Orientation defaults to auto-flat (the object's smallest dimension points down) and can be overridden with `--grid-orientation` euler degrees. `--grid-jitter` adds optional small position/angle noise for variety while staying tidy. Reuses the per-layer settle window (`--progressive-settle-frames`), `--final-relax-frames`, and the freeze toggle. Use this mode to generate orderly, structured scenes instead of random piles.
- `progressive` (PyBullet-style, recommended): drop one object at a time from `pile_top + bounding_radius + random(drop_clearance_min, drop_clearance_max)`, settle it, freeze it as a passive collider at its settled pose, then drop the next; a final all-active relaxation lets the pile self-adjust. Only one active body simulates per stage, so there are no mid-air collisions and impact energy stays low, which nearly eliminates out-of-bin ejections. Baked poses are read from the evaluated object, not basis `matrix_world`.
- Progressive freeze can be disabled with `--no-progressive-freeze`: settled objects stay ACTIVE and the whole pile re-settles each stage, so an object whose settle window was too short is never frozen mid-fall and left floating. This is slower (all active bodies re-simulate per stage) but produces tighter piles. Measured example (12 settle frames, no final relax, K41144): freeze on pile top 67.1 mm vs off 57.7 mm.
- `delayed` (legacy `--spawn-settle-frames > 0`): each object's rigid body stays disabled until its scheduled `spawn_frame`. Pending objects wait parked far outside the bin and are keyframe-teleported to their drop position 3 frames before activation, because disabled rigid bodies remain static colliders in Bullet.
- `batch`/`delayed`: when spawns coexist, drop positions enforce a hard 3D separation of `2 * bounding_radius + max(collision_margin, 0.0005)` and lift candidates upward instead of silently accepting overlaps.
- A physics explosion watchdog estimates per-object speed every frame and rejects the attempt early with `reason="physics_explosion"` when any activated object exceeds the speed limit or leaves the vertical sanity band.
- Out-of-bin uses a center-based test (`out_of_bin_check: world_center_xy`): an object is out of bin only when its recentered bbox center leaves the bin footprint, or its lowest point drops through the floor. Long parts leaning on the wall with their center inside are valid. The validator uses the same rule.
- `--record-failed-video` writes an MP4 of every rejected attempt into `<output>/rejected/` for failure investigation.
- Use an object-aware spawn edge margin and a best-candidate fallback when the requested per-layer spawn distance cannot be fully satisfied inside the bin.
- Render RGB visual reference from the RGB camera.
- Optionally render `spawn_simulation.mp4` from a dedicated debug camera so generator parameters can be tuned by watching objects fall and settle.
- Raycast from the depth camera to export depth, masks, normals, point cloud and labels.
- Store depth-camera and RGB-camera intrinsics/extrinsics separately while keeping legacy `camera_intrinsics`, `camera_to_world`, `world_to_camera`, and `object_to_camera` aliases for the depth camera.
- Export object pose ground truth.
- Reject/retry bad samples if any object leaves the bin or too few objects/points are visible.
- Keep the static Blender scene loaded once per generator run; per attempt only dynamic object copies are spawned and deleted.

Recommended generation command:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --samples 100 --objects 30 --width 320 --height 240 --bin-wall-height 0.14 --drop-height-min 0.12 --drop-height-max 0.34 --spawn-strategy layered --objects-per-layer 6 --spawn-min-distance 0.045 --collision-margin 0.0005 --min-visible-objects 12 --min-visible-points 8000 --max-sample-attempts 12 --settle-frames 260
```

Important generator options:

- `--model`: STL model path. Default remains `object-model/K41144.stl`.
- `--class-name`: semantic class name recorded in metadata. Defaults to the STL filename stem.
- `--model-scale`: scale applied to STL vertices before simulation. Use `0.001` for millimeter-authored STL files such as `bending_pipe.stl`.
- `--depth-camera-location`, `--depth-camera-target`, `--depth-camera-lens`: camera used for depth, masks, normals, point cloud, and pose labels.
- `--rgb-camera-location`, `--rgb-camera-target`, `--rgb-camera-lens`: camera used for `rgb.png`.
- `--debug-camera-location`, `--debug-camera-target`, `--debug-camera-lens`: camera used for simulation preview videos. It does not affect depth, masks, RGB, point cloud labels, or pose metadata.
- `--light-location`, `--light-energy`, `--light-size`: area-light controls.
- `--settings-file`: load a JSON generator preset before applying explicit CLI overrides.
- `--export-settings`: write the effective generator settings to a JSON preset.
- `--samples`: number of accepted samples to write.
- `--objects`: number of object copies to spawn before filtering.
- `--spawn-strategy layered`: spreads initial objects into height bands to avoid explosive overlaps.
- `--objects-per-layer`: number of objects per initial height band.
- `--spawn-settle-frames`: default `35`. If greater than zero, each object receives a scheduled `spawn_frame` separated by this many frames. Objects are hidden and rigid-body-disabled before their spawn frame, then become active; earlier objects remain active, so the pile can keep adjusting while later objects fall. Use `0` only when intentionally reverting to batch-active behavior for comparison.
- `--spawn-mode`: `progressive` (default), `grid`, `delayed`, or `batch`. See the simulation notes above.
- `--grid-spacing`: grid mode gap between objects in a row/column, in meters. Default `0.005`.
- `--grid-drop-clearance`: grid mode height each object/layer is placed above the floor/pile before settling. Default `0.005`.
- `--grid-jitter`: grid mode random position (m) and angle (rad) jitter for variety. Default `0.0` (perfectly tidy).
- `--grid-orientation`: grid mode object orientation as XYZ euler degrees. Omit for auto-flat (smallest dimension down).
- `--grid-layers`: grid mode number of stacked layers. `0` = auto (enough layers for all objects).
- `--drop-clearance-min`, `--drop-clearance-max`: progressive mode drop height above the current pile top, in meters. Defaults `0.03`/`0.12`. The bounding radius is added automatically.
- `--progressive-settle-frames`: progressive mode frames per object to fall and settle. Default `45`. Raise it if objects freeze before they settle.
- `--final-relax-frames`: progressive mode final all-active relaxation frames. Default `60`.
- `--progressive-freeze` / `--no-progressive-freeze`: freeze each settled object as a static collider (default, faster) or keep all objects active so the pile keeps re-settling (slower, avoids mid-fall freezing / floating).
- `--drop-height-min`, `--drop-height-max`: random absolute spawn height range for `delayed`/`batch` modes only.
- `--record-failed-video`: also render an MP4 of each rejected attempt into `<output>/rejected/`.
- `--collision-margin`: rigid-body margin in meters. Current recommended value is `0.0005` (0.5 mm). The old `0.00002` value was below what Bullet handles robustly and contributed to deep-penetration impulses and wall tunneling.
- `--object-restitution`: rigid-body bounce/restitution for spawned objects. Lower values reduce rebounds.
- `--object-mass`: rigid-body mass in kg. Default `0.08`.
- `--object-friction`: object rigid-body friction. Default `0.85`. Lower values (try `0.2`-`0.4`) let objects slide into gaps for a denser pile.
- `--object-linear-damping`: default `0.05`. The old hard-coded `0.35` made free fall unrealistically slow.
- `--object-angular-damping`: default `0.15` (old hard-coded value was `0.45`).
- `--bin-friction`, `--bin-restitution`: friction/restitution of the bin floor and walls. Defaults `0.9` / `0.05`.
- `--gravity`: world gravity along z. Default `-9.81`.
- `--physics-substeps`: rigid-body world substeps per frame. Default `60` (old hard-coded value was `20`); higher values reduce penetration depth and tunneling.
- `--physics-solver-iterations`: constraint solver iterations. Default `30`.
- `--explosion-speed-limit`: watchdog speed limit in m/s. Omit for an automatic limit `max(4.0, 1.5 * sqrt(2 g h_max))`; `0` disables the watchdog.
- `--bin-floor-thickness`, `--bin-wall-thickness`: bin geometry in meters. Defaults `0.01` / `0.012`.
- `--object-color`, `--object-metallic`, `--object-roughness`: object material. Defaults `(0.015,0.014,0.013)` / `0.85` / `0.85`.
- `--bin-color`, `--bin-roughness`: bin material. Defaults `(0.12,0.12,0.115)` / `0.82`.
- `--world-color`: world background color. Default `(0.018,0.018,0.02)`.
- `--light-type`: `AREA` (default), `SUN`, `POINT`, or `SPOT`.
- `--camera-sensor-width`, `--camera-clip-start`, `--camera-clip-end`: shared camera optics. Defaults `32.0` / `0.01` / `1.50`.
- `--cycles-samples`, `--view-transform`, `--view-look`, `--view-exposure`, `--view-gamma`: Cycles render and color management. Defaults `48` / `Filmic` / `Medium High Contrast` / `-1.2` / `1.0`.
- `--out-of-bin-tolerance`: tolerance around bin x/y bounds for the post-physics center check.
- `--out-of-bin-min-z`: lowest world z an object may reach before it counts as dropped through the floor. Default `-0.04`.
- `--allow-out-of-bin-filtering`: optional legacy/debug behavior that hides out-of-bin objects and accepts the remaining scene if visibility thresholds pass. Do not use it for training datasets. Default behavior is stricter: reject the whole attempt.
- `--min-visible-objects`: reject samples with fewer visible instances.
- `--min-visible-points`: reject samples with too few object points.
- `--max-sample-attempts`: retries before failing a sample.
- `--record-simulation-video`: render `spawn_simulation.mp4` for accepted samples using the debug camera. For tuning, prefer `--samples 1 --simulation-video-frame-step 1` so fast falling motion is not skipped. The preview temporarily uses a bright object material and hides bin walls during video render only; raw RGB, depth, masks, point clouds, and metadata are written before that preview-only override.

### Raw Dataset Validator

CLI script:

```text
scripts/validate_raw_dataset.py
```

Purpose:

- Validate required raw sample files and `sensor_data.npz` schema.
- Check instance ids across metadata, masks, and point labels.
- Check sparse samples and out-of-bin objects using metadata poses and STL bbox.
- Fail datasets generated with legacy out-of-bin filtering unless `--allow-legacy-filtering` is passed.

Recommended command:

```powershell
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\K41144_out_of_bin_reject_test2
```

### Open3D Point Cloud Viewer

Viewer script:

```text
scripts/view_point_cloud_open3d.py
```

Purpose:

- Read `sensor_data.npz`.
- Display `points_camera` or `points_world`.
- Color points by `point_instance_ids`.
- Optionally export a colored `.ply`.

Example:

```powershell
python .\scripts\view_point_cloud_open3d.py .\synthetic-data\K41144\sample_000001
```

Useful options:

```powershell
--world
--voxel-size 0.001
--save-ply output.ply
--no-window
```

### Augmented Point Cloud Viewer

GUI script:

```text
scripts/noise_tuning_gui.py
```

Purpose:

- Tune the `dataset.augment` YAML block visually against processed samples without editing YAML by hand.
- Load a processed dataset root plus a train config, then preview clean/noisy point clouds in the VTK `3D Point Cloud` tab with displacement and label-change stats.
- The 3D view supports `side-by-side` clean/noisy comparison and `overlay` mode; the older 2D projection tab is retained as a quick orthographic check.
- Read `dataset.root`, `dataset.normalize`, and `dataset.augment`; preview uses the same `src/data/augmentation.py` path as training. With `scene_center`, the GUI previews the normalized point cloud and supplies camera position in the normalized frame for camera-dependent sim-to-real noise.
- Save either an augment-only preset YAML or a full train config copy; in-place update is available but YAML comments may not be preserved.

Example:

```powershell
python .\scripts\noise_tuning_gui.py
```

Viewer script:

```text
scripts/view_augmented_point_cloud.py
```

Purpose:

- Review how train-time noise/augmentation changes a processed point cloud, since augmentation only runs inside the DataLoader and is not visible anywhere else.
- Apply the exact same `src/data/augmentation.py` used in training (xyz jitter, depth noise, point dropout, outliers, normal jitter, optional z-rotation) to a processed sample.
- Show the clean cloud and the noisy cloud side by side in Open3D, colored by semantic or instance, with optional red highlighting of changed points.
- Print stats: percent of points moved, mean/max displacement, semantic labels changed, object-to-background flips.

Example:

```powershell
python .\scripts\view_augmented_point_cloud.py .\processed-data\pointnet2_semseg_k41144\test\sample_000011.npz --config .\configs\train\pointnet2_semseg_k41144.yaml
```

Useful options:

```powershell
--point-dropout-prob 0.1 --outlier-ratio 0.01 --depth-noise-std 0.001   # override the config / try values
--highlight-changes                                                       # tint moved/relabeled points red
--save-ply-clean clean.ply --save-ply-noisy noisy.ply --no-window         # export instead of opening a window
```

Input is a processed dataset `.npz` (the array the DataLoader actually loads), not a raw `sensor_data.npz`. With `--config` it reads the YAML `augment` block; CLI flags override it, and any override auto-enables augmentation.

### PySide6/VTK PLY Viewer

Viewer script:

```text
scripts/view_ply_pyside_vtk.py
```

Purpose:

- Open individual `.ply` point-cloud files.
- Browse a folder of `.ply` files.
- Render colored point clouds with VTK in a PySide6 app.
- Use Trackball Camera interaction like the dataset GUI: left-drag rotate, middle-drag pan, right-drag or wheel zoom.
- Adjust point size and reset camera/interaction from the toolbar.
- Show point count, color availability, unique color count, bounds, center, and extent.

Run command:

```powershell
python .\scripts\view_ply_pyside_vtk.py .\experiments\pointnet2_semseg_k41144_20260606_180212\inference\processed_sample_000011\prediction.ply
```

### PySide6 Dataset GUI

GUI script:

```text
scripts/dataset_gui.py
```

Purpose:

- Run the Blender generator from a local desktop UI.
- Expose generator `--class-name`, `--model-scale`, depth-camera pose, RGB-camera pose, debug-camera pose for simulation videos, light controls, spawn settle frames, object restitution, and simulation-video controls in the Generation group so non-K41144 models and different sensor/physics setups can be generated without editing the script.
- Provide a `1 Sample + Video` action that forces one accepted sample, enables simulation video recording, and uses video frame step 1 for parameter tuning.
- Import/export generator presets as JSON from the Generation group. Preset keys match the generator CLI setting names, so the same file can be reused with `--settings-file`.
- Stream Blender generation logs while the process runs.
- Open any raw dataset folder and list `sample_xxxxxx` samples.
- Preview `rgb.png`, `label_overlay.png`, `depth_preview.png`, `instance_mask.png`, and `normal_camera.png`.
- Adjust preview layout with 1, 2, 4, or 5 simultaneous image views, configurable view size, zoom, and drag-to-pan image navigation.
- View object point clouds in a VTK-backed `3D View` tab when the optional `vtk` dependency is installed. The VTK widget explicitly uses Trackball Camera interaction: left-drag rotate, middle-drag pan, right-drag or wheel zoom.
- Show basic metadata stats per sample.
- Run built-in raw dataset validation for required files, NPZ schema, instance ids, sparse samples, and out-of-bin object checks.
- Keep the UI responsive by running dataset loading, sample preview loading, point-cloud loading, and validation in `QThread` workers. Blender generation uses `QProcess` signals.

Run command:

```powershell
python .\scripts\dataset_gui.py
```

GUI dependencies are listed in:

```text
requirements-gui.txt
```

The GUI does not manually edit raw samples. It creates new output folders through the existing generator CLI and reads existing datasets for inspection and validation.

## Raw Dataset Structure

Current raw dataset root:

```text
synthetic-data/K41144/
```

Each sample folder has:

```text
sample_xxxxxx/
  sensor_data.npz
  metadata.json
  rgb.png
  label_overlay.png
  depth_preview.png
  instance_mask.png
  normal_camera.png
  depth_preview.pgm
  instance_mask.ppm
  normal_camera.ppm
```

`rgb.png` is visual reference only. The current material uses black metallic K41144 objects.

`label_overlay.png` is for quick visual QA.

## `sensor_data.npz`

Each sample contains:

```text
depth_m              (H, W)       float32
instance_mask        (H, W)       uint16
normal_world         (H, W, 3)    float32
normal_camera        (H, W, 3)    float32
points_camera        (N, 3)       float32
points_world         (N, 3)       float32
point_instance_ids   (N,)         uint16
point_semantic_ids   (N,)         uint16
point_pixels         (N, 2)       uint16
```

Conventions:

- `instance_mask == 0` means background.
- `point_instance_ids` are visible object instance ids.
- `point_semantic_ids == 1` means class `K41144`.
- Camera point coordinates are in a positive-forward convention stored by the generator.

## `metadata.json`

Top-level metadata contains:

```text
model_path
class_name
class_id
camera_intrinsics
camera_to_world
world_to_camera
settings
object_count
visible_object_point_count
instances[]
files
```

Each instance contains:

```text
instance_id
name
class_id
collision_shape
collision_margin_m
spawn_layer
spawn_height
original_model_center_offset
original_model_to_body
body_to_world
object_to_world
object_to_camera
```

Pose convention:

- `body_to_world`: pose of the recentered physics body.
- `original_model_to_body`: transform from the original STL frame to the recentered body frame.
- `object_to_world`: pose of the original STL coordinate frame in world coordinates.
- `object_to_camera`: pose of the original STL coordinate frame in camera coordinates.

## Issues Found And Fixed

### Object Settled Upright

Symptom:

- Objects tended to stand upright after falling.

Cause:

- STL origin was not near the center of mass.
- Blender rigid body used that origin as the rigid body's center of mass.

Fix:

- Recenter mesh for physics.
- Preserve original STL frame in metadata.

### Unrealistic Object Gap

Symptom:

- Objects could appear to have a large collision gap.

Cause:

- Blender rigid body default `collision_margin` is about `0.04`.

Fix:

- Set `collision_margin = 0.00002`.
- Later correction (2026-06-10): `0.00002` was too small for stable Bullet contacts and contributed to deep-penetration impulses and wall tunneling. The recommended value is now `0.0005` (0.5 mm), which is still invisible at dataset resolution.

### Objects Ejected At High Speed During Drop

Symptom:

- Some spawned objects suddenly moved at very high speed in a direction unrelated to gravity and left the bin area. Behavior was random across attempts. Falling objects could also rest in mid-air.

Cause (verified by headless Blender probes, 2026-06-10):

- Delayed-activation pending objects (`rigid_body.enabled=False`, hidden) remain in the Bullet world as invisible static colliders, and all objects were placed at their drop positions at frame 1.
- Spawn spacing could not prevent overlap: `spawn_min_distance` was enforced only within one layer with a silent best-effort fallback, while two randomly rotated copies need up to `2 * bounding_radius` (K41144 `~0.096 m`, bending_pipe `~0.137 m`).
- An object activating while interpenetrating a pending static collider is ejected by Bullet penetration recovery (probe measured `38 m/s`).
- The tiny collision margin and 20 substeps amplified deep penetration and wall tunneling; `linear_damping 0.35` also made normal falls look slow.

Fix (see `docs/generator-physics-and-production-readiness-review.md` for full details):

- Park pending objects far outside the bin; teleport to the drop position 3 frames before activation with CONSTANT-interpolated location keyframes (zero handoff velocity).
- Enforce hard 3D drop-position separation with upward lift fallback whenever spawns coexist.
- New physics defaults: `collision_margin 0.0005`, `substeps 60`, `linear_damping 0.05`, `angular_damping 0.15`.
- Physics explosion watchdog rejects attempts early with `reason="physics_explosion"` and full diagnostics.

Validation outputs:

```text
synthetic-data/K41144_physics_fix_smoke_delayed: 2/2 OK, 0 rejected attempts, parked_spawns=9
synthetic-data/K41144_physics_fix_smoke_batch:   2/2 OK, 0 rejected attempts, z-lift active
```

Known pre-existing log artifact: Blender prints `rigidbody_get_shape_convexhull_from_mesh: no vertices to define Convex Hull collision shape with` twice per sample. This appears with the old generator code as well, does not affect physics correctness (hidden disabled bodies still collide; samples validate OK), and is unrelated to this fix.

### Too Many Objects Left The Bin

Symptom:

- A 100-sample dataset generated with 30 spawned objects had many sparse samples.
- Median final visible object count was about 10.
- Some samples had only 2-4 visible objects.

Cause:

- 30 objects were spawned in a way that caused strong overlap/collision impulses.
- The old inside-bin filter was too permissive.

Fix:

- Add `--spawn-strategy layered`.
- Add `--spawn-settle-frames` for sequential active spawning: each object is enabled after a delay, while previously spawned objects remain active rigid bodies.
- Add object-aware spawn edge margin and best-candidate XY sampling, so long objects such as `bending_pipe` do not spawn too close to the bin wall and impossible `spawn_min_distance` values do not fall back to arbitrary random positions.
- Add `--object-restitution` so bounce can be reduced for difficult object/bin combinations.
- Add stricter inside-bin filtering using `bin_x / 2 + out_of_bin_tolerance`.
- Add reject/retry thresholds.
- Later update: replace silent filtering with post-physics world-space bbox rejection. If any object leaves the bin, reject the whole attempt by default.
- Keep bin/camera/light/STL template loaded once per generator run; only dynamic K41144 copies are deleted and regenerated per attempt.

Validation output after the fix:

```text
synthetic-data/K41144_generator_validation/
```

Observed validation samples:

```text
sample_000000: 30 objects, 31051 object points
sample_000001: 29 objects, 29973 object points
sample_000002: 30 objects, 30539 object points
```

Validation output after strict out-of-bin rejection:

```text
synthetic-data/K41144_out_of_bin_reject_test2/
```

Observed validation samples:

```text
sample_000000: 30 objects, 30 visible instances, 30979 object points
sample_000001: 30 objects, 30 visible instances, 31552 object points
sample_000002: 30 objects, 30 visible instances, 30993 object points
```

Observed rejections:

```text
2 rejected attempts, both reason=out_of_bin with one object outside the bin.
```

### RGB Was White / Overexposed

Symptom:

- `rgb.png` was nearly all white.

Fix:

- K41144 material changed to black metallic.
- Use Filmic color management.
- Reduce exposure and light intensity.

Validation output:

```text
synthetic-data/K41144_rgb_material_test/
```

Observed test:

```text
12 / 12 instances visible
14038 object points
RGB shows black metallic objects
```

## Dataset Status

The older `synthetic-data/K41144/` 100-sample dataset was audited:

- File/schema consistency is clean.
- Metadata and point labels match.
- Collision margin is correctly set.
- Main issue: many sparse scenes due to old spawn strategy.

Use the old dataset only for smoke tests.

For real raw-data generation, regenerate with the latest generator command in this file.

Additional bending_pipe dataset created for instance-segmentation design checks:

```text
raw: synthetic-data/bending_pipe
processed: processed-data/pointnet2_semseg_bending_pipe
config: configs/train/pointnet2_semseg_bending_pipe.yaml
model: object-model/bending_pipe.stl
model_scale: 0.001
```

Generation profile:

```text
samples=30
objects=6
bin_x=0.36
bin_y=0.36
bin_wall_height=0.18
drop_height_min=0.08
drop_height_max=0.18
spawn_strategy=layered
objects_per_layer=3
spawn_min_distance=0.08
collision_margin=0.00002
min_visible_objects=3
min_visible_points=1500
settle_frames=180
```

Validation result:

```text
raw validation: 30/30 OK
visible objects: min=5, median=6, max=6
visible points: min=8777, median=10725.5, max=11625
processed split: train/val/test = 24/3/3
processed points/sample = 16384
sampled object/background = 10650/5734
instance audit: pass
instances = 178
instances below 64 points = 0
instances below 32 points = 0
```

K41144 and bending_pipe should be treated as separate single-class datasets. Do not mix them unless a future multi-class experiment explicitly requires it.

Recent bending_pipe generator finding:

- `synthetic-data/generator_preset_bending_pipe.json` uses `objects=30`, `objects_per_layer=7`, `spawn_min_distance=0.25`, `drop_height_max=1.0`, and a `0.42 x 0.32 m` bin.
- That preset is physically aggressive for `bending_pipe`: the requested per-layer distance is larger than the practical spawn region after edge margin, the drop height creates high-energy impacts, and 30 long curved parts create high reject rates.
- A smoke run with the corrected active sequential spawn and a safer 12-object profile passed validation:

```text
output: synthetic-data/bending_pipe_active_spawn_stable_12obj
objects=12
drop_height_min=0.18
drop_height_max=0.45
objects_per_layer=4
spawn_min_distance=0.09
spawn_settle_frames=30
settle_frames=240
validation: OK
visible_objects=12
out_of_bin_samples=0
```

For production-scale bending_pipe generation, prefer increasing object count gradually from this profile, raising `max_sample_attempts`, and only using 30 objects after a validation run shows an acceptable reject rate.

## Training Dataset Direction

Raw data should remain in `synthetic-data/`.

Detailed PointNet++ implementation plan:

```text
docs/pointnetplusplus-implementation-plan.md
```

Processed training-ready data should be generated later into a separate folder, for example:

```text
processed-data/
  pointnet_k41144/
  pprnet_k41144/
  yolo_k41144/
```

### For PointNet / PointNet++

PointNet-style training data should contain fixed-size sampled point clouds:

```text
points              (num_points, 3 or 6)
semantic_labels     (num_points,)
instance_labels     (num_points,)
```

Recommended MVP:

```text
points = points_camera
semantic_labels = point_semantic_ids
instance_labels = point_instance_ids
```

Implemented MVP converter:

```text
scripts/prepare_pointnet2_semseg_dataset.py
src/data/depth_unprojection.py
src/data/pointnet2_processing.py
src/data/augmentation.py
src/data/pointnet2_dataset.py
configs/train/pointnet2_semseg_k41144.yaml
```

Recommended conversion command:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\K41144 --out .\processed-data\pointnet2_semseg_k41144 --config .\configs\train\pointnet2_semseg_k41144.yaml
```

Current processed dataset:

```text
processed-data/pointnet2_semseg_k41144/
```

Observed stats:

```text
77 samples
train/val/test = 62/8/7
16384 sampled points per sample
10650 object points and 5734 background points per sample
```

Important correction:

- Semantic segmentation should reconstruct full-scene points from `depth_m` and label them with `instance_mask`.
- Do not train PointNet++ semantic segmentation directly from raw `points_camera`, because those raw arrays contain object-only points.

To include normals:

- Use `point_pixels` to index into `normal_camera[v, u]`.
- Concatenate `points_camera` and per-point normals to get `(N, 6)`.

In the processed PointNet++ MVP, normals are stored as a separate `features` array and the dataset loader exposes `model_input = xyz + features` when normals are enabled.

### PointNet++ Semantic Training MVP

Implemented files:

```text
scripts/train_pointnet2_semseg.py
src/models/pointnet2_ops.py
src/models/pointnet2_semseg.py
src/training/config.py
src/training/metrics.py
src/training/checkpoint.py
src/training/seed.py
```

Current environment:

```text
torch 2.12.0+cu126
pytorch3d 0.7.9
pyyaml 6.0.3
tqdm 4.67.3
scikit-learn 1.8.0
matplotlib already installed
```

CUDA is available in the installed PyTorch build:

```text
GPU: NVIDIA GeForce MX150
compute capability: (6, 1)
torch CUDA runtime: 12.6
torch arch list includes: sm_61
```

The MX150 has only 2GB VRAM, so start with `batch_size=1` and reduced/debug point counts when changing the model. Full 16384-point training fits with the current `pytorch3d` ops backend.

Runtime smoke tests passed:

- PyTorch import.
- `PointNet2SemSegDataset` loads processed smoke data.
- PointNet++ forward pass returns `(B, N, 2)` logits on CUDA.
- One-epoch CPU smoke training on `processed-data/pointnet2_semseg_k41144_smoke_config`.
- One-epoch CUDA smoke training on `processed-data/pointnet2_semseg_k41144_smoke_config`.

CPU smoke experiment:

```text
experiments/pointnet2_semseg_k41144_20260530_151203/
```

CUDA smoke experiment:

```text
experiments/pointnet2_semseg_k41144_20260530_153222/
```

Observed CUDA smoke metrics:

```text
train_loss=0.7583
val_loss=0.6951
val_miou=0.1750
val_object_recall=0.0000
```

This only verifies the end-to-end training path; it is not a quality baseline.

Full synthetic baseline:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_180212
dataset: processed-data/pointnet2_semseg_k41144
points per sample: 16384
epochs: 20
batch_size: 1
device: cuda
runtime: about 21.6 minutes on NVIDIA GeForce MX150
```

Best validation epoch:

```text
epoch=19
val_loss=0.00499
val_mean_iou=0.9961
val_object_iou=0.9973
val_object_recall=0.9984
```

Final epoch:

```text
epoch=20
val_loss=0.00518
val_mean_iou=0.9948
val_object_iou=0.9964
val_object_recall=0.9969
```

Validation previews:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
output: experiments/pointnet2_semseg_k41144_20260606_180212/previews/val
sample_count: 8
overall_accuracy=0.9982
mean_iou=0.9960
object_iou=0.9972
object_precision=0.9990
object_recall=0.9982
confusion_matrix=[[45789,83],[155,85045]]
```

This is the first meaningful PointNet++ semantic segmentation baseline. It passes the synthetic validation target by a wide margin, but PLY previews still need human visual inspection and test evaluation before sign-off.

Held-out test evaluation:

```text
script: scripts/eval_pointnet2_semseg.py
checkpoint: experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
data: processed-data/pointnet2_semseg_k41144
split: test
output: experiments/pointnet2_semseg_k41144_20260606_180212/test_metrics.json
sample_count: 7
device: cuda
ops_backend: pytorch3d
seed: 7
```

Test metrics:

```text
loss=0.0075
overall_accuracy=0.9972
mean_iou=0.9939
object_iou=0.9957
object_precision=0.9991
object_recall=0.9966
confusion_matrix=[[40074,64],[254,74296]]
```

Test previews:

```text
output: experiments/pointnet2_semseg_k41144_20260606_180212/previews/test
sample_count: 7
files: 7 gt PLY + 7 pred PLY + 7 error PLY + preview_summary.json
```

This passes the first held-out synthetic test gate. The remaining semantic-model sign-off item is human visual inspection of the PLY previews.

PointNet++ ops optimization:

```text
default ops backend: pytorch3d
fallback ops backend: pure_torch
config: configs/train/pointnet2_semseg_k41144.yaml
profile output: experiments/pointnet2_profile_pytorch3d_backend/
```

PyTorch3D backend check:

```text
pytorch3d._C available
CUDA forward/backward with pytorch3d backend: pass
train smoke with config backend=pytorch3d: pass
```

Profiler result on `processed-data/pointnet2_semseg_k41144`, 16384 points, batch size 1:

```text
pure_torch mean_seconds_per_batch=0.8340
pytorch3d mean_seconds_per_batch=0.1143
speedup: about 7.3x for the profiled training batch
```

PyTorch3D batch-size probe:

```text
batch_size=2 mean_seconds_per_batch=0.1872, max_reserved_mib=288.0
batch_size=4 mean_seconds_per_batch=0.4654, max_reserved_mib=572.0
```

Interpretation:

- Use `pytorch3d` for normal training on the current machine.
- Batch size 2 or 4 now fits on MX150 for the profiled config.
- Keep `pure_torch` for debugging and machines without PyTorch3D.
- Do not build a custom extension yet; tune batch size, DataLoader, and metric synchronization first.

PointNet++ inference API:

```text
module: src/inference/pointnet2_semseg_infer.py
script: scripts/infer_pointnet2_semseg.py
checkpoint: experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
ops_backend: pytorch3d
```

Processed sample inference smoke:

```text
sample: processed-data/pointnet2_semseg_k41144/test/sample_000011.npz
output: experiments/pointnet2_semseg_k41144_20260606_180212/inference/processed_sample_000011
points: 16384
predicted_object_points: 10632
ground_truth_object_points: 10650
elapsed_seconds: 0.4877
synthetic_debug_accuracy: 0.9976
files: prediction.npz, summary.json, prediction.ply
```

Raw sample inference smoke:

```text
sample: synthetic-data/K41144/sample_000011
output: experiments/pointnet2_semseg_k41144_20260606_180212/inference/raw_sample_000011
points: 16384
predicted_object_points: 6403
ground_truth_object_points: 6440
elapsed_seconds: 0.5222
synthetic_debug_accuracy: 0.9964
files: prediction.npz, summary.json, prediction.ply
```

Raw inference reconstructs points from `depth_m` and `camera_intrinsics`, samples without using labels, and uses `normal_camera` features when required. Synthetic masks are copied only for debug metrics.

### For PPRNet++-Style Training

Do not overwrite raw format. Create a converter later.

Potential training fields:

```text
points
semantic_labels
instance_labels
centroid_offsets
rotation_labels
visibility_labels
object_poses
instance_ids
```

The current raw dataset contains enough information to derive these fields.

### For YOLO Segmentation

The current raw data can be converted to YOLO segmentation format:

- Use `rgb.png` as input image.
- Use `instance_mask` from `sensor_data.npz` to create polygon masks.
- Since there is only one class, YOLO class id should be `0`.

Possible converter:

```text
scripts/export_yolo_segmentation.py
```

## 2026-06-30 K41144 Wave 2 Status

New K41144 raw roots were processed:

```text
synthetic-data/k41144_1 ... synthetic-data/k41144_8
```

Validation summary:

```text
k41144_1: 200/200 OK
k41144_2: 200 SPARSE samples, not corrupt; 10 visible objects, no out-of-bin
k41144_3: 200/200 OK
k41144_4: 200/200 OK
k41144_5: 107/107 OK
k41144_6: 200/200 OK
k41144_7: 200/200 OK
k41144_8: 87/88 OK; sample_000087 missing required raw files
```

`scripts/prepare_pointnet2_semseg_dataset.py` now has an opt-in `--skip-invalid-samples`
flag for trusted multi-root merges. It preserves strict default behavior but skipped
`synthetic-data/k41144_8/sample_000087` for this run and recorded the skip in
`conversion_config.json`.

Processed dataset:

```text
processed-data/pointnet2_semseg_k41144_wave2
samples: 1394
train/val/test: 1115/139/140
visible_instances min/median/max: 3/10/50
```

Instance checkpoint and test metrics:

```text
checkpoint: experiments/pointnet2_instance_k41144_wave2_20260630_231537/checkpoints/best.pt
training: 3 epochs, batch_size=4, device=cuda
test object_iou: 0.9932
test instance_precision: 0.8676
test instance_recall: 0.8054
test instance_mean_iou: 0.8600
```

Pose crop exports:

```text
GT train: experiments/wave2_pose_crops_k41144_gt_train, 18579 crops
GT val:   experiments/wave2_pose_crops_k41144_gt_val, 2251 crops
GT test:  experiments/wave2_pose_crops_k41144_gt_test, 2495 crops
Pred test: experiments/wave2_pose_crops_k41144_pred_test, 2181 crops
Pred matched_gt_iou mean: 0.7182
Pred matched_gt_iou >= 0.5: 1774/2181
```

Pose checkpoint and metrics:

```text
checkpoint: experiments/pointnet2_pose_k41144_20260630_233852/checkpoints/best_add.pt
source: 1-epoch fine-tune from experiments/pointnet2_pose_k41144_20260608_023605/checkpoints/best_add.pt

GT test old checkpoint:       ADD 4.406 mm, translation 4.434 mm, ADD_0.1d 0.973
GT test fine-tuned checkpoint ADD 3.099 mm, translation 3.199 mm, ADD_0.1d 0.961

Pred test IoU>=0.5: ADD 6.096 mm, translation 5.730 mm, ADD_0.1d 0.849
Pred test all:      ADD 10.817 mm, translation 11.280 mm, ADD_0.1d 0.732
```

Interpretation:

- The fine-tuned pose checkpoint improves GT-crop ADD and translation but slightly lowers GT ADD_0.1d versus the old checkpoint.
- Full predicted-crop performance is limited by crop quality; dense 50-object scenes remain difficult for the 3-epoch instance checkpoint.
- Do not promote K41144 v2 yet. Treat this as a provisional training/evaluation run.

Backlog:

1. Run longer K41144 Wave 2 instance training/sign-off and tune DBSCAN/min-cluster settings.
2. Run longer pose fine-tune and compare best ADD vs best ADD_0.1d selection.
3. Run optional hybrid refinement on Wave 2 GT and predicted cases, reported separately.
4. Run Phase 21 predicted-crop sweep on `pointnet2_semseg_k41144_wave2`.
5. Package K41144 bundle v2 only after full-pipeline metrics improve stably.

## Recommended Next Todo

1. Visually inspect validation/test/inference PLY previews.
2. Record visible failure cases, if any.
3. For K41144 Wave 2, complete the backlog above before promoting a v2 bundle.
4. When using real camera data, run the real-eval harness and revisit sim-to-real augmentation.

## 2026-07-01 Real-Depth Gap Note

User-provided real reference currently found as `real-data/scene_001.pcd`
(`real scene` folder was not present). It is a binary PCD with 459708 points and
mm-like coordinates; scale by `0.001` before feeding real-eval code that expects
metres. Quick stats after scaling:

```text
scene extent: 0.296 x 0.372 x 0.130 m
xy NN distance p50/p90/p99: 0.59 / 0.96 / 1.12 mm
local 8-neighbor z-gap p50/p90/p95/p99: 0.85 / 28.27 / 56.89 / 96.55 mm
edge-like points with z-gap > 5 mm: 19.56%
edge-like points with z-gap > 10 mm: 17.80%
```

Decision:

- Keep raw `synthetic-data/` clean and auditable.
- Train a separate K41144 Wave2 sim-to-real experiment with structured-light
  noise instead of promoting clean synthetic directly.
- Added config:
  `configs/train/pointnet2_instance_k41144_wave2_sim2real.yaml`.
- Added raw generator/GUI support for opt-in sensor artifacts:
  `sensor_noise_profile=none|structured_light|tof` plus quantization, range
  noise, random/edge/blob dropout, edge smear, flying pixels, and bridge
  artifacts. The generator writes `sensor_artifact_mask` and
  `sensor_noise_summary` when enabled.

Backlog:

1. Train/evaluate K41144 Wave2 sim-to-real config against the clean Wave2 model.
2. Convert real capture into the documented `depth.npy + intrinsics.json` format,
   or capture that format directly; a lone PCD is useful for stats but not full
   2D depth pipeline validation.
3. Calibrate both generator sensor-noise parameters and train-time augmentation
   against real frames, especially edge dropout, blob dropout, quantization,
   edge smear, flying pixels, and bridge artifacts for adjacent objects.
