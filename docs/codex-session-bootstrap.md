# Codex Session Bootstrap

Use this file at the beginning of a new Codex session to recover the minimum project context and work like a senior engineer on this repository.

## First Files To Read

Read these files in this order:

```text
docs/engineering-rules.md
docs/session-summary-synthetic-data.md
docs/pointnetplusplus-implementation-plan.md
```

Then inspect the current user request, usually under:

```text
docs/request/
```

If the task touches synthetic generation or dataset quality, inspect:

```text
scripts/generate_synthetic_blender.py
scripts/dataset_gui.py
src/data/raw_dataset.py
src/data/validation.py
```

If the task touches visualization, inspect:

```text
scripts/view_point_cloud_open3d.py
scripts/dataset_gui.py
```

If the task touches PointNet++ or training, inspect:

```text
docs/pointnetplusplus-implementation-plan.md
```

and then look for current `configs/`, `src/`, `scripts/`, `processed-data/`, and `experiments/` folders before assuming they exist.

## Non-Negotiable Project Rules

- Raw generated data under `synthetic-data/` is immutable. Do not manually edit samples.
- Do not train from raw Blender output directly. Convert raw data into `processed-data/`.
- Do not train until the dataset passes validation.
- Keep coordinate-frame direction explicit in names and docs, for example `object_to_camera`, `camera_to_world`, `body_to_world`.
- Use meters internally for geometry, point clouds, transforms, camera settings, and simulation settings.
- Keep scripts thin and move reusable logic into `src/`.
- Keep config outside reusable code where practical.
- When changing behavior, schema, CLI flags, generation settings, validation policy, or workflow, update related Markdown in the same task.
- Confirm with the user before non-trivial implementation or structural changes unless the user has already explicitly approved the plan.

## Current Project Shape

Project goal:

- Build a 6D pose estimation and robotic bin-picking pipeline for `object-model/K41144.stl`.
- Generate synthetic raw data with Blender.
- Use point cloud as the main training signal.
- Preserve enough raw signals to support PointNet++, PPRNet++-style pose regression, YOLO segmentation, and later pose refinement.

Important object facts:

- Object: `object-model/K41144.stl`.
- Approximate size: `30 mm x 93.6 mm x 20 mm`.
- STL origin is not centered. Generator recenters mesh vertices for physics and stores transforms so exported poses still refer to the original STL coordinate frame.

Important raw sample files:

```text
metadata.json
sensor_data.npz
rgb.png
label_overlay.png
depth_preview.png
instance_mask.png
normal_camera.png
```

Important `sensor_data.npz` keys:

```text
depth_m
instance_mask
normal_world
normal_camera
points_camera
points_world
point_instance_ids
point_semantic_ids
point_pixels
```

Important metadata pose fields:

```text
original_model_to_body
body_to_world
object_to_world
object_to_camera
camera_to_world
world_to_camera
```

## Senior Engineer Workflow

Before editing:

- Read the relevant files instead of guessing.
- Check whether the repo has existing helpers, schemas, or conventions.
- Keep changes scoped to the request.
- Do not refactor unrelated code while implementing a feature.
- Treat generated datasets and prior experiment outputs as historical artifacts.

During implementation:

- Prefer explicit data structures and type hints.
- Make CLI inputs and outputs clear.
- Fail loudly when data is invalid.
- Record rejection reasons and data-quality decisions.
- Use reusable modules for logic and keep GUI/scripts as orchestration layers.
- For GUI work, keep heavy file I/O and validation off the UI thread. Use Qt worker threads and signals; keep rendering/UI updates on the UI thread.
- For VTK/Qt work, keep VTK rendering in the UI thread and load point arrays in worker threads.

After editing:

- Run targeted compile/import checks.
- Run validators or smoke tests on existing datasets when possible.
- If a command would generate a new raw dataset, be explicit about the output folder and avoid overwriting existing data.
- Summarize what changed, what was verified, and what was not run.

## Dataset And Validation Expectations

Raw dataset validation should check at minimum:

- Required files exist.
- `metadata.json` is readable.
- `sensor_data.npz` has expected keys.
- Array shapes are consistent.
- Instance ids match between metadata, masks, and point labels.
- Semantic ids are valid.
- Visible point and object counts meet configured thresholds.
- Pose matrices are finite.
- Objects are not out of bin, using `object_to_world` and model geometry or bbox.

Dataset generation should:

- Use the Blender generator as the source of truth.
- Generate into a new output folder unless overwrite is explicitly requested.
- Reject attempts with out-of-bin objects by default.
- Record generator settings and rejection stats.

## GUI Notes

The PySide6 GUI entrypoint is:

```powershell
python .\scripts\dataset_gui.py
```

GUI responsibilities:

- Run the Blender generator via `QProcess`.
- Stream logs without blocking the UI.
- Browse raw datasets and preview samples.
- Load sample previews and point clouds in worker threads.
- Render VTK content on the UI thread.
- Run raw dataset validation in a worker thread.

The GUI must not manually edit raw samples.

## PointNet++ Direction

Do not start training from current raw point arrays without reading:

```text
docs/pointnetplusplus-implementation-plan.md
```

Important limitation:

- Current `points_camera` and `points_world` arrays are object-only.
- Semantic segmentation needs background/bin points too.
- A PointNet++ converter should reconstruct full-scene points from `depth_m` and `camera_intrinsics`, then use `instance_mask` for labels.

Processed datasets should go under:

```text
processed-data/
```

Experiments should go under:

```text
experiments/
```

## Useful Starting Commands

Inspect repository shape:

```powershell
Get-ChildItem -Force
Get-ChildItem scripts -File
Get-ChildItem docs -Recurse -File
```

Run GUI import smoke test:

```powershell
python -c "import PySide6; import numpy; import PIL; print('gui imports ok')"
```

Run Python compile checks after script/module edits:

```powershell
python -m py_compile scripts\dataset_gui.py src\data\raw_dataset.py src\data\validation.py
```

Run the Open3D point-cloud viewer:

```powershell
python .\scripts\view_point_cloud_open3d.py .\synthetic-data\K41144\sample_000001
```

Run the Blender generator manually:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --samples 3 --objects 30 --width 320 --height 240 --bin-wall-height 0.14 --drop-height-min 0.12 --drop-height-max 0.34 --spawn-strategy layered --objects-per-layer 6 --spawn-min-distance 0.045 --collision-margin 0.00002 --min-visible-objects 12 --min-visible-points 8000 --max-sample-attempts 12 --settle-frames 260 --output synthetic-data/K41144_test_new
```

## Handoff Checklist

At the end of a session, report:

- Files changed.
- Behavior changed.
- Commands/tests run.
- Any generated output folders.
- Any known risks or follow-up tasks.

