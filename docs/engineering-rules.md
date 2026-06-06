# Engineering Rules

These rules define how to design, implement, validate, and collaborate on this project.

## Collaboration Rule: Confirm Before Implementation

Before implementing non-trivial code or changing project structure, confirm the understood request and proposed approach with the user.

The confirmation should include:

- What Codex understands the user wants.
- Proposed implementation approach.
- Files or folders expected to be created or modified.
- Important trade-offs or risks.
- A request for explicit approval before implementation.

Read-only inspections, dataset audits, and small checks can be done directly. Small obvious bug fixes may be preceded by a concise implementation note.

## Documentation Rule: Keep Markdown In Sync

When code changes affect a module's behavior, data schema, CLI contract, generation settings, validation policy, or recommended workflow, update the related Markdown documentation in the same task.

Examples:

- Generator changes should update relevant notes in `docs/session-summary-synthetic-data.md`.
- Dataset schema changes should update raw/processed dataset docs.
- New validation or rejection behavior should be recorded with the validation result and output folder.
- New CLI flags should be documented with their purpose and default behavior.

Do not leave session summaries or engineering notes describing old behavior after changing the related module.

## Core Principles

### Raw Data Is Immutable

`synthetic-data/` contains raw generated scenes.

Rules:

- Do not manually edit generated samples.
- Do not silently overwrite important raw datasets.
- If generator logic changes, either generate a new dataset folder or record the exact config/version used.
- Raw data should remain rich and traceable, even if later training pipelines need simpler formats.

### Separate Raw Data And Training Data

Keep raw generated scenes separate from model-ready processed data.

Recommended structure:

```text
synthetic-data/      raw generated scenes
processed-data/      model-ready converted datasets
experiments/         training runs, logs, checkpoints
scripts/             command-line tools
src/                 reusable project modules
configs/             generator/training/eval configs
docs/                design notes and conventions
```

Training code should not depend directly on the raw Blender output format. Use conversion scripts to create stable processed datasets.

### Coordinate Frames Must Be Explicit

6D pose projects fail easily when transforms are ambiguous. Every transform variable, metadata field, and function should make transform direction explicit.

Prefer names like:

```text
object_to_camera
camera_to_world
world_to_camera
body_to_world
original_model_to_body
```

Avoid vague names like:

```text
pose
T
matrix
transform
```

unless the function scope makes the direction unambiguous.

Coordinate frames to document:

- Original STL object frame.
- Recentered physics body frame.
- Blender/world frame.
- Camera frame.
- Robot base frame, once robot integration starts.

### Use Meters Internally

All geometry, point clouds, transforms, camera parameters, and simulation settings should use meters by default.

If millimeters are used at an interface boundary, convert immediately and document it.

### Validate Before Training

Do not train on a dataset that has not passed validation.

Raw dataset validation should check:

- Required files exist.
- `metadata.json` and `sensor_data.npz` schemas match.
- Array shapes and dtypes are correct.
- Instance ids match across metadata, masks, and point labels.
- Semantic ids are valid.
- Point counts are reasonable.
- Object count distribution is reasonable.
- Min points per instance is reasonable.
- Pose matrices are finite and valid.
- Collision/generation settings are recorded.

### Config Outside Code

Avoid hard-coding experiment settings inside reusable code.

Prefer:

```text
configs/
  generator/k41144.yaml
  train/pointnet_k41144.yaml
```

or explicit CLI arguments for scripts.

Generator/training outputs should include the config used to produce them.

### Scripts Are Thin, Modules Are Reusable

Command-line scripts in `scripts/` should parse arguments and call reusable modules.

As the project grows, move reusable logic into `src/`.

Recommended future structure:

```text
src/
  data/
    raw_dataset.py
    validation.py
    pointnet_export.py
    pprnet_export.py
    yolo_export.py
  geometry/
    transforms.py
    camera.py
  visualization/
    open3d_viewer.py
```

### Every CLI Tool Has Clear Inputs And Outputs

Each script should have clear input/output arguments.

Examples:

```powershell
python scripts/validate_raw_dataset.py --data synthetic-data/K41144
python scripts/prepare_pointnet_dataset.py --raw synthetic-data/K41144 --out processed-data/pointnet_k41144
python scripts/export_yolo_segmentation.py --raw synthetic-data/K41144 --out processed-data/yolo_k41144
```

### Log Rejections And Data Quality Decisions

Data generation should not silently discard bad samples.

If a sample is rejected, record why:

- Too few visible objects.
- Too few visible points.
- Object outside bin.
- Invalid mask/pose/point cloud.

Dataset summaries should include acceptance/rejection statistics.

## Code Quality Rules

- Prefer readable code over clever code.
- Use explicit function inputs and outputs.
- Use type hints for reusable functions.
- Avoid hidden global state except clear constants.
- Avoid magic numbers; move them into config, CLI arguments, or named constants.
- Fail loudly when data is invalid.
- Keep comments short and useful.
- Do not mix unrelated refactors into task-specific changes.

## Dataset Rules

### Raw Dataset

Raw data should stay under:

```text
synthetic-data/
```

Raw sample folders may include:

```text
sensor_data.npz
metadata.json
rgb.png
label_overlay.png
depth_preview.png
instance_mask.png
normal_camera.png
```

Raw data should preserve all useful intermediate signals, even if not every model uses them.

### Processed Dataset

Processed model-ready datasets should stay under:

```text
processed-data/
```

Examples:

```text
processed-data/
  pointnet_k41144/
  pprnet_k41144/
  yolo_k41144/
```

Processed datasets should include:

- Split files or explicit train/val/test folders.
- Dataset stats.
- A copy of the conversion config.
- A clear schema description.

### Do Not Train Directly From Raw Format

Raw format is for traceability and conversion. Training should use processed datasets with stable model-specific schemas.

## Experiment Rules

Each training or evaluation run should have its own folder:

```text
experiments/
  pointnet_k41144_YYYY-MM-DD_001/
    config.yaml
    metrics.json
    train.log
    checkpoints/
    samples_preview/
```

Rules:

- Do not overwrite old experiments unless explicitly requested.
- Save config with every run.
- Save metrics in machine-readable form.
- Save previews when useful for segmentation/pose debugging.

## Testing And Validation Rules

Minimum validation targets:

- Camera projection and unprojection.
- Transform composition and inversion.
- Raw dataset schema.
- Processed dataset schema.
- Instance id consistency.
- Pose matrix validity.
- Point cloud bounds and units.

At this stage, validators are more important than heavy unit test coverage.

## Current Non-Negotiable Rules

1. Coordinate frame direction must be clear in names and docs.
2. Raw data must not be manually modified.
3. Do not train until the dataset passes validation.
4. Confirm with the user before non-trivial implementation or structural changes.
5. Keep related Markdown documentation in sync when changing module behavior.
