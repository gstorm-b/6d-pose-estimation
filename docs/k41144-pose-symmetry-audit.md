# K41144 Pose Symmetry Audit

Date: 2026-06-07

Purpose: define the initial symmetry assumptions for the next pose-estimation stage, before adding a rotation/pose regression head.

## Mesh Frame

Mesh: `object-model/K41144.stl`

Measured STL bounds:

```text
bounds_min = [-0.014999636, 0.0, -0.010000000]
bounds_max = [ 0.014999636, 0.093630873, 0.010000000]
dims       = [ 0.029999273, 0.093630873, 0.020000000] m
bbox_center = [0.0, 0.046815436, 0.0]
```

The object is elongated along local `+Y`. This matches the generator metadata field `original_model_center_offset`.

## Symmetry Probe

The audit sampled STL vertices and compared the mesh against rotations around the bbox center using nearest-neighbor Chamfer-style distances.

K41144 result:

```text
rotation     mean distance   p95 distance    max distance
x 180 deg    0.002646 m      0.005326 m      0.008986 m
y 180 deg    0.000128 m      0.000317 m      0.000847 m
z 180 deg    0.002646 m      0.005327 m      0.008987 m
y  90 deg    0.000486 m      0.001774 m      0.009172 m
y 270 deg    0.000486 m      0.001774 m      0.009172 m
```

Interpretation:

- `R_y(180 deg)` is the only strong geometric symmetry candidate.
- `R_y(90 deg)` and `R_y(270 deg)` have low mean error but high local max error, so they should not be treated as exact pose equivalences for training/evaluation.
- `R_x(180 deg)` and `R_z(180 deg)` are too different for an exact symmetry metric.

As a sanity comparison, `bending_pipe.stl` is strongly asymmetric under 180-degree rotations, so it should initially use no rotational symmetry equivalence.

## Recommended Pose Metric

For K41144, evaluate pose with a discrete symmetry set:

```text
S_k41144 = {
  I,
  R_y(pi)
}
```

When comparing a predicted `object_to_camera` against GT `object_to_camera`, compute the pose error under both GT-equivalent transforms and keep the lower error:

```text
object_to_camera_gt_equiv = object_to_camera_gt @ S
```

The same symmetry set can be used for ADD-S-like model-point distance:

```text
min_{S in S_k41144} mean || transform(pred, model_points) - transform(gt @ S, model_points) ||
```

Do not apply a continuous axial symmetry assumption. K41144 is elongated and visually close to axial symmetry in some sampled distances, but its rectangular dimensions and local features make 90-degree rotations non-equivalent.

## Grasp-Equivalence Assumption

Initial assumption for bin picking:

- 180-degree rotation around the local long axis `Y` is grasp-equivalent.
- 90-degree rotations around local `Y` are not grasp-equivalent unless a future gripper-specific grasp planner proves they are acceptable.
- Flips around local `X` or `Z` are not pose-equivalent.

The pose stage should keep the raw `object_to_camera` matrix in outputs, and apply symmetry only in loss/metric code. This avoids destroying information needed by downstream grasp planning.
