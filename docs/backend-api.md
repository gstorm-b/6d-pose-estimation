# Pose Backend HTTP API (v1)

The backend serves ranked 6D object poses for one SKU per request. It is a thin
stdlib `http.server` layer (no FastAPI/uvicorn dependency) in front of the
in-memory `PosePipeline`; all inference is serialized through a single GPU lock.

## Running

```bash
python scripts/run_backend_service.py                 # reads configs/backend/service.yaml
python scripts/run_backend_service.py --host 0.0.0.0 --port 9000 --warmup
```

Config (`configs/backend/service.yaml`): `registry.root`, `registry.vram_budget_bundles`,
`inference.device` (`cuda`, CPU fallback), `service.host/port/request_timeout_s/max_queue_depth`.
CLI flags override the config. `--warmup` pre-loads and warms the latest bundle of every SKU.

Security: v1 has no auth. Bind to a trusted cell network only (`service.host`).

## Endpoints

### `GET /v1/health`

Liveness, device, and what is resident.

```json
{ "status": "ok", "device": "cuda", "loaded_bundles": ["bending_pipe:v2"],
  "available_skus": ["K41144", "bending_pipe"] }
```

### `GET /v1/skus`

Registry listing with gate status.

```json
{ "skus": {
    "bending_pipe": { "versions": ["v1", "v2"], "latest": "v2",
                      "symmetry": "bending_pipe", "diameter_m": 0.1391, "gates_met": true } } }
```

### `POST /v1/poses`

Body must be a JSON object. Provide the scene **either** as points or as a depth frame.

Points (positive-forward camera frame, metres):

```json
{ "sku": "bending_pipe",
  "version": null,
  "options": { "max_instances": 5, "refine": true, "min_confidence": 0.0 },
  "points_camera": [[x, y, z], ...],
  "features": [[nx, ny, nz], ...] }
```

Depth frame (base64 of an `np.savez` buffer with `depth_m`, optional `normal_camera`):

```json
{ "sku": "bending_pipe",
  "intrinsics": { "fx": 1066.8, "fy": 1067.5, "cx": 312.9, "cy": 241.3, "width": 640, "height": 480 },
  "depth_npz_b64": "<base64 npz>" }
```

- `version` optional; omitted -> latest. `options` optional; defaults `max_instances=10, refine=true, min_confidence=0.0`.
- `features`, when given, must match `points_camera` length. `intrinsics` is required with `depth_npz_b64`.

Response (instances ranked by confidence, highest first):

```json
{ "sku": "bending_pipe",
  "model_version": "v2",
  "instances": [
    { "instance_id": 7,
      "object_to_camera": [[r,r,r,tx],[r,r,r,ty],[r,r,r,tz],[0,0,0,1]],
      "confidence": 0.787,
      "point_count": 1140,
      "diagnostics": { "center_vote_dispersion": 0.0167, "keypoint_dispersion": 0.0207,
                       "cluster_isolation": 0.4698 } } ],
  "timings_ms": { "instance_ms": 5899.9, "crops_ms": 3.3, "pose_ms": 570.3, "wall_ms": 6474.2 },
  "warnings": [] }
```

`object_to_camera` is a 4x4 row-major rigid transform in the Blender metadata camera
convention. An empty bin returns `200` with `instances: []` and a `"no instances found"` warning
(an empty bin is a normal condition, not an error).

## Errors

Errors return `{ "error": { "code": <code>, "message": <text> } }` with an HTTP status:

| HTTP | code            | when |
|------|-----------------|------|
| 400  | `invalid_input` | malformed body, missing `sku`, bad `points_camera` shape, feature/length mismatch, bad `options`, depth without `intrinsics` |
| 404  | `unknown_sku`   | SKU or requested `version` not in the registry |
| 413  | `invalid_input` | request body over 256 MB |
| 429  | `internal`      | `max_queue_depth` requests already in flight (backpressure) |
| 500  | `internal`      | bundle load failure or an unexpected inference error |

## Client snippet

```python
import json, urllib.request
import numpy as np

points = my_scene_points_camera()                      # (N, 3) float32, positive-forward metres
payload = {"sku": "bending_pipe", "options": {"max_instances": 5},
           "points_camera": points.astype(np.float32).tolist()}
req = urllib.request.Request("http://127.0.0.1:8080/v1/poses",
                             data=json.dumps(payload).encode(), method="POST",
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read())
for inst in result["instances"]:                       # already ranked; pick the first reachable one
    T = np.asarray(inst["object_to_camera"])           # 4x4 object->camera
    print(inst["instance_id"], inst["confidence"], T[:3, 3])
```

First call for an uncached SKU pays the bundle-load latency; subsequent calls reuse the cached pipeline.
```
