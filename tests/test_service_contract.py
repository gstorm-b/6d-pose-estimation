"""Contract tests for the backend service runtime + HTTP layer.

Covers every error code via ServiceRuntime.infer (no network), admission-control
backpressure, a live stdlib-HTTP smoke test (health/skus/404/bad-body, no models
required), and a guarded real-inference round-trip on the packaged bundle. Runs
under pytest or standalone:

    python tests/test_service_contract.py
"""

from __future__ import annotations

import glob
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.app import ServiceState, build_server  # noqa: E402
from src.service.runtime import ServiceRuntime  # noqa: E402
from src.service.schemas import ErrorCode  # noqa: E402

MODELS_ROOT = PROJECT_ROOT / "models"
PROCESSED_BENDING = PROJECT_ROOT / "processed-data" / "pointnet2_semseg_bending_pipe_wave2" / "test"


def _cpu_runtime(root: str | Path = "models") -> ServiceRuntime:
    return ServiceRuntime(root, device="cpu")


def _valid_points_payload(sku: str = "bending_pipe") -> dict:
    points = (np.random.default_rng(0).random((128, 3)).astype(np.float32) * 0.1).tolist()
    return {"sku": sku, "points_camera": points}


# -- error codes via runtime.infer (no network, no models) --------------------
def test_missing_sku() -> None:
    rt = _cpu_runtime()
    status, body = rt.infer({"points_camera": [[0.0, 0.0, 0.1]]})
    assert status == 400, status
    assert body["error"]["code"] == ErrorCode.INVALID_INPUT, body


def test_no_points_or_depth() -> None:
    rt = _cpu_runtime()
    status, body = rt.infer({"sku": "bending_pipe"})
    assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body
    assert "points_camera" in body["error"]["message"], body


def test_bad_points_shape() -> None:
    rt = _cpu_runtime()
    status, body = rt.infer({"sku": "bending_pipe", "points_camera": [[0.0, 0.0]]})
    assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body


def test_features_length_mismatch() -> None:
    rt = _cpu_runtime()
    payload = {"sku": "bending_pipe", "points_camera": [[0.0, 0.0, 0.1], [0.0, 0.1, 0.1]], "features": [[1.0, 0.0, 0.0]]}
    status, body = rt.infer(payload)
    assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body


def test_bad_options() -> None:
    rt = _cpu_runtime()
    payload = _valid_points_payload()
    payload["options"] = {"min_confidence": 2.0}
    status, body = rt.infer(payload)
    assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body


def test_depth_without_intrinsics() -> None:
    import io

    rt = _cpu_runtime()
    buf = io.BytesIO()
    np.savez(buf, depth_m=np.zeros((4, 4), dtype=np.float32))
    import base64

    payload = {"sku": "bending_pipe", "depth_npz_b64": base64.b64encode(buf.getvalue()).decode("ascii")}
    status, body = rt.infer(payload)
    assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body
    assert "intrinsics" in body["error"]["message"], body


def test_unknown_sku() -> None:
    rt = _cpu_runtime()
    status, body = rt.infer(_valid_points_payload(sku="does_not_exist_sku"))
    assert status == 404, status
    assert body["error"]["code"] == ErrorCode.UNKNOWN_SKU, body


# -- admission control --------------------------------------------------------
def test_admission_control() -> None:
    state = ServiceState(_cpu_runtime(), max_queue_depth=2)
    assert state.try_admit() is True
    assert state.try_admit() is True
    assert state.try_admit() is False, "should reject beyond max_queue_depth"
    state.release()
    assert state.try_admit() is True, "slot should free after release"
    assert state.inflight == 2


# -- live HTTP smoke (no models needed) ---------------------------------------
def _spawn_server():
    root = tempfile.mkdtemp(prefix="pose_service_test_")
    runtime = ServiceRuntime(root, device="cpu")
    server, state = build_server(runtime, host="127.0.0.1", port=0, max_queue_depth=4)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, port, state


def _get(port: int, path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(port: int, path: str, raw: bytes, content_type: str = "application/json"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=raw, method="POST")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_smoke() -> None:
    server, thread, port, _state = _spawn_server()
    try:
        status, body = _get(port, "/v1/health")
        assert status == 200 and body["status"] == "ok" and "device" in body, body

        status, body = _get(port, "/v1/skus")
        assert status == 200 and "skus" in body, body

        status, body = _get(port, "/v1/unknown")
        assert status == 404 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body

        status, body = _post(port, "/v1/poses", b"not json{")
        assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body

        status, body = _post(port, "/v1/poses", b"")
        assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body

        status, body = _post(port, "/v1/poses", json.dumps({"points_camera": [[0, 0, 0.1]]}).encode())
        assert status == 400 and body["error"]["code"] == ErrorCode.INVALID_INPUT, body  # missing sku

        status, body = _post(port, "/v1/wrong", b"{}")
        assert status == 404, status
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- guarded real inference round-trip ----------------------------------------
def test_real_inference_roundtrip() -> None:
    if not (MODELS_ROOT / "bending_pipe").exists() or not glob.glob(str(PROCESSED_BENDING / "*.npz")):
        print("  (skip: missing bending_pipe bundle or processed wave2 test sample)")
        return
    from src.inference.device import resolve_device

    runtime = ServiceRuntime(MODELS_ROOT, device=resolve_device("cuda"))
    sample = sorted(glob.glob(str(PROCESSED_BENDING / "*.npz")))[0]
    data = np.load(sample)
    payload = {
        "sku": "bending_pipe",
        "points_camera": data["points"].astype(np.float32).tolist(),
        "features": data["features"].astype(np.float32).tolist(),
        "options": {"max_instances": 5},
    }
    status, body = runtime.infer(payload)
    assert status == 200, (status, body)
    # JSON round-trip must succeed (no numpy leaking into the response).
    body = json.loads(json.dumps(body))
    assert body["sku"] == "bending_pipe" and body["model_version"], body
    assert set(body.keys()) == {"sku", "model_version", "instances", "timings_ms", "warnings"}, body
    assert len(body["instances"]) <= 5, body
    for inst in body["instances"]:
        T = np.asarray(inst["object_to_camera"], dtype=np.float64)
        assert T.shape == (4, 4) and np.all(np.isfinite(T)), inst
        assert 0.0 <= inst["confidence"] <= 1.0, inst
        assert isinstance(inst["point_count"], int) and inst["point_count"] > 0, inst
    print(
        f"  real inference: {len(body['instances'])} instances, "
        f"top_conf={body['instances'][0]['confidence']:.3f} "
        f"wall_ms={body['timings_ms'].get('wall_ms')}" if body["instances"] else "  real inference: 0 instances"
    )


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
