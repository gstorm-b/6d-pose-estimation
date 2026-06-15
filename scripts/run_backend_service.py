"""Launch the pose-estimation backend HTTP service.

Reads configs/backend/service.yaml (registry root, device, bind host/port,
queue depth), builds a ServiceRuntime, and serves the stdlib HTTP API. CLI flags
override the config for ad-hoc runs.

    python scripts/run_backend_service.py
    python scripts/run_backend_service.py --host 0.0.0.0 --port 9000 --warmup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.service.app import build_server  # noqa: E402
from src.service.runtime import ServiceRuntime  # noqa: E402


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pose-estimation backend service")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "backend" / "service.yaml",
        help="Path to service.yaml",
    )
    parser.add_argument("--host", default=None, help="Override bind host")
    parser.add_argument("--port", type=int, default=None, help="Override bind port")
    parser.add_argument("--registry-root", default=None, help="Override bundle registry root")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--warmup", action="store_true", help="Pre-load + warm up the latest bundle of every SKU")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    registry_cfg = config.get("registry", {})
    inference_cfg = config.get("inference", {})
    service_cfg = config.get("service", {})

    registry_root = args.registry_root or registry_cfg.get("root", "models")
    device = args.device or inference_cfg.get("device", "cuda")
    vram_budget = int(registry_cfg.get("vram_budget_bundles", 2))
    host = args.host or service_cfg.get("host", "127.0.0.1")
    port = args.port if args.port is not None else int(service_cfg.get("port", 8080))
    max_queue_depth = int(service_cfg.get("max_queue_depth", 8))
    request_timeout_s = float(service_cfg.get("request_timeout_s", 30))

    runtime = ServiceRuntime(
        registry_root,
        device=device,
        vram_budget_bundles=vram_budget,
        gpu_acquire_timeout_s=request_timeout_s,
    )
    skus = runtime.registry.list_skus()
    print(f"[service] device={runtime.device} registry_root={registry_root} skus={skus}")

    if args.warmup:
        for sku in skus:
            try:
                bundle = runtime.registry.resolve(sku)
                pipeline = runtime._pipeline_for_bundle(bundle)
                pipeline.warmup()
                print(f"[service] warmed up {sku}:{bundle.version}")
            except Exception as exc:  # noqa: BLE001 - warmup is best-effort
                print(f"[service] warmup failed for {sku}: {type(exc).__name__}: {exc}")

    server, _state = build_server(runtime, host=host, port=port, max_queue_depth=max_queue_depth)
    print(f"[service] listening on http://{host}:{port}  (POST /v1/poses, GET /v1/skus, GET /v1/health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[service] shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
