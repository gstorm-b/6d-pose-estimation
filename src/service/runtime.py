"""Service runtime: resolves SKUs to pipelines, serializes GPU work, runs inference.

Stateless HTTP handlers call into one shared ServiceRuntime. Inference is
serialized by a single GPU lock (one GPU assumption); when the lock is busy the
runtime reports backpressure so the handler can return 429.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.device import resolve_device
from src.registry.bundle_schema import BundleValidationError
from src.registry.model_registry import ModelRegistry
from src.service.schemas import (
    CameraIntrinsics,
    ErrorCode,
    InferenceOptions,
    PoseInstanceResult,
    PoseResponse,
    SchemaError,
)


class BackpressureError(RuntimeError):
    """Raised when the GPU is busy and the request cannot be admitted."""


@dataclass
class _CachedPipeline:
    version: str
    pipeline: Any


def _jsonify_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy sub-score detail and coerce numpy scalars to JSON-safe types."""

    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if key == "confidence_terms":
            continue
        if value is None or isinstance(value, (bool, str)):
            out[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


class ServiceRuntime:
    """Shared runtime backing the HTTP service."""

    def __init__(
        self,
        registry_root: str | Path = "models",
        *,
        device: str = "cuda",
        vram_budget_bundles: int = 2,
        gpu_acquire_timeout_s: float = 30.0,
    ) -> None:
        self.registry = ModelRegistry(registry_root, vram_budget_bundles=vram_budget_bundles)
        self.device = resolve_device(device)
        self.gpu_acquire_timeout_s = float(gpu_acquire_timeout_s)
        self._gpu_lock = threading.Lock()
        self._pipelines: dict[str, _CachedPipeline] = {}
        self._pipelines_lock = threading.Lock()

    def list_skus(self) -> dict[str, Any]:
        skus = {}
        for sku in self.registry.list_skus():
            versions = self.registry.versions(sku)
            latest = self.registry.resolve(sku)
            skus[sku] = {
                "versions": versions,
                "latest": latest.version,
                "symmetry": latest.manifest.model.symmetry_name,
                "diameter_m": latest.manifest.model.diameter_m,
                "gates_met": bool(latest.manifest.eval_metrics),
            }
        return {"skus": skus}

    def health(self) -> dict[str, Any]:
        loaded = [f"{sku}:{ver}" for (sku, ver, _dev) in self.registry._cache.keys()]
        return {
            "status": "ok",
            "device": self.device,
            "loaded_bundles": loaded,
            "available_skus": self.registry.list_skus(),
        }

    def _pipeline_for_bundle(self, bundle: Any) -> Any:
        from src.inference.pose_pipeline import PosePipeline

        cache_key = f"{bundle.sku}:{bundle.version}"
        with self._pipelines_lock:
            cached = self._pipelines.get(cache_key)
            if cached is not None:
                return cached.pipeline
        loaded = self.registry.load(bundle, device=self.device)
        pipeline = PosePipeline.from_loaded_bundle(loaded)
        with self._pipelines_lock:
            self._pipelines[cache_key] = _CachedPipeline(version=bundle.version, pipeline=pipeline)
        return pipeline

    @staticmethod
    def _decode_points(payload: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
        """Return (points_camera, features, depth_payload) from the request body."""

        if "points_camera" in payload:
            points = np.asarray(payload["points_camera"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise SchemaError("points_camera must be (N, 3)")
            features = None
            if payload.get("features") is not None:
                features = np.asarray(payload["features"], dtype=np.float32)
                if features.shape[0] != points.shape[0]:
                    raise SchemaError("features length must match points_camera")
            return points, features, None
        if "depth_npz_b64" in payload:
            raw = base64.b64decode(payload["depth_npz_b64"])
            with np.load(io.BytesIO(raw)) as data:
                depth = np.asarray(data["depth_m"], dtype=np.float32)
                normal = np.asarray(data["normal_camera"], dtype=np.float32) if "normal_camera" in data.files else None
            return None, None, {"depth_m": depth, "normal_camera": normal}
        raise SchemaError("request must include points_camera or depth_npz_b64")

    def infer(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Run one inference request. Returns (http_status, body)."""

        try:
            sku = str(payload["sku"]) if "sku" in payload else None
            if not sku:
                raise SchemaError("sku is required")
            options = InferenceOptions.from_dict(payload.get("options"))
            options.validate()
            version = payload.get("version")
            points, features, depth_payload = self._decode_points(payload)
            if depth_payload is not None:
                if "intrinsics" not in payload:
                    raise SchemaError("intrinsics is required with depth_npz_b64")
                intrinsics = CameraIntrinsics.from_dict(payload["intrinsics"])
                intrinsics.validate()
        except SchemaError as exc:
            return 400, {"error": {"code": ErrorCode.INVALID_INPUT, "message": str(exc)}}

        try:
            bundle = self.registry.resolve(sku, version)
        except BundleValidationError as exc:
            return 404, {"error": {"code": ErrorCode.UNKNOWN_SKU, "message": str(exc)}}
        model_version = bundle.version
        try:
            pipeline = self._pipeline_for_bundle(bundle)
        except Exception as exc:  # noqa: BLE001 - bundle load failure is a structured 500
            return 500, {"error": {"code": ErrorCode.INTERNAL, "message": f"failed to load bundle: {exc}"}}

        acquired = self._gpu_lock.acquire(timeout=self.gpu_acquire_timeout_s)
        if not acquired:
            return 429, {"error": {"code": ErrorCode.INTERNAL, "message": "GPU busy; retry later"}}
        try:
            started = time.perf_counter()
            if depth_payload is not None:
                result = pipeline.infer_scene(
                    depth_payload["depth_m"],
                    intrinsics.to_dict(),
                    depth_payload["normal_camera"],
                    options=options,
                )
            else:
                result = pipeline.infer_from_points(points, features, options=options)
            wall_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - surface as a structured 500
            return 500, {"error": {"code": ErrorCode.INTERNAL, "message": f"{type(exc).__name__}: {exc}"}}
        finally:
            self._gpu_lock.release()

        timings = {k: round(float(v), 3) for k, v in {**result.timings_ms, "wall_ms": wall_ms}.items()}
        if not result.instances:
            return 200, PoseResponse(
                sku=sku,
                model_version=model_version,
                instances=[],
                timings_ms=timings,
                warnings=result.warnings + ["no instances found"],
            ).to_dict()

        instances = [
            PoseInstanceResult(
                instance_id=inst.instance_id,
                object_to_camera=[[float(v) for v in row] for row in inst.object_to_camera],
                confidence=float(inst.confidence) if inst.confidence is not None else 0.0,
                point_count=inst.point_count,
                diagnostics=_jsonify_diagnostics(inst.diagnostics),
            )
            for inst in result.instances
        ]
        response = PoseResponse(
            sku=sku,
            model_version=model_version,
            instances=instances,
            timings_ms=timings,
            warnings=result.warnings,
        )
        return 200, response.to_dict()
