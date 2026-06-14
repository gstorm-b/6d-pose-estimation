"""Versioned per-SKU model registry for the pose-estimation backend.

Discovers bundles under a root directory (`models/<sku>/<version>/`), resolves a
SKU to a bundle, and loads its checkpoints with an LRU cache bounded by a bundle
budget. Listing/resolving needs no heavy deps; `load` imports torch lazily.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bundle_schema import BundleManifest, BundleValidationError, read_manifest


@dataclass(frozen=True)
class Bundle:
    """A resolved (but not yet loaded) bundle on disk."""

    sku: str
    version: str
    directory: Path
    manifest: BundleManifest

    @property
    def key(self) -> tuple[str, str]:
        return (self.sku, self.version)


@dataclass
class LoadedBundle:
    """A bundle with its models and geometry loaded onto a device."""

    bundle: Bundle
    device: str
    instance_model: Any
    instance_config: dict[str, Any]
    pose_model: Any
    pose_config: dict[str, Any]
    model_points_object: np.ndarray
    keypoints_object: np.ndarray
    diameter_m: float


def _version_sort_key(version: str) -> tuple[int, Any]:
    """Sort numeric-suffixed versions (v1, v2, ...) numerically, others lexically."""

    digits = "".join(ch for ch in version if ch.isdigit())
    if digits and digits == version.lstrip("vV"):
        return (1, int(digits))
    return (0, version)


class ModelRegistry:
    """Discover and load per-SKU model bundles with a bounded in-memory cache."""

    def __init__(self, root: str | Path = "models", *, vram_budget_bundles: int = 2) -> None:
        self.root = Path(root)
        self.vram_budget_bundles = max(1, int(vram_budget_bundles))
        self._cache: "OrderedDict[tuple[str, str, str], LoadedBundle]" = OrderedDict()

    def list_skus(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and self._versions_dir(p))

    def _versions_dir(self, sku_dir: Path) -> list[Path]:
        return [v for v in sku_dir.iterdir() if v.is_dir() and (v / "manifest.json").exists()]

    def versions(self, sku: str) -> list[str]:
        sku_dir = self.root / sku
        if not sku_dir.exists():
            return []
        names = [v.name for v in self._versions_dir(sku_dir)]
        return sorted(names, key=_version_sort_key)

    def resolve(self, sku: str, version: str | None = None) -> Bundle:
        available = self.versions(sku)
        if not available:
            raise BundleValidationError(f"unknown sku or no versions: {sku}")
        chosen = available[-1] if version is None else version
        if chosen not in available:
            raise BundleValidationError(f"version {version} not found for sku {sku}; have {available}")
        directory = self.root / sku / chosen
        manifest = read_manifest(directory, check_files=True)
        return Bundle(sku=sku, version=chosen, directory=directory, manifest=manifest)

    def load(self, bundle: Bundle, *, device: str = "cuda", ops_backend: str = "pytorch3d") -> LoadedBundle:
        cache_key = (bundle.sku, bundle.version, device)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        loaded = self._load_uncached(bundle, device=device, ops_backend=ops_backend)
        self._cache[cache_key] = loaded
        self._cache.move_to_end(cache_key)
        self._evict_if_needed()
        return loaded

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.vram_budget_bundles:
            _key, evicted = self._cache.popitem(last=False)
            self._release(evicted)

    @staticmethod
    def _release(loaded: LoadedBundle) -> None:
        try:
            import torch

            del loaded.instance_model
            del loaded.pose_model
            if loaded.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - eviction must never crash the registry
            pass

    def _load_uncached(self, bundle: Bundle, *, device: str, ops_backend: str) -> LoadedBundle:
        from src.inference.device import resolve_device
        from src.models.pointnet2_instance_seg import build_pointnet2_instance_seg_from_config
        from src.models.pointnet2_pose_voting import build_pointnet2_pose_voting_from_config
        from src.training.checkpoint import load_checkpoint

        resolved_device = resolve_device(device)

        manifest = bundle.manifest
        directory = bundle.directory

        instance_ckpt = load_checkpoint(manifest.file_path(directory, "instance_seg"), map_location="cpu")
        instance_config = dict(instance_ckpt.get("config", {}))
        if ops_backend:
            instance_config.setdefault("model", {})["ops_backend"] = ops_backend
        instance_model = build_pointnet2_instance_seg_from_config(instance_config)
        instance_model.load_state_dict(instance_ckpt["model_state"])
        instance_model.to(resolved_device).eval()

        pose_ckpt = load_checkpoint(manifest.file_path(directory, "pose_voting"), map_location="cpu")
        pose_config = dict(pose_ckpt.get("config", {}))
        if ops_backend:
            pose_config.setdefault("model", {})["ops_backend"] = ops_backend
        pose_model = build_pointnet2_pose_voting_from_config(pose_config)
        pose_model.load_state_dict(pose_ckpt["model_state"])
        pose_model.to(resolved_device).eval()

        model_points = np.load(manifest.file_path(directory, "model_points")).astype(np.float32)
        keypoints = np.load(manifest.file_path(directory, "keypoints")).astype(np.float32)

        return LoadedBundle(
            bundle=bundle,
            device=resolved_device,
            instance_model=instance_model,
            instance_config=instance_config,
            pose_model=pose_model,
            pose_config=pose_config,
            model_points_object=model_points,
            keypoints_object=keypoints,
            diameter_m=float(manifest.model.diameter_m),
        )

    def clear_cache(self) -> None:
        while self._cache:
            _key, evicted = self._cache.popitem(last=False)
            self._release(evicted)
