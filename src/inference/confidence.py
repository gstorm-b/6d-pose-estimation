"""Per-instance confidence scoring and top-K pick-success metrics.

Commercial bin-picking needs the chosen pick to be correct, not the average
crop. This module turns per-instance pipeline diagnostics into one explainable
confidence in [0, 1] (v1: a hand-tuned monotonic combination of bounded terms,
fully defined by a config so it is tunable per bundle without code), and scores
how often the highest-confidence pick(s) are correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ConfidenceConfig:
    """Reference scales and weights for the v1 confidence score.

    Each term maps a diagnostic to a sub-score in (0, 1]; the confidence is the
    weighted geometric mean of the available sub-scores.
    """

    center_dispersion_scale: float = 0.06   # normalized origin-vote std where the score halves-ish
    keypoint_dispersion_scale: float = 0.12  # normalized keypoint-vote std scale
    point_count_ref: float = 400.0           # crop points for a saturated point score
    isolation_ref: float = 0.8               # centroid gap (in diameters) for full isolation score
    refinement_inlier_ref: float = 0.6       # refinement inlier fraction for full score
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "center_vote": 1.0,
            "keypoint": 1.0,
            "point_count": 0.5,
            "isolation": 0.5,
            "object_prob": 0.5,
            "refinement": 0.5,
        }
    )

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "ConfidenceConfig":
        if not value:
            return cls()
        base = cls()
        for key in ("center_dispersion_scale", "keypoint_dispersion_scale", "point_count_ref", "isolation_ref", "refinement_inlier_ref"):
            if key in value:
                setattr(base, key, float(value[key]))
        if isinstance(value.get("weights"), dict):
            base.weights = {**base.weights, **{k: float(v) for k, v in value["weights"].items()}}
        return base


def _saturating(value: float, reference: float) -> float:
    if reference <= 0:
        return 1.0
    return float(min(1.0, max(0.0, value) / reference))


def compute_instance_confidence(features: dict[str, Any], config: ConfidenceConfig | None = None) -> tuple[float, dict[str, float]]:
    """Return (confidence, per-term sub-scores) from instance diagnostics.

    Recognized feature keys (all optional except the two vote dispersions):
    center_vote_dispersion, keypoint_dispersion (normalized by diameter),
    point_count, cluster_isolation (in diameters), object_probability,
    refinement_inlier_fraction.
    """

    config = config or ConfidenceConfig()
    sub: dict[str, float] = {}

    if "center_vote_dispersion" in features:
        sub["center_vote"] = float(np.exp(-float(features["center_vote_dispersion"]) / max(config.center_dispersion_scale, 1e-6)))
    if "keypoint_dispersion" in features:
        sub["keypoint"] = float(np.exp(-float(features["keypoint_dispersion"]) / max(config.keypoint_dispersion_scale, 1e-6)))
    if "point_count" in features:
        sub["point_count"] = _saturating(float(features["point_count"]), config.point_count_ref)
    if features.get("cluster_isolation") is not None:
        sub["isolation"] = _saturating(float(features["cluster_isolation"]), config.isolation_ref)
    if features.get("object_probability") is not None:
        sub["object_prob"] = float(min(1.0, max(0.0, float(features["object_probability"]))))
    if features.get("refinement_inlier_fraction") is not None:
        sub["refinement"] = _saturating(float(features["refinement_inlier_fraction"]), config.refinement_inlier_ref)

    if not sub:
        return 0.0, sub
    log_sum = 0.0
    weight_sum = 0.0
    for name, score in sub.items():
        weight = float(config.weights.get(name, 1.0))
        log_sum += weight * np.log(max(score, 1e-6))
        weight_sum += weight
    confidence = float(np.exp(log_sum / max(weight_sum, 1e-6)))
    return confidence, sub


def cluster_isolation(centroids: np.ndarray, diameter_m: float) -> np.ndarray:
    """Distance (in diameters) from each centroid to its nearest other centroid."""

    centroids = np.asarray(centroids, dtype=np.float64)
    n = centroids.shape[0]
    if n <= 1:
        return np.full(n, np.inf, dtype=np.float64)
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1) / max(float(diameter_m), 1e-8)


def top_k_pick_success(scenes: list[list[tuple[float, bool]]], k: int = 3) -> dict[str, float]:
    """Aggregate top-1 and top-k pick success over scenes.

    Each scene is a list of (confidence, is_correct). For each scene the
    instances are ranked by confidence; top-1 success is whether the highest
    is correct, top-k success is whether any of the top k are correct.
    """

    top1 = []
    topk = []
    for scene in scenes:
        if not scene:
            continue
        ranked = sorted(scene, key=lambda item: item[0], reverse=True)
        top1.append(1.0 if ranked[0][1] else 0.0)
        topk.append(1.0 if any(correct for _conf, correct in ranked[:k]) else 0.0)
    return {
        "scenes": float(len(top1)),
        "top1_pick_success": float(np.mean(top1)) if top1 else 0.0,
        f"top{k}_pick_success": float(np.mean(topk)) if topk else 0.0,
    }
