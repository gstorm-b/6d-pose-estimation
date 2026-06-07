"""Instance clustering and point-set matching for voted centers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VotedCenterClusteringConfig:
    object_probability_threshold: float = 0.5
    dbscan_eps_m: float = 0.006
    dbscan_min_samples: int = 8
    min_cluster_points: int = 32

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "VotedCenterClusteringConfig":
        config = value or {}
        return cls(
            object_probability_threshold=float(config.get("object_probability_threshold", cls.object_probability_threshold)),
            dbscan_eps_m=float(config.get("dbscan_eps_m", cls.dbscan_eps_m)),
            dbscan_min_samples=int(config.get("dbscan_min_samples", cls.dbscan_min_samples)),
            min_cluster_points=int(config.get("min_cluster_points", cls.min_cluster_points)),
        )


@dataclass(frozen=True)
class InstanceMetricConfig:
    instance_iou_threshold: float = 0.5
    ignore_gt_instances_below_points: int = 32
    merge_split_overlap_threshold: float = 0.1

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "InstanceMetricConfig":
        config = value or {}
        return cls(
            instance_iou_threshold=float(config.get("instance_iou_threshold", cls.instance_iou_threshold)),
            ignore_gt_instances_below_points=int(config.get("ignore_gt_instances_below_points", cls.ignore_gt_instances_below_points)),
            merge_split_overlap_threshold=float(config.get("merge_split_overlap_threshold", cls.merge_split_overlap_threshold)),
        )


def cluster_voted_centers(
    points_camera: np.ndarray,
    object_probabilities: np.ndarray,
    offsets: np.ndarray,
    config: VotedCenterClusteringConfig | dict[str, Any] | None = None,
) -> dict[str, np.ndarray | int | float]:
    """Cluster object-point votes and return per-point instance labels.

    Predicted instance labels use `0` for background/noise and compact positive
    ids for accepted clusters.
    """

    cfg = config if isinstance(config, VotedCenterClusteringConfig) else VotedCenterClusteringConfig.from_mapping(config)
    points = np.asarray(points_camera, dtype=np.float32)
    probabilities = np.asarray(object_probabilities, dtype=np.float32)
    pred_offsets = np.asarray(offsets, dtype=np.float32)
    if points.shape != pred_offsets.shape:
        raise ValueError(f"points and offsets shape mismatch: {points.shape} vs {pred_offsets.shape}")
    if probabilities.shape[0] != points.shape[0]:
        raise ValueError("object_probabilities length must match points")

    voted_centers = points + pred_offsets
    object_mask = probabilities >= cfg.object_probability_threshold
    point_labels = np.zeros(points.shape[0], dtype=np.int64)
    if not np.any(object_mask):
        return {
            "instance_labels": point_labels,
            "voted_centers": voted_centers,
            "object_mask": object_mask,
            "raw_cluster_labels": np.full(points.shape[0], -1, dtype=np.int64),
            "cluster_count": 0,
            "dbscan_eps_m": cfg.dbscan_eps_m,
        }

    raw_object_labels = _dbscan(voted_centers[object_mask], cfg.dbscan_eps_m, cfg.dbscan_min_samples)
    raw_labels = np.full(points.shape[0], -1, dtype=np.int64)
    raw_labels[object_mask] = raw_object_labels

    next_cluster_id = 1
    for raw_label in sorted(int(label) for label in np.unique(raw_object_labels) if label >= 0):
        cluster_mask = object_mask & (raw_labels == raw_label)
        if int(np.count_nonzero(cluster_mask)) < cfg.min_cluster_points:
            continue
        point_labels[cluster_mask] = next_cluster_id
        next_cluster_id += 1

    return {
        "instance_labels": point_labels,
        "voted_centers": voted_centers,
        "object_mask": object_mask,
        "raw_cluster_labels": raw_labels,
        "cluster_count": next_cluster_id - 1,
        "dbscan_eps_m": cfg.dbscan_eps_m,
    }


def evaluate_instance_predictions(
    target_instance_labels: np.ndarray,
    predicted_instance_labels: np.ndarray,
    config: InstanceMetricConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predicted instance labels with Hungarian point-set IoU matching."""

    cfg = config if isinstance(config, InstanceMetricConfig) else InstanceMetricConfig.from_mapping(config)
    target = np.asarray(target_instance_labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predicted_instance_labels, dtype=np.int64).reshape(-1)
    if target.shape != pred.shape:
        raise ValueError(f"target and prediction shape mismatch: {target.shape} vs {pred.shape}")

    all_gt_ids = [int(value) for value in np.unique(target) if int(value) > 0]
    gt_ids: list[int] = []
    ignored_gt_ids: list[int] = []
    for gt_id in all_gt_ids:
        count = int(np.count_nonzero(target == gt_id))
        if count < cfg.ignore_gt_instances_below_points:
            ignored_gt_ids.append(gt_id)
        else:
            gt_ids.append(gt_id)

    pred_ids = [int(value) for value in np.unique(pred) if int(value) > 0]
    iou_matrix = compute_instance_iou_matrix(target, pred, gt_ids, pred_ids)
    matches = hungarian_matches(iou_matrix, gt_ids, pred_ids, cfg.instance_iou_threshold)
    matched_gt = {match["gt_id"] for match in matches}
    matched_pred = {match["pred_id"] for match in matches}
    unmatched_gt = [gt_id for gt_id in gt_ids if gt_id not in matched_gt]
    unmatched_pred = [pred_id for pred_id in pred_ids if pred_id not in matched_pred]

    precision = len(matches) / max(len(pred_ids), 1)
    recall = len(matches) / max(len(gt_ids), 1)
    mean_iou = float(np.mean([match["iou"] for match in matches])) if matches else 0.0
    merge_count, split_count = merge_split_counts(iou_matrix, cfg.merge_split_overlap_threshold)
    return {
        "cluster_count_gt": len(gt_ids),
        "cluster_count_pred": len(pred_ids),
        "ignored_gt_instances": len(ignored_gt_ids),
        "small_gt_instances": len(ignored_gt_ids),
        "instance_precision": float(precision),
        "instance_recall": float(recall),
        "instance_mean_iou": mean_iou,
        "matched_instance_count": len(matches),
        "unmatched_gt_count": len(unmatched_gt),
        "unmatched_pred_count": len(unmatched_pred),
        "merge_count": int(merge_count),
        "split_count": int(split_count),
        "matches": matches,
        "unmatched_gt_ids": unmatched_gt,
        "unmatched_pred_ids": unmatched_pred,
        "ignored_gt_ids": ignored_gt_ids,
        "gt_ids": gt_ids,
        "pred_ids": pred_ids,
        "iou_matrix": iou_matrix.tolist(),
    }


def compute_instance_iou_matrix(
    target: np.ndarray,
    pred: np.ndarray,
    gt_ids: list[int],
    pred_ids: list[int],
) -> np.ndarray:
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    if not gt_ids or not pred_ids:
        return iou_matrix
    gt_masks = [target == gt_id for gt_id in gt_ids]
    pred_masks = [pred == pred_id for pred_id in pred_ids]
    for row, gt_mask in enumerate(gt_masks):
        for col, pred_mask in enumerate(pred_masks):
            intersection = int(np.count_nonzero(gt_mask & pred_mask))
            union = int(np.count_nonzero(gt_mask | pred_mask))
            iou_matrix[row, col] = float(intersection / union) if union > 0 else 0.0
    return iou_matrix


def hungarian_matches(
    iou_matrix: np.ndarray,
    gt_ids: list[int],
    pred_ids: list[int],
    threshold: float,
) -> list[dict[str, float | int]]:
    if iou_matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:  # noqa: BLE001 - scipy is optional for import-time use.
        return greedy_matches(iou_matrix, gt_ids, pred_ids, threshold)
    rows, cols = linear_sum_assignment(-iou_matrix)
    matches: list[dict[str, float | int]] = []
    for row, col in zip(rows, cols):
        iou = float(iou_matrix[row, col])
        if iou >= threshold:
            matches.append({"gt_id": int(gt_ids[row]), "pred_id": int(pred_ids[col]), "iou": iou})
    return matches


def greedy_matches(
    iou_matrix: np.ndarray,
    gt_ids: list[int],
    pred_ids: list[int],
    threshold: float,
) -> list[dict[str, float | int]]:
    candidates = [
        (float(iou_matrix[row, col]), row, col)
        for row in range(iou_matrix.shape[0])
        for col in range(iou_matrix.shape[1])
        if float(iou_matrix[row, col]) >= threshold
    ]
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[dict[str, float | int]] = []
    for iou, row, col in sorted(candidates, reverse=True):
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        matches.append({"gt_id": int(gt_ids[row]), "pred_id": int(pred_ids[col]), "iou": iou})
    return matches


def merge_split_counts(iou_matrix: np.ndarray, overlap_threshold: float) -> tuple[int, int]:
    if iou_matrix.size == 0:
        return 0, 0
    overlaps = iou_matrix >= overlap_threshold
    split_count = int(np.count_nonzero(overlaps.sum(axis=1) > 1))
    merge_count = int(np.count_nonzero(overlaps.sum(axis=0) > 1))
    return merge_count, split_count


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(points).astype(np.int64)
    except Exception:  # noqa: BLE001 - fallback keeps preview usable without sklearn.
        return _dbscan_with_kdtree(points, eps, min_samples)


def _dbscan_with_kdtree(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    labels = np.full(points.shape[0], -1, dtype=np.int64)
    visited = np.zeros(points.shape[0], dtype=np.bool_)
    cluster_id = 0
    for point_index in range(points.shape[0]):
        if visited[point_index]:
            continue
        visited[point_index] = True
        neighbors = tree.query_ball_point(points[point_index], eps)
        if len(neighbors) < min_samples:
            continue
        labels[point_index] = cluster_id
        seeds = list(neighbors)
        cursor = 0
        while cursor < len(seeds):
            neighbor_index = int(seeds[cursor])
            cursor += 1
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = tree.query_ball_point(points[neighbor_index], eps)
                if len(neighbor_neighbors) >= min_samples:
                    seeds.extend(int(value) for value in neighbor_neighbors if int(value) not in seeds)
            if labels[neighbor_index] < 0:
                labels[neighbor_index] = cluster_id
        cluster_id += 1
    return labels
