"""Segmentation metrics for semantic point labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationMeter:
    num_classes: int

    def __post_init__(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        pred = np.asarray(predictions).reshape(-1)
        target = np.asarray(targets).reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        encoded = target[valid] * self.num_classes + pred[valid]
        counts = np.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | list[list[int]]]:
        conf = self.confusion.astype(np.float64)
        true_positive = np.diag(conf)
        support = conf.sum(axis=1)
        predicted = conf.sum(axis=0)
        union = support + predicted - true_positive
        total = conf.sum()

        class_accuracy = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
        iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
        precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
        recall = class_accuracy

        metrics: dict[str, float | list[list[int]]] = {
            "overall_accuracy": float(true_positive.sum() / total) if total > 0 else 0.0,
            "mean_class_accuracy": float(class_accuracy.mean()) if class_accuracy.size else 0.0,
            "mean_iou": float(iou.mean()) if iou.size else 0.0,
            "confusion_matrix": self.confusion.astype(int).tolist(),
        }
        for idx in range(self.num_classes):
            metrics[f"class_{idx}_iou"] = float(iou[idx])
            metrics[f"class_{idx}_precision"] = float(precision[idx])
            metrics[f"class_{idx}_recall"] = float(recall[idx])
        if self.num_classes > 1:
            metrics["object_iou"] = float(iou[1])
            metrics["object_precision"] = float(precision[1])
            metrics["object_recall"] = float(recall[1])
        return metrics
