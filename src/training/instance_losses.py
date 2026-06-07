"""Losses for PointNet++ instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InstanceSegLossConfig:
    semantic_weight: float = 1.0
    offset_weight: float = 1.0
    embedding_weight: float = 0.1
    offset_smooth_l1_beta: float = 0.01
    delta_var: float = 0.5
    delta_dist: float = 1.5
    regularization_weight: float = 0.001
    min_points_per_instance: int = 16
    centroid_distance_weighting: bool = False
    centroid_distance_weight_min: float = 0.5
    centroid_distance_weight_max: float = 1.5
    semantic_class_weights: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "InstanceSegLossConfig":
        loss_config = dict(config or {})
        class_weights = loss_config.pop("semantic_class_weights", None)
        if class_weights is None:
            class_weights = loss_config.pop("class_weights", None)
        if class_weights is not None:
            class_weights = tuple(float(value) for value in class_weights)
        return cls(
            semantic_weight=float(loss_config.get("semantic_weight", cls.semantic_weight)),
            offset_weight=float(loss_config.get("offset_weight", cls.offset_weight)),
            embedding_weight=float(loss_config.get("embedding_weight", cls.embedding_weight)),
            offset_smooth_l1_beta=float(loss_config.get("offset_smooth_l1_beta", cls.offset_smooth_l1_beta)),
            delta_var=float(loss_config.get("delta_var", cls.delta_var)),
            delta_dist=float(loss_config.get("delta_dist", cls.delta_dist)),
            regularization_weight=float(loss_config.get("regularization_weight", cls.regularization_weight)),
            min_points_per_instance=int(loss_config.get("min_points_per_instance", cls.min_points_per_instance)),
            centroid_distance_weighting=bool(loss_config.get("centroid_distance_weighting", cls.centroid_distance_weighting)),
            centroid_distance_weight_min=float(loss_config.get("centroid_distance_weight_min", cls.centroid_distance_weight_min)),
            centroid_distance_weight_max=float(loss_config.get("centroid_distance_weight_max", cls.centroid_distance_weight_max)),
            semantic_class_weights=class_weights,
        )


class PointNet2InstanceLoss(nn.Module):
    """Composite semantic, centroid-offset, and embedding loss."""

    def __init__(self, config: InstanceSegLossConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, InstanceSegLossConfig) else InstanceSegLossConfig.from_mapping(config)
        if self.config.semantic_class_weights is None:
            self.register_buffer("semantic_class_weights", None)
        else:
            weights = torch.tensor(self.config.semantic_class_weights, dtype=torch.float32)
            self.register_buffer("semantic_class_weights", weights)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        semantic_logits = outputs["semantic_logits"]
        pred_offsets = outputs["offsets"]
        embeddings = outputs["embeddings"]
        semantic_labels = batch["semantic_labels"].to(device=semantic_logits.device, dtype=torch.long)
        target_offsets = batch["centroid_offsets"].to(device=pred_offsets.device, dtype=pred_offsets.dtype)
        valid_instance_mask = batch["valid_instance_mask"].to(device=pred_offsets.device, dtype=torch.bool)
        instance_labels = batch["instance_labels"].to(device=embeddings.device, dtype=torch.long)

        semantic_loss = F.cross_entropy(
            semantic_logits.transpose(1, 2),
            semantic_labels,
            weight=self._semantic_weights_for(semantic_logits),
        )
        offset_loss = self._offset_loss(pred_offsets, target_offsets, valid_instance_mask, batch)
        embedding_parts = self._embedding_loss(embeddings, instance_labels, valid_instance_mask)
        embedding_loss = embedding_parts["embedding_loss"]

        total_loss = (
            self.config.semantic_weight * semantic_loss
            + self.config.offset_weight * offset_loss
            + self.config.embedding_weight * embedding_loss
        )
        report = {
            "total_loss": total_loss,
            "semantic_loss": semantic_loss,
            "offset_loss": offset_loss,
            "embedding_loss": embedding_loss,
            "offset_l1_object": self._offset_l1(pred_offsets, target_offsets, valid_instance_mask),
            **self._near_far_offset_l1(pred_offsets, target_offsets, valid_instance_mask),
            **embedding_parts,
        }
        report["embedding_weighted_loss"] = self.config.embedding_weight * embedding_loss
        return report

    def _semantic_weights_for(self, semantic_logits: torch.Tensor) -> torch.Tensor | None:
        if self.semantic_class_weights is None:
            return None
        return self.semantic_class_weights.to(device=semantic_logits.device, dtype=semantic_logits.dtype)

    def _offset_loss(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        valid_instance_mask: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if not bool(valid_instance_mask.any()):
            return pred_offsets.sum() * 0.0

        per_coord_loss = F.smooth_l1_loss(
            pred_offsets,
            target_offsets,
            reduction="none",
            beta=self.config.offset_smooth_l1_beta,
        )
        per_point_loss = per_coord_loss.mean(dim=-1)
        masked_loss = per_point_loss[valid_instance_mask]
        if not self.config.centroid_distance_weighting:
            return masked_loss.mean()

        weights = batch["centroid_distance_weights"].to(device=pred_offsets.device, dtype=pred_offsets.dtype)
        weights = weights.clamp(
            min=self.config.centroid_distance_weight_min,
            max=self.config.centroid_distance_weight_max,
        )
        masked_weights = weights[valid_instance_mask]
        return (masked_loss * masked_weights).sum() / masked_weights.sum().clamp_min(1e-8)

    def _embedding_loss(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor,
        valid_instance_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.config.embedding_weight <= 0:
            zero = embeddings.sum() * 0.0
            return {
                "embedding_loss": zero,
                "embedding_pull": zero,
                "embedding_push": zero,
                "embedding_reg": zero,
                "embedding_valid_instances": zero,
            }

        batch_pull: list[torch.Tensor] = []
        batch_push: list[torch.Tensor] = []
        batch_reg: list[torch.Tensor] = []
        valid_instance_counts: list[torch.Tensor] = []

        for sample_embeddings, sample_labels, sample_valid_mask in zip(embeddings, instance_labels, valid_instance_mask, strict=True):
            means: list[torch.Tensor] = []
            pull_terms: list[torch.Tensor] = []
            unique_ids = torch.unique(sample_labels[sample_valid_mask])
            for instance_id in unique_ids:
                if int(instance_id.item()) <= 0:
                    continue
                mask = sample_valid_mask & (sample_labels == instance_id)
                if int(mask.sum().item()) < self.config.min_points_per_instance:
                    continue
                instance_embeddings = sample_embeddings[mask]
                mean = instance_embeddings.mean(dim=0)
                means.append(mean)
                distances = torch.linalg.norm(instance_embeddings - mean.unsqueeze(0), dim=1)
                pull_terms.append(torch.relu(distances - self.config.delta_var).pow(2).mean())

            if not means:
                continue

            means_tensor = torch.stack(means, dim=0)
            batch_pull.append(torch.stack(pull_terms).mean())
            batch_reg.append(torch.linalg.norm(means_tensor, dim=1).mean())
            valid_instance_counts.append(torch.as_tensor(float(len(means)), device=embeddings.device, dtype=embeddings.dtype))
            if means_tensor.shape[0] > 1:
                distances = torch.pdist(means_tensor, p=2)
                batch_push.append(torch.relu(2.0 * self.config.delta_dist - distances).pow(2).mean())
            else:
                batch_push.append(embeddings.sum() * 0.0)

        if not batch_pull:
            zero = embeddings.sum() * 0.0
            return {
                "embedding_loss": zero,
                "embedding_pull": zero,
                "embedding_push": zero,
                "embedding_reg": zero,
                "embedding_valid_instances": zero,
            }

        pull = torch.stack(batch_pull).mean()
        push = torch.stack(batch_push).mean()
        reg = torch.stack(batch_reg).mean()
        embedding_loss = pull + push + self.config.regularization_weight * reg
        valid_instances = torch.stack(valid_instance_counts).sum()
        return {
            "embedding_loss": embedding_loss,
            "embedding_pull": pull,
            "embedding_push": push,
            "embedding_reg": reg,
            "embedding_valid_instances": valid_instances,
        }

    def _offset_l1(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(mask.any()):
            return pred_offsets.sum() * 0.0
        return torch.abs(pred_offsets[mask] - target_offsets[mask]).mean()

    def _near_far_offset_l1(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        valid_instance_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not bool(valid_instance_mask.any()):
            zero = pred_offsets.sum() * 0.0
            return {
                "offset_l1_near_centroid": zero,
                "offset_l1_far_centroid": zero,
            }

        distances = torch.linalg.norm(target_offsets, dim=-1)
        valid_distances = distances[valid_instance_mask]
        threshold = valid_distances.median()
        near_mask = valid_instance_mask & (distances <= threshold)
        far_mask = valid_instance_mask & (distances > threshold)
        return {
            "offset_l1_near_centroid": self._offset_l1(pred_offsets, target_offsets, near_mask),
            "offset_l1_far_centroid": self._offset_l1(pred_offsets, target_offsets, far_mask),
        }


def build_pointnet2_instance_loss_from_config(config: dict[str, Any]) -> PointNet2InstanceLoss:
    return PointNet2InstanceLoss(config.get("loss", {}))
