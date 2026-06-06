"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)
