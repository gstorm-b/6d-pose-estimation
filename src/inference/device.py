"""Device resolution shared across the backend.

Default to GPU for both development and deployment; fall back to CPU only when
CUDA is unavailable.
"""

from __future__ import annotations


def resolve_device(preferred: str = "cuda") -> str:
    """Return `preferred` if usable, otherwise fall back to cpu.

    A `cuda`/`cuda:N` request falls back to `cpu` when no CUDA device is present.
    Importing torch is deferred so non-inference code paths stay lightweight.
    """

    requested = str(preferred or "cuda")
    if not requested.startswith("cuda"):
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return requested
    except Exception:  # noqa: BLE001 - torch missing or broken -> cpu
        pass
    return "cpu"
