"""Service-layer contracts for the pose-estimation backend."""

from .schemas import (
    CameraIntrinsics,
    ErrorCode,
    ErrorResponse,
    InferenceOptions,
    PoseInstanceResult,
    PoseRequest,
    PoseResponse,
    SchemaError,
)

__all__ = [
    "CameraIntrinsics",
    "ErrorCode",
    "ErrorResponse",
    "InferenceOptions",
    "PoseInstanceResult",
    "PoseRequest",
    "PoseResponse",
    "SchemaError",
]
