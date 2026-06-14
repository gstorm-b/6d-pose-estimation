"""Request/response contracts for the pose-estimation backend service.

Defined with plain dataclasses (no pydantic dependency) so the contract is
usable from the library and the future FastAPI layer alike. Depth transport
(base64 npz vs raw buffer) is a P7 service concern; here the schema carries the
camera intrinsics, options, and structured results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class SchemaError(ValueError):
    """Raised when a request payload violates the contract."""


class ErrorCode:
    """Stable error codes returned by the service."""

    UNKNOWN_SKU = "unknown_sku"
    INVALID_INPUT = "invalid_input"
    NO_OBJECT_POINTS = "no_object_points"
    NO_INSTANCES = "no_instances"
    INTERNAL = "internal"

    ALL = frozenset({UNKNOWN_SKU, INVALID_INPUT, NO_OBJECT_POINTS, NO_INSTANCES, INTERNAL})


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels; image size in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def validate(self) -> None:
        if not (self.fx > 0 and self.fy > 0):
            raise SchemaError("intrinsics fx/fy must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise SchemaError("intrinsics width/height must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        try:
            return cls(
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
                width=int(data["width"]),
                height=int(data["height"]),
            )
        except KeyError as exc:
            raise SchemaError(f"intrinsics missing field: {exc}") from exc


@dataclass
class InferenceOptions:
    """Per-request inference knobs."""

    max_instances: int = 10
    refine: bool = True
    min_confidence: float = 0.0

    def validate(self) -> None:
        if self.max_instances <= 0:
            raise SchemaError("max_instances must be > 0")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise SchemaError("min_confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "InferenceOptions":
        if not data:
            return cls()
        return cls(
            max_instances=int(data.get("max_instances", 10)),
            refine=bool(data.get("refine", True)),
            min_confidence=float(data.get("min_confidence", 0.0)),
        )


@dataclass
class PoseRequest:
    """Metadata of a pose request. Point/depth payload is transported separately."""

    sku: str
    intrinsics: CameraIntrinsics
    options: InferenceOptions = field(default_factory=InferenceOptions)
    version: str | None = None

    def validate(self) -> None:
        if not self.sku:
            raise SchemaError("sku is required")
        self.intrinsics.validate()
        self.options.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "version": self.version,
            "intrinsics": self.intrinsics.to_dict(),
            "options": self.options.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseRequest":
        if "sku" not in data:
            raise SchemaError("request missing field: sku")
        if "intrinsics" not in data:
            raise SchemaError("request missing field: intrinsics")
        return cls(
            sku=str(data["sku"]),
            intrinsics=CameraIntrinsics.from_dict(data["intrinsics"]),
            options=InferenceOptions.from_dict(data.get("options")),
            version=data.get("version"),
        )


@dataclass
class PoseInstanceResult:
    """One ranked object instance pose."""

    instance_id: int
    object_to_camera: list[list[float]]  # 4x4 row-major
    confidence: float
    point_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoseResponse:
    """Ranked poses for one scene, highest confidence first."""

    sku: str
    model_version: str
    instances: list[PoseInstanceResult]
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "model_version": self.model_version,
            "instances": [instance.to_dict() for instance in self.instances],
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
        }


@dataclass
class ErrorResponse:
    """Structured error payload."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if self.code not in ErrorCode.ALL:
            raise SchemaError(f"unknown error code: {self.code}")

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}
