"""Generate a small synthetic bin-picking dataset with Blender.

Run from the project root:

    blender --background --python scripts/generate_synthetic_blender.py -- --samples 3

On this machine Blender is available at:

    C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector


OBJECT_CLASS_ID = 1
BACKGROUND_ID = 0
ARTIFACT_RANDOM_DROPOUT = 1
ARTIFACT_EDGE_DROPOUT = 2
ARTIFACT_BLOB_DROPOUT = 4
ARTIFACT_EDGE_SMEAR = 8
ARTIFACT_FLYING_PIXEL = 16
ARTIFACT_BRIDGE = 32
ARTIFACT_RANGE_NOISE = 64
ARTIFACT_QUANTIZATION = 128

# Pending rigid bodies stay in the Bullet world as static colliders, so they are
# parked outside the bin and teleported to the drop position this many frames
# before activation. The static frames zero the kinematic->dynamic handoff velocity.
PARK_STATIC_LEAD_FRAMES = 3
SPAWN_SEPARATION_XY_TRIES = 150
SPAWN_SEPARATION_LIFT_TRIES = 25
# Below this spawn spacing (~0.5 s at 24 fps) the previous object may still be
# falling inside the drop band when the next one teleports in, so drop positions
# must keep the hard 3D separation even for parked objects.
SPAWN_FALL_CLEAR_FRAMES = 12


@dataclass
class GenerationScene:
    template: bpy.types.Object
    depth_camera: bpy.types.Object
    rgb_camera: bpy.types.Object
    debug_camera: bpy.types.Object
    class_name: str
    stereo_right_camera: bpy.types.Object | None = None
    stereo_projector: bpy.types.Object | None = None


@dataclass
class SimulationVideoRecorder:
    enabled: bool
    frame_step: int = 4
    objects: list[bpy.types.Object] | None = None
    spawn_frames: dict[str, int] | None = None
    samples: dict[str, dict[int, tuple[tuple[float, float, float], tuple[float, float, float], bool]]] | None = None
    last_recorded_frame: int = -1

    def __post_init__(self) -> None:
        if self.objects is None:
            self.objects = []
        if self.spawn_frames is None:
            self.spawn_frames = {}
        if self.samples is None:
            self.samples = {}
        self.frame_step = max(1, int(self.frame_step))

    def register_object(self, obj: bpy.types.Object, *, spawn_frame: int | None = None) -> None:
        if not self.enabled:
            return
        assert self.objects is not None
        assert self.spawn_frames is not None
        assert self.samples is not None
        self.objects.append(obj)
        self.spawn_frames[obj.name] = int(spawn_frame if spawn_frame is not None else bpy.context.scene.frame_current)
        self.samples.setdefault(obj.name, {})

    def record_current(self, *, force: bool = False, frame_override: int | None = None) -> None:
        # Progressive baking re-bakes each object from frame 1, so the caller
        # passes a monotonically increasing global frame to keep one continuous
        # video timeline instead of overwriting frame 1..window every stage.
        if not self.enabled:
            return
        frame = int(frame_override if frame_override is not None else bpy.context.scene.frame_current)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        assert self.objects is not None
        assert self.samples is not None
        for obj in self.objects:
            if obj.name not in bpy.data.objects:
                continue
            # Read the evaluated transform so baked rigid-body motion is captured
            # even in progressive mode, where the basis matrix_world stays at spawn.
            matrix_world = obj.evaluated_get(depsgraph).matrix_world.copy()
            location = matrix_world.to_translation()
            rotation = matrix_world.to_euler()
            self.samples.setdefault(obj.name, {})[frame] = (
                tuple(float(v) for v in location),
                tuple(float(v) for v in rotation),
                bool(obj.hide_viewport or obj.hide_render),
            )
        self.last_recorded_frame = frame

    def apply_animation_keyframes(self, *, frame_start: int, frame_end: int) -> int:
        if not self.enabled:
            return 0
        assert self.objects is not None
        assert self.spawn_frames is not None
        assert self.samples is not None
        inserted = 0
        for obj in self.objects:
            if obj.name not in bpy.data.objects:
                continue
            obj.animation_data_clear()
            spawn_frame = max(int(frame_start), int(self.spawn_frames.get(obj.name, frame_start)))
            obj_samples = self.samples.get(obj.name, {})
            if spawn_frame > frame_start:
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_start)
                obj.keyframe_insert(data_path="hide_render", frame=frame_start)
                obj.keyframe_insert(data_path="hide_viewport", frame=spawn_frame - 1)
                obj.keyframe_insert(data_path="hide_render", frame=spawn_frame - 1)
                inserted += 2
            for frame in sorted(f for f in obj_samples if spawn_frame <= f <= frame_end):
                location, rotation, hidden = obj_samples[frame]
                obj.location = location
                obj.rotation_euler = rotation
                obj.hide_viewport = hidden
                obj.hide_render = hidden
                obj.keyframe_insert(data_path="location", frame=frame)
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                obj.keyframe_insert(data_path="hide_render", frame=frame)
                inserted += 1
            if obj.animation_data and obj.animation_data.action:
                set_animation_interpolation(obj.animation_data.action)
        return inserted

    def summary(self, *, frame_start: int, frame_end: int) -> dict[str, Any]:
        if not self.enabled:
            return {"objects": 0, "visible_samples": 0, "z_min": None, "z_max": None}
        assert self.samples is not None
        visible_samples = 0
        z_values: list[float] = []
        for obj_samples in self.samples.values():
            for frame, (location, _rotation, hidden) in obj_samples.items():
                if frame_start <= frame <= frame_end and not hidden:
                    visible_samples += 1
                    z_values.append(float(location[2]))
        return {
            "objects": len(self.samples),
            "visible_samples": visible_samples,
            "z_min": min(z_values) if z_values else None,
            "z_max": max(z_values) if z_values else None,
        }


def set_animation_interpolation(action: bpy.types.Action, *, location_interpolation: str = "LINEAR") -> None:
    constant_paths = {"hide_viewport", "hide_render", "rigid_body.enabled", "rigid_body.kinematic"}
    for fcurve in action.fcurves:
        if fcurve.data_path in constant_paths:
            interpolation = "CONSTANT"
        elif fcurve.data_path == "location":
            interpolation = location_interpolation
        else:
            interpolation = "LINEAR"
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = interpolation


@dataclass
class ObjectPhysicsSettings:
    """Rigid-body parameters applied to every spawned object copy."""

    mass_kg: float
    friction: float
    restitution: float
    linear_damping: float
    angular_damping: float
    collision_margin: float
    collision_shape: str


@dataclass
class ObjectAppearance:
    """Material parameters for the spawned object copies."""

    color_rgb: tuple[float, float, float]
    metallic: float
    roughness: float


@dataclass
class RenderSettings:
    """Camera optics and Cycles/color-management settings shared by all cameras."""

    sensor_width: float
    clip_start: float
    clip_end: float
    cycles_samples: int
    view_transform: str
    view_look: str
    view_exposure: float
    view_gamma: float


def evaluated_translation(obj: bpy.types.Object) -> Vector:
    """World-space origin of an object after rigid-body evaluation.

    Baked rigid-body results are written to the evaluated object, not the basis
    matrix_world, so progressive baking must read the evaluated transform.
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(depsgraph).matrix_world.translation.copy()


@dataclass
class PhysicsExplosionWatchdog:
    """Abort an attempt early when solver impulses eject an object at unphysical speed.

    Speed is estimated from per-frame position deltas of activated objects only,
    so parking teleports of still-pending objects are never measured.
    """

    enabled: bool
    speed_limit_m_s: float
    z_min_limit_m: float
    z_max_limit_m: float
    fps: float
    objects: list[bpy.types.Object] = field(default_factory=list)
    spawn_frames: dict[str, int] = field(default_factory=dict)
    previous_locations: dict[str, Vector] = field(default_factory=dict)
    violation: dict[str, Any] | None = None
    use_evaluated: bool = False

    def check_current_frame(self) -> dict[str, Any] | None:
        if not self.enabled or self.violation is not None:
            return self.violation
        frame = int(bpy.context.scene.frame_current)
        for obj in self.objects:
            if obj.name not in bpy.data.objects:
                continue
            if frame < int(self.spawn_frames.get(obj.name, 1)):
                continue
            location = evaluated_translation(obj) if self.use_evaluated else obj.matrix_world.translation.copy()
            previous = self.previous_locations.get(obj.name)
            self.previous_locations[obj.name] = location
            if previous is None:
                continue
            speed = (location - previous).length * self.fps
            if speed > self.speed_limit_m_s or location.z < self.z_min_limit_m or location.z > self.z_max_limit_m:
                self.violation = {
                    "object": obj.name,
                    "frame": frame,
                    "speed_m_s": round(float(speed), 3),
                    "speed_limit_m_s": round(float(self.speed_limit_m_s), 3),
                    "location_m": [round(float(v), 4) for v in location],
                    "z_limits_m": [round(float(self.z_min_limit_m), 4), round(float(self.z_max_limit_m), 4)],
                }
                return self.violation
        return None


def auto_explosion_speed_limit(max_spawn_z_m: float) -> float:
    """Speed limit from free-fall energy: no legitimate motion should exceed it."""

    free_fall_speed = math.sqrt(2.0 * 9.81 * max(max_spawn_z_m, 0.05))
    return max(4.0, 1.5 * free_fall_speed)


def synthetic_log(message: str) -> None:
    print(message, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-file", help="Load generator settings from a JSON preset before applying CLI overrides.")
    parser.add_argument("--export-settings", help="Write the effective generator settings to a JSON preset.")
    parser.add_argument("--model", default="object-model/K41144.stl")
    parser.add_argument("--class-name", help="Semantic class name. Defaults to the model filename stem.")
    parser.add_argument("--model-scale", type=float, default=1.0, help="Scale applied to STL vertices before simulation. Use 0.001 for millimeter STL files.")
    parser.add_argument("--output", default="synthetic-data/K41144")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--objects", type=int, default=12)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--depth-camera-location", type=float, nargs=3, default=(0.0, -0.05, 0.42))
    parser.add_argument("--depth-camera-target", type=float, nargs=3, default=(0.0, 0.0, 0.025))
    parser.add_argument("--depth-camera-lens", type=float, default=35.0)
    # Active-stereo simulation (Route A): render a rectified L/R pair from the
    # depth camera plus a random-dot projector, so a real SGM/SGBM matcher can
    # produce the depth (scripts/stereo_depth_postprocess.py) with the same
    # artifact physics as the Basler stereo camera. GT stays raycast-perfect.
    parser.add_argument("--stereo", action=argparse.BooleanOptionalAction, default=False,
                        help="Render a stereo pair + dot projector for SGM depth simulation.")
    parser.add_argument("--stereo-baseline", type=float, default=0.0998,
                        help="Stereo baseline in metres (Basler STA-100: 0.0998).")
    parser.add_argument("--stereo-samples", type=int, default=24,
                        help="Cycles samples for the L/R renders (denoise off; noise is wanted).")
    parser.add_argument("--stereo-projector-strength", type=float, default=40.0,
                        help="Dot projector spotlight power in watts.")
    parser.add_argument("--stereo-projector-dot-density", type=float, default=0.04,
                        help="Fraction of the pattern texture covered by dots.")
    parser.add_argument("--stereo-spot-size-deg", type=float, default=65.0,
                        help="Projector cone angle in degrees (should cover the camera FOV).")
    parser.add_argument("--ground-plane", action=argparse.BooleanOptionalAction, default=None,
                        help="Add a table surface under the bin (default: on when --stereo).")
    parser.add_argument("--ground-plane-size", type=float, default=1.6)
    parser.add_argument("--ground-plane-color", type=float, nargs=3, default=(0.28, 0.28, 0.29))
    parser.add_argument("--ground-plane-roughness", type=float, default=0.6)
    parser.add_argument("--rgb-camera-location", type=float, nargs=3, default=(0.0, -0.05, 0.42))
    parser.add_argument("--rgb-camera-target", type=float, nargs=3, default=(0.0, 0.0, 0.025))
    parser.add_argument("--rgb-camera-lens", type=float, default=35.0)
    parser.add_argument("--debug-camera-location", type=float, nargs=3, default=(0.45, -0.55, 0.45))
    parser.add_argument("--debug-camera-target", type=float, nargs=3, default=(0.0, 0.0, 0.06))
    parser.add_argument("--debug-camera-lens", type=float, default=22.0)
    parser.add_argument("--light-location", type=float, nargs=3, default=(0.0, -0.22, 0.45))
    parser.add_argument("--light-energy", type=float, default=90.0)
    parser.add_argument("--light-size", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bin-x", type=float, default=0.24)
    parser.add_argument("--bin-y", type=float, default=0.24)
    parser.add_argument("--bin-wall-height", type=float, default=0.10)
    parser.add_argument("--drop-height", type=float, default=0.16)
    parser.add_argument("--drop-height-jitter", type=float, default=0.0)
    parser.add_argument("--drop-height-min", type=float, default=None)
    parser.add_argument("--drop-height-max", type=float, default=None)
    parser.add_argument("--spawn-strategy", choices=("random", "layered"), default="layered")
    parser.add_argument("--objects-per-layer", type=int, default=6)
    parser.add_argument("--spawn-min-distance", type=float, default=0.045)
    parser.add_argument(
        "--spawn-mode",
        choices=("progressive", "grid", "delayed", "batch"),
        default="progressive",
        help=(
            "progressive: drop one object at a time just above the current pile, settle, freeze "
            "as a passive collider, then drop the next (PyBullet-style, fewest out-of-bin ejections). "
            "grid: place objects in tidy rows/columns at a fixed orientation, settle per layer. "
            "delayed: legacy parked delayed-activation. batch: all objects active from frame 1."
        ),
    )
    parser.add_argument("--grid-spacing", type=float, default=0.005, help="Grid mode: gap between objects in a row/column, in meters.")
    parser.add_argument(
        "--grid-drop-clearance",
        type=float,
        default=0.005,
        help="Grid mode: height each object/layer is placed above the floor or pile before settling, in meters.",
    )
    parser.add_argument(
        "--grid-jitter",
        type=float,
        default=0.0,
        help="Grid mode: random position (m) and orientation (rad) jitter for variety. 0 = perfectly tidy.",
    )
    parser.add_argument(
        "--grid-orientation",
        type=float,
        nargs=3,
        default=None,
        help="Grid mode: object orientation as XYZ euler degrees. Omit for auto-flat (smallest dimension down).",
    )
    parser.add_argument(
        "--grid-layers",
        type=int,
        default=0,
        help="Grid mode: number of stacked layers. 0 = auto (enough layers to place all objects).",
    )
    parser.add_argument(
        "--drop-clearance-min",
        type=float,
        default=0.03,
        help="Progressive mode: minimum drop height above the current pile top, in meters.",
    )
    parser.add_argument(
        "--drop-clearance-max",
        type=float,
        default=0.12,
        help="Progressive mode: maximum drop height above the current pile top, in meters.",
    )
    parser.add_argument(
        "--progressive-settle-frames",
        type=int,
        default=45,
        help="Progressive mode: frames simulated for each object to fall and settle before freezing it.",
    )
    parser.add_argument(
        "--final-relax-frames",
        type=int,
        default=60,
        help="Progressive mode: frames of a final all-active relaxation so the pile self-adjusts.",
    )
    parser.add_argument(
        "--progressive-freeze",
        dest="progressive_freeze",
        action="store_true",
        help="Progressive mode: freeze each object as a passive collider once settled (faster).",
    )
    parser.add_argument(
        "--no-progressive-freeze",
        dest="progressive_freeze",
        action="store_false",
        help="Progressive mode: keep settled objects active so the pile keeps re-settling (slower, avoids floating).",
    )
    parser.set_defaults(progressive_freeze=True)
    parser.add_argument(
        "--spawn-settle-frames",
        type=int,
        default=35,
        help="delayed/batch modes: if > 0, enable one active rigid body at a time after this many settle frames.",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.0005,
        help="Rigid-body collision margin in meters. 0.5 mm keeps Bullet contacts stable without visible gaps.",
    )
    parser.add_argument("--collision-shape", choices=("CONVEX_HULL", "MESH"), default="CONVEX_HULL")
    parser.add_argument("--object-restitution", type=float, default=0.05, help="Rigid-body bounce/restitution for spawned objects.")
    parser.add_argument("--object-mass", type=float, default=0.08, help="Rigid-body mass in kg for spawned objects.")
    parser.add_argument(
        "--object-linear-damping",
        type=float,
        default=0.05,
        help="Rigid-body linear damping for spawned objects. High values make free fall unrealistically slow.",
    )
    parser.add_argument("--object-angular-damping", type=float, default=0.15, help="Rigid-body angular damping for spawned objects.")
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=60,
        help="Rigid-body world substeps per frame. Higher values reduce penetration depth and tunneling.",
    )
    parser.add_argument("--physics-solver-iterations", type=int, default=30, help="Rigid-body constraint solver iterations per substep.")
    parser.add_argument(
        "--object-friction",
        type=float,
        default=0.85,
        help="Rigid-body friction for spawned objects. Lower values let objects slide into gaps for a denser pile.",
    )
    parser.add_argument("--bin-friction", type=float, default=0.9, help="Rigid-body friction for the bin floor and walls.")
    parser.add_argument("--bin-restitution", type=float, default=0.05, help="Rigid-body restitution (bounce) for the bin floor and walls.")
    parser.add_argument("--gravity", type=float, default=-9.81, help="World gravity along z in m/s^2.")
    parser.add_argument(
        "--out-of-bin-min-z",
        type=float,
        default=-0.04,
        help="Lowest world z an object may reach before it counts as having dropped through the floor.",
    )
    parser.add_argument("--bin-floor-thickness", type=float, default=0.01, help="Bin floor thickness in meters.")
    parser.add_argument("--bin-wall-thickness", type=float, default=0.012, help="Bin wall thickness in meters.")
    parser.add_argument("--camera-sensor-width", type=float, default=32.0, help="Camera sensor width in mm for all cameras.")
    parser.add_argument("--camera-clip-start", type=float, default=0.01, help="Camera near clip distance in meters.")
    parser.add_argument("--camera-clip-end", type=float, default=1.50, help="Camera far clip distance in meters.")
    parser.add_argument(
        "--sensor-noise-profile",
        choices=("none", "structured_light", "tof"),
        default="none",
        help="Apply a post-raycast depth sensor artifact profile to raw depth/mask/points.",
    )
    parser.add_argument("--depth-quantization-m", type=float, default=0.001, help="Quantize valid depth to this step in metres when sensor noise is enabled.")
    parser.add_argument("--depth-quadratic-noise", type=float, default=0.004, help="Depth std coefficient: std = coef * depth^2, in metres.")
    parser.add_argument("--depth-random-dropout-prob", type=float, default=0.03, help="Random valid-pixel depth dropout probability.")
    parser.add_argument("--edge-gap-m", type=float, default=0.005, help="Local neighbor depth gap that marks a depth edge, in metres.")
    parser.add_argument("--edge-dropout-prob", type=float, default=0.40, help="Drop valid pixels near depth discontinuities.")
    parser.add_argument("--edge-smear-prob", type=float, default=0.25, help="Replace edge depth with a foreground/background mixture.")
    parser.add_argument("--edge-smear-radius-px", type=int, default=1, help="Dilate edge zones by this many pixels before smear/dropout.")
    parser.add_argument("--flying-pixel-prob", type=float, default=0.25, help="Create mixed-depth flying pixels near depth discontinuities.")
    parser.add_argument("--flying-pixel-alpha-min", type=float, default=0.20, help="Minimum foreground/background mix alpha for flying pixels.")
    parser.add_argument("--flying-pixel-alpha-max", type=float, default=0.80, help="Maximum foreground/background mix alpha for flying pixels.")
    parser.add_argument("--bridge-prob", type=float, default=0.15, help="Fill some background pixels near object edges with object-like mixed depth.")
    parser.add_argument("--bridge-max-gap-px", type=int, default=3, help="Search radius in pixels for bridge artifacts between adjacent objects.")
    parser.add_argument("--blob-dropout-count", type=int, default=3, help="Number of circular sensor dropout blobs per frame.")
    parser.add_argument("--blob-dropout-radius-px", type=int, default=8, help="Radius of each circular dropout blob in pixels.")
    parser.add_argument("--cycles-samples", type=int, default=48, help="Cycles render samples for rgb.png.")
    parser.add_argument("--view-transform", default="Filmic", help="Color management view transform, e.g. Filmic, Standard, AgX.")
    parser.add_argument("--view-look", default="Medium High Contrast", help="Color management look, e.g. None, Medium High Contrast.")
    parser.add_argument("--view-exposure", type=float, default=-1.2, help="Color management exposure.")
    parser.add_argument("--view-gamma", type=float, default=1.0, help="Color management gamma.")
    parser.add_argument("--object-color", type=float, nargs=3, default=(0.015, 0.014, 0.013), help="Object base color RGB in 0..1.")
    parser.add_argument("--object-metallic", type=float, default=0.85, help="Object material metallic in 0..1.")
    parser.add_argument("--object-roughness", type=float, default=0.85, help="Object material roughness in 0..1.")
    parser.add_argument("--bin-color", type=float, nargs=3, default=(0.12, 0.12, 0.115), help="Bin base color RGB in 0..1.")
    parser.add_argument("--bin-roughness", type=float, default=0.82, help="Bin material roughness in 0..1.")
    parser.add_argument("--light-type", choices=("AREA", "SUN", "POINT", "SPOT"), default="AREA", help="Light type for the scene light.")
    parser.add_argument("--world-color", type=float, nargs=3, default=(0.018, 0.018, 0.02), help="World background color RGB in 0..1.")
    parser.add_argument(
        "--explosion-speed-limit",
        type=float,
        default=None,
        help=(
            "Reject the attempt when any activated object moves faster than this many m/s. "
            "Omit for an automatic limit derived from drop height; 0 disables the watchdog."
        ),
    )
    parser.add_argument("--out-of-bin-tolerance", type=float, default=0.015)
    parser.add_argument(
        "--allow-out-of-bin-filtering",
        dest="allow_out_of_bin_filtering",
        action="store_true",
        help="Keep old behavior: hide out-of-bin objects and accept the remaining scene if visibility thresholds pass.",
    )
    parser.add_argument("--no-allow-out-of-bin-filtering", dest="allow_out_of_bin_filtering", action="store_false")
    parser.set_defaults(allow_out_of_bin_filtering=False)
    parser.add_argument("--min-visible-objects", type=int, default=8)
    parser.add_argument("--min-visible-points", type=int, default=2000)
    parser.add_argument("--max-sample-attempts", type=int, default=8)
    parser.add_argument("--settle-frames", type=int, default=220)
    parser.add_argument("--record-simulation-video", action="store_true", help="Render an MP4 preview of the accepted sample's object drop/settle animation.")
    parser.add_argument(
        "--record-failed-video",
        action="store_true",
        help="Also render an MP4 of each rejected attempt into <output>/rejected/ for failure investigation.",
    )
    parser.add_argument("--simulation-video-name", default="spawn_simulation.mp4")
    parser.add_argument("--simulation-video-frame-step", type=int, default=4, help="Render every Nth simulation frame in the preview video.")
    parser.add_argument("--simulation-video-fps", type=int, default=24)
    parser.add_argument("--simulation-video-resolution-percent", type=int, default=70)
    return parser


def blender_script_argv() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def load_settings_file(path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        settings = json.load(f)
    if not isinstance(settings, dict):
        raise ValueError(f"Settings file must contain a JSON object: {path}")

    valid_keys = {action.dest for action in parser._actions if action.dest != "help"}
    loaded = {str(key).replace("-", "_"): value for key, value in settings.items()}
    unknown = sorted(key for key in loaded if key not in valid_keys)
    if unknown:
        raise ValueError(f"Unknown generator setting(s) in {path}: {', '.join(unknown)}")
    return loaded


def parse_args() -> argparse.Namespace:
    argv = blender_script_argv()
    parser = build_arg_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--settings-file")
    pre_args, _unknown = pre_parser.parse_known_args(argv)
    if pre_args.settings_file:
        parser.set_defaults(**load_settings_file(Path(pre_args.settings_file), parser))
    return parser.parse_args(argv)


def generator_settings_dict(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"settings_file", "export_settings"}
    settings: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key in excluded:
            continue
        if isinstance(value, tuple):
            settings[key] = list(value)
        elif isinstance(value, Path):
            settings[key] = str(value)
        else:
            settings[key] = value
    return settings


def sanitize_blender_name(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "object"


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.72,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def import_stl(path: Path, class_name: str, model_scale: float) -> bpy.types.Object:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"Could not import STL: {path}")
    obj = imported[0]
    safe_name = sanitize_blender_name(class_name)
    obj.name = f"{safe_name}_template"
    obj.data.name = f"{safe_name}_mesh"
    center_mesh_for_physics(obj, model_scale)
    return obj


def center_mesh_for_physics(obj: bpy.types.Object, model_scale: float) -> None:
    """Move mesh vertices around their bbox center while preserving original STL pose metadata.

    Blender rigid bodies use the object origin as the body's center of mass. Some STL files
    have origins far from the physical center, so without this shift the object can behave
    like a pendulum. We keep the offset so metadata can still report poses for the original
    STL coordinate frame.
    """
    vertices = obj.data.vertices
    if not vertices:
        return

    mins = Vector((min(v.co.x for v in vertices), min(v.co.y for v in vertices), min(v.co.z for v in vertices)))
    maxs = Vector((max(v.co.x for v in vertices), max(v.co.y for v in vertices), max(v.co.z for v in vertices)))
    center = (mins + maxs) * 0.5
    for vertex in vertices:
        vertex.co = (vertex.co - center) * model_scale
    obj.data.update()
    obj["original_model_center_offset"] = [float(center.x), float(center.y), float(center.z)]
    obj["model_scale"] = float(model_scale)


def configure_rigid_body_margin(obj: bpy.types.Object, collision_margin: float) -> None:
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = collision_margin


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    collision_margin: float,
    friction: float,
    restitution: float,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "PASSIVE"
    obj.rigid_body.collision_shape = "BOX"
    obj.rigid_body.friction = friction
    obj.rigid_body.restitution = restitution
    configure_rigid_body_margin(obj, collision_margin)
    obj.select_set(False)
    return obj


def create_bin(
    bin_x: float,
    bin_y: float,
    wall_h: float,
    collision_margin: float,
    floor_thickness: float,
    wall_thickness: float,
    color_rgb: tuple[float, float, float],
    roughness: float,
    friction: float,
    restitution: float,
) -> list[bpy.types.Object]:
    mat = make_material("mat_bin_dark_gray", (color_rgb[0], color_rgb[1], color_rgb[2], 1.0), roughness=roughness)
    floor_t = floor_thickness
    wall_t = wall_thickness
    parts = [
        add_box("bin_floor", (0.0, 0.0, -floor_t / 2.0), (bin_x, bin_y, floor_t), mat, collision_margin, friction, restitution),
        add_box("bin_wall_pos_y", (0.0, bin_y / 2.0 + wall_t / 2.0, wall_h / 2.0), (bin_x + 2 * wall_t, wall_t, wall_h), mat, collision_margin, friction, restitution),
        add_box("bin_wall_neg_y", (0.0, -bin_y / 2.0 - wall_t / 2.0, wall_h / 2.0), (bin_x + 2 * wall_t, wall_t, wall_h), mat, collision_margin, friction, restitution),
        add_box("bin_wall_pos_x", (bin_x / 2.0 + wall_t / 2.0, 0.0, wall_h / 2.0), (wall_t, bin_y, wall_h), mat, collision_margin, friction, restitution),
        add_box("bin_wall_neg_x", (-bin_x / 2.0 - wall_t / 2.0, 0.0, wall_h / 2.0), (wall_t, bin_y, wall_h), mat, collision_margin, friction, restitution),
    ]
    return parts


def create_ground_plane(size: float, color_rgb: tuple[float, float, float], roughness: float) -> bpy.types.Object:
    """A table surface under/around the bin.

    Real captures always have geometry outside the bin (table, rig frame); a
    scene that ends at the bin walls leaves the stereo matcher free to
    hallucinate garbage correspondences on the empty world background. The
    plane sits flush under the bin floor, is labeled background by the raycast
    (it is not in the objects list), and needs no rigid body - the walls keep
    parts from ever reaching it.
    """

    mat = make_material("mat_ground_table", (color_rgb[0], color_rgb[1], color_rgb[2], 1.0), roughness=roughness)
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, -0.012 - 0.002))
    ground = bpy.context.object
    ground.name = "ground_table"
    ground.scale = (size / 2.0, size / 2.0, 0.002)
    bpy.ops.object.transform_apply(scale=True)
    ground.data.materials.append(mat)
    return ground


def random_rotation() -> tuple[float, float, float]:
    return (
        random.uniform(0.0, math.tau),
        random.uniform(0.0, math.tau),
        random.uniform(0.0, math.tau),
    )


def sample_drop_height(
    idx: int,
    count: int,
    drop_height: float,
    drop_height_jitter: float,
    drop_height_min: float | None,
    drop_height_max: float | None,
    spawn_strategy: str,
    objects_per_layer: int,
) -> tuple[float, int]:
    if drop_height_min is not None or drop_height_max is not None:
        min_h = drop_height if drop_height_min is None else drop_height_min
        max_h = drop_height if drop_height_max is None else drop_height_max
    else:
        min_h = drop_height - drop_height_jitter
        max_h = drop_height + drop_height_jitter

    if spawn_strategy == "layered":
        objects_per_layer = max(1, objects_per_layer)
        layer = idx // objects_per_layer
        layer_count = max(1, math.ceil(count / objects_per_layer))
        if layer_count == 1:
            center_h = (min_h + max_h) * 0.5
            band = max(0.002, max_h - min_h)
        else:
            span = max_h - min_h
            center_h = min_h + span * layer / (layer_count - 1)
            band = max(0.002, span / (layer_count * 3.0))
        return center_h + random.uniform(-band, band), layer

    return random.uniform(min_h, max_h), 0


def object_bounding_radius(template: bpy.types.Object) -> float:
    """Worst-case distance from the recentered origin to any vertex, in meters."""

    if not template.data.vertices:
        return 0.035
    return max(float(vertex.co.length) for vertex in template.data.vertices)


def object_spawn_edge_margin(template: bpy.types.Object, bin_x: float, bin_y: float, collision_margin: float) -> float:
    radius = object_bounding_radius(template)
    desired = max(0.035, radius + collision_margin)
    return min(desired, min(bin_x, bin_y) * 0.45)


def sample_separated_spawn_position(
    bin_x: float,
    bin_y: float,
    margin: float,
    z_base: float,
    existing_positions: list[tuple[float, float, float]],
    min_separation: float,
) -> tuple[float, float, float, int]:
    """Sample a drop position with a hard 3D separation against coexisting drops.

    Coexisting drops happen in batch mode (spawn_settle_frames=0) and for objects
    whose spawn frame is too early to park. Interpenetration at activation causes
    ballistic solver ejections, so when the bin footprint cannot satisfy the
    separation the candidate is lifted upward instead of accepting an overlap.
    Returns (x, y, z, lift_steps).
    """

    x_min = -bin_x / 2.0 + margin
    x_max = bin_x / 2.0 - margin
    y_min = -bin_y / 2.0 + margin
    y_max = bin_y / 2.0 - margin
    if x_min > x_max or y_min > y_max:
        return 0.0, 0.0, z_base, 0
    if not existing_positions:
        return random.uniform(x_min, x_max), random.uniform(y_min, y_max), z_base, 0

    lift_step = max(0.5 * min_separation, 0.02)
    for lift_idx in range(SPAWN_SEPARATION_LIFT_TRIES + 1):
        z = z_base + lift_idx * lift_step
        for _ in range(SPAWN_SEPARATION_XY_TRIES):
            candidate = (random.uniform(x_min, x_max), random.uniform(y_min, y_max), z)
            if all(math.dist(candidate, existing) >= min_separation for existing in existing_positions):
                return candidate[0], candidate[1], z, lift_idx
    # Guaranteed-clear fallback: rise above the highest existing drop position.
    z = max(existing[2] for existing in existing_positions) + min_separation
    return random.uniform(x_min, x_max), random.uniform(y_min, y_max), z, SPAWN_SEPARATION_LIFT_TRIES + 1


def configure_parked_spawn_location(
    obj: bpy.types.Object,
    *,
    spawn_frame: int,
    park_location: tuple[float, float, float],
    drop_location: tuple[float, float, float],
) -> bool:
    """Keep a pending object parked far outside the bin until shortly before activation.

    Pending rigid bodies (enabled=False) remain in the Bullet world as static
    colliders, so they must not wait inside the drop volume. The object is
    teleported to its drop position PARK_STATIC_LEAD_FRAMES before activation and
    held static there, which zeroes the kinematic->dynamic handoff velocity.
    Location keyframes use CONSTANT interpolation so the parked body never glides
    through the scene. Returns False when the spawn frame is too early to park.
    """

    teleport_frame = int(spawn_frame) - PARK_STATIC_LEAD_FRAMES
    if teleport_frame < 2:
        return False
    obj.location = park_location
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = drop_location
    obj.keyframe_insert(data_path="location", frame=teleport_frame)
    return True


def sample_spawn_xy(
    bin_x: float,
    bin_y: float,
    margin: float,
    layer_xy: list[tuple[float, float]],
    spawn_min_distance: float,
) -> tuple[float, float]:
    x_min = -bin_x / 2.0 + margin
    x_max = bin_x / 2.0 - margin
    y_min = -bin_y / 2.0 + margin
    y_max = bin_y / 2.0 - margin
    if x_min > x_max or y_min > y_max:
        return 0.0, 0.0

    best_candidate = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
    best_distance = -1.0
    for _ in range(120):
        candidate = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
        if not layer_xy:
            return candidate
        nearest = min(math.dist(candidate, existing) for existing in layer_xy)
        if nearest >= spawn_min_distance:
            return candidate
        if nearest > best_distance:
            best_candidate = candidate
            best_distance = nearest
    return best_candidate


def set_rigid_body_enabled_keyframe(obj: bpy.types.Object, *, frame: int, enabled: bool) -> None:
    obj.hide_viewport = not enabled
    obj.hide_render = not enabled
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)
    if obj.rigid_body is not None:
        obj.rigid_body.enabled = enabled
        obj.rigid_body.kinematic = not enabled
        obj.rigid_body.keyframe_insert(data_path="enabled", frame=frame)
        obj.rigid_body.keyframe_insert(data_path="kinematic", frame=frame)


def configure_delayed_rigid_body_activation(obj: bpy.types.Object, spawn_frame: int) -> None:
    spawn_frame = max(1, int(spawn_frame))
    if spawn_frame > 1:
        set_rigid_body_enabled_keyframe(obj, frame=1, enabled=False)
        set_rigid_body_enabled_keyframe(obj, frame=spawn_frame - 1, enabled=False)
    set_rigid_body_enabled_keyframe(obj, frame=spawn_frame, enabled=True)
    if obj.animation_data and obj.animation_data.action:
        # Parking keyframes must hold their value, not glide between positions.
        set_animation_interpolation(obj.animation_data.action, location_interpolation="CONSTANT")


def apply_object_material(template: bpy.types.Object, class_name: str, appearance: ObjectAppearance) -> str:
    """Assign the configured material to the template and return its safe name."""

    safe_name = sanitize_blender_name(class_name)
    color = appearance.color_rgb
    mat = make_material(
        f"mat_{safe_name}_object",
        (color[0], color[1], color[2], 1.0),
        metallic=appearance.metallic,
        roughness=appearance.roughness,
    )
    template.data.materials.clear()
    template.data.materials.append(mat)
    return safe_name


def spawn_active_object_copy(
    template: bpy.types.Object,
    safe_name: str,
    idx: int,
    location: tuple[float, float, float],
    physics: ObjectPhysicsSettings,
    rotation_euler: tuple[float, float, float] | None = None,
) -> bpy.types.Object:
    """Create one linked-mesh copy as an active rigid body at the given location.

    `rotation_euler` (radians) sets a fixed orientation; None uses a random one.
    """

    obj = template.copy()
    obj.data = template.data
    obj.name = f"{safe_name}_{idx + 1:03d}"
    obj.location = location
    obj.rotation_euler = random_rotation() if rotation_euler is None else rotation_euler
    obj.hide_viewport = False
    obj.hide_render = False
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    obj.select_set(False)
    obj.rigid_body.type = "ACTIVE"
    obj.rigid_body.mass = physics.mass_kg
    obj.rigid_body.friction = physics.friction
    obj.rigid_body.restitution = physics.restitution
    obj.rigid_body.linear_damping = physics.linear_damping
    obj.rigid_body.angular_damping = physics.angular_damping
    obj.rigid_body.collision_shape = physics.collision_shape
    configure_rigid_body_margin(obj, physics.collision_margin)
    return obj


def pile_top_z(objects: list[bpy.types.Object]) -> float:
    """Highest world-space z of any already-placed object bbox, floor at 0."""

    top = 0.0
    for obj in objects:
        if obj.name not in bpy.data.objects:
            continue
        _min_corner, max_corner = object_world_bounds(obj)
        top = max(top, float(max_corner.z))
    return top


def freeze_object_as_passive(obj: bpy.types.Object) -> None:
    """Bake the current settled pose into the object base and make it a static collider.

    Progressive baking re-bakes from frame 1 each time a new object is added, so a
    settled object must hold its pose with zero motion. A PASSIVE rigid body at its
    settled base pose stays perfectly still (verified drift 0.000 mm) while still
    colliding with later falling objects.
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()
    settled = obj.evaluated_get(depsgraph).matrix_world.copy()
    bpy.context.scene.frame_set(1)
    obj.animation_data_clear()
    obj.location = settled.to_translation()
    obj.rotation_euler = settled.to_euler()
    if obj.rigid_body is not None:
        obj.rigid_body.type = "PASSIVE"
        obj.rigid_body.kinematic = False


def rebase_active_objects_to_settled(objects: list[bpy.types.Object]) -> None:
    """Update every object's base pose to its current settled pose, keeping it ACTIVE.

    This is the no-freeze path: instead of turning settled objects into static
    colliders, they stay dynamic so the whole pile keeps re-settling each time a
    new object lands. That avoids objects being frozen mid-fall and left floating,
    at the cost of re-simulating all active bodies every stage. All evaluated
    poses are read before `frame_set(1)` resets the evaluation.
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()
    settled_poses = {
        obj.name: obj.evaluated_get(depsgraph).matrix_world.copy()
        for obj in objects
        if obj.name in bpy.data.objects
    }
    bpy.context.scene.frame_set(1)
    for obj in objects:
        pose = settled_poses.get(obj.name)
        if pose is None:
            continue
        obj.animation_data_clear()
        obj.location = pose.to_translation()
        obj.rotation_euler = pose.to_euler()
        if obj.rigid_body is not None:
            obj.rigid_body.type = "ACTIVE"
            obj.rigid_body.kinematic = False


def simulate_progressive(
    template: bpy.types.Object,
    class_name: str,
    count: int,
    bin_x: float,
    bin_y: float,
    drop_clearance_min: float,
    drop_clearance_max: float,
    progressive_settle_frames: int,
    final_relax_frames: int,
    physics: ObjectPhysicsSettings,
    appearance: ObjectAppearance,
    substeps: int,
    solver_iterations: int,
    gravity_z: float,
    watchdog_speed_limit: float,
    out_of_bin_min_z: float,
    freeze_settled: bool = True,
    recorder: SimulationVideoRecorder | None = None,
) -> tuple[list[bpy.types.Object], dict[str, Any] | None, int]:
    """Drop objects one at a time, settling each before the next.

    Returns (objects, explosion_violation_or_None, total_global_frames). Each
    object falls a short clearance above the current pile top, so impact energy
    stays low and objects rarely roll out of the bin.

    When `freeze_settled` is True (default), each settled object becomes a static
    passive collider, so only one active body simulates per stage (fast). When
    False, settled objects stay active and the whole pile re-settles each stage,
    which avoids objects being frozen mid-fall and left floating, at higher cost.
    """

    safe_name = apply_object_material(template, class_name, appearance)
    margin = object_spawn_edge_margin(template, bin_x, bin_y, physics.collision_margin)
    radius = object_bounding_radius(template)
    x_min = -bin_x / 2.0 + margin
    x_max = bin_x / 2.0 - margin
    y_min = -bin_y / 2.0 + margin
    y_max = bin_y / 2.0 - margin
    window = max(1, int(progressive_settle_frames))
    fps = float(bpy.context.scene.render.fps)
    scene = bpy.context.scene

    objects: list[bpy.types.Object] = []
    global_frame = 1
    for idx in range(count):
        top = pile_top_z(objects)
        clearance = random.uniform(drop_clearance_min, drop_clearance_max)
        # The mesh is recentered on its bbox center, so any rotation can put a
        # vertex up to `radius` below the origin. Drop the origin a full radius
        # above the pile so the object never spawns interpenetrating it.
        drop_z = top + radius + clearance
        if x_min > x_max or y_min > y_max:
            x = y = 0.0
        else:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
        obj = spawn_active_object_copy(template, safe_name, idx, (x, y, drop_z), physics)
        obj["spawn_layer"] = idx
        obj["spawn_height"] = drop_z
        obj["spawn_edge_margin"] = margin
        obj["spawn_z_lift_steps"] = 0
        obj["spawn_parked"] = False
        obj["spawn_frame"] = global_frame
        objects.append(obj)
        if recorder is not None:
            recorder.register_object(obj, spawn_frame=global_frame)

        bake_physics_window(window, substeps, solver_iterations, gravity_z)
        watchdog = PhysicsExplosionWatchdog(
            enabled=watchdog_speed_limit > 0.0,
            speed_limit_m_s=watchdog_speed_limit,
            z_min_limit_m=out_of_bin_min_z,
            z_max_limit_m=drop_z + 0.30,
            fps=fps,
            objects=[obj],
            spawn_frames={obj.name: 1},
            use_evaluated=True,
        )
        violation: dict[str, Any] | None = None
        for frame in range(1, window + 1):
            scene.frame_set(frame)
            # Force the rigid-body result into matrix_world. Without recording,
            # frame_set after free_bake_all leaves matrix_world stale, so the
            # watchdog and freeze would read the spawn pose instead of the fall.
            bpy.context.view_layer.update()
            if recorder is not None:
                recorder.record_current(frame_override=global_frame)
            global_frame += 1
            violation = watchdog.check_current_frame()
            if violation is not None:
                break
        if violation is not None:
            violation["stage"] = idx + 1
            return objects, violation, global_frame

        settled_z = float(evaluated_translation(obj).z)
        if freeze_settled:
            freeze_object_as_passive(obj)
        else:
            # Keep the whole pile dynamic so it re-settles when the next lands.
            rebase_active_objects_to_settled(objects)
        if idx == 0 or (idx + 1) % 5 == 0 or idx + 1 == count:
            synthetic_log(
                f"[synthetic] progressive dropped object {idx + 1}/{count} "
                f"pile_top={top:.3f} drop_z={drop_z:.3f} settled_z={settled_z:.3f} freeze={freeze_settled}"
            )

    template.hide_viewport = True
    template.hide_render = True

    relax_frames = max(0, int(final_relax_frames))
    if relax_frames > 0 and objects:
        for obj in objects:
            obj.animation_data_clear()
            if obj.rigid_body is not None:
                obj.rigid_body.type = "ACTIVE"
        bake_physics_window(relax_frames, substeps, solver_iterations, gravity_z)
        relax_watchdog = PhysicsExplosionWatchdog(
            enabled=watchdog_speed_limit > 0.0,
            speed_limit_m_s=watchdog_speed_limit,
            z_min_limit_m=out_of_bin_min_z,
            z_max_limit_m=pile_top_z(objects) + 0.30,
            fps=fps,
            objects=list(objects),
            spawn_frames={obj.name: 1 for obj in objects},
            use_evaluated=True,
        )
        for frame in range(1, relax_frames + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            if recorder is not None:
                recorder.record_current(frame_override=global_frame)
            global_frame += 1
            violation = relax_watchdog.check_current_frame()
            if violation is not None:
                violation["stage"] = "final_relax"
                return objects, violation, global_frame

    return objects, None, global_frame


def template_local_bbox(template: bpy.types.Object) -> tuple[Vector, Vector]:
    """Local-space bbox (min, max) of the recentered template mesh."""

    corners = [Vector(corner) for corner in template.bound_box]
    if not corners:
        return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))
    min_corner = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_corner = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return min_corner, max_corner


def auto_flat_orientation(template: bpy.types.Object) -> tuple[float, float, float]:
    """Euler (radians) that lays the object flattest: its smallest dimension points down."""

    min_corner, max_corner = template_local_bbox(template)
    dims = max_corner - min_corner
    smallest_axis = min(range(3), key=lambda axis: dims[axis])
    if smallest_axis == 2:
        return (0.0, 0.0, 0.0)
    if smallest_axis == 0:
        return (0.0, math.pi / 2.0, 0.0)  # x -> z
    return (math.pi / 2.0, 0.0, 0.0)  # y -> z


def oriented_footprint(template: bpy.types.Object, euler_rad: tuple[float, float, float]) -> tuple[float, float, float]:
    """Axis-aligned (x, y, z) extents of the template bbox rotated by the given euler."""

    min_corner, max_corner = template_local_bbox(template)
    rotation = Euler(euler_rad, "XYZ").to_matrix()
    corners = [
        rotation @ Vector((x, y, z))
        for x in (min_corner.x, max_corner.x)
        for y in (min_corner.y, max_corner.y)
        for z in (min_corner.z, max_corner.z)
    ]
    extent_x = max(c.x for c in corners) - min(c.x for c in corners)
    extent_y = max(c.y for c in corners) - min(c.y for c in corners)
    extent_z = max(c.z for c in corners) - min(c.z for c in corners)
    return float(extent_x), float(extent_y), float(extent_z)


def simulate_grid(
    template: bpy.types.Object,
    class_name: str,
    count: int,
    bin_x: float,
    bin_y: float,
    grid_spacing: float,
    grid_drop_clearance: float,
    grid_jitter: float,
    grid_orientation_deg: tuple[float, float, float] | None,
    grid_layers: int,
    final_relax_frames: int,
    layer_settle_frames: int,
    physics: ObjectPhysicsSettings,
    appearance: ObjectAppearance,
    substeps: int,
    solver_iterations: int,
    gravity_z: float,
    watchdog_speed_limit: float,
    out_of_bin_min_z: float,
    freeze_settled: bool = True,
    recorder: SimulationVideoRecorder | None = None,
) -> tuple[list[bpy.types.Object], dict[str, Any] | None, int]:
    """Place objects in tidy rows/columns at a fixed orientation, settling per layer.

    Returns (objects, explosion_violation_or_None, total_global_frames). Objects fill
    a row-major grid sized to the bin footprint; extra objects stack in further layers.
    Each layer is placed just above the current pile, then settled (and optionally
    frozen) before the next layer. With `grid_jitter > 0`, small position/orientation
    noise is added for variety while keeping the arrangement tidy.
    """

    safe_name = apply_object_material(template, class_name, appearance)
    if grid_orientation_deg is not None:
        orientation = tuple(math.radians(float(a)) for a in grid_orientation_deg)
    else:
        orientation = auto_flat_orientation(template)
    footprint_x, footprint_y, footprint_z = oriented_footprint(template, orientation)
    pitch_x = footprint_x + max(0.0, grid_spacing)
    pitch_y = footprint_y + max(0.0, grid_spacing)

    # Center positions must keep the whole footprint inside the bin.
    usable_x = max(0.0, bin_x - footprint_x)
    usable_y = max(0.0, bin_y - footprint_y)
    cols = max(1, int(usable_x / pitch_x) + 1)
    rows = max(1, int(usable_y / pitch_y) + 1)
    per_layer = cols * rows
    if grid_layers and grid_layers > 0:
        layers = int(grid_layers)
    else:
        layers = max(1, math.ceil(count / per_layer))
    span_x = (cols - 1) * pitch_x
    span_y = (rows - 1) * pitch_y

    synthetic_log(
        f"[synthetic] grid layout: cols={cols} rows={rows} per_layer={per_layer} layers={layers} "
        f"footprint=({footprint_x:.3f},{footprint_y:.3f},{footprint_z:.3f}) "
        f"orientation_deg=({math.degrees(orientation[0]):.0f},{math.degrees(orientation[1]):.0f},{math.degrees(orientation[2]):.0f})"
    )

    window = max(1, int(layer_settle_frames))
    fps = float(bpy.context.scene.render.fps)
    scene = bpy.context.scene
    objects: list[bpy.types.Object] = []
    global_frame = 1
    placed = 0
    for layer in range(layers):
        if placed >= count:
            break
        top = pile_top_z(objects)
        layer_z = top + footprint_z / 2.0 + max(0.0, grid_drop_clearance)
        layer_objects: list[bpy.types.Object] = []
        for slot in range(per_layer):
            if placed >= count:
                break
            col = slot % cols
            row = slot // cols
            x = -span_x / 2.0 + col * pitch_x
            y = -span_y / 2.0 + row * pitch_y
            rot = list(orientation)
            if grid_jitter > 0.0:
                x += random.uniform(-grid_jitter, grid_jitter)
                y += random.uniform(-grid_jitter, grid_jitter)
                rot = [angle + random.uniform(-grid_jitter, grid_jitter) for angle in orientation]
            obj = spawn_active_object_copy(
                template, safe_name, placed, (x, y, layer_z), physics, rotation_euler=tuple(rot)
            )
            obj["spawn_layer"] = layer
            obj["spawn_height"] = layer_z
            obj["spawn_edge_margin"] = 0.0
            obj["spawn_z_lift_steps"] = 0
            obj["spawn_parked"] = False
            obj["spawn_frame"] = global_frame
            objects.append(obj)
            layer_objects.append(obj)
            placed += 1
            if recorder is not None:
                recorder.register_object(obj, spawn_frame=global_frame)

        bake_physics_window(window, substeps, solver_iterations, gravity_z)
        watchdog = PhysicsExplosionWatchdog(
            enabled=watchdog_speed_limit > 0.0,
            speed_limit_m_s=watchdog_speed_limit,
            z_min_limit_m=out_of_bin_min_z,
            z_max_limit_m=layer_z + 0.30,
            fps=fps,
            objects=list(layer_objects),
            spawn_frames={obj.name: 1 for obj in layer_objects},
            use_evaluated=True,
        )
        violation: dict[str, Any] | None = None
        for frame in range(1, window + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            if recorder is not None:
                recorder.record_current(frame_override=global_frame)
            global_frame += 1
            violation = watchdog.check_current_frame()
            if violation is not None:
                break
        if violation is not None:
            violation["stage"] = f"layer_{layer + 1}"
            return objects, violation, global_frame

        if freeze_settled and layer < layers - 1:
            for obj in layer_objects:
                freeze_object_as_passive(obj)
        else:
            rebase_active_objects_to_settled(objects)
        synthetic_log(f"[synthetic] grid placed layer {layer + 1}/{layers}, objects so far {placed}/{count}")

    template.hide_viewport = True
    template.hide_render = True

    relax_frames = max(0, int(final_relax_frames))
    if relax_frames > 0 and objects:
        for obj in objects:
            obj.animation_data_clear()
            if obj.rigid_body is not None:
                obj.rigid_body.type = "ACTIVE"
        bake_physics_window(relax_frames, substeps, solver_iterations, gravity_z)
        relax_watchdog = PhysicsExplosionWatchdog(
            enabled=watchdog_speed_limit > 0.0,
            speed_limit_m_s=watchdog_speed_limit,
            z_min_limit_m=out_of_bin_min_z,
            z_max_limit_m=pile_top_z(objects) + 0.30,
            fps=fps,
            objects=list(objects),
            spawn_frames={obj.name: 1 for obj in objects},
            use_evaluated=True,
        )
        for frame in range(1, relax_frames + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            if recorder is not None:
                recorder.record_current(frame_override=global_frame)
            global_frame += 1
            violation = relax_watchdog.check_current_frame()
            if violation is not None:
                violation["stage"] = "final_relax"
                return objects, violation, global_frame

    return objects, None, global_frame


def create_object_instances(
    template: bpy.types.Object,
    class_name: str,
    count: int,
    bin_x: float,
    bin_y: float,
    drop_height: float,
    drop_height_jitter: float,
    drop_height_min: float | None,
    drop_height_max: float | None,
    spawn_strategy: str,
    objects_per_layer: int,
    spawn_min_distance: float,
    spawn_settle_frames: int,
    physics: ObjectPhysicsSettings,
    appearance: ObjectAppearance,
    recorder: SimulationVideoRecorder | None = None,
) -> list[bpy.types.Object]:
    safe_name = apply_object_material(template, class_name, appearance)

    objects = []
    spawn_xy_by_layer: dict[int, list[tuple[float, float]]] = {}
    coexisting_drop_positions: list[tuple[float, float, float]] = []
    margin = object_spawn_edge_margin(template, bin_x, bin_y, physics.collision_margin)
    radius = object_bounding_radius(template)
    # Two randomly rotated copies can touch up to 2 * bounding radius apart.
    min_separation = 2.0 * radius + max(physics.collision_margin, 0.0005)
    park_x_base = max(bin_x, bin_y) + 1.0
    park_x_step = radius * 2.0 + 0.05
    # With closely spaced spawns the previous object may still be falling inside
    # the drop band, so drop positions must keep the hard separation anyway.
    separation_required = int(spawn_settle_frames) < SPAWN_FALL_CLEAR_FRAMES
    bpy.context.scene.frame_set(1)
    for idx in range(count):
        obj = template.copy()
        obj.data = template.data
        obj.name = f"{safe_name}_{idx + 1:03d}"
        spawn_frame = 1 + idx * max(0, int(spawn_settle_frames))
        can_park = spawn_frame - PARK_STATIC_LEAD_FRAMES >= 2
        z, layer = sample_drop_height(
            idx,
            count,
            drop_height,
            drop_height_jitter,
            drop_height_min,
            drop_height_max,
            spawn_strategy,
            objects_per_layer,
        )
        lift_steps = 0
        if can_park and not separation_required:
            # Parked objects never coexist mid-air, so the per-layer distance is
            # only a spread heuristic, not a collision-safety requirement.
            layer_xy = spawn_xy_by_layer.setdefault(layer, [])
            x, y = sample_spawn_xy(bin_x, bin_y, margin, layer_xy, spawn_min_distance)
            layer_xy.append((x, y))
        else:
            x, y, z, lift_steps = sample_separated_spawn_position(
                bin_x,
                bin_y,
                margin,
                z,
                coexisting_drop_positions,
                min_separation,
            )
            coexisting_drop_positions.append((x, y, z))
        drop_location = (x, y, z)
        obj.location = drop_location
        obj["spawn_layer"] = layer
        obj["spawn_height"] = z
        obj["spawn_edge_margin"] = margin
        obj["spawn_z_lift_steps"] = int(lift_steps)
        obj.rotation_euler = random_rotation()
        obj.hide_viewport = False
        obj.hide_render = False
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.select_set(False)
        obj.rigid_body.type = "ACTIVE"
        obj.rigid_body.mass = physics.mass_kg
        obj.rigid_body.friction = physics.friction
        obj.rigid_body.restitution = physics.restitution
        obj.rigid_body.linear_damping = physics.linear_damping
        obj.rigid_body.angular_damping = physics.angular_damping
        obj.rigid_body.collision_shape = physics.collision_shape
        configure_rigid_body_margin(obj, physics.collision_margin)
        parked = False
        if can_park:
            park_location = (park_x_base + idx * park_x_step, 0.0, z)
            parked = configure_parked_spawn_location(
                obj,
                spawn_frame=spawn_frame,
                park_location=park_location,
                drop_location=drop_location,
            )
        obj["spawn_parked"] = bool(parked)
        configure_delayed_rigid_body_activation(obj, spawn_frame)
        obj["spawn_frame"] = int(spawn_frame)
        objects.append(obj)
        if recorder is not None:
            recorder.register_object(obj, spawn_frame=spawn_frame)
        if idx == 0 or (idx + 1) % 5 == 0 or idx + 1 == count:
            synthetic_log(
                f"[synthetic] scheduled object {idx + 1}/{count} "
                f"spawn_frame={spawn_frame} parked={parked} z_lift_steps={lift_steps}"
            )

    template.hide_viewport = True
    template.hide_render = True
    return objects


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(
    name: str,
    width: int,
    height: int,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    lens: float,
    render: "RenderSettings",
) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = name
    look_at(camera, Vector(target))
    camera.data.lens = lens
    camera.data.sensor_width = render.sensor_width
    camera.data.clip_start = render.clip_start
    camera.data.clip_end = render.clip_end
    setup_render_settings(width, height, render)
    return camera


def setup_render_settings(width: int, height: int, render: "RenderSettings") -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(render.cycles_samples))
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = render.view_transform
    scene.view_settings.look = render.view_look
    scene.view_settings.exposure = render.view_exposure
    scene.view_settings.gamma = render.view_gamma
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False


def setup_lighting(
    location: tuple[float, float, float],
    energy: float,
    size: float,
    light_type: str,
    world_color: tuple[float, float, float],
) -> None:
    bpy.ops.object.light_add(type=light_type, location=location)
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = energy
    if hasattr(light.data, "size"):
        light.data.size = size

    bpy.context.scene.world.color = (world_color[0], world_color[1], world_color[2])


def build_dot_pattern_image(name: str, size: int, density: float, seed: int) -> bpy.types.Image:
    """A random-dot projector pattern as an in-memory Blender image."""

    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size), dtype=np.float32)
    dot_count = max(1, int(size * size * density / 4.0))  # ~2x2 px per dot
    ys = rng.integers(0, size - 1, size=dot_count)
    xs = rng.integers(0, size - 1, size=dot_count)
    for dy in (0, 1):
        for dx in (0, 1):
            canvas[np.clip(ys + dy, 0, size - 1), np.clip(xs + dx, 0, size - 1)] = 1.0

    image = bpy.data.images.new(name, width=size, height=size, alpha=False, float_buffer=True)
    rgba = np.empty((size * size, 4), dtype=np.float32)
    rgba[:, 0] = rgba[:, 1] = rgba[:, 2] = canvas.reshape(-1)
    rgba[:, 3] = 1.0
    image.pixels.foreach_set(rgba.reshape(-1))
    image.pack()
    return image


def setup_stereo_rig(left_camera: bpy.types.Object, args: argparse.Namespace) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Rectified right camera + random-dot projector for active-stereo simulation.

    The right camera shares the left camera's rotation and optics and is offset
    by the baseline along the camera's local +x (image right), which matches the
    OpenCV left/right SGBM convention (positive disparities). The projector sits
    between the two cameras like the Basler's pattern module and is kept
    `hide_render=True` except while the stereo pair is rendered.
    """

    from mathutils import Matrix

    bpy.ops.object.camera_add()
    right = bpy.context.object
    right.name = "stereo_right_camera"
    right.data = left_camera.data.copy()
    right.matrix_world = left_camera.matrix_world @ Matrix.Translation((float(args.stereo_baseline), 0.0, 0.0))

    bpy.ops.object.light_add(type="SPOT")
    projector = bpy.context.object
    projector.name = "stereo_dot_projector"
    projector.matrix_world = left_camera.matrix_world @ Matrix.Translation((float(args.stereo_baseline) / 2.0, 0.0, 0.0))
    spot = projector.data
    spot.energy = float(args.stereo_projector_strength)
    spot.spot_size = math.radians(float(args.stereo_spot_size_deg))
    spot.spot_blend = 0.15
    spot.shadow_soft_size = 0.002  # near-point source: sharp dots + hard occlusion shadows

    pattern = build_dot_pattern_image(
        "stereo_dot_pattern", 1024, float(args.stereo_projector_dot_density), seed=20260705
    )
    spot.use_nodes = True
    tree = spot.node_tree
    emission = next(node for node in tree.nodes if node.type == "EMISSION")
    texcoord = tree.nodes.new("ShaderNodeTexCoord")
    transform = tree.nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "VECTOR"
    transform.convert_from = "WORLD"
    transform.convert_to = "OBJECT"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    # Perspective divide: u = x / -z, v = y / -z (light emits along local -z).
    div_u = tree.nodes.new("ShaderNodeMath")
    div_u.operation = "DIVIDE"
    div_v = tree.nodes.new("ShaderNodeMath")
    div_v.operation = "DIVIDE"
    neg_z = tree.nodes.new("ShaderNodeMath")
    neg_z.operation = "MULTIPLY"
    neg_z.inputs[1].default_value = -1.0
    scale = 0.5 / math.tan(math.radians(float(args.stereo_spot_size_deg)) / 2.0)
    map_u = tree.nodes.new("ShaderNodeMath")
    map_u.operation = "MULTIPLY_ADD"
    map_u.inputs[1].default_value = scale
    map_u.inputs[2].default_value = 0.5
    map_v = tree.nodes.new("ShaderNodeMath")
    map_v.operation = "MULTIPLY_ADD"
    map_v.inputs[1].default_value = scale
    map_v.inputs[2].default_value = 0.5
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = pattern
    tex.extension = "REPEAT"
    tex.interpolation = "Closest"

    links = tree.links
    links.new(texcoord.outputs["Normal"], transform.inputs["Vector"])
    links.new(transform.outputs["Vector"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], neg_z.inputs[0])
    links.new(separate.outputs["X"], div_u.inputs[0])
    links.new(neg_z.outputs["Value"], div_u.inputs[1])
    links.new(separate.outputs["Y"], div_v.inputs[0])
    links.new(neg_z.outputs["Value"], div_v.inputs[1])
    links.new(div_u.outputs["Value"], map_u.inputs[0])
    links.new(div_v.outputs["Value"], map_v.inputs[0])
    links.new(map_u.outputs["Value"], combine.inputs["X"])
    links.new(map_v.outputs["Value"], combine.inputs["Y"])
    links.new(combine.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], emission.inputs["Color"])

    projector.hide_render = True
    return right, projector


def render_stereo_pair(
    sample_dir: Path,
    left_camera: bpy.types.Object,
    right_camera: bpy.types.Object,
    projector: bpy.types.Object,
    args: argparse.Namespace,
) -> None:
    """Render the L/R grayscale pair with the projector on and denoising off.

    Render noise is deliberately kept (low sample count, no denoiser): it plays
    the role of sensor noise and is what makes SGM produce realistic ripple.
    """

    scene = bpy.context.scene
    saved = {
        "samples": scene.cycles.samples,
        "denoise": scene.cycles.use_denoising,
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "camera": scene.camera,
        "filepath": scene.render.filepath,
    }
    try:
        scene.cycles.samples = max(1, int(args.stereo_samples))
        scene.cycles.use_denoising = False
        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.color_depth = "8"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        projector.hide_render = False
        for camera, filename in ((left_camera, "stereo_left.png"), (right_camera, "stereo_right.png")):
            scene.camera = camera
            scene.render.filepath = str(sample_dir / filename)
            bpy.ops.render.render(write_still=True)
    finally:
        projector.hide_render = True
        scene.cycles.samples = saved["samples"]
        scene.cycles.use_denoising = saved["denoise"]
        scene.render.image_settings.color_mode = saved["color_mode"]
        scene.render.image_settings.color_depth = saved["color_depth"]
        scene.view_settings.view_transform = saved["view_transform"]
        scene.view_settings.look = saved["look"]
        scene.camera = saved["camera"]
        scene.render.filepath = saved["filepath"]


def configure_physics(
    frame_end: int = 260,
    substeps_per_frame: int = 60,
    solver_iterations: int = 30,
    gravity_z: float = -9.81,
) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.gravity = (0.0, 0.0, gravity_z)
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.time_scale = 1.0
    scene.rigidbody_world.substeps_per_frame = max(1, int(substeps_per_frame))
    scene.rigidbody_world.solver_iterations = max(1, int(solver_iterations))
    scene.rigidbody_world.point_cache.frame_start = 1
    scene.rigidbody_world.point_cache.frame_end = frame_end


def reset_physics_cache(frames: int, substeps_per_frame: int = 60, solver_iterations: int = 30, gravity_z: float = -9.81) -> None:
    configure_physics(frames, substeps_per_frame, solver_iterations, gravity_z)
    bpy.context.scene.frame_set(1)
    try:
        bpy.ops.ptcache.free_bake_all()
    except RuntimeError:
        pass


def bake_physics_window(frames: int, substeps_per_frame: int, solver_iterations: int, gravity_z: float = -9.81) -> None:
    """Free and re-bake the rigid-body cache for a fresh [1, frames] window.

    Progressive baking adds one rigid body and re-bakes from frame 1 each stage.
    An explicit bake_all is required: after free_bake_all, plain frame stepping
    can leave matrix_world at the spawn pose in background mode.
    """

    reset_physics_cache(frames, substeps_per_frame, solver_iterations, gravity_z)
    bpy.context.view_layer.update()
    try:
        bpy.ops.ptcache.bake_all(bake=True)
    except RuntimeError:
        pass


def settle_scene(frames: int, recorder: SimulationVideoRecorder | None = None) -> None:
    scene = bpy.context.scene
    scene.frame_set(1)
    if recorder is not None:
        recorder.record_current(force=True)
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        if recorder is not None:
            recorder.record_current()


def advance_scene_frames(
    frames: int,
    recorder: SimulationVideoRecorder | None = None,
    watchdog: PhysicsExplosionWatchdog | None = None,
) -> dict[str, Any] | None:
    """Step the simulation forward. Returns a watchdog violation dict on abort."""

    scene = bpy.context.scene
    start = scene.frame_current
    for frame in range(start + 1, start + frames + 1):
        scene.frame_set(frame)
        if recorder is not None:
            recorder.record_current()
        if watchdog is not None:
            violation = watchdog.check_current_frame()
            if violation is not None:
                return violation
    return None


def simulation_frame_count(args: argparse.Namespace) -> int:
    if args.spawn_mode == "progressive":
        return args.objects * max(1, args.progressive_settle_frames) + max(0, args.final_relax_frames)
    if args.spawn_mode == "grid":
        # Layout (layers) is computed at runtime from the object footprint; this is
        # only a logging estimate assuming a few objects per layer.
        approx_layers = max(1, math.ceil(args.objects / 8))
        return approx_layers * max(1, args.progressive_settle_frames) + max(0, args.final_relax_frames)
    if args.spawn_settle_frames <= 0:
        return args.settle_frames
    return 1 + args.settle_frames + max(0, args.objects - 1) * args.spawn_settle_frames


def keep_objects_inside_bin(
    objects: list[bpy.types.Object],
    bin_x: float,
    bin_y: float,
    out_of_bin_tolerance: float,
    min_z: float = -0.04,
) -> list[bpy.types.Object]:
    kept = []
    for obj in objects:
        loc = obj.matrix_world.translation
        in_bounds = (
            abs(float(loc.x)) <= bin_x / 2.0 + out_of_bin_tolerance
            and abs(float(loc.y)) <= bin_y / 2.0 + out_of_bin_tolerance
            and float(loc.z) >= min_z
        )
        if in_bounds:
            kept.append(obj)
        else:
            obj.hide_viewport = True
            obj.hide_render = True
    return kept


def object_world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    min_corner = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_corner = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return min_corner, max_corner


def find_out_of_bin_objects(
    objects: list[bpy.types.Object],
    bin_x: float,
    bin_y: float,
    out_of_bin_tolerance: float,
    min_z: float = -0.04,
) -> list[bpy.types.Object]:
    """Center-based out-of-bin test.

    An object counts as out of bin only when its center (recentered bbox origin)
    leaves the bin footprint, or its lowest point drops through the floor. A long
    part lying at an angle with its center inside but its bbox poking past the
    wall is a valid bin pose, not an ejection.
    """

    x_limit = bin_x / 2.0 + out_of_bin_tolerance
    y_limit = bin_y / 2.0 + out_of_bin_tolerance
    out_of_bin = []
    for obj in objects:
        center = evaluated_translation(obj)
        min_corner, _max_corner = object_world_bounds(obj)
        in_bounds = (
            abs(float(center.x)) <= x_limit
            and abs(float(center.y)) <= y_limit
            and float(min_corner.z) >= min_z
        )
        if not in_bounds:
            out_of_bin.append(obj)
    return out_of_bin


def describe_out_of_bin(
    objects: list[bpy.types.Object],
    out_of_bin_objects: list[bpy.types.Object],
    bin_x: float,
    bin_y: float,
    out_of_bin_tolerance: float,
) -> list[dict[str, Any]]:
    """Per-object diagnostics: is the center still inside, and how far the bbox overshoots.

    This distinguishes a real ejection (center outside the bin) from a long part
    whose bbox merely pokes past the wall while its center stays inside.
    """

    details = []
    for obj in out_of_bin_objects:
        center = evaluated_translation(obj)
        min_corner, _max_corner = object_world_bounds(obj)
        center_overshoot_mm = 1000.0 * max(
            abs(float(center.x)) - bin_x / 2.0,
            abs(float(center.y)) - bin_y / 2.0,
            0.0,
        )
        details.append(
            {
                "id": objects.index(obj) + 1,
                "center_xy": [round(float(center.x), 4), round(float(center.y), 4)],
                "center_overshoot_mm": round(center_overshoot_mm, 1),
                "min_z": round(float(min_corner.z), 4),
            }
        )
    return details


def hide_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        obj.hide_viewport = True
        obj.hide_render = True


def delete_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def camera_intrinsics(camera: bpy.types.Object, width: int, height: int) -> dict[str, float]:
    angle_x = camera.data.angle_x
    angle_y = camera.data.angle_y
    fx = width / (2.0 * math.tan(angle_x / 2.0))
    fy = height / (2.0 * math.tan(angle_y / 2.0))
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }


def matrix_to_list(matrix: Matrix) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def original_model_to_body_matrix(obj: bpy.types.Object) -> Matrix:
    offset = obj.get("original_model_center_offset", [0.0, 0.0, 0.0])
    model_scale = float(obj.get("model_scale", 1.0))
    scale_matrix = Matrix.Diagonal((model_scale, model_scale, model_scale, 1.0))
    return scale_matrix @ Matrix.Translation(Vector((-float(offset[0]), -float(offset[1]), -float(offset[2]))))


def save_ppm(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    with path.open("wb") as f:
        f.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def save_pgm(path: Path, gray: np.ndarray) -> None:
    gray = np.asarray(np.clip(gray, 0, 255), dtype=np.uint8)
    with path.open("wb") as f:
        f.write(f"P5\n{gray.shape[1]} {gray.shape[0]}\n255\n".encode("ascii"))
        f.write(gray.tobytes())


def save_png_rgb(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    height, width = rgb.shape[:2]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb.astype(np.float32) / 255.0
    image = bpy.data.images.new(path.stem, width=width, height=height, alpha=True)
    image.pixels.foreach_set(rgba.reshape(-1))
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def raycast_sensor_data(
    camera: bpy.types.Object,
    objects: list[bpy.types.Object],
    width: int,
    height: int,
) -> dict[str, np.ndarray | dict[str, float]]:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    intr = camera_intrinsics(camera, width, height)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    depth = np.zeros((height, width), dtype=np.float32)
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    normal_world = np.zeros((height, width, 3), dtype=np.float32)
    points_camera = []
    points_world = []
    point_instances = []
    point_pixels = []
    object_id = {obj.name: idx + 1 for idx, obj in enumerate(objects)}

    camera_origin = camera.matrix_world.translation
    camera_rot = camera.matrix_world.to_quaternion()
    world_to_camera = camera.matrix_world.inverted()
    camera_normal_rot = world_to_camera.to_3x3()
    max_distance = camera.data.clip_end

    for v in range(height):
        y_cam = -((v - cy) / fy)
        for u in range(width):
            x_cam = (u - cx) / fx
            ray_dir_camera = Vector((x_cam, y_cam, -1.0)).normalized()
            ray_dir_world = camera_rot @ ray_dir_camera
            hit, location, normal, _face_index, hit_object, _matrix = scene.ray_cast(
                depsgraph,
                camera_origin,
                ray_dir_world,
                distance=max_distance,
            )
            if not hit:
                continue
            point_camera_blender = world_to_camera @ location
            z_forward = -float(point_camera_blender.z)
            if z_forward <= 0:
                continue

            depth[v, u] = z_forward
            normal_world[v, u] = np.array(normal, dtype=np.float32)
            instance_id = object_id.get(hit_object.name, BACKGROUND_ID)
            instance_mask[v, u] = instance_id
            if instance_id != BACKGROUND_ID:
                points_camera.append([x_cam * z_forward, y_cam * z_forward, z_forward])
                points_world.append([float(location.x), float(location.y), float(location.z)])
                point_instances.append(instance_id)
                point_pixels.append([u, v])

    normals_camera = normal_world.reshape(-1, 3) @ np.array(camera_normal_rot.transposed(), dtype=np.float32)
    normals_camera = normals_camera.reshape(height, width, 3)
    points_camera_arr = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    points_world_arr = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    point_instances_arr = np.asarray(point_instances, dtype=np.uint16)
    semantic_ids = np.where(point_instances_arr > 0, OBJECT_CLASS_ID, BACKGROUND_ID).astype(np.uint16)
    point_pixels_arr = np.asarray(point_pixels, dtype=np.uint16).reshape(-1, 2)

    return {
        "intrinsics": intr,
        "depth_m": depth,
        "instance_mask": instance_mask,
        "normal_world": normal_world,
        "normal_camera": normals_camera.astype(np.float32),
        "points_camera": points_camera_arr,
        "points_world": points_world_arr,
        "point_instance_ids": point_instances_arr,
        "point_semantic_ids": semantic_ids,
        "point_pixels": point_pixels_arr,
    }


def _shift_2d(array: np.ndarray, dy: int, dx: int, fill: float | int) -> np.ndarray:
    out = np.full(array.shape, fill, dtype=array.dtype)
    h, w = array.shape[:2]
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def _dilate_bool(mask: np.ndarray, radius_px: int) -> np.ndarray:
    radius = max(0, int(radius_px))
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out |= _shift_2d(mask, dy, dx, False)
    return out


def _depth_edge_context(
    depth: np.ndarray,
    instance_mask: np.ndarray,
    normal_world: np.ndarray,
    normal_camera: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = depth > 0
    min_depth = np.full(depth.shape, np.inf, dtype=np.float32)
    max_depth = np.full(depth.shape, -np.inf, dtype=np.float32)
    min_instance = np.zeros(depth.shape, dtype=np.uint16)
    max_instance = np.zeros(depth.shape, dtype=np.uint16)
    min_normal_world = np.zeros(normal_world.shape, dtype=np.float32)
    max_normal_world = np.zeros(normal_world.shape, dtype=np.float32)
    min_normal_camera = np.zeros(normal_camera.shape, dtype=np.float32)
    max_normal_camera = np.zeros(normal_camera.shape, dtype=np.float32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted_depth = depth if (dy == 0 and dx == 0) else _shift_2d(depth, dy, dx, 0.0)
            shifted_instance = instance_mask if (dy == 0 and dx == 0) else _shift_2d(instance_mask, dy, dx, 0)
            shifted_world = normal_world if (dy == 0 and dx == 0) else _shift_2d(normal_world, dy, dx, 0.0)
            shifted_camera = normal_camera if (dy == 0 and dx == 0) else _shift_2d(normal_camera, dy, dx, 0.0)
            shifted_valid = shifted_depth > 0
            smaller = shifted_valid & (shifted_depth < min_depth)
            larger = shifted_valid & (shifted_depth > max_depth)
            min_depth[smaller] = shifted_depth[smaller]
            max_depth[larger] = shifted_depth[larger]
            min_instance[smaller] = shifted_instance[smaller]
            max_instance[larger] = shifted_instance[larger]
            min_normal_world[smaller] = shifted_world[smaller]
            max_normal_world[larger] = shifted_world[larger]
            min_normal_camera[smaller] = shifted_camera[smaller]
            max_normal_camera[larger] = shifted_camera[larger]

    depth_gap = np.where(valid & np.isfinite(min_depth) & np.isfinite(max_depth), max_depth - min_depth, 0.0)
    return {
        "depth_gap": depth_gap.astype(np.float32),
        "min_depth": min_depth,
        "max_depth": max_depth,
        "min_instance": min_instance,
        "max_instance": max_instance,
        "min_normal_world": min_normal_world,
        "max_normal_world": max_normal_world,
        "min_normal_camera": min_normal_camera,
        "max_normal_camera": max_normal_camera,
    }


def _copy_mixed_normals(
    target: np.ndarray,
    choose_min: np.ndarray,
    normal_min: np.ndarray,
    normal_max: np.ndarray,
) -> np.ndarray:
    out = target.copy()
    out[choose_min] = normal_min[choose_min]
    out[~choose_min] = normal_max[~choose_min]
    return out


def _apply_bridge_artifacts(
    depth: np.ndarray,
    instance_mask: np.ndarray,
    normal_world: np.ndarray,
    normal_camera: np.ndarray,
    artifact_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    prob: float,
    radius_px: int,
    edge_gap_m: float,
) -> None:
    radius = max(0, int(radius_px))
    if prob <= 0 or radius <= 0:
        return
    h, w = depth.shape
    obj_depth_sum = np.zeros_like(depth, dtype=np.float32)
    obj_depth_count = np.zeros_like(depth, dtype=np.uint16)
    nearest_id = np.zeros_like(instance_mask, dtype=np.uint16)
    nearest_depth = np.zeros_like(depth, dtype=np.float32)
    nearest_world = np.zeros_like(normal_world, dtype=np.float32)
    nearest_camera = np.zeros_like(normal_camera, dtype=np.float32)
    best_dist2 = np.full(depth.shape, 10**9, dtype=np.int32)
    object_pixels = (depth > 0) & (instance_mask > 0)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            dist2 = dy * dy + dx * dx
            if dist2 > radius * radius:
                continue
            shifted_object = _shift_2d(object_pixels, dy, dx, False)
            shifted_depth = _shift_2d(depth, dy, dx, 0.0)
            shifted_id = _shift_2d(instance_mask, dy, dx, 0)
            shifted_world = _shift_2d(normal_world, dy, dx, 0.0)
            shifted_camera = _shift_2d(normal_camera, dy, dx, 0.0)
            obj_depth_sum[shifted_object] += shifted_depth[shifted_object]
            obj_depth_count[shifted_object] += 1
            better = shifted_object & (dist2 < best_dist2)
            best_dist2[better] = dist2
            nearest_id[better] = shifted_id[better]
            nearest_depth[better] = shifted_depth[better]
            nearest_world[better] = shifted_world[better]
            nearest_camera[better] = shifted_camera[better]

    mean_depth = np.divide(
        obj_depth_sum,
        np.maximum(obj_depth_count.astype(np.float32), 1.0),
        out=np.zeros_like(obj_depth_sum),
        where=obj_depth_count > 0,
    )
    background_near_object = (instance_mask == 0) & (obj_depth_count >= 2) & (nearest_id > 0)
    if edge_gap_m > 0:
        background_near_object &= np.abs(mean_depth - depth) > edge_gap_m
    selected = background_near_object & (rng.random((h, w)) < prob)
    if not np.any(selected):
        return
    alpha = rng.uniform(0.35, 0.75, size=(h, w)).astype(np.float32)
    base_depth = np.where(depth > 0, depth, mean_depth)
    depth[selected] = (alpha[selected] * nearest_depth[selected]) + ((1.0 - alpha[selected]) * base_depth[selected])
    instance_mask[selected] = nearest_id[selected]
    normal_world[selected] = nearest_world[selected]
    normal_camera[selected] = nearest_camera[selected]
    artifact_mask[selected] |= ARTIFACT_BRIDGE


def _rebuild_sensor_points(
    sensor: dict[str, Any],
    camera: bpy.types.Object,
) -> None:
    depth = np.asarray(sensor["depth_m"], dtype=np.float32)
    instance_mask = np.asarray(sensor["instance_mask"], dtype=np.uint16)
    intr = sensor["intrinsics"]
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    v, u = np.nonzero((depth > 0) & (instance_mask > 0))
    if v.size == 0:
        sensor["points_camera"] = np.zeros((0, 3), dtype=np.float32)
        sensor["points_world"] = np.zeros((0, 3), dtype=np.float32)
        sensor["point_instance_ids"] = np.zeros((0,), dtype=np.uint16)
        sensor["point_semantic_ids"] = np.zeros((0,), dtype=np.uint16)
        sensor["point_pixels"] = np.zeros((0, 2), dtype=np.uint16)
        return

    z = depth[v, u]
    x = ((u.astype(np.float32) - cx) / fx) * z
    y = -((v.astype(np.float32) - cy) / fy) * z
    points_camera = np.stack([x, y, z], axis=1).astype(np.float32)
    points_camera_blender = np.stack([x, y, -z, np.ones_like(z)], axis=1).astype(np.float32)
    camera_to_world = np.array(camera.matrix_world, dtype=np.float32)
    points_world = (points_camera_blender @ camera_to_world.T)[:, :3].astype(np.float32)
    point_instances = instance_mask[v, u].astype(np.uint16)

    sensor["points_camera"] = points_camera
    sensor["points_world"] = points_world
    sensor["point_instance_ids"] = point_instances
    sensor["point_semantic_ids"] = np.where(point_instances > 0, OBJECT_CLASS_ID, BACKGROUND_ID).astype(np.uint16)
    sensor["point_pixels"] = np.stack([u, v], axis=1).astype(np.uint16)


def apply_depth_sensor_artifacts(
    sensor: dict[str, Any],
    camera: bpy.types.Object,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    if args.sensor_noise_profile == "none":
        return sensor

    rng = np.random.default_rng(seed + 17_171)
    noisy = dict(sensor)
    depth = np.asarray(sensor["depth_m"], dtype=np.float32).copy()
    instance_mask = np.asarray(sensor["instance_mask"], dtype=np.uint16).copy()
    normal_world = np.asarray(sensor["normal_world"], dtype=np.float32).copy()
    normal_camera = np.asarray(sensor["normal_camera"], dtype=np.float32).copy()
    artifact_mask = np.zeros(depth.shape, dtype=np.uint16)
    h, w = depth.shape

    context = _depth_edge_context(depth, instance_mask, normal_world, normal_camera)
    edge = context["depth_gap"] > float(args.edge_gap_m)
    edge = _dilate_bool(edge, int(args.edge_smear_radius_px))

    if args.edge_smear_prob > 0:
        selected = edge & (depth > 0) & (rng.random((h, w)) < float(args.edge_smear_prob))
        if np.any(selected):
            alpha = rng.uniform(float(args.flying_pixel_alpha_min), float(args.flying_pixel_alpha_max), size=(h, w)).astype(np.float32)
            depth[selected] = (
                alpha[selected] * context["min_depth"][selected]
                + (1.0 - alpha[selected]) * context["max_depth"][selected]
            )
            artifact_mask[selected] |= ARTIFACT_EDGE_SMEAR

    if args.flying_pixel_prob > 0:
        selected = edge & (depth > 0) & (rng.random((h, w)) < float(args.flying_pixel_prob))
        if np.any(selected):
            alpha_min = min(float(args.flying_pixel_alpha_min), float(args.flying_pixel_alpha_max))
            alpha_max = max(float(args.flying_pixel_alpha_min), float(args.flying_pixel_alpha_max))
            alpha = rng.uniform(alpha_min, alpha_max, size=(h, w)).astype(np.float32)
            depth[selected] = (
                alpha[selected] * context["min_depth"][selected]
                + (1.0 - alpha[selected]) * context["max_depth"][selected]
            )
            choose_min = rng.random((h, w)) < 0.5
            replace_id = np.where(choose_min, context["min_instance"], context["max_instance"]).astype(np.uint16)
            replace = selected & (replace_id > 0)
            instance_mask[replace] = replace_id[replace]
            normal_world[replace] = _copy_mixed_normals(
                normal_world, choose_min, context["min_normal_world"], context["max_normal_world"]
            )[replace]
            normal_camera[replace] = _copy_mixed_normals(
                normal_camera, choose_min, context["min_normal_camera"], context["max_normal_camera"]
            )[replace]
            artifact_mask[selected] |= ARTIFACT_FLYING_PIXEL

    _apply_bridge_artifacts(
        depth,
        instance_mask,
        normal_world,
        normal_camera,
        artifact_mask,
        rng,
        prob=float(args.bridge_prob),
        radius_px=int(args.bridge_max_gap_px),
        edge_gap_m=float(args.edge_gap_m),
    )

    dropout = np.zeros(depth.shape, dtype=bool)
    if args.depth_random_dropout_prob > 0:
        selected = (depth > 0) & (rng.random((h, w)) < float(args.depth_random_dropout_prob))
        dropout |= selected
        artifact_mask[selected] |= ARTIFACT_RANDOM_DROPOUT
    if args.edge_dropout_prob > 0:
        selected = edge & (depth > 0) & (rng.random((h, w)) < float(args.edge_dropout_prob))
        dropout |= selected
        artifact_mask[selected] |= ARTIFACT_EDGE_DROPOUT
    if int(args.blob_dropout_count) > 0 and int(args.blob_dropout_radius_px) > 0:
        valid_y, valid_x = np.nonzero(depth > 0)
        if valid_y.size > 0:
            count = min(int(args.blob_dropout_count), int(valid_y.size))
            centers = rng.choice(valid_y.size, size=count, replace=False)
            yy, xx = np.ogrid[:h, :w]
            radius2 = int(args.blob_dropout_radius_px) ** 2
            for center in centers:
                cy, cx = int(valid_y[center]), int(valid_x[center])
                selected = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius2
                dropout |= selected
                artifact_mask[selected] |= ARTIFACT_BLOB_DROPOUT
    if np.any(dropout):
        depth[dropout] = 0.0
        instance_mask[dropout] = 0
        normal_world[dropout] = 0.0
        normal_camera[dropout] = 0.0

    valid = depth > 0
    if args.depth_quadratic_noise > 0 and np.any(valid):
        std = float(args.depth_quadratic_noise) * (depth[valid] ** 2)
        depth[valid] += rng.normal(0.0, std).astype(np.float32)
        depth[depth < 0] = 0.0
        artifact_mask[valid] |= ARTIFACT_RANGE_NOISE
    valid = depth > 0
    if args.depth_quantization_m > 0 and np.any(valid):
        step = float(args.depth_quantization_m)
        depth[valid] = np.round(depth[valid] / step) * step
        artifact_mask[valid] |= ARTIFACT_QUANTIZATION
    invalid_after_noise = depth <= 0
    if np.any(invalid_after_noise):
        instance_mask[invalid_after_noise] = 0
        normal_world[invalid_after_noise] = 0.0
        normal_camera[invalid_after_noise] = 0.0

    noisy["depth_m"] = depth.astype(np.float32)
    noisy["instance_mask"] = instance_mask
    noisy["normal_world"] = normal_world.astype(np.float32)
    noisy["normal_camera"] = normal_camera.astype(np.float32)
    noisy["sensor_artifact_mask"] = artifact_mask
    noisy["sensor_noise_summary"] = {
        "profile": args.sensor_noise_profile,
        "artifact_pixels": int(np.count_nonzero(artifact_mask)),
        "artifact_pixel_fraction": float(np.count_nonzero(artifact_mask)) / float(max(1, artifact_mask.size)),
        "valid_pixels_before": int(np.count_nonzero(np.asarray(sensor["depth_m"]) > 0)),
        "valid_pixels_after": int(np.count_nonzero(depth > 0)),
        "object_pixels_before": int(np.count_nonzero(np.asarray(sensor["instance_mask"]) > 0)),
        "object_pixels_after": int(np.count_nonzero(instance_mask > 0)),
    }
    _rebuild_sensor_points(noisy, camera)
    return noisy


def render_rgb(path: Path, camera: bpy.types.Object) -> None:
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def remove_rigid_bodies_for_animation(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        if obj.name not in bpy.data.objects or obj.rigid_body is None:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.rigidbody.object_remove()
        except RuntimeError:
            pass
        obj.select_set(False)


def render_simulation_video(
    path: Path,
    camera: bpy.types.Object,
    objects: list[bpy.types.Object],
    *,
    recorder: SimulationVideoRecorder | None,
    frame_end: int,
    frame_step: int,
    fps: int,
    resolution_percent: int,
) -> None:
    if not objects:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    render = scene.render
    old_settings = {
        "camera": scene.camera,
        "engine": scene.render.engine,
        "filepath": render.filepath,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "frame_step": scene.frame_step,
        "fps": render.fps,
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
        "resolution_percentage": render.resolution_percentage,
        "file_format": render.image_settings.file_format,
        "ffmpeg_format": render.ffmpeg.format,
        "ffmpeg_codec": render.ffmpeg.codec,
        "ffmpeg_constant_rate_factor": render.ffmpeg.constant_rate_factor,
    }
    old_object_colors = {obj.name: tuple(obj.color) for obj in objects if obj.name in bpy.data.objects}
    old_materials_by_data: dict[str, list[bpy.types.Material]] = {}
    preview_hidden_objects = [obj for obj in bpy.data.objects if obj.name.startswith("bin_wall_")]
    old_preview_visibility = {
        obj.name: (bool(obj.hide_viewport), bool(obj.hide_render))
        for obj in preview_hidden_objects
    }
    shading = getattr(scene.display, "shading", None)
    old_shading = {
        "light": getattr(shading, "light", None) if shading is not None else None,
        "color_type": getattr(shading, "color_type", None) if shading is not None else None,
    }
    remove_rigid_bodies_for_animation(objects)
    keyframes = recorder.apply_animation_keyframes(frame_start=1, frame_end=frame_end) if recorder is not None else 0
    recorder_summary = recorder.summary(frame_start=1, frame_end=frame_end) if recorder is not None else {}
    for obj in preview_hidden_objects:
        obj.hide_viewport = True
        obj.hide_render = True
    for idx, obj in enumerate(objects):
        if obj.name in bpy.data.objects:
            obj.color = (0.95, 0.62 + 0.12 * (idx % 2), 0.18, 1.0)
            data_name = obj.data.name
            if data_name not in old_materials_by_data:
                old_materials_by_data[data_name] = [mat for mat in obj.data.materials]
            preview_mat = make_material("mat_simulation_video_orange", (0.95, 0.62, 0.16, 1.0), roughness=0.5)
            obj.data.materials.clear()
            obj.data.materials.append(preview_mat)
    scene.camera = camera
    scene.frame_start = 1
    scene.frame_end = max(1, int(frame_end))
    scene.frame_step = max(1, int(frame_step))
    render.fps = max(1, int(fps))
    resolution_scale = max(5, min(100, int(resolution_percent))) / 100.0
    render.resolution_x = max(2, int(round(render.resolution_x * resolution_scale)) // 2 * 2)
    render.resolution_y = max(2, int(round(render.resolution_y * resolution_scale)) // 2 * 2)
    render.resolution_percentage = 100
    render.filepath = str(path)
    render.image_settings.file_format = "FFMPEG"
    render.ffmpeg.format = "MPEG4"
    render.ffmpeg.codec = "H264"
    render.ffmpeg.constant_rate_factor = "MEDIUM"
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        if shading is not None:
            shading.light = "STUDIO"
            shading.color_type = "OBJECT"
    except TypeError:
        pass
    synthetic_log(
        f"[synthetic] rendering simulation video: {path} "
        f"frames=1..{scene.frame_end} step={scene.frame_step} fps={render.fps} "
        f"sampled_keyframes={keyframes} recorder={recorder_summary}"
    )
    try:
        bpy.ops.render.render(animation=True)
    finally:
        scene.camera = old_settings["camera"]
        scene.render.engine = old_settings["engine"]
        render.filepath = old_settings["filepath"]
        scene.frame_start = old_settings["frame_start"]
        scene.frame_end = old_settings["frame_end"]
        scene.frame_step = old_settings["frame_step"]
        render.fps = old_settings["fps"]
        render.resolution_x = old_settings["resolution_x"]
        render.resolution_y = old_settings["resolution_y"]
        render.resolution_percentage = old_settings["resolution_percentage"]
        render.image_settings.file_format = old_settings["file_format"]
        render.ffmpeg.format = old_settings["ffmpeg_format"]
        render.ffmpeg.codec = old_settings["ffmpeg_codec"]
        render.ffmpeg.constant_rate_factor = old_settings["ffmpeg_constant_rate_factor"]
        for obj in preview_hidden_objects:
            if obj.name in bpy.data.objects and obj.name in old_preview_visibility:
                obj.hide_viewport, obj.hide_render = old_preview_visibility[obj.name]
        for obj in objects:
            if obj.name in bpy.data.objects and obj.name in old_object_colors:
                obj.color = old_object_colors[obj.name]
            if obj.name in bpy.data.objects and obj.data.name in old_materials_by_data:
                obj.data.materials.clear()
                for mat in old_materials_by_data[obj.data.name]:
                    obj.data.materials.append(mat)
        if shading is not None:
            if old_shading["light"] is not None:
                shading.light = old_shading["light"]
            if old_shading["color_type"] is not None:
                shading.color_type = old_shading["color_type"]


def write_sample_outputs(
    sample_dir: Path,
    model_path: Path,
    class_name: str,
    depth_camera: bpy.types.Object,
    rgb_camera: bpy.types.Object,
    debug_camera: bpy.types.Object,
    objects: list[bpy.types.Object],
    width: int,
    height: int,
    settings: dict[str, Any],
    sensor: dict[str, np.ndarray | dict[str, float]] | None = None,
    simulation_video_file: str | None = None,
    stereo_right_camera: bpy.types.Object | None = None,
    stereo_projector: bpy.types.Object | None = None,
    stereo_args: argparse.Namespace | None = None,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    render_rgb(sample_dir / "rgb.png", rgb_camera)
    stereo_rendered = False
    if stereo_right_camera is not None and stereo_projector is not None and stereo_args is not None:
        render_stereo_pair(sample_dir, depth_camera, stereo_right_camera, stereo_projector, stereo_args)
        stereo_rendered = True

    if sensor is None:
        sensor = raycast_sensor_data(depth_camera, objects, width, height)
    np.savez_compressed(
        sample_dir / "sensor_data.npz",
        depth_m=sensor["depth_m"],
        instance_mask=sensor["instance_mask"],
        normal_world=sensor["normal_world"],
        normal_camera=sensor["normal_camera"],
        points_camera=sensor["points_camera"],
        points_world=sensor["points_world"],
        point_instance_ids=sensor["point_instance_ids"],
        point_semantic_ids=sensor["point_semantic_ids"],
        point_pixels=sensor["point_pixels"],
        sensor_artifact_mask=sensor.get("sensor_artifact_mask", np.zeros((height, width), dtype=np.uint16)),
    )

    depth = sensor["depth_m"]
    valid = depth > 0
    depth_preview = np.zeros_like(depth, dtype=np.uint8)
    if np.any(valid):
        d_min = float(depth[valid].min())
        d_max = float(depth[valid].max())
        if d_max > d_min:
            depth_preview[valid] = 255 - ((depth[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_preview[valid] = 255
    save_pgm(sample_dir / "depth_preview.pgm", depth_preview)
    save_png_rgb(sample_dir / "depth_preview.png", np.repeat(depth_preview[:, :, None], 3, axis=2))

    rng = np.random.default_rng(123)
    palette = np.zeros((len(objects) + 1, 3), dtype=np.uint8)
    palette[1:] = rng.integers(40, 255, size=(len(objects), 3), dtype=np.uint8)
    mask_rgb = palette[np.asarray(sensor["instance_mask"], dtype=np.int32)]
    save_ppm(sample_dir / "instance_mask.ppm", mask_rgb)
    save_png_rgb(sample_dir / "instance_mask.png", mask_rgb)

    normal_preview = ((np.asarray(sensor["normal_camera"]) + 1.0) * 127.5).astype(np.uint8)
    normal_preview[~valid] = 0
    save_ppm(sample_dir / "normal_camera.ppm", normal_preview)
    save_png_rgb(sample_dir / "normal_camera.png", normal_preview)

    overlay = np.repeat(depth_preview[:, :, None], 3, axis=2)
    object_pixels = np.asarray(sensor["instance_mask"]) > 0
    overlay[object_pixels] = (0.35 * overlay[object_pixels] + 0.65 * mask_rgb[object_pixels]).astype(np.uint8)
    save_png_rgb(sample_dir / "label_overlay.png", overlay)

    metadata = {
        "model_path": str(model_path),
        "class_name": class_name,
        "class_id": OBJECT_CLASS_ID,
        "camera_intrinsics": sensor["intrinsics"],
        "camera_to_world": matrix_to_list(depth_camera.matrix_world),
        "world_to_camera": matrix_to_list(depth_camera.matrix_world.inverted()),
        "depth_camera_intrinsics": sensor["intrinsics"],
        "depth_camera_to_world": matrix_to_list(depth_camera.matrix_world),
        "depth_world_to_camera": matrix_to_list(depth_camera.matrix_world.inverted()),
        "rgb_camera_intrinsics": camera_intrinsics(rgb_camera, width, height),
        "rgb_camera_to_world": matrix_to_list(rgb_camera.matrix_world),
        "rgb_world_to_camera": matrix_to_list(rgb_camera.matrix_world.inverted()),
        "debug_camera_intrinsics": camera_intrinsics(debug_camera, width, height),
        "debug_camera_to_world": matrix_to_list(debug_camera.matrix_world),
        "debug_world_to_camera": matrix_to_list(debug_camera.matrix_world.inverted()),
        "settings": settings,
        "sensor_noise_summary": sensor.get("sensor_noise_summary", {"profile": "none"}),
        "object_count": len(objects),
        "visible_object_point_count": int(np.asarray(sensor["point_instance_ids"]).shape[0]),
        "instances": [],
        "stereo": None if not stereo_rendered else {
            "baseline_m": float(stereo_args.stereo_baseline),
            "left_image": "stereo_left.png",
            "right_image": "stereo_right.png",
            "samples": int(stereo_args.stereo_samples),
            "projector_strength": float(stereo_args.stereo_projector_strength),
            "projector_dot_density": float(stereo_args.stereo_projector_dot_density),
            "spot_size_deg": float(stereo_args.stereo_spot_size_deg),
        },
        "files": {
            "rgb": "rgb.png",
            "sensor_data": "sensor_data.npz",
            "depth_preview": "depth_preview.pgm",
            "depth_preview_png": "depth_preview.png",
            "instance_mask_preview": "instance_mask.ppm",
            "instance_mask_preview_png": "instance_mask.png",
            "normal_preview": "normal_camera.ppm",
            "normal_preview_png": "normal_camera.png",
            "label_overlay_png": "label_overlay.png",
        },
    }
    if simulation_video_file:
        metadata["files"]["simulation_video"] = simulation_video_file
    depth_world_to_camera = depth_camera.matrix_world.inverted()
    rgb_world_to_camera = rgb_camera.matrix_world.inverted()
    for idx, obj in enumerate(objects):
        original_to_body = original_model_to_body_matrix(obj)
        original_to_world = obj.matrix_world @ original_to_body
        object_to_depth_camera = depth_world_to_camera @ original_to_world
        metadata["instances"].append(
            {
                "instance_id": idx + 1,
                "name": obj.name,
                "class_id": OBJECT_CLASS_ID,
                "collision_shape": obj.rigid_body.collision_shape,
                "collision_margin_m": float(obj.rigid_body.collision_margin),
                "object_restitution": float(obj.rigid_body.restitution),
                "spawn_layer": int(obj.get("spawn_layer", -1)),
                "spawn_height": float(obj.get("spawn_height", 0.0)),
                "spawn_frame": int(obj.get("spawn_frame", 1)),
                "spawn_edge_margin": float(obj.get("spawn_edge_margin", 0.0)),
                "spawn_parked": bool(obj.get("spawn_parked", False)),
                "spawn_z_lift_steps": int(obj.get("spawn_z_lift_steps", 0)),
                "original_model_center_offset": [float(v) for v in obj.get("original_model_center_offset", [0.0, 0.0, 0.0])],
                "model_scale": float(obj.get("model_scale", 1.0)),
                "original_model_to_body": matrix_to_list(original_to_body),
                "body_to_world": matrix_to_list(obj.matrix_world),
                "object_to_world": matrix_to_list(original_to_world),
                "object_to_camera": matrix_to_list(object_to_depth_camera),
                "object_to_depth_camera": matrix_to_list(object_to_depth_camera),
                "object_to_rgb_camera": matrix_to_list(rgb_world_to_camera @ original_to_world),
            }
        )
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def sample_visibility_stats(sensor: dict[str, np.ndarray | dict[str, float]]) -> dict[str, int]:
    instance_ids = np.asarray(sensor["point_instance_ids"])
    visible_ids = set(int(i) for i in np.unique(instance_ids) if int(i) != BACKGROUND_ID)
    return {
        "visible_objects": len(visible_ids),
        "visible_points": int(instance_ids.shape[0]),
    }


def render_settings_from_args(args: argparse.Namespace) -> RenderSettings:
    return RenderSettings(
        sensor_width=args.camera_sensor_width,
        clip_start=args.camera_clip_start,
        clip_end=args.camera_clip_end,
        cycles_samples=args.cycles_samples,
        view_transform=args.view_transform,
        view_look=args.view_look,
        view_exposure=args.view_exposure,
        view_gamma=args.view_gamma,
    )


def object_appearance_from_args(args: argparse.Namespace) -> ObjectAppearance:
    return ObjectAppearance(
        color_rgb=tuple(args.object_color),
        metallic=args.object_metallic,
        roughness=args.object_roughness,
    )


def setup_generation_scene(args: argparse.Namespace, model_path: Path) -> GenerationScene:
    clean_scene()
    configure_physics(args.settle_frames, args.physics_substeps, args.physics_solver_iterations, args.gravity)
    create_bin(
        args.bin_x,
        args.bin_y,
        args.bin_wall_height,
        args.collision_margin,
        args.bin_floor_thickness,
        args.bin_wall_thickness,
        tuple(args.bin_color),
        args.bin_roughness,
        args.bin_friction,
        args.bin_restitution,
    )
    ground_enabled = args.ground_plane if args.ground_plane is not None else bool(getattr(args, "stereo", False))
    if ground_enabled:
        create_ground_plane(args.ground_plane_size, tuple(args.ground_plane_color), args.ground_plane_roughness)
    template = import_stl(model_path, args.class_name, args.model_scale)
    render = render_settings_from_args(args)
    depth_camera = setup_camera(
        "depth_camera",
        args.width,
        args.height,
        tuple(args.depth_camera_location),
        tuple(args.depth_camera_target),
        args.depth_camera_lens,
        render,
    )
    rgb_camera = setup_camera(
        "rgb_camera",
        args.width,
        args.height,
        tuple(args.rgb_camera_location),
        tuple(args.rgb_camera_target),
        args.rgb_camera_lens,
        render,
    )
    debug_camera = setup_camera(
        "debug_camera",
        args.width,
        args.height,
        tuple(args.debug_camera_location),
        tuple(args.debug_camera_target),
        args.debug_camera_lens,
        render,
    )
    bpy.context.scene.camera = rgb_camera
    setup_lighting(tuple(args.light_location), args.light_energy, args.light_size, args.light_type, tuple(args.world_color))
    stereo_right_camera = None
    stereo_projector = None
    if getattr(args, "stereo", False):
        stereo_right_camera, stereo_projector = setup_stereo_rig(depth_camera, args)
    return GenerationScene(
        template=template,
        depth_camera=depth_camera,
        rgb_camera=rgb_camera,
        debug_camera=debug_camera,
        class_name=args.class_name,
        stereo_right_camera=stereo_right_camera,
        stereo_projector=stereo_projector,
    )


def simulate_attempt(
    args: argparse.Namespace,
    scene_context: GenerationScene,
    physics_settings: ObjectPhysicsSettings,
    appearance: ObjectAppearance,
    recorder: SimulationVideoRecorder,
) -> tuple[list[bpy.types.Object], dict[str, Any] | None, float, int, dict[str, int]]:
    """Run the configured spawn mode and return objects, explosion, speed limit, video frame end, spawn stats."""

    total_frames = simulation_frame_count(args)
    watchdog_z_min = args.out_of_bin_min_z - 0.01
    if args.spawn_mode == "progressive":
        if args.explosion_speed_limit is None:
            # Objects fall only the drop clearance above the pile, so the speed
            # bound comes from the clearance, not the absolute pile height.
            effective_speed_limit = auto_explosion_speed_limit(args.drop_clearance_max)
        else:
            effective_speed_limit = float(args.explosion_speed_limit)
        objects, violation, used_frames = simulate_progressive(
            scene_context.template,
            scene_context.class_name,
            args.objects,
            args.bin_x,
            args.bin_y,
            args.drop_clearance_min,
            args.drop_clearance_max,
            args.progressive_settle_frames,
            args.final_relax_frames,
            physics_settings,
            appearance,
            args.physics_substeps,
            args.physics_solver_iterations,
            args.gravity,
            effective_speed_limit,
            watchdog_z_min,
            freeze_settled=args.progressive_freeze,
            recorder=recorder,
        )
        video_frame_end = max(used_frames, recorder.last_recorded_frame)
        return objects, violation, effective_speed_limit, video_frame_end, {"parked_spawns": 0, "spawn_z_lift_events": 0}

    if args.spawn_mode == "grid":
        if args.explosion_speed_limit is None:
            # Objects are placed a short clearance above the floor/pile, so the
            # speed bound comes from the drop clearance, not the bin height.
            effective_speed_limit = auto_explosion_speed_limit(args.grid_drop_clearance)
        else:
            effective_speed_limit = float(args.explosion_speed_limit)
        objects, violation, used_frames = simulate_grid(
            scene_context.template,
            scene_context.class_name,
            args.objects,
            args.bin_x,
            args.bin_y,
            args.grid_spacing,
            args.grid_drop_clearance,
            args.grid_jitter,
            tuple(args.grid_orientation) if args.grid_orientation is not None else None,
            args.grid_layers,
            args.final_relax_frames,
            args.progressive_settle_frames,
            physics_settings,
            appearance,
            args.physics_substeps,
            args.physics_solver_iterations,
            args.gravity,
            effective_speed_limit,
            watchdog_z_min,
            freeze_settled=args.progressive_freeze,
            recorder=recorder,
        )
        video_frame_end = max(used_frames, recorder.last_recorded_frame)
        return objects, violation, effective_speed_limit, video_frame_end, {"parked_spawns": 0, "spawn_z_lift_events": 0}

    reset_physics_cache(total_frames, args.physics_substeps, args.physics_solver_iterations, args.gravity)
    objects = create_object_instances(
        scene_context.template,
        scene_context.class_name,
        args.objects,
        args.bin_x,
        args.bin_y,
        args.drop_height,
        args.drop_height_jitter,
        args.drop_height_min,
        args.drop_height_max,
        args.spawn_strategy,
        args.objects_per_layer,
        args.spawn_min_distance,
        args.spawn_settle_frames,
        physics_settings,
        appearance,
        recorder=recorder,
    )
    max_spawn_z = max((float(obj.get("spawn_height", 0.0)) for obj in objects), default=0.0)
    if args.explosion_speed_limit is None:
        effective_speed_limit = auto_explosion_speed_limit(max_spawn_z)
    else:
        effective_speed_limit = float(args.explosion_speed_limit)
    watchdog = PhysicsExplosionWatchdog(
        enabled=effective_speed_limit > 0.0,
        speed_limit_m_s=effective_speed_limit,
        z_min_limit_m=watchdog_z_min,
        z_max_limit_m=max_spawn_z + 0.25,
        fps=float(bpy.context.scene.render.fps),
        objects=objects,
        spawn_frames={obj.name: int(obj.get("spawn_frame", 1)) for obj in objects},
    )
    spawn_stats = {
        "parked_spawns": sum(1 for obj in objects if bool(obj.get("spawn_parked", False))),
        "spawn_z_lift_events": sum(1 for obj in objects if int(obj.get("spawn_z_lift_steps", 0)) > 0),
    }
    recorder.record_current(force=True)
    violation = advance_scene_frames(
        max(0, total_frames - int(bpy.context.scene.frame_current)),
        recorder=recorder,
        watchdog=watchdog,
    )
    if violation is None:
        bpy.context.scene.frame_set(total_frames)
        recorder.record_current(force=True)
    video_frame_end = max(total_frames, int(bpy.context.scene.frame_current), recorder.last_recorded_frame)
    return objects, violation, effective_speed_limit, video_frame_end, spawn_stats


def build_sample(
    args: argparse.Namespace,
    scene_context: GenerationScene,
    sample_idx: int,
    attempt_idx: int,
    model_path: Path,
    output_dir: Path,
) -> tuple[bool, dict[str, Any]]:
    seed = args.seed + sample_idx * 1009 + attempt_idx
    random.seed(seed)
    np.random.seed(seed)
    total_frames = simulation_frame_count(args)
    recorder = SimulationVideoRecorder(
        enabled=bool(args.record_simulation_video or args.record_failed_video),
        frame_step=int(args.simulation_video_frame_step),
    )
    physics_settings = ObjectPhysicsSettings(
        mass_kg=args.object_mass,
        friction=args.object_friction,
        restitution=args.object_restitution,
        linear_damping=args.object_linear_damping,
        angular_damping=args.object_angular_damping,
        collision_margin=args.collision_margin,
        collision_shape=args.collision_shape,
    )
    appearance = object_appearance_from_args(args)
    synthetic_log(
        f"[synthetic] sample {sample_idx:06d} attempt {attempt_idx + 1}: "
        f"spawn_mode={args.spawn_mode}, objects={args.objects}, simulation_frames={total_frames}"
    )
    objects, violation, effective_speed_limit, video_frame_end, spawn_stats = simulate_attempt(
        args, scene_context, physics_settings, appearance, recorder
    )
    all_objects = objects
    parked_spawns = int(spawn_stats["parked_spawns"])
    spawn_z_lift_events = int(spawn_stats["spawn_z_lift_events"])
    try:
        stats: dict[str, Any] = {
            "spawned_objects": len(objects),
            "parked_spawns": parked_spawns,
            "spawn_z_lift_events": spawn_z_lift_events,
        }
        reject_reason: str | None = None
        sensor: dict[str, np.ndarray | dict[str, float]] | None = None
        visible_objects = objects

        if violation is not None:
            reject_reason = "physics_explosion"
            stats["reason"] = "physics_explosion"
            stats["explosion"] = violation
        else:
            out_of_bin_objects = find_out_of_bin_objects(objects, args.bin_x, args.bin_y, args.out_of_bin_tolerance, args.out_of_bin_min_z)
            stats["in_bin_objects"] = len(objects) - len(out_of_bin_objects)
            stats["out_of_bin_objects"] = len(out_of_bin_objects)
            stats["out_of_bin_ids"] = [objects.index(obj) + 1 for obj in out_of_bin_objects]
            if out_of_bin_objects:
                stats["out_of_bin_detail"] = describe_out_of_bin(
                    objects, out_of_bin_objects, args.bin_x, args.bin_y, args.out_of_bin_tolerance
                )
            if out_of_bin_objects and not args.allow_out_of_bin_filtering:
                reject_reason = "out_of_bin"
                stats["reason"] = "out_of_bin"
            else:
                if out_of_bin_objects:
                    hide_objects(out_of_bin_objects)
                    visible_objects = [obj for obj in objects if obj not in out_of_bin_objects]
                sensor = raycast_sensor_data(scene_context.depth_camera, visible_objects, args.width, args.height)
                sensor = apply_depth_sensor_artifacts(sensor, scene_context.depth_camera, args, seed=seed)
                stats.update(sample_visibility_stats(sensor))
                if "sensor_noise_summary" in sensor:
                    stats["sensor_noise"] = sensor["sensor_noise_summary"]
                target_visible_objects = min(args.objects, args.min_visible_objects)
                if int(stats["visible_objects"]) < target_visible_objects or int(stats["visible_points"]) < args.min_visible_points:
                    reject_reason = "visibility"
                    stats["reason"] = "visibility"

        if reject_reason is not None:
            if args.record_failed_video and all_objects:
                rejected_dir = output_dir / "rejected"
                fail_video = rejected_dir / f"sample_{sample_idx:06d}_attempt_{attempt_idx + 1:02d}_{reject_reason}.mp4"
                synthetic_log(f"[synthetic] recording failed-attempt video: {fail_video}")
                render_simulation_video(
                    fail_video,
                    scene_context.debug_camera,
                    all_objects,
                    recorder=recorder,
                    frame_end=video_frame_end,
                    frame_step=args.simulation_video_frame_step,
                    fps=args.simulation_video_fps,
                    resolution_percent=args.simulation_video_resolution_percent,
                )
            return False, stats

        settings = {
            "seed": seed,
            "attempt_idx": attempt_idx,
            "bin_x": args.bin_x,
            "bin_y": args.bin_y,
            "bin_wall_height": args.bin_wall_height,
            "drop_height": args.drop_height,
            "drop_height_jitter": args.drop_height_jitter,
            "drop_height_min": args.drop_height_min,
            "drop_height_max": args.drop_height_max,
            "drop_clearance_min": args.drop_clearance_min,
            "drop_clearance_max": args.drop_clearance_max,
            "spawn_strategy": args.spawn_strategy,
            "spawn_mode": args.spawn_mode,
            "progressive_settle_frames": args.progressive_settle_frames,
            "final_relax_frames": args.final_relax_frames,
            "progressive_freeze": args.progressive_freeze,
            "grid_spacing": args.grid_spacing,
            "grid_drop_clearance": args.grid_drop_clearance,
            "grid_jitter": args.grid_jitter,
            "grid_orientation": list(args.grid_orientation) if args.grid_orientation is not None else None,
            "grid_layers": args.grid_layers,
            "objects_per_layer": args.objects_per_layer,
            "spawn_min_distance": args.spawn_min_distance,
            "spawn_settle_frames": args.spawn_settle_frames,
            "collision_margin": args.collision_margin,
            "model_scale": args.model_scale,
            "collision_shape": args.collision_shape,
            "object_restitution": args.object_restitution,
            "object_mass": args.object_mass,
            "object_friction": args.object_friction,
            "object_linear_damping": args.object_linear_damping,
            "object_angular_damping": args.object_angular_damping,
            "object_color": list(args.object_color),
            "object_metallic": args.object_metallic,
            "object_roughness": args.object_roughness,
            "bin_friction": args.bin_friction,
            "bin_restitution": args.bin_restitution,
            "bin_floor_thickness": args.bin_floor_thickness,
            "bin_wall_thickness": args.bin_wall_thickness,
            "bin_color": list(args.bin_color),
            "bin_roughness": args.bin_roughness,
            "gravity": args.gravity,
            "physics_substeps": args.physics_substeps,
            "physics_solver_iterations": args.physics_solver_iterations,
            "explosion_speed_limit_m_s": effective_speed_limit,
            "spawn_parking_lead_frames": PARK_STATIC_LEAD_FRAMES,
            "parked_spawns": parked_spawns,
            "spawn_z_lift_events": spawn_z_lift_events,
            "camera_sensor_width": args.camera_sensor_width,
            "camera_clip_start": args.camera_clip_start,
            "camera_clip_end": args.camera_clip_end,
            "sensor_noise_profile": args.sensor_noise_profile,
            "depth_quantization_m": args.depth_quantization_m,
            "depth_quadratic_noise": args.depth_quadratic_noise,
            "depth_random_dropout_prob": args.depth_random_dropout_prob,
            "edge_gap_m": args.edge_gap_m,
            "edge_dropout_prob": args.edge_dropout_prob,
            "edge_smear_prob": args.edge_smear_prob,
            "edge_smear_radius_px": args.edge_smear_radius_px,
            "flying_pixel_prob": args.flying_pixel_prob,
            "flying_pixel_alpha_min": args.flying_pixel_alpha_min,
            "flying_pixel_alpha_max": args.flying_pixel_alpha_max,
            "bridge_prob": args.bridge_prob,
            "bridge_max_gap_px": args.bridge_max_gap_px,
            "blob_dropout_count": args.blob_dropout_count,
            "blob_dropout_radius_px": args.blob_dropout_radius_px,
            "cycles_samples": args.cycles_samples,
            "view_transform": args.view_transform,
            "view_look": args.view_look,
            "view_exposure": args.view_exposure,
            "view_gamma": args.view_gamma,
            "depth_camera_location": list(args.depth_camera_location),
            "depth_camera_target": list(args.depth_camera_target),
            "depth_camera_lens": args.depth_camera_lens,
            "stereo": bool(getattr(args, "stereo", False)),
            "stereo_baseline": args.stereo_baseline,
            "stereo_samples": args.stereo_samples,
            "stereo_projector_strength": args.stereo_projector_strength,
            "stereo_projector_dot_density": args.stereo_projector_dot_density,
            "stereo_spot_size_deg": args.stereo_spot_size_deg,
            "rgb_camera_location": list(args.rgb_camera_location),
            "rgb_camera_target": list(args.rgb_camera_target),
            "rgb_camera_lens": args.rgb_camera_lens,
            "debug_camera_location": list(args.debug_camera_location),
            "debug_camera_target": list(args.debug_camera_target),
            "debug_camera_lens": args.debug_camera_lens,
            "light_location": list(args.light_location),
            "light_energy": args.light_energy,
            "light_size": args.light_size,
            "light_type": args.light_type,
            "world_color": list(args.world_color),
            "out_of_bin_tolerance": args.out_of_bin_tolerance,
            "out_of_bin_min_z": args.out_of_bin_min_z,
            "allow_out_of_bin_filtering": args.allow_out_of_bin_filtering,
            "out_of_bin_policy": "filter" if args.allow_out_of_bin_filtering else "reject_attempt",
            "out_of_bin_check": "world_center_xy",
            "min_visible_objects": args.min_visible_objects,
            "min_visible_points": args.min_visible_points,
            "settle_frames": args.settle_frames,
            "simulation_frames": total_frames,
            "record_simulation_video": args.record_simulation_video,
            "record_failed_video": args.record_failed_video,
            "simulation_video_name": args.simulation_video_name,
            "simulation_video_frame_step": args.simulation_video_frame_step,
            "simulation_video_fps": args.simulation_video_fps,
            "simulation_video_resolution_percent": args.simulation_video_resolution_percent,
        }
        sample_dir = output_dir / f"sample_{sample_idx:06d}"
        simulation_video_file = Path(args.simulation_video_name).name if args.record_simulation_video else None
        write_sample_outputs(
            sample_dir,
            model_path,
            scene_context.class_name,
            scene_context.depth_camera,
            scene_context.rgb_camera,
            scene_context.debug_camera,
            visible_objects,
            args.width,
            args.height,
            settings,
            sensor=sensor,
            simulation_video_file=simulation_video_file,
            stereo_right_camera=scene_context.stereo_right_camera,
            stereo_projector=scene_context.stereo_projector,
            stereo_args=args if getattr(args, "stereo", False) else None,
        )
        if args.record_simulation_video and simulation_video_file:
            render_simulation_video(
                sample_dir / simulation_video_file,
                scene_context.debug_camera,
                visible_objects,
                recorder=recorder,
                frame_end=video_frame_end,
                frame_step=args.simulation_video_frame_step,
                fps=args.simulation_video_fps,
                resolution_percent=args.simulation_video_resolution_percent,
            )
        return True, stats
    finally:
        delete_objects(all_objects)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    project_root = Path.cwd()
    model_path = (project_root / args.model).resolve()
    output_dir = (project_root / args.output).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    args.class_name = args.class_name or model_path.stem
    if args.export_settings:
        settings_path = (project_root / args.export_settings).resolve()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(generator_settings_dict(args), f, indent=2)
        synthetic_log(f"[synthetic] exported generator settings: {settings_path}")
    if args.samples <= 0:
        # Export-only / dry-run: write the full preset without generating samples.
        synthetic_log("[synthetic] samples <= 0, skipping generation (export-only run)")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_context = setup_generation_scene(args, model_path)
    total_frames = simulation_frame_count(args)
    synthetic_log(
        f"[synthetic] config: samples={args.samples}, objects={args.objects}, "
        f"resolution={args.width}x{args.height}, collision_shape={args.collision_shape}, "
        f"spawn_mode={args.spawn_mode}, simulation_frames={total_frames}"
    )

    accepted = 0
    rejected = 0
    rejection_stats: list[dict[str, int | str | list[int]]] = []
    for sample_idx in range(args.samples):
        synthetic_log(f"[synthetic] generating sample {sample_idx + 1}/{args.samples}")
        last_stats: dict[str, int | str | list[int]] = {"visible_objects": 0, "visible_points": 0}
        for attempt_idx in range(args.max_sample_attempts):
            ok, stats = build_sample(args, scene_context, sample_idx, attempt_idx, model_path, output_dir)
            last_stats = stats
            if ok:
                accepted += 1
                if attempt_idx > 0:
                    synthetic_log(f"[synthetic] accepted sample {sample_idx:06d} after {attempt_idx + 1} attempts: {stats}")
                break
            rejected += 1
            rejection_stats.append(stats)
            synthetic_log(f"[synthetic] rejected sample {sample_idx:06d} attempt {attempt_idx + 1}: {stats}")
        else:
            raise RuntimeError(
                f"Could not generate sample_{sample_idx:06d} after {args.max_sample_attempts} attempts. "
                f"Last stats: {last_stats}. Lower --min-visible-objects or improve spawn settings."
            )

    summary = {
        "model": str(model_path),
        "class_name": args.class_name,
        "output": str(output_dir),
        "samples": args.samples,
        "objects_per_scene": args.objects,
        "resolution": [args.width, args.height],
        "seed": args.seed,
        "drop_height": args.drop_height,
        "drop_height_min": args.drop_height_min,
        "drop_height_max": args.drop_height_max,
        "drop_clearance_min": args.drop_clearance_min,
        "drop_clearance_max": args.drop_clearance_max,
        "spawn_strategy": args.spawn_strategy,
        "spawn_mode": args.spawn_mode,
        "progressive_settle_frames": args.progressive_settle_frames,
        "final_relax_frames": args.final_relax_frames,
        "progressive_freeze": args.progressive_freeze,
        "grid_spacing": args.grid_spacing,
        "grid_drop_clearance": args.grid_drop_clearance,
        "grid_jitter": args.grid_jitter,
        "grid_orientation": list(args.grid_orientation) if args.grid_orientation is not None else None,
        "grid_layers": args.grid_layers,
        "objects_per_layer": args.objects_per_layer,
        "spawn_min_distance": args.spawn_min_distance,
        "spawn_settle_frames": args.spawn_settle_frames,
        "collision_margin": args.collision_margin,
        "model_scale": args.model_scale,
        "collision_shape": args.collision_shape,
        "object_restitution": args.object_restitution,
        "object_mass": args.object_mass,
        "object_friction": args.object_friction,
        "object_linear_damping": args.object_linear_damping,
        "object_angular_damping": args.object_angular_damping,
        "object_color": list(args.object_color),
        "object_metallic": args.object_metallic,
        "object_roughness": args.object_roughness,
        "bin_friction": args.bin_friction,
        "bin_restitution": args.bin_restitution,
        "bin_floor_thickness": args.bin_floor_thickness,
        "bin_wall_thickness": args.bin_wall_thickness,
        "bin_color": list(args.bin_color),
        "bin_roughness": args.bin_roughness,
        "gravity": args.gravity,
        "physics_substeps": args.physics_substeps,
        "physics_solver_iterations": args.physics_solver_iterations,
        "explosion_speed_limit": args.explosion_speed_limit if args.explosion_speed_limit is not None else "auto",
        "spawn_parking_lead_frames": PARK_STATIC_LEAD_FRAMES,
        "camera_sensor_width": args.camera_sensor_width,
        "camera_clip_start": args.camera_clip_start,
        "camera_clip_end": args.camera_clip_end,
        "sensor_noise_profile": args.sensor_noise_profile,
        "depth_quantization_m": args.depth_quantization_m,
        "depth_quadratic_noise": args.depth_quadratic_noise,
        "depth_random_dropout_prob": args.depth_random_dropout_prob,
        "edge_gap_m": args.edge_gap_m,
        "edge_dropout_prob": args.edge_dropout_prob,
        "edge_smear_prob": args.edge_smear_prob,
        "edge_smear_radius_px": args.edge_smear_radius_px,
        "flying_pixel_prob": args.flying_pixel_prob,
        "flying_pixel_alpha_min": args.flying_pixel_alpha_min,
        "flying_pixel_alpha_max": args.flying_pixel_alpha_max,
        "bridge_prob": args.bridge_prob,
        "bridge_max_gap_px": args.bridge_max_gap_px,
        "blob_dropout_count": args.blob_dropout_count,
        "blob_dropout_radius_px": args.blob_dropout_radius_px,
        "cycles_samples": args.cycles_samples,
        "view_transform": args.view_transform,
        "view_look": args.view_look,
        "view_exposure": args.view_exposure,
        "view_gamma": args.view_gamma,
        "depth_camera_location": list(args.depth_camera_location),
        "depth_camera_target": list(args.depth_camera_target),
        "depth_camera_lens": args.depth_camera_lens,
            "stereo": bool(getattr(args, "stereo", False)),
            "stereo_baseline": args.stereo_baseline,
            "stereo_samples": args.stereo_samples,
            "stereo_projector_strength": args.stereo_projector_strength,
            "stereo_projector_dot_density": args.stereo_projector_dot_density,
            "stereo_spot_size_deg": args.stereo_spot_size_deg,
        "rgb_camera_location": list(args.rgb_camera_location),
        "rgb_camera_target": list(args.rgb_camera_target),
        "rgb_camera_lens": args.rgb_camera_lens,
        "debug_camera_location": list(args.debug_camera_location),
        "debug_camera_target": list(args.debug_camera_target),
        "debug_camera_lens": args.debug_camera_lens,
        "light_location": list(args.light_location),
        "light_energy": args.light_energy,
        "light_size": args.light_size,
        "light_type": args.light_type,
        "world_color": list(args.world_color),
        "out_of_bin_tolerance": args.out_of_bin_tolerance,
        "out_of_bin_min_z": args.out_of_bin_min_z,
        "allow_out_of_bin_filtering": args.allow_out_of_bin_filtering,
        "out_of_bin_policy": "filter" if args.allow_out_of_bin_filtering else "reject_attempt",
        "out_of_bin_check": "world_center_xy",
        "min_visible_objects": args.min_visible_objects,
        "min_visible_points": args.min_visible_points,
        "max_sample_attempts": args.max_sample_attempts,
        "settle_frames": args.settle_frames,
        "simulation_frames": total_frames,
        "record_simulation_video": args.record_simulation_video,
        "record_failed_video": args.record_failed_video,
        "simulation_video_name": args.simulation_video_name,
        "simulation_video_frame_step": args.simulation_video_frame_step,
        "simulation_video_fps": args.simulation_video_fps,
        "simulation_video_resolution_percent": args.simulation_video_resolution_percent,
        "accepted_samples": accepted,
        "rejected_attempts": rejected,
        "rejection_stats": rejection_stats[:25],
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    synthetic_log(f"[synthetic] done: {output_dir}")


if __name__ == "__main__":
    main()
