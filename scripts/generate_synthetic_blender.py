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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


OBJECT_CLASS_ID = 1
BACKGROUND_ID = 0


@dataclass
class GenerationScene:
    template: bpy.types.Object
    depth_camera: bpy.types.Object
    rgb_camera: bpy.types.Object
    class_name: str


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
    parser.add_argument("--rgb-camera-location", type=float, nargs=3, default=(0.0, -0.05, 0.42))
    parser.add_argument("--rgb-camera-target", type=float, nargs=3, default=(0.0, 0.0, 0.025))
    parser.add_argument("--rgb-camera-lens", type=float, default=35.0)
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
        "--spawn-settle-frames",
        type=int,
        default=35,
        help="If > 0, enable one active rigid body at a time after this many settle frames before enabling the next object.",
    )
    parser.add_argument("--collision-margin", type=float, default=0.00002)
    parser.add_argument("--collision-shape", choices=("CONVEX_HULL", "MESH"), default="CONVEX_HULL")
    parser.add_argument("--object-restitution", type=float, default=0.05, help="Rigid-body bounce/restitution for spawned objects.")
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
    obj.rigid_body.friction = 0.9
    obj.rigid_body.restitution = 0.05
    configure_rigid_body_margin(obj, collision_margin)
    obj.select_set(False)
    return obj


def create_bin(bin_x: float, bin_y: float, wall_h: float, collision_margin: float) -> list[bpy.types.Object]:
    mat = make_material("mat_bin_dark_gray", (0.12, 0.12, 0.115, 1.0), roughness=0.82)
    floor_t = 0.01
    wall_t = 0.012
    parts = [
        add_box("bin_floor", (0.0, 0.0, -floor_t / 2.0), (bin_x, bin_y, floor_t), mat, collision_margin),
        add_box("bin_wall_pos_y", (0.0, bin_y / 2.0 + wall_t / 2.0, wall_h / 2.0), (bin_x + 2 * wall_t, wall_t, wall_h), mat, collision_margin),
        add_box("bin_wall_neg_y", (0.0, -bin_y / 2.0 - wall_t / 2.0, wall_h / 2.0), (bin_x + 2 * wall_t, wall_t, wall_h), mat, collision_margin),
        add_box("bin_wall_pos_x", (bin_x / 2.0 + wall_t / 2.0, 0.0, wall_h / 2.0), (wall_t, bin_y, wall_h), mat, collision_margin),
        add_box("bin_wall_neg_x", (-bin_x / 2.0 - wall_t / 2.0, 0.0, wall_h / 2.0), (wall_t, bin_y, wall_h), mat, collision_margin),
    ]
    return parts


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


def object_spawn_edge_margin(template: bpy.types.Object, bin_x: float, bin_y: float, collision_margin: float) -> float:
    if not template.data.vertices:
        return 0.035
    radius = max(float(vertex.co.length) for vertex in template.data.vertices)
    desired = max(0.035, radius + collision_margin)
    return min(desired, min(bin_x, bin_y) * 0.45)


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
    collision_margin: float,
    collision_shape: str,
    object_restitution: float,
) -> list[bpy.types.Object]:
    safe_name = sanitize_blender_name(class_name)
    mat = make_material(f"mat_{safe_name}_black_metal", (0.015, 0.014, 0.013, 1.0), metallic=0.85, roughness=0.85)
    template.data.materials.clear()
    template.data.materials.append(mat)

    objects = []
    spawn_xy_by_layer: dict[int, list[tuple[float, float]]] = {}
    margin = object_spawn_edge_margin(template, bin_x, bin_y, collision_margin)
    for idx in range(count):
        obj = template.copy()
        obj.data = template.data
        obj.name = f"{safe_name}_{idx + 1:03d}"
        obj.hide_viewport = False
        obj.hide_render = False
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
        layer_xy = spawn_xy_by_layer.setdefault(layer, [])
        x, y = sample_spawn_xy(bin_x, bin_y, margin, layer_xy, spawn_min_distance)
        layer_xy.append((x, y))
        obj.location = (x, y, z)
        obj["spawn_layer"] = layer
        obj["spawn_height"] = z
        obj["spawn_edge_margin"] = margin
        obj.rotation_euler = random_rotation()
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.select_set(False)
        obj.rigid_body.type = "ACTIVE"
        obj.rigid_body.mass = 0.08
        obj.rigid_body.friction = 0.85
        obj.rigid_body.restitution = object_restitution
        obj.rigid_body.linear_damping = 0.35
        obj.rigid_body.angular_damping = 0.45
        obj.rigid_body.collision_shape = collision_shape
        configure_rigid_body_margin(obj, collision_margin)
        obj["spawn_frame"] = int(bpy.context.scene.frame_current)
        objects.append(obj)
        if idx == 0 or (idx + 1) % 5 == 0 or idx + 1 == count:
            synthetic_log(
                f"[synthetic] spawned object {idx + 1}/{count} "
                f"at frame {bpy.context.scene.frame_current}"
            )
        if spawn_settle_frames > 0 and idx < count - 1:
            advance_scene_frames(spawn_settle_frames)

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
) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = name
    look_at(camera, Vector(target))
    camera.data.lens = lens
    camera.data.sensor_width = 32.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1.50
    setup_render_settings(width, height)
    return camera


def setup_render_settings(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -1.2
    scene.view_settings.gamma = 1.0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False


def setup_lighting(location: tuple[float, float, float], energy: float, size: float) -> None:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = energy
    light.data.size = size

    bpy.context.scene.world.color = (0.018, 0.018, 0.02)


def configure_physics(frame_end: int = 260) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.gravity = (0.0, 0.0, -9.81)
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.time_scale = 1.0
    scene.rigidbody_world.substeps_per_frame = 20
    scene.rigidbody_world.solver_iterations = 30
    scene.rigidbody_world.point_cache.frame_start = 1
    scene.rigidbody_world.point_cache.frame_end = frame_end


def reset_physics_cache(frames: int) -> None:
    configure_physics(frames)
    bpy.context.scene.frame_set(1)
    try:
        bpy.ops.ptcache.free_bake_all()
    except RuntimeError:
        pass


def settle_scene(frames: int) -> None:
    scene = bpy.context.scene
    scene.frame_set(1)
    for frame in range(1, frames + 1):
        scene.frame_set(frame)


def advance_scene_frames(frames: int) -> None:
    scene = bpy.context.scene
    start = scene.frame_current
    for frame in range(start + 1, start + frames + 1):
        scene.frame_set(frame)


def simulation_frame_count(args: argparse.Namespace) -> int:
    if args.spawn_settle_frames <= 0:
        return args.settle_frames
    return args.settle_frames + max(0, args.objects - 1) * args.spawn_settle_frames


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
    x_limit = bin_x / 2.0 + out_of_bin_tolerance
    y_limit = bin_y / 2.0 + out_of_bin_tolerance
    out_of_bin = []
    for obj in objects:
        min_corner, max_corner = object_world_bounds(obj)
        in_bounds = (
            float(min_corner.x) >= -x_limit
            and float(max_corner.x) <= x_limit
            and float(min_corner.y) >= -y_limit
            and float(max_corner.y) <= y_limit
            and float(min_corner.z) >= min_z
        )
        if not in_bounds:
            out_of_bin.append(obj)
    return out_of_bin


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


def render_rgb(path: Path, camera: bpy.types.Object) -> None:
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def write_sample_outputs(
    sample_dir: Path,
    model_path: Path,
    class_name: str,
    depth_camera: bpy.types.Object,
    rgb_camera: bpy.types.Object,
    objects: list[bpy.types.Object],
    width: int,
    height: int,
    settings: dict[str, Any],
    sensor: dict[str, np.ndarray | dict[str, float]] | None = None,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    render_rgb(sample_dir / "rgb.png", rgb_camera)

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
        "settings": settings,
        "object_count": len(objects),
        "visible_object_point_count": int(np.asarray(sensor["point_instance_ids"]).shape[0]),
        "instances": [],
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


def setup_generation_scene(args: argparse.Namespace, model_path: Path) -> GenerationScene:
    clean_scene()
    configure_physics(args.settle_frames)
    create_bin(args.bin_x, args.bin_y, args.bin_wall_height, args.collision_margin)
    template = import_stl(model_path, args.class_name, args.model_scale)
    depth_camera = setup_camera(
        "depth_camera",
        args.width,
        args.height,
        tuple(args.depth_camera_location),
        tuple(args.depth_camera_target),
        args.depth_camera_lens,
    )
    rgb_camera = setup_camera(
        "rgb_camera",
        args.width,
        args.height,
        tuple(args.rgb_camera_location),
        tuple(args.rgb_camera_target),
        args.rgb_camera_lens,
    )
    bpy.context.scene.camera = rgb_camera
    setup_lighting(tuple(args.light_location), args.light_energy, args.light_size)
    return GenerationScene(template=template, depth_camera=depth_camera, rgb_camera=rgb_camera, class_name=args.class_name)


def build_sample(
    args: argparse.Namespace,
    scene_context: GenerationScene,
    sample_idx: int,
    attempt_idx: int,
    model_path: Path,
    output_dir: Path,
) -> tuple[bool, dict[str, int | str | list[int]]]:
    seed = args.seed + sample_idx * 1009 + attempt_idx
    random.seed(seed)
    np.random.seed(seed)
    total_frames = simulation_frame_count(args)
    synthetic_log(
        f"[synthetic] sample {sample_idx:06d} attempt {attempt_idx + 1}: "
        f"objects={args.objects}, simulation_frames={total_frames}"
    )
    reset_physics_cache(total_frames)
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
        args.collision_margin,
        args.collision_shape,
        args.object_restitution,
    )
    all_objects = objects
    try:
        if args.spawn_settle_frames > 0:
            synthetic_log(
                f"[synthetic] sample {sample_idx:06d} attempt {attempt_idx + 1}: "
                f"final settling {args.settle_frames} frames from frame {bpy.context.scene.frame_current}"
            )
            advance_scene_frames(args.settle_frames)
        else:
            settle_scene(args.settle_frames)
        out_of_bin_objects = find_out_of_bin_objects(objects, args.bin_x, args.bin_y, args.out_of_bin_tolerance)
        stats: dict[str, int | str | list[int]] = {
            "spawned_objects": len(objects),
            "in_bin_objects": len(objects) - len(out_of_bin_objects),
            "out_of_bin_objects": len(out_of_bin_objects),
            "out_of_bin_ids": [objects.index(obj) + 1 for obj in out_of_bin_objects],
        }
        if out_of_bin_objects and not args.allow_out_of_bin_filtering:
            stats["reason"] = "out_of_bin"
            return False, stats
        if out_of_bin_objects:
            hide_objects(out_of_bin_objects)
            objects = [obj for obj in objects if obj not in out_of_bin_objects]

        sensor = raycast_sensor_data(scene_context.depth_camera, objects, args.width, args.height)
        stats.update(sample_visibility_stats(sensor))
        target_visible_objects = min(args.objects, args.min_visible_objects)
        if int(stats["visible_objects"]) < target_visible_objects or int(stats["visible_points"]) < args.min_visible_points:
            stats["reason"] = "visibility"
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
            "spawn_strategy": args.spawn_strategy,
            "objects_per_layer": args.objects_per_layer,
            "spawn_min_distance": args.spawn_min_distance,
            "spawn_settle_frames": args.spawn_settle_frames,
            "collision_margin": args.collision_margin,
            "model_scale": args.model_scale,
            "collision_shape": args.collision_shape,
            "object_restitution": args.object_restitution,
            "depth_camera_location": list(args.depth_camera_location),
            "depth_camera_target": list(args.depth_camera_target),
            "depth_camera_lens": args.depth_camera_lens,
            "rgb_camera_location": list(args.rgb_camera_location),
            "rgb_camera_target": list(args.rgb_camera_target),
            "rgb_camera_lens": args.rgb_camera_lens,
            "light_location": list(args.light_location),
            "light_energy": args.light_energy,
            "light_size": args.light_size,
            "out_of_bin_tolerance": args.out_of_bin_tolerance,
            "allow_out_of_bin_filtering": args.allow_out_of_bin_filtering,
            "out_of_bin_policy": "filter" if args.allow_out_of_bin_filtering else "reject_attempt",
            "out_of_bin_check": "world_bbox_xy",
            "min_visible_objects": args.min_visible_objects,
            "min_visible_points": args.min_visible_points,
            "settle_frames": args.settle_frames,
            "simulation_frames": total_frames,
        }
        sample_dir = output_dir / f"sample_{sample_idx:06d}"
        write_sample_outputs(
            sample_dir,
            model_path,
            scene_context.class_name,
            scene_context.depth_camera,
            scene_context.rgb_camera,
            objects,
            args.width,
            args.height,
            settings,
            sensor=sensor,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_context = setup_generation_scene(args, model_path)
    total_frames = simulation_frame_count(args)
    synthetic_log(
        f"[synthetic] config: samples={args.samples}, objects={args.objects}, "
        f"resolution={args.width}x{args.height}, collision_shape={args.collision_shape}, "
        f"spawn_settle_frames={args.spawn_settle_frames}, simulation_frames={total_frames}"
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
        "spawn_strategy": args.spawn_strategy,
        "objects_per_layer": args.objects_per_layer,
        "spawn_min_distance": args.spawn_min_distance,
        "spawn_settle_frames": args.spawn_settle_frames,
        "collision_margin": args.collision_margin,
        "model_scale": args.model_scale,
        "collision_shape": args.collision_shape,
        "object_restitution": args.object_restitution,
        "depth_camera_location": list(args.depth_camera_location),
        "depth_camera_target": list(args.depth_camera_target),
        "depth_camera_lens": args.depth_camera_lens,
        "rgb_camera_location": list(args.rgb_camera_location),
        "rgb_camera_target": list(args.rgb_camera_target),
        "rgb_camera_lens": args.rgb_camera_lens,
        "light_location": list(args.light_location),
        "light_energy": args.light_energy,
        "light_size": args.light_size,
        "out_of_bin_tolerance": args.out_of_bin_tolerance,
        "allow_out_of_bin_filtering": args.allow_out_of_bin_filtering,
        "out_of_bin_policy": "filter" if args.allow_out_of_bin_filtering else "reject_attempt",
        "out_of_bin_check": "world_bbox_xy",
        "min_visible_objects": args.min_visible_objects,
        "min_visible_points": args.min_visible_points,
        "max_sample_attempts": args.max_sample_attempts,
        "settle_frames": args.settle_frames,
        "simulation_frames": total_frames,
        "accepted_samples": accepted,
        "rejected_attempts": rejected,
        "rejection_stats": rejection_stats[:25],
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    synthetic_log(f"[synthetic] done: {output_dir}")


if __name__ == "__main__":
    main()
