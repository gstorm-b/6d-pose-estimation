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
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


OBJECT_CLASS_ID = 1
BACKGROUND_ID = 0


@dataclass
class GenerationScene:
    template: bpy.types.Object
    camera: bpy.types.Object


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="object-model/K41144.stl")
    parser.add_argument("--output", default="synthetic-data/K41144")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--objects", type=int, default=12)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
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
    parser.add_argument("--collision-margin", type=float, default=0.00002)
    parser.add_argument("--collision-shape", choices=("CONVEX_HULL", "MESH"), default="CONVEX_HULL")
    parser.add_argument("--out-of-bin-tolerance", type=float, default=0.015)
    parser.add_argument(
        "--allow-out-of-bin-filtering",
        action="store_true",
        help="Keep old behavior: hide out-of-bin objects and accept the remaining scene if visibility thresholds pass.",
    )
    parser.add_argument("--min-visible-objects", type=int, default=8)
    parser.add_argument("--min-visible-points", type=int, default=2000)
    parser.add_argument("--max-sample-attempts", type=int, default=8)
    parser.add_argument("--settle-frames", type=int, default=220)
    return parser.parse_args(argv)


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


def import_stl(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"Could not import STL: {path}")
    obj = imported[0]
    obj.name = "K41144_template"
    obj.data.name = "K41144_mesh"
    center_mesh_for_physics(obj)
    return obj


def center_mesh_for_physics(obj: bpy.types.Object) -> None:
    """Move mesh vertices around their bbox center while preserving original STL pose metadata.

    Blender rigid bodies use the object origin as the body's center of mass. The K41144 STL
    has its origin near one end of the part, so without this shift the object behaves like a
    pendulum and settles upright. We keep the offset so metadata can still report poses for
    the original STL coordinate frame.
    """
    vertices = obj.data.vertices
    if not vertices:
        return

    mins = Vector((min(v.co.x for v in vertices), min(v.co.y for v in vertices), min(v.co.z for v in vertices)))
    maxs = Vector((max(v.co.x for v in vertices), max(v.co.y for v in vertices), max(v.co.z for v in vertices)))
    center = (mins + maxs) * 0.5
    for vertex in vertices:
        vertex.co -= center
    obj.data.update()
    obj["original_model_center_offset"] = [float(center.x), float(center.y), float(center.z)]


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


def create_object_instances(
    template: bpy.types.Object,
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
    collision_margin: float,
    collision_shape: str,
) -> list[bpy.types.Object]:
    mat = make_material("mat_k41144_black_metal", (0.015, 0.014, 0.013, 1.0), metallic=0.85, roughness=0.85)
    template.data.materials.clear()
    template.data.materials.append(mat)

    objects = []
    spawn_xy_by_layer: dict[int, list[tuple[float, float]]] = {}
    for idx in range(count):
        obj = template.copy()
        obj.data = template.data
        obj.name = f"K41144_{idx + 1:03d}"
        obj.hide_viewport = False
        obj.hide_render = False
        margin = 0.035
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
        x = random.uniform(-bin_x / 2.0 + margin, bin_x / 2.0 - margin)
        y = random.uniform(-bin_y / 2.0 + margin, bin_y / 2.0 - margin)
        for _ in range(80):
            candidate = (
                random.uniform(-bin_x / 2.0 + margin, bin_x / 2.0 - margin),
                random.uniform(-bin_y / 2.0 + margin, bin_y / 2.0 - margin),
            )
            if all(math.dist(candidate, existing) >= spawn_min_distance for existing in layer_xy):
                x, y = candidate
                break
        layer_xy.append((x, y))
        obj.location = (x, y, z)
        obj["spawn_layer"] = layer
        obj["spawn_height"] = z
        obj.rotation_euler = random_rotation()
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.select_set(False)
        obj.rigid_body.type = "ACTIVE"
        obj.rigid_body.mass = 0.08
        obj.rigid_body.friction = 0.85
        obj.rigid_body.restitution = 0.05
        obj.rigid_body.linear_damping = 0.35
        obj.rigid_body.angular_damping = 0.45
        obj.rigid_body.collision_shape = collision_shape
        configure_rigid_body_margin(obj, collision_margin)
        objects.append(obj)

    template.hide_viewport = True
    template.hide_render = True
    return objects


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(width: int, height: int) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, -0.05, 0.42))
    camera = bpy.context.object
    look_at(camera, Vector((0.0, 0.0, 0.025)))
    camera.data.lens = 35.0
    camera.data.sensor_width = 32.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1.50
    bpy.context.scene.camera = camera

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
    return camera


def setup_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(0.0, -0.22, 0.45))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 90
    light.data.size = 0.45

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
    return Matrix.Translation(Vector((-float(offset[0]), -float(offset[1]), -float(offset[2]))))


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


def render_rgb(path: Path) -> None:
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def write_sample_outputs(
    sample_dir: Path,
    model_path: Path,
    camera: bpy.types.Object,
    objects: list[bpy.types.Object],
    width: int,
    height: int,
    settings: dict[str, float | int | str | None],
    sensor: dict[str, np.ndarray | dict[str, float]] | None = None,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    render_rgb(sample_dir / "rgb.png")

    if sensor is None:
        sensor = raycast_sensor_data(camera, objects, width, height)
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
        "class_name": "K41144",
        "class_id": OBJECT_CLASS_ID,
        "camera_intrinsics": sensor["intrinsics"],
        "camera_to_world": matrix_to_list(camera.matrix_world),
        "world_to_camera": matrix_to_list(camera.matrix_world.inverted()),
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
    world_to_camera = camera.matrix_world.inverted()
    for idx, obj in enumerate(objects):
        original_to_body = original_model_to_body_matrix(obj)
        original_to_world = obj.matrix_world @ original_to_body
        metadata["instances"].append(
            {
                "instance_id": idx + 1,
                "name": obj.name,
                "class_id": OBJECT_CLASS_ID,
                "collision_shape": obj.rigid_body.collision_shape,
                "collision_margin_m": float(obj.rigid_body.collision_margin),
                "spawn_layer": int(obj.get("spawn_layer", -1)),
                "spawn_height": float(obj.get("spawn_height", 0.0)),
                "original_model_center_offset": [float(v) for v in obj.get("original_model_center_offset", [0.0, 0.0, 0.0])],
                "original_model_to_body": matrix_to_list(original_to_body),
                "body_to_world": matrix_to_list(obj.matrix_world),
                "object_to_world": matrix_to_list(original_to_world),
                "object_to_camera": matrix_to_list(world_to_camera @ original_to_world),
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
    template = import_stl(model_path)
    camera = setup_camera(args.width, args.height)
    setup_lighting()
    return GenerationScene(template=template, camera=camera)


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
    reset_physics_cache(args.settle_frames)
    objects = create_object_instances(
        scene_context.template,
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
        args.collision_margin,
        args.collision_shape,
    )
    all_objects = objects
    try:
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

        sensor = raycast_sensor_data(scene_context.camera, objects, args.width, args.height)
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
            "collision_margin": args.collision_margin,
            "collision_shape": args.collision_shape,
            "out_of_bin_tolerance": args.out_of_bin_tolerance,
            "allow_out_of_bin_filtering": args.allow_out_of_bin_filtering,
            "out_of_bin_policy": "filter" if args.allow_out_of_bin_filtering else "reject_attempt",
            "out_of_bin_check": "world_bbox_xy",
            "min_visible_objects": args.min_visible_objects,
            "min_visible_points": args.min_visible_points,
            "settle_frames": args.settle_frames,
        }
        sample_dir = output_dir / f"sample_{sample_idx:06d}"
        write_sample_outputs(
            sample_dir,
            model_path,
            scene_context.camera,
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
    args = parse_args()
    project_root = Path.cwd()
    model_path = (project_root / args.model).resolve()
    output_dir = (project_root / args.output).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_context = setup_generation_scene(args, model_path)

    accepted = 0
    rejected = 0
    rejection_stats: list[dict[str, int | str | list[int]]] = []
    for sample_idx in range(args.samples):
        print(f"[synthetic] generating sample {sample_idx + 1}/{args.samples}")
        last_stats: dict[str, int | str | list[int]] = {"visible_objects": 0, "visible_points": 0}
        for attempt_idx in range(args.max_sample_attempts):
            ok, stats = build_sample(args, scene_context, sample_idx, attempt_idx, model_path, output_dir)
            last_stats = stats
            if ok:
                accepted += 1
                if attempt_idx > 0:
                    print(f"[synthetic] accepted sample {sample_idx:06d} after {attempt_idx + 1} attempts: {stats}")
                break
            rejected += 1
            rejection_stats.append(stats)
            print(f"[synthetic] rejected sample {sample_idx:06d} attempt {attempt_idx + 1}: {stats}")
        else:
            raise RuntimeError(
                f"Could not generate sample_{sample_idx:06d} after {args.max_sample_attempts} attempts. "
                f"Last stats: {last_stats}. Lower --min-visible-objects or improve spawn settings."
            )

    summary = {
        "model": str(model_path),
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
        "collision_margin": args.collision_margin,
        "collision_shape": args.collision_shape,
        "out_of_bin_tolerance": args.out_of_bin_tolerance,
        "allow_out_of_bin_filtering": args.allow_out_of_bin_filtering,
        "out_of_bin_policy": "filter" if args.allow_out_of_bin_filtering else "reject_attempt",
        "out_of_bin_check": "world_bbox_xy",
        "min_visible_objects": args.min_visible_objects,
        "min_visible_points": args.min_visible_points,
        "max_sample_attempts": args.max_sample_attempts,
        "accepted_samples": accepted,
        "rejected_attempts": rejected,
        "rejection_stats": rejection_stats[:25],
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[synthetic] done: {output_dir}")


if __name__ == "__main__":
    main()
