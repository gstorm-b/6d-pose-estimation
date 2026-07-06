"""
Recover the pinhole intrinsic matrix K of the Basler camera from paired
organized point clouds (*-points_intensity.pcd) and range images
(*-range_color.tiff) in this folder.

Each organized PCD stores one 3D point per pixel of its WIDTH x HEIGHT grid,
in the same row-major order as the range image. That gives an exact,
known correspondence (u, v) <-> (X, Y, Z) for every valid pixel, with no
calibration target needed - we are recovering the parameters the camera/SDK
already used to build the cloud from its range data, by fitting the pinhole
model:

    u * Z = fx * X + cx * Z
    v * Z = fy * Y + cy * Z

These two equations are independent (2 unknowns each), so they are solved
as two small linear least-squares systems. All frames in the folder are
accumulated into the same normal equations (memory-light: each file is
read, reduced to sums, and discarded), since the task states every file
comes from the same physical camera.

Usage:
    python find_intrinsics.py [folder] [--recursive] [--limit N]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PCD_TYPE_MAP = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
}


def read_organized_pcd_xyz(path: Path):
    """Parse a binary PCD header + data, returning (xyz float64 Nx3, width, height)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path.name}: unexpected EOF while reading header")
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            header[key] = rest
            if key == "DATA":
                break

        if header["DATA"].strip() != "binary":
            raise ValueError(f"{path.name}: only 'DATA binary' PCDs are supported")

        fields = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header["COUNT"].split()]
        width = int(header["WIDTH"])
        height = int(header["HEIGHT"])
        n_points = int(header["POINTS"])

        dtype_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            np_type = PCD_TYPE_MAP[(typ, size)]
            dtype_fields.append((name, np_type) if count == 1 else (name, np_type, count))
        dtype = np.dtype(dtype_fields)

        raw = np.fromfile(f, dtype=dtype, count=n_points)

    if raw.size != n_points:
        raise ValueError(f"{path.name}: truncated data ({raw.size}/{n_points} points read)")

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    return xyz, width, height


def new_accumulator():
    keys = [
        "Su_xx", "Su_xz", "Su_zz", "Su_xb", "Su_zb", "Su_bb",
        "Sv_yy", "Sv_yz", "Sv_zz", "Sv_yb", "Sv_zb", "Sv_bb",
        "n",
    ]
    return {k: 0.0 for k in keys}


def accumulate(acc, xyz, width, height):
    n = xyz.shape[0]
    if n != width * height:
        raise ValueError(f"point count {n} != width*height ({width}x{height}); not organized")

    row = np.repeat(np.arange(height, dtype=np.float64), width)
    col = np.tile(np.arange(width, dtype=np.float64), height)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z != 0)
    if not np.any(valid):
        return 0

    x, y, z = x[valid], y[valid], z[valid]
    u, v = col[valid], row[valid]
    bu, bv = u * z, v * z

    acc["Su_xx"] += np.dot(x, x)
    acc["Su_xz"] += np.dot(x, z)
    acc["Su_zz"] += np.dot(z, z)
    acc["Su_xb"] += np.dot(x, bu)
    acc["Su_zb"] += np.dot(z, bu)
    acc["Su_bb"] += np.dot(bu, bu)

    acc["Sv_yy"] += np.dot(y, y)
    acc["Sv_yz"] += np.dot(y, z)
    acc["Sv_zz"] += np.dot(z, z)
    acc["Sv_yb"] += np.dot(y, bv)
    acc["Sv_zb"] += np.dot(z, bv)
    acc["Sv_bb"] += np.dot(bv, bv)

    acc["n"] += x.size
    return x.size


def solve(acc):
    Au = np.array([[acc["Su_xx"], acc["Su_xz"]],
                    [acc["Su_xz"], acc["Su_zz"]]])
    bu = np.array([acc["Su_xb"], acc["Su_zb"]])
    fx, cx = np.linalg.solve(Au, bu)
    rss_u = max(acc["Su_bb"] - fx * acc["Su_xb"] - cx * acc["Su_zb"], 0.0)

    Av = np.array([[acc["Sv_yy"], acc["Sv_yz"]],
                    [acc["Sv_yz"], acc["Sv_zz"]]])
    bv = np.array([acc["Sv_yb"], acc["Sv_zb"]])
    fy, cy = np.linalg.solve(Av, bv)
    rss_v = max(acc["Sv_bb"] - fy * acc["Sv_yb"] - cy * acc["Sv_zb"], 0.0)

    n = acc["n"]
    rmse_u = np.sqrt(rss_u / n)
    rmse_v = np.sqrt(rss_v / n)
    return fx, fy, cx, cy, rmse_u, rmse_v


def find_pairs(folder: Path, recursive: bool):
    glob = folder.rglob if recursive else folder.glob
    for pcd_path in sorted(glob("*-points_intensity.pcd")):
        tiff_path = Path(str(pcd_path).replace("-points_intensity.pcd", "-range_color.tiff"))
        yield pcd_path, tiff_path if tiff_path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default=".", help="folder containing the pcd/tiff pairs")
    ap.add_argument("--recursive", action="store_true", help="also search subfolders")
    ap.add_argument("--limit", type=int, default=None, help="only use the first N pairs (default: all)")
    args = ap.parse_args()

    folder = Path(args.folder)
    acc = new_accumulator()
    n_files = 0
    expected_wh = None

    for pcd_path, tiff_path in find_pairs(folder, args.recursive):
        if args.limit is not None and n_files >= args.limit:
            break
        try:
            xyz, width, height = read_organized_pcd_xyz(pcd_path)
        except ValueError as e:
            print(f"skip {pcd_path.name}: {e}")
            continue

        if tiff_path is not None:
            with Image.open(tiff_path) as img:
                if img.size != (width, height):
                    print(f"warn {pcd_path.name}: tiff size {img.size} != pcd grid {(width, height)}")
        else:
            print(f"warn {pcd_path.name}: no matching range_color.tiff found")

        if expected_wh is None:
            expected_wh = (width, height)
        elif (width, height) != expected_wh:
            print(f"skip {pcd_path.name}: grid {(width, height)} differs from {expected_wh}, "
                  f"would corrupt the fit")
            continue

        n_valid = accumulate(acc, xyz, width, height)
        n_files += 1
        print(f"[{n_files}] {pcd_path.name}: {n_valid} valid points")

    if n_files == 0 or acc["n"] == 0:
        raise SystemExit("No valid organized pcd/tiff pairs found.")

    fx, fy, cx, cy, rmse_u, rmse_v = solve(acc)

    print()
    print(f"Used {n_files} frame(s), {int(acc['n'])} valid points total, grid {expected_wh[0]}x{expected_wh[1]}")
    print(f"Reprojection fit error: RMSE_u = {rmse_u:.4f} px, RMSE_v = {rmse_v:.4f} px")
    print()
    print("Intrinsic matrix K:")
    print(f"[[{fx:12.4f}, {0:12.4f}, {cx:12.4f}],")
    print(f" [{0:12.4f}, {fy:12.4f}, {cy:12.4f}],")
    print(f" [{0:12.4f}, {0:12.4f}, {1:12.4f}]]")
    print()
    print(f"fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")
    print()
    print("Unproject pixel (u, v) with depth Z (same sign/unit convention as this PCD's z field) to a 3D point:")
    print("  X = (u - cx) * Z / fx")
    print("  Y = (v - cy) * Z / fy")


if __name__ == "__main__":
    main()
