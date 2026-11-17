"""从场景点云中按密度峰自动定位主体并居中裁剪（任意物体，不限植物）。

用法：python crop_center.py fused.ply out_prefix [--radius 0.5]

产物：<out_prefix>_cropped.npy（xyz）、<out_prefix>_cropped.png（俯/侧视核验图）
 stdout 输出主体点数、包围盒尺寸。
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage


def read_ply(path):
    props = []
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, t))
            elif line == "end_header":
                break
        dtype = np.dtype([(nm, {"float": "<f4", "uchar": "u1", "int": "<i4"}[t])
                          for nm, t in props])
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out_prefix")
    ap.add_argument("--radius", type=float, default=0.5,
                    help="裁剪半径（重建坐标单位，先用 view_ply 估尺寸）")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = read_ply(args.ply)
    print(f"points: {len(xyz)}")

    # xy 平面密度峰定位主体
    H, xedges, yedges = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=120)
    Hs = ndimage.gaussian_filter(H, sigma=2)
    pi, pj = np.unravel_index(np.argmax(Hs), Hs.shape)
    cx = (xedges[pi] + xedges[pi + 1]) / 2
    cy = (yedges[pj] + yedges[pj + 1]) / 2
    print(f"density peak at ({cx:.2f}, {cy:.2f})")

    d = np.linalg.norm(xyz[:, :2] - [cx, cy], axis=1)
    obj = xyz[d <= args.radius]
    if len(obj) < 2000:  # 太瘦就放大半径
        obj = xyz[d <= args.radius * 2]
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    print(f"cropped object: {len(obj)} points, bbox size = {hi - lo}")

    s = obj if len(obj) <= 80000 else obj[np.random.default_rng(0).choice(
        len(obj), 80000, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (i, j), t in zip(axes, [(0, 1), (0, 2)], ["object top (xy)", "object side (xz)"]):
        ax.scatter(s[:, i], s[:, j], s=0.8, c="#9ecfff", linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.set_facecolor("#111111")
    fig.savefig(args.out_prefix + "_cropped.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    np.save(args.out_prefix + "_cropped.npy", obj)
    print("saved", args.out_prefix + "_cropped.png / .npy")


if __name__ == "__main__":
    main()
