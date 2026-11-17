"""PLY 点云 -> 俯视/侧视核验图（不用装 Meshlab 也能快速检查结果）。

用法：python view_ply.py fused.ply out.png [--max-points 40000]
"""
from __future__ import annotations

import argparse

import numpy as np


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
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(float)
    if "red" in data.dtype.names:
        rgb = np.stack([data["red"], data["green"], data["blue"]],
                       axis=1).astype(float) / 255.0
    else:
        rgb = np.ones((len(xyz), 3)) * 0.7
    return xyz, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out_png")
    ap.add_argument("--max-points", type=int, default=40000)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz, rgb = read_ply(args.ply)
    n = len(xyz)
    idx = np.random.default_rng(0).choice(n, min(args.max_points, n), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (i, j), t in zip(axes, [(0, 1), (0, 2)], ["top view (xy)", "side view (xz)"]):
        ax.scatter(xyz[idx, i], xyz[idx, j], s=0.5, c=rgb[idx], linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{t}  n={n}")
        ax.set_facecolor("#111111")
    fig.savefig(args.out_png, dpi=110, bbox_inches="tight", facecolor="white")
    print("saved", args.out_png, "points:", n)


if __name__ == "__main__":
    main()
