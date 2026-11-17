"""扫描管线自检器：真值模型 -> 模拟手持拍摄 -> 重建 -> 对齐打分。

闭环验证：用一个已知模型（fused.ply / .npy）当"真值"，渲染成带手持抖动的
环绕帧，走和生产环境完全相同的 scan.reconstruct 代码路径，再把重建结果与
真值 ICP 对齐，输出完整度/精度/注册率三张成绩单。

用法：
  python selfcheck.py reference.ply --out check1 [--views 60] [--jitter handheld]

grades（抖动等级）：
  clean      无抖动（管线上限）
  handheld   位姿抖动 + 亮度抖动 + 轻微模糊（模拟手持绕拍，默认）
  noisy      handheld 再加重（测管线底线）

产物：
  <out>/frames/          模拟拍摄帧
  <out>/recon/fused.ply  重建点云
  <out>/report.json      成绩单
  <out>/compare.png       对齐后对照图（真值蓝 / 重建橙）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from scan import reconstruct
from view_ply import read_ply

W, H = 960, 720

JITTER = {
    "clean":    dict(pose_deg=0.0,  pose_frac=0.0,  bright=0.0,  blur=0),
    "handheld": dict(pose_deg=1.2,  pose_frac=0.01, bright=0.06, blur=1),
    "noisy":    dict(pose_deg=3.0,  pose_frac=0.03, bright=0.12, blur=2),
}


def umeyama(src, dst):
    """相似变换（旋转+平移+统一缩放）把 src 对齐到 dst。返回 (R, t, s)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = S.sum() / (X ** 2).sum()
    t = mu_d - s * R @ mu_s
    return R, t, s


def icp_align(src, dst, iterations=30):
    """最近邻 ICP。src 为重建点云，dst 为真值。返回对齐后的 src（在 dst 原始坐标系）。"""
    from scipy.spatial import cKDTree
    mu_d = dst.mean(0)
    src = src - src.mean(0)
    dst = dst - mu_d
    # 初始尺度：包围盒对角线比值
    s0 = np.linalg.norm(dst.max(0) - dst.min(0)) / max(
        1e-9, np.linalg.norm(src.max(0) - src.min(0)))
    src = src * s0
    R_all, t_all, s_all = np.eye(3), np.zeros(3), 1.0
    sub = src[::max(1, len(src) // 20000)]
    tree = cKDTree(dst[::max(1, len(dst) // 20000)])
    dsub = dst[::max(1, len(dst) // 20000)]
    for it in range(iterations):
        dd, idx = tree.query(sub)
        # 截尾：丢掉距离最远的 30% 对应，抗离群点和错误配对
        keep = dd < np.percentile(dd, 70)
        R, t, s = umeyama(sub[keep], dsub[idx[keep]])
        if it >= 3:
            s = 1.0  # 尺度只在头几轮估计，之后冻结——否则部分重叠时持续缩小导致塌缩
        src = s * (src @ R.T) + t
        sub = s * (sub @ R.T) + t
        R_all = R @ R_all
        t_all = s * (t_all @ R.T) + t
        s_all = s * s_all
    return src + mu_d  # 对齐在居中坐标系完成，加回真值质心回到原始坐标系


def render_views(xyz, rgb, frames_dir, n_views, jitter_cfg, seed=0):
    """环绕渲染 + 手持抖动退化，输出 jpg 帧。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    center = xyz.mean(0)
    span = np.linalg.norm(xyz.max(0) - xyz.min(0))
    r_obj = np.linalg.norm(xyz[:, :2] - center[:2], axis=1).max()
    f = float(W)
    radius = 1.9 * r_obj  # 目标占画面约一半（太小 SIFT 特征不足）
    order = rng.permutation(len(xyz))

    for k in range(n_views):
        az = 2 * np.pi * k / n_views
        el = np.radians(8.0 if k % 2 == 0 else -5.0)
        # 手持抖动：方位/俯仰加噪，半径伸缩
        az += np.radians(rng.normal(0, jitter_cfg["pose_deg"]))
        el += np.radians(rng.normal(0, jitter_cfg["pose_deg"]))
        rad = radius * (1 + rng.normal(0, jitter_cfg["pose_frac"]))
        eye = np.array([center[0] + rad * np.cos(az),
                        center[1] + rad * np.sin(az),
                        center[2] + rad * np.sin(el)])
        target = center
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        Rm = np.stack([right, up, fwd])
        p = (Rm @ (xyz - eye).T).T
        valid = p[:, 2] > 1e-6
        u = f * p[:, 0] / np.maximum(p[:, 2], 1e-9) + W / 2
        v = f * p[:, 1] / np.maximum(p[:, 2], 1e-9) + H / 2
        depth = np.where(valid, p[:, 2], np.inf)

        img = np.ones((H, W, 3))
        zbuf = np.full((H, W), np.inf)
        idx = order.copy()
        idx = idx[valid[idx]]
        idx = idx[np.argsort(depth[idx])]
        px = np.round(u[idx]).astype(int)
        py = np.round(v[idx]).astype(int)
        keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px, py = px[keep], py[keep]
        # 向量化 splat：2x2 像素块 + 简单 z-buffer（1px 点太稀，SIFT 没特征）
        dsel = depth[idx[keep]]
        for dx in (0, 1):
            for dy in (0, 1):
                qx, qy = px + dx, py + dy
                ok = (qx < W) & (qy < H)
                closer = dsel[ok] < zbuf[qy[ok], qx[ok]]
                m = np.where(ok)[0][closer]
                zbuf[qy[m], qx[m]] = dsel[ok][closer]
                img[qy[m], qx[m]] = rgb[idx[keep]][ok][closer]
        # 传感器颗粒噪声：给纯色表面制造微纹理，SIFT 才有特征可提
        img += rng.normal(0, 0.03, img.shape)
        # 亮度抖动
        if jitter_cfg["bright"] > 0:
            img *= 1 + rng.normal(0, jitter_cfg["bright"])
        img = np.clip(img, 0, 1)
        # 轻微模糊（手持抖动）
        if jitter_cfg["blur"] > 0:
            img = gaussian_filter(img, sigma=(jitter_cfg["blur"],) * 2 + (0,))
        fig = plt.figure(figsize=(W / 120, H / 120), dpi=120)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")
        fig.savefig(os.path.join(frames_dir, f"f_{k:04d}.jpg"), dpi=120)
        plt.close(fig)
    return radius, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="真值点云（fused.ply 或 .npy，xyz[+rgb]）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=60)
    ap.add_argument("--grade", choices=list(JITTER), default="handheld")
    ap.add_argument("--colmap-bin", default=os.environ.get("COLMAP_BIN", "colmap"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[check] 载入真值 {args.reference}")
    if args.reference.endswith(".npy"):
        arr = np.load(args.reference)
        xyz = arr[:, :3].astype(float)
        rgb = arr[:, 3:6].astype(float) / (255.0 if arr[:, 3:6].max() > 1.5 else 1.0) \
            if arr.shape[1] >= 6 else None
    else:
        xyz, rgb = read_ply(args.reference)
    if rgb is None or rgb.std() < 1e-3:
        print("[check] 警告：真值无颜色/颜色为常数，渲染帧将缺乏纹理，SfM 可能失败")
    n_ref = len(xyz)

    print(f"[check] 模拟拍摄：{args.views} 帧，等级={args.grade}")
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    if not os.listdir(frames_dir):
        render_views(xyz, rgb, frames_dir, args.views, JITTER[args.grade], args.seed)
    n_frames = len(os.listdir(frames_dir))

    print("[check] 走 scan.reconstruct 生产代码路径重建")
    recon_dir = os.path.join(args.out, "recon")
    fused, n_reg, _ = reconstruct(frames_dir, recon_dir, args.colmap_bin)

    print("[check] ICP 对齐重建 <-> 真值")
    rec, _ = read_ply(fused)
    # 只保留主体附近的点（去掉重建飘出的离群）
    c = xyz.mean(0)
    span = np.linalg.norm(xyz.max(0) - xyz.min(0))
    rec_c = rec[np.linalg.norm(rec - rec.mean(0), axis=1) < 2.5 * span]
    aligned = icp_align(rec_c, xyz)

    from scipy.spatial import cKDTree
    tree_ref = cKDTree(xyz)
    tree_rec = cKDTree(aligned)
    d_rec2ref, _ = tree_ref.query(aligned)          # 精度：重建点离真值多远
    d_ref2rec, _ = tree_rec.query(xyz)              # 完整度：真值点被覆盖多远
    scale = span  # 归一化单位
    report = {
        "reference": os.path.basename(args.reference),
        "grade": args.grade,
        "views": n_frames,
        "registered": int(n_reg),
        "registration_rate": n_reg / n_frames,
        "reference_points": int(n_ref),
        "reconstructed_points": int(len(rec)),
        "accuracy_mean": float(d_rec2ref.mean() / scale),
        "accuracy_p95": float(np.percentile(d_rec2ref, 95) / scale),
        "completeness_mean": float(d_ref2rec.mean() / scale),
        "completeness_within_2pct": float((d_ref2rec < 0.02 * scale).mean()),
        "completeness_within_5pct": float((d_ref2rec < 0.05 * scale).mean()),
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 成绩单 =====")
    print(f"帧注册率          : {report['registration_rate']:.0%} ({n_reg}/{n_frames})")
    print(f"精度(重建->真值)  : mean {report['accuracy_mean']:.3f} / "
          f"p95 {report['accuracy_p95']:.3f} （单位=物体尺寸）")
    print(f"完整度(真值被覆盖): 2% 内 {report['completeness_within_2pct']:.0%}，"
          f"5% 内 {report['completeness_within_5pct']:.0%}")
    print(f"报告: {os.path.join(args.out, 'report.json')}")

    # 对照图：真值蓝 / 重建橙（对齐后，侧视）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sub_r = xyz[::max(1, n_ref // 20000)]
    sub_a = aligned[::max(1, len(aligned) // 20000)]
    for ax, (i, j), t in zip(axes, [(0, 1), (0, 2)], ["top (xy)", "side (xz)"]):
        ax.scatter(sub_r[:, i], sub_r[:, j], s=0.5, c="#4fa3ff", label="reference", linewidths=0)
        ax.scatter(sub_a[:, i], sub_a[:, j], s=0.5, c="#ffa04f", label="reconstructed",
                   alpha=0.6, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.legend(markerscale=6)
    fig.savefig(os.path.join(args.out, "compare.png"), dpi=110,
                bbox_inches="tight", facecolor="white")
    print("对照图:", os.path.join(args.out, "compare.png"))


if __name__ == "__main__":
    sys.exit(main())
