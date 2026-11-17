"""视频 -> 稠密点云：一键驱动 ffmpeg 抽帧 + COLMAP 稀疏/稠密重建。

用法：
  python scan.py --video VID.mp4 --out out_dir [--fps 2.5] [--max-frames 150]

依赖（PATH 可找到，或用 --colmap-bin / --ffmpeg-bin 指定）：
  - ffmpeg / ffprobe
  - COLMAP >= 3.7（4.x 自动适配参数名；建议带 CUDA 的版本）

产物：
  out_dir/frames/        抽帧 jpg
  out_dir/sparse/0/      COLMAP 稀疏模型（.bin + .txt）
  out_dir/dense/fused.ply  稠密点云（标准 PLY，Meshlab/CloudCompare 可开）
  各阶段日志 out_dir/log_*.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-5:])
        raise RuntimeError(f"命令失败({r.returncode}): {' '.join(cmd)}\n{tail}")


def feature_gpu_flag(colmap):
    """COLMAP 4.x 用 FeatureExtraction.use_gpu，3.x 用 SiftExtraction.use_gpu。"""
    help_text = subprocess.run([colmap, "feature_extractor", "--help"],
                               capture_output=True, text=True).stdout
    if "--FeatureExtraction.use_gpu" in help_text:
        return "--FeatureExtraction.use_gpu"
    return "--SiftExtraction.use_gpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.5, help="抽帧速率")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--colmap-bin", default=os.environ.get("COLMAP_BIN", "colmap"))
    ap.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    args = ap.parse_args()

    frames_dir = os.path.join(args.out, "frames")
    sparse_dir = os.path.join(args.out, "sparse")
    dense_dir = os.path.join(args.out, "dense")
    for d in (frames_dir, sparse_dir, dense_dir):
        os.makedirs(d, exist_ok=True)

    print(f"[1/6] 抽帧 fps={args.fps}")
    run([args.ffmpeg_bin, "-v", "error", "-i", args.video,
         "-vf", f"fps={args.fps}", "-q:v", "2",
         os.path.join(frames_dir, "f_%04d.jpg")],
        os.path.join(args.out, "log_frames.txt"))
    n = len(os.listdir(frames_dir))
    if n > args.max_frames:  # 均匀抽稀到 max_frames
        keep = sorted(os.listdir(frames_dir))[::int(n / args.max_frames) + 1][:args.max_frames]
        for f in os.listdir(frames_dir):
            if f not in keep:
                os.remove(os.path.join(frames_dir, f))
        n = len(os.listdir(frames_dir))
    print(f"      {n} 帧")

    gpu = feature_gpu_flag(args.colmap_bin)
    db = os.path.join(args.out, "db.db")
    if os.path.exists(db):
        os.remove(db)

    print("[2/6] SIFT 特征提取 (GPU)")
    run([args.colmap_bin, "feature_extractor", "--database_path", db,
         "--image_path", frames_dir, gpu, "1"],
        os.path.join(args.out, "log_feat.txt"))
    print("[3/6] 顺序匹配（视频帧小基线专用）")
    run([args.colmap_bin, "sequential_matcher", "--database_path", db,
         gpu.replace("Extraction", "Matching"), "1"],
        os.path.join(args.out, "log_match.txt"))
    print("[4/6] 增量式建图（SfM）")
    run([args.colmap_bin, "mapper", "--database_path", db,
         "--image_path", frames_dir, "--output_path", sparse_dir],
        os.path.join(args.out, "log_map.txt"))
    model = os.path.join(sparse_dir, "0")
    if not os.path.isdir(model):
        raise RuntimeError("SfM 失败：没有生成 sparse/0，请看 log_map.txt")
    run([args.colmap_bin, "model_converter", "--input_path", model,
         "--output_path", model, "--output_type", "TXT"],
        os.path.join(args.out, "log_conv.txt"))
    n_img = sum(1 for ln in open(os.path.join(model, "images.txt"))
                if ln.strip() and not ln.startswith("#")) // 2
    print(f"      注册 {n_img}/{n} 帧")

    print("[5/6] 去畸变 + GPU Patch Match 稠密重建")
    run([args.colmap_bin, "image_undistorter", "--image_path", frames_dir,
         "--input_path", model, "--output_path", dense_dir],
        os.path.join(args.out, "log_undist.txt"))
    run([args.colmap_bin, "patch_match_stereo", "--workspace_path", dense_dir],
        os.path.join(args.out, "log_pm.txt"))
    print("[6/6] 点云融合 -> fused.ply")
    run([args.colmap_bin, "stereo_fusion", "--workspace_path", dense_dir,
         "--output_path", os.path.join(dense_dir, "fused.ply")],
        os.path.join(args.out, "log_fuse.txt"))
    print("完成：", os.path.join(dense_dir, "fused.ply"))


if __name__ == "__main__":
    sys.exit(main())
