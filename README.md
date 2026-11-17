# scan-anything-3d

**拿手机绕任何物体拍一段视频，一键变成带颜色的稠密三维点云。**
Turn a casual orbiting video of any object into a dense, colored, measurable 3D point cloud — one command, no coding.

不局限于植物：摆件、手办、工具、家具、雕塑、石头、花盆……任何有纹理、不反光、不动的东西都能扫。

## 工作原理

```
视频 (20–30 s, 手持环绕)
  → ffmpeg 抽帧 (默认 2.5 fps，得 50–80 张)
  → COLMAP SIFT 特征 (GPU)
  → 顺序匹配器（视频帧小基线专用）
  → 增量式 SfM（相机位姿 + 稀疏点云）
  → GPU Patch Match 稠密重建
  → 融合成 fused.ply（Meshlab / CloudCompare / Blender 直接打开）
```

本质是把专业摄影测量流程（SfM + MVS）封装成"视频进、点云出"，并附两个后处理小工具
（查看、居中裁剪）。

## 安装

需要两个外部工具（PATH 可找到，或用环境变量 `COLMAP_BIN` / `FFMPEG_BIN` 指定路径）：

| 依赖 | 版本 | 说明 |
|------|------|------|
| [COLMAP](https://colmap.github.io/) | ≥ 3.7（4.x 自动适配参数名） | 带 CUDA 的版本快 10 倍；没有 GPU 也能跑（CPU 模式） |
| [ffmpeg](https://ffmpeg.org/) | 任意近期版本 | 抽帧用 |

```bash
pip install numpy scipy matplotlib   # 后处理小工具
```

## 用法

```bash
# 1. 扫描：视频 -> 点云
python scan.py --video cup.mp4 --out my_cup
# 产物：my_cup/dense/fused.ply

# 2. 查看：点云 -> 俯视/侧视核验图
python view_ply.py my_cup/dense/fused.ply my_cup/check.png

# 3. 居中裁剪：把主体从场景里抠出来（密度峰自动定位）
python crop_center.py my_cup/dense/fused.ply my_cup/object --radius 0.5
# 产物：my_cup/object_cropped.npy + .png
```

## 拍摄规范（决定成败，务必看 [docs/SHOOTING.md](docs/SHOOTING.md)）

要点速览：

1. **环绕一整圈**，20–30 秒，慢而匀速（每圈 15–25 秒）
2. 目标占画面 **1/3 以上**；越小细节越少
3. 物体**绝对不能动**；手持可以，脚步要稳
4. 均匀光照，避免逆光、强反光、透明材质
5. 想要真实尺寸：画面里放一把尺子，重建后做尺度归一

## 实测记录

| 场景 | 视频 | 抽帧 | 注册率 | 稠密点云 |
|------|------|------|--------|---------|
| 田间草莓苗（35 s, 720p） | 87 帧 | **87/87** | 1,899,397 点 |
| 田间草莓苗（19 s, 1080p 60fps） | 48 帧 | **48/48** | 2,223,869 点 |

## 已知边界（诚实清单）

- 单目重建只给**相对尺寸**；绝对尺寸需要画面内参照物。
- **搞不定的物体**：纯透明（玻璃）、强反光（镜面、漆面）、无纹理纯色表面、
  会动的东西（宠物、窗帘、风吹的枝叶）、尺度极小的物体（<5 cm 细节受限）。
- 点云是"彩色点"不是网格；要实体模型需要再做表面重建（Poisson 等）。

## 许可

MIT（见 LICENSE）。COLMAP 与 ffmpeg 为各自项目许可。
