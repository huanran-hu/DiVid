# DiVid

DiVid 是一个面向视频生成模型的六维多样性评测工具。给定同一个生成 prompt 和一组 `k >= 2` 个视频，工具输出六个维度的 Mean Pairwise Distance (MPD)：

```text
semantic, style, subject, scene, motion, camera
```

MPD 计算同一 prompt 下所有视频对的平均距离，分数越高表示该维度的跨视频变化越大。每个维度同时返回基于同一相似度 kernel 的 Vendi score 和相似度矩阵。

## 评测原理

- **Semantic**：OpenCLIP 视频特征与 prompt 文本特征构造条件 kernel，去除共同的 prompt 对齐成分。
- **Style**：InceptionV3 pool3 视频特征比较整体构图、纹理、色彩、光照和外观。
- **Subject**：GroundingDINO 定位主体，SAM2 跟踪 mask，DINOv2 编码主体裁剪区域。
- **Scene**：复用主体 mask，将主体区域模糊后，用 DINOv2 编码背景环境。
- **Motion v4.1**：CoTracker 提取主体相对背景轨迹，并在主体 canonical 化后用 RAFT 提取非刚性形变；两者的 RBF kernel 等权融合。
- **Camera v3**：CoTracker 跟踪背景点，使用 RANSAC 拟合全局 affine/homography 变换，比较平移、缩放、旋转等运镜描述符。

Subject/Scene/Motion/Camera 使用同一套主体定位结果，但后续输入区域和描述符不同：Subject 只编码主体，Scene 只编码被抑制主体后的背景，Motion 使用主体相对背景的运动，Camera 使用背景的全局变换。

## 安装

建议使用 Python 3.10 或更高版本，并根据硬件安装对应 CUDA 版本的 PyTorch。

```bash
git clone https://github.com/your-org/DiVid.git
cd DiVid
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install git+https://github.com/facebookresearch/co-tracker.git
```

## 模型

OpenCLIP 和 Hugging Face 模型会按默认配置自动下载。为复现实验中的视觉和动态评测，请将以下权重放在项目的 `models/` 目录：

| 模型 | 下载地址 |
| --- | --- |
| InceptionV3 | [PyTorch checkpoint](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth) |
| RAFT-Large | [PyTorch checkpoint](https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth) |
| CoTracker3 scaled offline | [Hugging Face checkpoint](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth) |
| GroundingDINO | [Hugging Face model](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| SAM 2.1 | [Hugging Face model](https://huggingface.co/facebook/sam2.1-hiera-large) |
| DINOv2 | [Hugging Face model](https://huggingface.co/facebook/dinov2-large) |

```bash
mkdir -p models
wget -O models/inception_v3_google-0cc3c7bd.pth \
  https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
wget -O models/raft_large_C_T_SKHT_V2-ff5fadd5.pth \
  https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth
wget -O models/scaled_offline.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

第三方模型遵循各自的许可证和使用条款；发布或商用前请分别确认其许可范围。

## 使用指南

### 评测一个视频目录

目录中放置同一个 prompt 生成的至少两个视频：

```bash
python -m divid.cli \
  --video-dir ./videos/example \
  --prompt "A dog running through a snowy forest" \
  --subjects "a dog" \
  --output ./results/example.json
```

`--subjects` 是 Subject、Scene、Motion、Camera 的主体查询。多主体可以写成多个参数：

```bash
python -m divid.cli \
  --video-dir ./videos/example \
  --prompt "Two dogs playing with a red ball in a park" \
  --subjects "a dog" "a red ball"
```

如果不提供 `--subjects`，程序会把完整 prompt 作为 GroundingDINO 查询；对复杂 prompt，建议显式提供主体短语。

### 直接指定视频

```bash
python -m divid.cli \
  --videos ./videos/a.mp4 ./videos/b.mp4 ./videos/c.mp4 \
  --prompt "A red car turning on a city street" \
  --subjects "a red car"
```

### 只运行部分维度

```bash
python -m divid.cli \
  --video-dir ./videos/example \
  --prompt "A bird flying over a lake" \
  --subjects "a bird" \
  --dimensions semantic style motion camera \
  --device cuda
```

无 GPU 时可使用 `--device cpu`，但 Subject/Scene、Motion 和 Camera 会明显变慢。

## 输出

结果为 JSON，主要字段如下：

```json
{
  "prompt": "...",
  "video_count": 10,
  "semantic_mpd": 0.21,
  "style_mpd": 0.34,
  "subject_mpd": 0.18,
  "scene_mpd": 0.27,
  "motion_mpd": 0.15,
  "camera_mpd": 0.09
}
```

不同维度的 MPD 不应直接跨维度比较绝对值；模型比较应在同一维度、相同 prompt 和相同视频数下进行。当前版本只包含六维多样性核心评测，不包含质量、忠实度或旧版动态指标。

## Python API

```python
from divid import evaluate

scores = evaluate(
    ["videos/a.mp4", "videos/b.mp4", "videos/c.mp4"],
    "A cat jumping onto a chair",
    subjects=["a cat"],
    device="cuda",
)
print(scores["semantic_mpd"], scores["camera_mpd"])
```

## 许可证

DiVid 代码采用 MIT License。GroundingDINO、SAM2、DINOv2、CoTracker、OpenCLIP、InceptionV3 和 RAFT 权重及代码仍受其各自许可证约束。
