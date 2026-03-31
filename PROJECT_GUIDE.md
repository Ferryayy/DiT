# DiT 项目使用指南

> **Scalable Diffusion Models with Transformers** — Meta (Facebook Research)
> 
> 一句话概括：用 Transformer 替代 U-Net 做扩散模型的骨干网络，在 ImageNet 上生成高质量的类别条件图像。

---

## 目录

- [这个项目能做什么？](#这个项目能做什么)
- [整体架构一览](#整体架构一览)
- [项目文件结构说明](#项目文件结构说明)
- [VAE 在哪里？](#vae-在哪里)
- [环境安装](#环境安装)
- [数据集要求](#数据集要求)
- [如何训练](#如何训练)
- [如何推理（生成图像）](#如何推理生成图像)
- [如何评估（FID 等指标）](#如何评估fid-等指标)
- [可用的模型变体](#可用的模型变体)
- [关键参数速查表](#关键参数速查表)

---

## 这个项目能做什么？

| 功能 | 说明 |
|------|------|
| **类别条件图像生成** | 给定一个 ImageNet 类别编号（0~999），生成该类别的高质量图像 |
| **支持 256×256 和 512×512** | 两种分辨率的预训练模型均可直接使用 |
| **从零训练** | 提供完整的 DDP 分布式训练脚本，可在自己的数据上训练 |
| **批量采样 + 评估** | 支持多 GPU 并行采样数万张图，输出 `.npz` 文件用于计算 FID/IS |

**典型用途**：你想生成某个类别的图像（如金毛犬、火山、披萨等），只需指定 ImageNet 类别 ID 即可。

---

## 整体架构一览

```
┌─────────────────────────────────────────────────────────┐
│                    完整生成流程                           │
│                                                         │
│  训练时：                                                │
│  原始图像 ──→ [VAE Encoder] ──→ 潜空间特征(latent)       │
│                                      │                  │
│                                      ▼                  │
│                              [DiT Transformer]          │
│                              (学习去噪过程)              │
│                                                         │
│  推理时：                                                │
│  随机噪声 ──→ [DiT Transformer] ──→ 去噪后的latent       │
│              (逐步去噪 250步)              │              │
│                                           ▼             │
│                                   [VAE Decoder]         │
│                                           │             │
│                                           ▼             │
│                                      生成的图像          │
└─────────────────────────────────────────────────────────┘
```

**关键点**：DiT 只负责"在潜空间中去噪"这一步，VAE 负责"图像 ↔ 潜空间"的转换。

---

## 项目文件结构说明

```
DiT/
├── models.py              # ⭐ DiT 模型定义（Transformer 骨干网络）
├── train.py               # ⭐ 训练脚本（PyTorch DDP 分布式训练）
├── sample.py              # ⭐ 单机推理脚本（生成少量图像）
├── sample_ddp.py          # ⭐ 多 GPU 批量采样脚本（用于 FID 评估）
├── download.py            # 预训练模型自动下载工具
├── run_DiT.ipynb          # Colab/Jupyter 演示 notebook
├── environment.yml        # Conda 环境依赖
├── diffusion/             # 扩散过程实现（来自 OpenAI）
│   ├── __init__.py        #   create_diffusion() 入口函数
│   ├── gaussian_diffusion.py  # 高斯扩散核心（加噪、去噪、损失计算）
│   ├── respace.py         #   时间步重采样（加速采样用）
│   ├── timestep_sampler.py#   训练时的时间步采样策略
│   └── diffusion_utils.py #   KL散度、对数似然等数学工具
└── visuals/               # 示例生成图像
```

### 各部分的作用

| 文件/模块 | 做什么用 |
|-----------|---------|
| `models.py` | 定义 DiT 模型结构。包含 12 种模型变体（S/B/L/XL × patch 2/4/8）。这是"去噪网络"本身 |
| `train.py` | 训练入口。加载 ImageNet 数据 → VAE 编码到潜空间 → DiT 学习去噪 → 保存 checkpoint |
| `sample.py` | 推理入口。加载预训练 DiT → 从随机噪声去噪 → VAE 解码回图像 → 保存 `sample.png` |
| `sample_ddp.py` | 大规模采样。多 GPU 并行生成 50K 张图，输出 `.npz` 文件用于 FID 计算 |
| `download.py` | 自动从 Facebook 服务器下载预训练权重到 `pretrained_models/` 目录 |
| `diffusion/` | 扩散过程的数学实现（加噪调度、去噪采样、损失函数）。来自 OpenAI 的 ADM/IDDPM 代码 |

---

## VAE 在哪里？

**VAE 不在本项目代码中**，而是通过 `diffusers` 库直接加载 Stability AI 的预训练 VAE：

```python
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
```

具体来说：
- **VAE 来源**：Stable Diffusion 的 VAE（`stabilityai/sd-vae-ft-ema` 或 `stabilityai/sd-vae-ft-mse`）
- **VAE 作用**：将 256×256 的图像压缩为 32×32×4 的潜空间特征（缩小 8 倍）
- **VAE 不参与训练**：在训练和推理中 VAE 权重都是冻结的，只有 DiT 部分被训练
- **两个 VAE 变体**：
  - `ema`：用 EMA 微调的 VAE（训练时默认用这个）
  - `mse`：用 MSE 损失微调的 VAE（推理时默认用这个，FID 更好）

**首次运行时**，VAE 权重会从 HuggingFace 自动下载（约 335MB）。

---

## 环境安装

### 方式一：Conda（推荐）

```bash
git clone https://github.com/facebookresearch/DiT.git
cd DiT
conda env create -f environment.yml
conda activate DiT
```

### 方式二：pip 手动安装

```bash
pip install torch torchvision   # 需要 >= 1.13，建议装 CUDA 版本
pip install timm                # Vision Transformer 组件
pip install diffusers           # 用于加载 Stable Diffusion 的 VAE
pip install accelerate          # diffusers 的依赖
```

### 依赖总结

| 包 | 用途 |
|----|------|
| `pytorch >= 1.13` | 深度学习框架 |
| `torchvision` | 数据加载、图像变换 |
| `timm` | 提供 PatchEmbed、Attention 等 ViT 组件 |
| `diffusers` | 加载 Stable Diffusion 的 VAE 模型 |
| `accelerate` | diffusers 的运行时依赖 |

---

## 数据集要求

### 训练需要什么数据集？

**默认使用 ImageNet-1K 数据集**（ILSVRC 2012），这是一个包含 1000 个类别、约 128 万张图像的分类数据集。

### 数据集格式要求

训练脚本使用 PyTorch 的 `ImageFolder`，所以数据必须按以下目录结构组织：

```
/path/to/imagenet/train/
├── n01440764/          # 类别文件夹（ImageNet synset ID）
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   └── ...
├── n01443537/
│   ├── ...
│   └── ...
├── n01484850/
│   └── ...
└── ... (共 1000 个类别文件夹)
```

**关键要求**：
- 每个子文件夹 = 一个类别
- 文件夹名就是类别标签（`ImageFolder` 会自动按文件夹名排序分配 0~999 的标签）
- 图像格式：JPEG/PNG 均可
- 图像尺寸：任意（训练时会自动 center crop 到 256×256 或 512×512）

### 如何获取 ImageNet 数据集？

1. **官方渠道**：前往 [image-net.org](https://image-net.org/) 注册账号，申请下载 ILSVRC2012
2. **学术机构**：很多大学/实验室有内部镜像
3. **HuggingFace**：`huggingface-cli download imagenet-1k`（需要申请访问权限）
4. **Kaggle**：搜索 "ImageNet Object Localization Challenge"

### 能用自定义数据集吗？

**可以！** 只要满足 `ImageFolder` 格式即可。但需要注意：
- 修改 `--num-classes` 参数为你的类别数
- 如果类别数不是 1000，推理时也要相应修改
- 预训练模型是在 ImageNet-1K 上训练的，自定义数据需要从头训练

### 数据预处理

训练时自动执行以下预处理（在 `train.py` 中）：
1. **Center Crop**：将图像裁剪为正方形（256×256 或 512×512）
2. **随机水平翻转**：数据增强
3. **归一化**：像素值归一化到 [-1, 1]
4. **VAE 编码**：在训练循环中，图像会被 VAE 编码为 32×32×4 的潜空间特征

---

## 如何训练

### 基本训练命令

```bash
# 单节点 N 卡训练 DiT-XL/2（256×256）
torchrun --nnodes=1 --nproc_per_node=N train.py \
    --model DiT-XL/2 \
    --data-path /path/to/imagenet/train \
    --image-size 256
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-path` | **必填** | ImageNet 训练集路径 |
| `--model` | `DiT-XL/2` | 模型变体，可选见下方表格 |
| `--image-size` | `256` | 图像分辨率，可选 256 或 512 |
| `--epochs` | `1400` | 训练轮数 |
| `--global-batch-size` | `256` | 全局 batch size（会自动分配到各 GPU） |
| `--num-classes` | `1000` | 类别数（ImageNet 为 1000） |
| `--vae` | `ema` | VAE 变体（`ema` 或 `mse`，训练时无影响） |
| `--results-dir` | `results` | 输出目录 |
| `--log-every` | `100` | 每 N 步打印一次日志 |
| `--ckpt-every` | `50000` | 每 N 步保存一次 checkpoint |
| `--global-seed` | `0` | 随机种子 |

### 训练输出

```
results/
└── 000-DiT-XL-2/
    ├── checkpoints/
    │   ├── 0050000.pt    # 每 50K 步保存一次
    │   ├── 0100000.pt
    │   └── ...
    └── log.txt           # 训练日志
```

每个 checkpoint 包含：`model`（模型权重）、`ema`（EMA 权重）、`opt`（优化器状态）、`args`（训练参数）

### 训练资源参考

| 模型 | 推荐 GPU | 论文训练步数 | 预计时间 |
|------|---------|-------------|---------|
| DiT-XL/2 | 8× A100 | 7M steps | 数周 |
| DiT-B/4 | 4× A100 | 400K steps | 数天 |

### 注意事项

- **必须使用 `torchrun` 启动**（DDP 分布式训练），即使单卡也需要
- **A100 用户**：脚本已默认开启 TF32 加速，训练速度大幅提升
- **不支持断点续训**（原版代码未实现，可参考 [fast-DiT](https://github.com/chuanyangjin/fast-DiT)）

---

## 如何推理（生成图像）

### 方式一：使用预训练模型（最简单）

```bash
# 生成 256×256 图像（自动下载预训练权重）
python sample.py --image-size 256 --seed 1

# 生成 512×512 图像
python sample.py --image-size 512 --seed 1
```

运行后会在当前目录生成 `sample.png`，包含 8 张图像（默认类别：金毛犬、狼、雪豹等）。

### 方式二：使用自己训练的模型

```bash
python sample.py --model DiT-XL/2 --image-size 256 --ckpt /path/to/your/checkpoint.pt
```

### 方式三：Jupyter Notebook

打开 `run_DiT.ipynb`，可以在 Colab 或本地 Jupyter 中交互式运行。

### 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `DiT-XL/2` | 模型变体 |
| `--image-size` | `256` | 生成图像分辨率（256 或 512） |
| `--vae` | `mse` | VAE 变体（推理默认用 `mse`，FID 更好） |
| `--cfg-scale` | `4.0` | Classifier-Free Guidance 强度。越大图像越"清晰"但多样性降低 |
| `--num-sampling-steps` | `250` | 去噪步数。越多质量越好但越慢 |
| `--seed` | `0` | 随机种子 |
| `--ckpt` | `None` | 自定义 checkpoint 路径（不填则自动下载预训练模型） |

### 如何指定生成的类别？

在 `sample.py` 第 38 行修改类别列表：

```python
class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
```

这些数字是 ImageNet 类别 ID。常用类别示例：

| ID | 类别 |
|----|------|
| 207 | 金毛犬 |
| 360 | 水獭 |
| 387 | 大象 |
| 974 | 火山 |
| 88 | 金刚鹦鹉 |
| 979 | 山谷 |
| 417 | 气球 |
| 279 | 北极狐 |

完整 ImageNet 类别列表：[查看](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

### 预训练模型下载

首次运行 `sample.py` 时会自动下载，也可以手动下载：

| 模型 | 分辨率 | FID | 下载链接 |
|------|--------|-----|---------|
| DiT-XL/2 | 256×256 | 2.27 | [下载](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) |
| DiT-XL/2 | 512×512 | 3.04 | [下载](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) |

或使用下载脚本：

```bash
python download.py   # 下载所有预训练模型到 pretrained_models/
```

---

## 如何评估（FID 等指标）

使用 `sample_ddp.py` 批量生成图像，然后用 ADM 评估工具计算指标：

```bash
# 多 GPU 采样 50K 张图
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
    --model DiT-XL/2 \
    --num-fid-samples 50000 \
    --image-size 256 \
    --cfg-scale 1.5

# 生成的 .npz 文件可用 ADM 评估套件计算 FID/IS
# https://github.com/openai/guided-diffusion/tree/main/evaluations
```

---

## 可用的模型变体

模型命名规则：`DiT-{规模}/{patch大小}`

| 模型 | 深度 | 隐藏维度 | 注意力头数 | Patch 大小 | 参数量级 |
|------|------|---------|-----------|-----------|---------|
| DiT-S/2, S/4, S/8 | 12 | 384 | 6 | 2/4/8 | 小 |
| DiT-B/2, B/4, B/8 | 12 | 768 | 12 | 2/4/8 | 中 |
| DiT-L/2, L/4, L/8 | 24 | 1024 | 16 | 2/4/8 | 大 |
| DiT-XL/2, XL/4, XL/8 | 28 | 1152 | 16 | 2/4/8 | 超大 |

- **Patch 大小越小**（如 /2），输入 token 越多，计算量越大，效果越好
- **预训练权重只提供 DiT-XL/2**（256×256 和 512×512 两个版本）

---

## 关键参数速查表

### 训练快速启动

```bash
# 最小可运行命令（单节点 1 卡）
torchrun --nnodes=1 --nproc_per_node=1 train.py \
    --model DiT-B/4 \
    --data-path /path/to/imagenet/train \
    --image-size 256 \
    --global-batch-size 32 \
    --epochs 100
```

### 推理快速启动

```bash
# 最简单的推理（自动下载模型）
python sample.py --image-size 256 --seed 42

# 快速采样（减少步数，质量略降）
python sample.py --image-size 256 --num-sampling-steps 50 --cfg-scale 4.0

# 高质量采样
python sample.py --image-size 256 --num-sampling-steps 250 --cfg-scale 4.0
```

### cfg-scale 调参建议

| cfg-scale | 效果 |
|-----------|------|
| 1.0 | 无引导，多样性最高，质量一般 |
| 1.5 | 轻度引导 |
| 4.0 | 默认值，质量与多样性平衡 |
| 7.0+ | 强引导，图像更清晰但多样性低 |

---

## 常见问题

**Q: 没有 GPU 能跑吗？**
A: 推理可以在 CPU 上跑（很慢），训练必须有 GPU。

**Q: 能生成任意内容的图像吗？**
A: 不能。这是类别条件模型，只能生成 ImageNet 1000 个类别中的图像，不支持文本描述。

**Q: 和 Stable Diffusion 什么关系？**
A: DiT 使用了 Stable Diffusion 的 VAE 来做图像编解码，但去噪网络用的是 Transformer 而非 U-Net。两者共享"潜空间扩散"的思路。

**Q: 显存需要多少？**
A: 推理 DiT-XL/2 约需 8GB 显存；训练 DiT-XL/2 建议 8×A100（80GB）。小模型如 DiT-B/4 训练资源需求小很多。

---

## CelebA-HQ 用于 DiT 训练：类别怎么处理？

### 核心问题

DiT 是一个**类别条件**生成模型，它的架构中 `y`（类别标签）是**必须的输入**：

```python
# models.py - DiT.forward()
c = t + y   # 时间步嵌入 + 类别嵌入，两者相加作为条件
```

类别标签 `y` 通过 `LabelEmbedder`（一个 `nn.Embedding` 查找表）转换为向量，和时间步嵌入相加后，控制每个 Transformer Block 的 AdaLN 调制。

而 CelebA-HQ 是一个**无类别标签的人脸数据集**（只有人脸图像，没有分类标签），所以你需要选择一种策略来处理。

---

### 方案对比

| 方案 | 做法 | 适用场景 | 推荐度 |
|------|------|---------|--------|
| **方案 A：无条件生成** | 所有图像标签设为 0，`num_classes=1` | 只想生成人脸，不区分属性 | ⭐⭐⭐⭐⭐ |
| **方案 B：属性条件生成** | 用 CelebA 的 40 个属性组合出类别 | 想控制生成的人脸属性（如性别、是否戴眼镜） | ⭐⭐⭐ |

---

### 方案 A：无条件生成（最推荐）

这是最简单也最常用的做法。**把所有图像当作同一个类别**。

#### 1. 数据集目录结构

```
/path/to/celeba_hq/
└── train/
    └── 0/                    # 只有一个子文件夹，文件夹名为 "0"
        ├── 00000.jpg
        ├── 00001.jpg
        ├── 00002.jpg
        └── ... (30000 张人脸图像)
```

把所有图像放进一个名为 `0` 的子文件夹即可，`ImageFolder` 会自动给所有图像分配标签 `0`。

#### 2. 训练命令

```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
    --model DiT-XL/2 \
    --data-path /path/to/celeba_hq/train \
    --image-size 256 \
    --num-classes 1
```

关键：`--num-classes 1`

#### 3. 推理时的修改

`sample.py` 需要做两处修改：
[读取文件中]
好的，现在我来完整说明。

---

## CelebA-HQ + DiT：类别处理完整方案

### 核心问题

DiT 的 `forward` 中，类别标签 `y` 是**必须的输入**：

```python
# models.py 第 236-237 行
y = self.y_embedder(y, self.training)    # (N, D)
c = t + y                                # 时间步 + 类别 → 条件向量
```

`y` 经过 `LabelEmbedder`（一个 `nn.Embedding(num_classes+1, hidden_size)` 查找表）变成向量，和时间步嵌入相加后控制整个去噪过程。CelebA-HQ 没有类别标签，所以需要处理。

---

### 三种方案

#### 方案 A：无条件生成（最推荐 ⭐⭐⭐⭐⭐）

**思路**：所有图像都是"人脸"这一个类别，设 `num_classes=1`，所有标签都是 `0`。

**1. 数据目录结构**

```
/path/to/celeba_hq/train/
└── 0/                          # 唯一的子文件夹
    ├── 00000.jpg
    ├── 00001.jpg
    └── ... (所有 30K 张图)
```

只需把所有图片放进一个名为 `0` 的文件夹，`ImageFolder` 自动分配标签 `0`。

**2. 训练命令**

```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
    --model DiT-XL/2 \
    --data-path /path/to/celeba_hq/train \
    --image-size 256 \
    --num-classes 1
```

**3. 推理命令**

```bash
python sample.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --num-classes 1 \
    --ckpt /path/to/your/checkpoint.pt
```

但 `sample.py` 中有两处硬编码需要修改：

- 第 50 行 `class_labels = [207, 360, ...]` → 改为 `[0, 0, 0, ...]`
- 第 59 行 `y_null = torch.tensor([1000] * n, ...)` → 空类别 ID 应该是 `num_classes` 的值，即 `1`

修改如下：

```python
# 第 50 行：所有标签都是 0（唯一类别）
class_labels = [0] * 8   # 生成 8 张人脸

# 第 59 行：null class 的 ID = num_classes = 1
y_null = torch.tensor([args.num_classes] * n, device=device)
```

---

#### 方案 B：属性条件生成（可控生成）

**思路**：CelebA-HQ 继承了 CelebA 的 **40 个二值属性**（如 Male、Eyeglasses、Smiling 等）。你可以选几个关键属性组合成类别。

**举例**：选 3 个属性（Male、Smiling、Eyeglasses），组合出 2³ = 8 个类别：

| 类别 ID | Male | Smiling | Eyeglasses | 含义 |
|---------|------|---------|------------|------|
| 0 | ✗ | ✗ | ✗ | 女性、不笑、无眼镜 |
| 1 | ✗ | ✗ | ✓ | 女性、不笑、戴眼镜 |
| 2 | ✗ | ✓ | ✗ | 女性、微笑、无眼镜 |
| 3 | ✗ | ✓ | ✓ | 女性、微笑、戴眼镜 |
| 4 | ✓ | ✗ | ✗ | 男性、不笑、无眼镜 |
| 5 | ✓ | ✗ | ✓ | 男性、不笑、戴眼镜 |
| 6 | ✓ | ✓ | ✗ | 男性、微笑、无眼镜 |
| 7 | ✓ | ✓ | ✓ | 男性、微笑、戴眼镜 |

**做法**：
1. 读取 CelebA 的 `list_attr_celeba.txt` 属性文件
2. 写一个预处理脚本，根据属性组合把图片分到对应的子文件夹
3. 训练时设 `--num-classes 8`

```python
# preprocess_celeba.py - 示例预处理脚本
import os, shutil, pandas as pd

attr = pd.read_csv("list_attr_celeba.txt", sep=r"\s+", skiprows=1)
# CelebA 属性值为 -1/1，转为 0/1
attr = (attr + 1) // 2

src_dir = "/path/to/celeba_hq/images"
dst_dir = "/path/to/celeba_hq/train"

for idx, row in attr.iterrows():
    # 3 个属性组合成类别 ID (0~7)
    class_id = row["Male"] * 4 + row["Smiling"] * 2 + row["Eyeglasses"]
    class_dir = os.path.join(dst_dir, str(class_id))
    os.makedirs(class_dir, exist_ok=True)
    shutil.copy(os.path.join(src_dir, idx), class_dir)
```

训练：`--num-classes 8`，推理时指定想要的类别 ID 即可控制生成属性。

**对于大多数人脸生成任务，方案 A（无条件生成）就够了**。CelebA-HQ 只有 3 万张图，数据量不大，类别越少模型越容易学好。