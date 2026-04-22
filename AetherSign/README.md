# AetherSign MSKA-SLR v0

这是基于原始 `MSKA` 仓库整理出的一个**自包含**版本，专门用于你们当前的
`关键点序列 -> Gloss 序列` 复现任务。

目标是尽量保留原论文和原代码中的核心设计：

- 四流关键点注意力主干
- 四个识别 head
- CTC 损失
- 自蒸馏损失
- 原始训练超参数

同时只对以下部分做必要适配：

- 数据读取方式：直接读取 `dataset/npy/*.npy` 与 `dataset/label/*.csv`
- 关键点流划分：从原始 79 点改为适配你们的 65 点
- 坐标预处理：支持从归一化图像坐标转换到论文使用的中心化坐标
- 设备适配：移除原代码中的硬编码 `.cuda()`

## 目录结构

```text
AetherSign/
├─ configs/
│  └─ aethersign_s2g_v0.yaml
├─ mska_v0/
│  ├─ data.py
│  ├─ engine.py
│  ├─ metrics.py
│  ├─ model.py
│  ├─ optimizer.py
│  ├─ recognition.py
│  ├─ tokenizer.py
│  ├─ utils.py
│  └─ visual_head.py
├─ scripts/
│  ├─ build_gloss_vocab.py
│  ├─ train_s2g.py
│  └─ infer_s2g.py
├─ outputs/
├─ QuickStart.md
├─ 代码导读.md
├─ Prompt.md
└─ project-5.md
```

## 与原仓库的主要差异

1. 只保留 `S2G` 任务，不包含 `SLT`、mBART、翻译相关代码。
2. 原仓库使用 pickle 格式数据文件，这里改为直接读取：
   - `dataset/label/train.csv`
   - `dataset/label/dev.csv`
   - `dataset/label/test.csv`
   - `dataset/npy/{name}.npy`
3. 原仓库 `DSTA` 中写死了 `26/27/79` 个节点数，这里改成从配置中的
   `left/right/face/body` 自动推导。
4. 你们的 65 点关键点流划分固定为：
   - `left = [0, 2, 7, 11, 13, 15, 23..43]`
   - `right = [0, 5, 8, 12, 14, 16, 44..64]`
   - `face = [0..10]`
   - `body = [0..64]`
5. 推理脚本支持两种模式：
   - 对 `dev/test` split 批量评估
   - 对单个 `.npy` 文件直接推理
6. 如果环境中安装了 TensorFlow，则继续使用原计划中的 CTC beam search 解码；
   如果没有 TensorFlow，则自动回退到 greedy CTC 解码，保证训练和基本推理流程可运行。

## 默认数据约定

默认假设数据放在：

```text
AetherSign/dataset/
├─ label/
│  ├─ gloss_map.txt
│  ├─ train.csv
│  ├─ dev.csv
│  ├─ test.csv
│  └─ gloss2ids.pkl
└─ npy/
   ├─ S000001_P0000_T00.npy
   └─ ...
```

如果你的数据不在这个位置，可以直接修改
`configs/aethersign_s2g_v0.yaml` 里的路径字段。

## 建议使用顺序

1. 先阅读 [QuickStart.md](./QuickStart.md)
2. 再阅读 [代码导读.md](./代码导读.md)
3. 确认 `gloss_map.txt`、`train/dev/test.csv` 和 `npy/` 都准备好
4. 先运行词表构建脚本，再开始训练
