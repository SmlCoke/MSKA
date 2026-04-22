# Quick Start Guide

## 1. 准备环境

建议使用单独虚拟环境，然后安装最小依赖：

```powershell
pip install -r AetherSign/requirements_v0.txt
```

如果你已经有和原始 MSKA 仓库类似的环境，只要其中包含这些核心依赖即可：

- `torch`
- `tensorflow`：可选，用于 TensorFlow CTC beam search 解码；如果缺失，代码会自动回退到 greedy CTC 解码
- `numpy`
- `PyYAML`

## 2. 准备数据

默认目录：

```text
AetherSign/dataset/
├─ label/
│  ├─ gloss_map.txt
│  ├─ train.csv
│  ├─ dev.csv
│  └─ test.csv
└─ npy/
```

如果数据不在默认位置，请修改：

`AetherSign/configs/aethersign_s2g_v0.yaml`

重点字段：

- `data.dataset_root`
- `data.label_dir`
- `data.npy_dir`
- `gloss.gloss2id_file`

## 3. 构建词表

先由 `gloss_map.txt` 生成 `gloss2ids.pkl`：

```powershell
python AetherSign/scripts/build_gloss_vocab.py --config AetherSign/configs/aethersign_s2g_v0.yaml
```

生成后的特殊 token 固定为：

- `<si>: 0`
- `<unk>: 1`
- `<pad>: 2`
- `</s>: 3`

## 4. 开始训练

最基本训练命令：

```powershell
python AetherSign/scripts/train_s2g.py --config AetherSign/configs/aethersign_s2g_v0.yaml
```

常用覆盖参数：

```powershell
python AetherSign/scripts/train_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --device cuda `
  --batch-size 8 `
  --epochs 100 `
  --num-workers 4
```

继续训练：

```powershell
python AetherSign/scripts/train_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --resume AetherSign/outputs/aethersign_mska_s2g_v0/checkpoint_last.pth
```

## 5. 批量评估 dev/test

对 `dev` 集评估：

```powershell
python AetherSign/scripts/infer_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --checkpoint AetherSign/outputs/aethersign_mska_s2g_v0/checkpoint_best.pth `
  --split dev
```

对 `test` 集评估：

```powershell
python AetherSign/scripts/infer_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --checkpoint AetherSign/outputs/aethersign_mska_s2g_v0/checkpoint_best.pth `
  --split test
```

如果你想指定推理 head：

```powershell
python AetherSign/scripts/infer_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --checkpoint AetherSign/outputs/aethersign_mska_s2g_v0/checkpoint_best.pth `
  --split test `
  --prediction-head ensemble_last
```

## 6. 对单个样本推理

只给 `.npy`：

```powershell
python AetherSign/scripts/infer_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --checkpoint AetherSign/outputs/aethersign_mska_s2g_v0/checkpoint_best.pth `
  --npy AetherSign/dataset/npy/S000009_P0000_T00.npy
```

如果你还想顺便和真值比较：

```powershell
python AetherSign/scripts/infer_s2g.py `
  --config AetherSign/configs/aethersign_s2g_v0.yaml `
  --checkpoint AetherSign/outputs/aethersign_mska_s2g_v0/checkpoint_best.pth `
  --npy AetherSign/dataset/npy/S000009_P0000_T00.npy `
  --gloss "你 老师 是"
```

## 7. 输出文件说明

训练输出目录默认是：

`AetherSign/outputs/aethersign_mska_s2g_v0`

常见文件：

- `checkpoint_last.pth`：最后一个 epoch 的权重
- `checkpoint_best.pth`：按验证集最优 WER 保存的权重
- `metrics.jsonl`：逐 epoch 日志
- `dev_predictions_epoch_XXX.jsonl`：每轮验证集预测
- `test_predictions_best.jsonl`：最佳模型测试集预测
- `test_summary_best.json`：最佳模型测试集汇总指标
