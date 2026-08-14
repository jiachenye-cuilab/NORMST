# Count-aware 全基因预训练编码器

该目录是一个独立实验原型，不修改现有 NORMST 训练路径，也不会向 `Data/`
写入内容。它从标准 Visium raw counts 学习一个可冻结的 256 维表示：

- 前 255 维：基因组成编码；
- 最后 1 维：`log1p(total UMI)`；
- 解码器输出基因组成概率，并以 `total UMI * composition` 重建预期 counts。

组成编码器使用全部“训练集中至少出现一次、且所有切片共有”的基因。验证和测试
表达不参与基因过滤、初始化、latent 标准化或 checkpoint 选择。

## 为什么不是普通自编码器

每个 spot 的 UMI 被二项 thinning 成两个互补子样本 A/B。模型用 A 预测 B、用 B
预测 A，主损失是按目标 UMI 归一化的 composition cross-entropy；可选的 NB 项使用
逐基因 dispersion。这样不会通过复现输入中的抽样零值获得主要收益。

组成分支内部使用 `log1p(scale * counts / total_UMI)`，但 total UMI 同时作为独立标量
保留。因此该分解不会丢弃测序深度；真正不可逆的部分只有 255 维瓶颈。

## 预训练

从 `src` 目录运行：

```powershell
conda run -n normst python pre-train/train.py `
  --manifest diagnostics/phase1_startup_gate/C4_seed2027/manifest.json `
  --output-dir pre-train/runs/seed2027 `
  --composition-dim 255
```

输出目录必须为空且不能位于任何输入切片目录内。主要产物包括：

- `best.pt` / `last.pt`：模型、优化器和完整配置；
- `genes.txt`：固定基因顺序；
- `latent_statistics.npz`：仅由训练集拟合的 latent/library 标准化参数；
- `history.json`、`val_metrics.json`、`test_metrics.json`。

默认宽度为 `255 + 1`，可用 `--composition-dim 128` 或 `511` 做瓶颈消融。
checkpoint 只根据固定-thinning validation loss 选择，test 仅在选择完成后评估。

## Pair-aware LODO

`manifests/lodo_d1.json`、`lodo_d2.json`、`lodo_d3.json` 使用严格的嵌套分组：

- test：一个donor的4张切片全部留出；
- train/validation：剩余两个donor各贡献一整对serial sections；
- 同一serial pair绝不会跨train、validation或test。

当前冻结分组为：

| fold | train | validation | test donor |
|---|---|---|---|
| lodo_d1 | 151671-72, 151673-74 | 151669-70, 151675-76 | Br5292 / 151507-10 |
| lodo_d2 | 151509-10, 151673-74 | 151507-08, 151675-76 | Br5595 / 151669-72 |
| lodo_d3 | 151507-08, 151671-72 | 151509-10, 151669-70 | Br8100 / 151673-76 |

审计本地checked-in manifests和数据路径：

```powershell
conda run -n normst python pre-train/lodo.py audit `
  pre-train/manifests/lodo_d1.json `
  pre-train/manifests/lodo_d2.json `
  pre-train/manifests/lodo_d3.json `
  --check-paths
```

如果服务器数据不位于仓库同级的`Data/visium`，生成仅适用于该服务器的manifest：

```bash
python pre-train/lodo.py generate \
  --visium-root /absolute/path/to/Data/visium \
  --output-dir /absolute/path/to/lodo_manifests
```

生成器拒绝覆盖已有manifest；训练入口会自动审计带
`protocol=pair_grouped_lodo`的manifest，pair或donor角色被改坏时直接终止。

## 导出冻结特征

```powershell
conda run -n normst python pre-train/export_features.py `
  --checkpoint pre-train/runs/seed2027/best.pt `
  --manifest diagnostics/phase1_startup_gate/C4_seed2027/manifest.json `
  --output-dir pre-train/features/seed2027
```

每个切片输出一个 NPZ，包含原始组成 latent、`log_library`、标准化后的完整特征
`feature_standardized`、spot barcode 和切片角色。后续模型应预测标准化特征；冻结
decoder 仍应用于 count-space 辅助损失和生物学指标验证。

`frozen.py` 提供 `FrozenCountRepresentation`：

- `encode_target(raw_counts)` 生成标准化的 256 维监督目标；
- `latent_loss(prediction, target)` 分开记录组成和 UMI 标量损失；
- `decode_feature(prediction)` 将下游预测反标准化并还原预期 counts；
- `count_auxiliary_loss(prediction, raw_counts)` 让梯度穿过冻结 decoder 回到下游模型。

冻结是针对 encoder/decoder 参数；decoder 运算本身不放在 `no_grad` 中，因此辅助损失
仍能训练下游空间网络。

## 逻辑测试

```powershell
conda run -n normst python -m unittest discover -s pre-train/tests -v
```
