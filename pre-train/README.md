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
conda run -n normst python pre-train/pre-train.py `
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

## scVI + scArches 基线

`scvi_baseline.py` 使用官方 `scvi-tools`。默认 preset 是
`scvi-modern-hvg`，按当前 scVI/scArches reference-mapping 工作流配置输入与模型：

- 候选基因先取所有切片共有、且 train 中总 count 大于零的基因；
- 只在 LODO train slices 的原始 integer counts 上运行 Scanpy Seurat-v3 HVG，
  `batch_key=donor`，取前 3000；validation/test 表达不参与拟合或排序；
- `n_hidden=128`、`n_layers=2`、`n_latent=10`、`dropout=0.2`，encoder/decoder
  使用 layer norm、关闭 batch norm，并编码 donor covariate；
- gene likelihood 使用 `ZINB`，mini-batch 为 128，Adam 学习率 `1e-3`、
  `eps=0.01`、`weight_decay=1e-6`；
- reference 先进行 100 epochs KL warm-up，不在这段选择 checkpoint；随后在固定
  `KL weight=1` 的最多 150 epochs 内按整切片 validation ELBO 早停，patience 30；
- scArches query adaptation 固定 200 epochs，并继续使用 `weight_decay=0`。

`scvi-paper-brain-small` 仍作为兼容 preset 保留：它复现 2018 年 BRAIN-SMALL
notebook 的 raw-count standard-deviation 排序、1 层网络和 dropout 0.1。该旧 notebook
不是当前默认值；当前正式 LODO 实验使用上面的 train-only、donor-aware Seurat-v3 HVG。

参考：[scVI 原论文](https://www.nature.com/articles/s41592-018-0229-2)、
[官方复现仓库](https://github.com/romain-lopez/scVI-reproducibility)、
[官方 scArches reference mapping 教程](https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/scarches_scvi_tools.html)。

reference scVI 以 donor 为 batch covariate，只用 manifest 中的 train spots 更新权重、
用整张 validation slices 选择 checkpoint。

对 held-out donor 输出三组结果：

- `scvi_zero_shot`：不使用 query gradient；把 query 分别按训练中两个 donor 的
  batch 条件解码并平均，避免为未见 donor 使用随机新 batch 权重；
- `scarches_unadapted_initialization`：扩展新 donor batch 后、尚未使用 query
  gradient 的诊断结果；
- `scarches_query_adapted`：固定 epoch 的 scArches 适配结果，明确记录
  `query_expression_used_for_weight_adaptation=true`。

query adaptation 固定使用 `weight_decay=0`。scArches 通过 gradient hook 冻结旧
权重的部分区域；输出中的 `scarches_update_audit.json` 逐参数记录适配前后的实际
变化元素数。

两组重建都使用 posterior-mean latent 和输入 spot 的真实 total UMI，并复用 AE 的
`ReconstructionMetrics`。scArches 是转导式结果，不能当作纯 zero-shot LODO。

从 `src` 运行默认的 modern 单折：

```powershell
conda run -n normst-scvi python pre-train/scvi_baseline.py `
  --manifest pre-train/manifests/lodo_d1.json `
  --output-dir pre-train/runs/scvi_scarches_modern_lodo_d1_seed2027_hvg3000_latent10 `
  --device cuda
```

每次 modern 运行额外保存 `gene_selection.csv`（rank、train raw mean/variance、
normalized variance 和 donor-batch 支持数）及
固定顺序的 `genes.txt`。新的重建指标只覆盖这 3000 个输入基因，不能直接与旧的
约 2.4 万基因 AE 指标比较；需要把 AE/PCA 在同一个 `genes.txt` 上重新评估，或比较
共同的下游空间任务指标。

matched AE 必须直接复用同折 scVI 的基因顺序，例如：

```powershell
conda run -n normst python pre-train/pre-train.py `
  --manifest pre-train/manifests/lodo_d1.json `
  --fixed-genes pre-train/runs/scvi_scarches_modern_lodo_d1_seed2027_hvg3000_latent10/genes.txt `
  --output-dir pre-train/runs/ae_modern_lodo_d1_seed2027_hvg3000_latent10 `
  --composition-dim 10 --hidden-dims 128,128 --dropout 0.2 `
  --batch-size 128 --epochs 250 --patience 30 --lr 1e-3 `
  --weight-decay 1e-6 --seed 2027 --device cuda
```

`config.json` 会记录外部 `genes.txt` 的绝对路径、SHA256 和最终输出基因 SHA256；
同折两个 hash 必须一致，才允许把重建指标解释为模型差异。

若要保持 modern 模型超参数不变、只把基因筛选切换为原始 BRAIN-SMALL 的
raw-count variance/std 排序，使用：

```powershell
conda run -n normst-scvi python pre-train/scvi_baseline.py `
  --preset scvi-variance-hvg `
  --manifest pre-train/manifests/lodo_d1.json `
  --n-top-genes 3000 `
  --min-train-detection-fraction 0.05 `
  --output-dir pre-train/runs/scvi_variance_lodo_d1_seed2027_hvg3000_latent10 `
  --device cuda
```

方差与标准差产生完全相同的排序；实现保存两列。检测率阈值只在 train spots 上拟合，
用于排除由极少数超高 count spot 驱动的基因。正式 matched AE 继续通过
`--fixed-genes <scVI-run>/genes.txt` 复用同折基因。当前 DLPFC 三折的纯方差 top-3000
最低 train detection fraction 已高于 5%，所以 `0.05` 只作为保护条件，不改变入选结果。

要精确复现此前的 AE-matched 全基因入口，显式指定兼容 preset：

```powershell
conda run -n normst-scvi python pre-train/scvi_baseline.py `
  --preset ae-matched `
  --manifest pre-train/manifests/lodo_d1.json `
  --ae-run-dir pre-train/runs/lodo_d1_seed2027_dim255 `
  --output-dir pre-train/runs/scvi_scarches_lodo_d1_seed2027_dim255_compat
```

该入口拒绝覆盖非空目录；`Data/` 始终只读。scVI 专用测试使用：

```powershell
conda run -n normst-scvi python -m unittest discover -s pre-train/tests -v
```

## 单次 8:2:2 横向降维比较

`manifests/random_pair_8_2_2_seed2027.json` 是独立于 LODO 的单次横向比较切分。
它先用 `numpy.default_rng(2027)` 随机排列六个完整 serial pairs，再按 4:1:1
分给 train/validation/test，因此对应 8/2/2 张切片且不拆开相邻 serial sections。
该切分的 test donor 已在 train 出现，不构成跨 donor 泛化证据；scVI 直接按已观察
donor 条件推断，并跳过 scArches query adaptation。

三种方法统一使用 train-only donor-aware Seurat-v3 选择的 3000 个基因、seed 2027，
表示均为 31 个 composition/score 维度加 1 个 `log1p(total UMI)` 维度。先运行
`scvi_baseline.py --latent-dim 31` 产生冻结的 `genes.txt`，AE 和 GLM-PCA 再通过
`--fixed-genes` 精确复用该顺序。

当前接入 AE-NORMST 的独立 AE 使用同一 8:2:2 manifest、train-only Seurat-v3
HVG、最低 10% train-spot 检出率（等价于 train zero fraction 不超过 90%）、1000
个基因和 32 维 composition latent。Linux 服务器从仓库根目录依次运行：

```bash
python pre-train/select_hvgs.py \
  --manifest pre-train/manifests/random_pair_8_2_2_seed2027.json \
  --output-dir pre-train/runs/hvg1000_min_detect10pct_random822_seed2027 \
  --n-top-genes 1000 \
  --min-train-detection-fraction 0.1

python pre-train/pre-train.py \
  --manifest pre-train/manifests/random_pair_8_2_2_seed2027.json \
  --fixed-genes pre-train/runs/hvg1000_min_detect10pct_random822_seed2027/genes.txt \
  --output-dir pre-train/runs/ae_random822_seed2027_hvg1000_detect10_latent32_linear_varcov005 \
  --composition-dim 32 \
  --hidden-dims 128,128 \
  --decoder-type linear \
  --dropout 0.2 \
  --thinning-probability 0.5 \
  --nb-weight 0.1 \
  --consistency-weight 0.05 \
  --latent-variance-weight 0.05 \
  --latent-covariance-weight 0.05 \
  --batch-size 128 \
  --epochs 250 \
  --patience 30 \
  --lr 0.001 \
  --weight-decay 1e-6 \
  --gradient-clip 1 \
  --workers 0 \
  --seed 2027 \
  --device cuda \
  --amp
```

`checkpoint_io.py` 会在保存前把 `Path` 元数据递归转换为字符串，因此 Windows
产生的新 checkpoint 可直接在 Linux 加载；加载器也兼容修复前含 `WindowsPath`
元数据的旧 checkpoint。输出目录仍拒绝覆盖已有非空目录，重跑时请使用一个新的
目录名或先人工确认并处理旧目录。

`glmpca_baseline.py` 是本地 conditional-multinomial GLM-PCA 重实现，不是官方 R
`glmpca` 包的运行结果。共享的 gene intercept、donor fixed effects 和 loadings 只由
train expression 更新；validation/test expression 只优化各自的 spot score。这样保持
真实 selected-gene library，并允许复用同一套 count-space 指标，但新 spot 需要迭代投影，
不能把其运行时开销等同于 AE/scVI 的单次 encoder forward。

```powershell
conda run -n normst python pre-train/glmpca_baseline.py `
  --manifest pre-train/manifests/random_pair_8_2_2_seed2027.json `
  --fixed-genes pre-train/runs/scvi_random822_seed2027_hvg3000_latent31/genes.txt `
  --output-dir pre-train/runs/glmpca_random822_seed2027_hvg3000_latent31 `
  --latent-dim 31 --seed 2027 --device cuda
```

`export_scvi_features.py` 和 `analyze_matched_features.py` 用于冻结特征导出及相同
test spots 上的有效秩/技术量依赖诊断；诊断明确排除最后一维显式 library 特征。
