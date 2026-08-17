# Progressive NORMST 模型与训练合同

状态：**核心合同已冻结，direct-512 实现已接入，尚未启动真实数据训练**
最后更新：2026-08-18

本文是 Progressive NORMST 第一阶段唯一的技术合同，统一记录模型、数据、训练和评估规则。旧 [`CURRENT_PLAN.md`](../CURRENT_PLAN.md) 中的讨论选项不再构成合同；其中 frozen-AE/latent 路线已被 direct Shared-512 决策取代。

本文不授权启动训练、修改 `Data/`、同步 `NORMST/` 或复用旧 checkpoint。任何未明确记录的细节不得从旧训练入口静默继承。

## 1. 任务、证据与信息边界

- 任务是标准 Visium masked-spot recovery，不是真实 dense super-resolution。
- 输入、输出、target 和 loss 均为同序 Shared-512 expression。
- query truth 只进入 final gene loss，不构造 representation target，也不进入任何 forward feature。
- global key/value 永远只来自 original-visible nodes；query prediction 不反馈 visible。
- frontier 按轮同步更新，每个 query 只在首次到达时激活一次；同轮 query 不互相读取。
- 最大轮数 `T=4`；Depth `>=5`、无 visible path 或 all-query component 始终为 global-only。
- 现有 `src/models/progressive_normst.py` 只是 synthetic-tested frozen-AE prototype，保持原样用于历史兼容；本文另以 `ProNORMST` 实现 direct-512，不实例化 `ProgressiveNORMST`/`FrozenLatentEncoder`，只复用其中已验证且与AE无关的 geometry、global-attention 和 local-operator primitives。
- 旧 NORMST、AE-NORMST、AE 和 progressive prototype checkpoint 均与本合同不兼容。

证据必须明确区分：static/synthetic validation、real-data smoke、pilot 和完整 LODO 实验；前两者不构成性能或生物学结论。

## 2. 数据协议与运行单位

### 2.1 Primary 与 pilot

- Primary：三折 `pair_grouped_lodo`，每折 `4 train / 4 validation / 4 test`；完整 donor 和 serial pair 不跨角色。
- Pilot：pair-aware `8/2/2`，split seed `2027`；只作工程和模型健康检查，不支持 donor-generalization 结论。
- Pilot 固定使用 initialization seed `2027`；LODO 每折使用 `2027/2028/2029`。每个独立训练的神经方法均有 `3 folds x 3 seeds = 9` 个正式 runs；四个神经方法共36个，IDW不训练。
- initialization、mask、data-order 和其他显式 RNG 相互独立；data-order seed固定为 `2027`，matched runs复用相同 masks 与数据顺序。

### 2.2 Validation、test 与失败运行

- Pilot validation 只判断预先冻结的健康门槛；进入 LODO 前锁定唯一候选和完整合同。
- LODO validation 只用于 early stopping 和 best-checkpoint 选择。
- 每个正式 run 的候选与 checkpoint hash 锁定后，test 只成功提交一次并保存 raw predictions。prediction、metrics 与 completion marker 必须先在 staging 目录完整生成，再作为同一 `test_artifacts/` 原子提交；提交前技术中断允许对同一锁定 checkpoint 安全重试，已完整提交且哈希一致时只作幂等读取，不计为第二次 test。
- 技术中断可从兼容的 `last` 恢复，或以相同 run identity 从头重启；不得因表现差、NaN/Inf 或 OOM 更换 seed。
- 除第3.1节披露的固定 panel 开发来源外，test expression 不得参与 fold-specific preprocessing/统计量拟合、训练、validation、超参数选择或 checkpoint 选择。数据适配器允许通过 Scanpy 在初始化时加载 test expression，但必须按 `role=test` 逻辑只读隔离；只有候选与 best-checkpoint 锁定后才能将其传入 test evaluation。

### 2.3 Optimization sample unit

- 一个 microbatch = 一张完整 train slice × 一个 mask，forward batch size为1。
- 每个 optimizer step 依次处理全部 train slices，step loss为各 slice-mask loss的等权平均；不按 spot、visible 或 query 数加权。
- 全部 slice microbatches 完成 backward 后才执行 gradient clipping、optimizer step 和 scaler update。

## 3. Gene、表达空间与 direct state

### 3.1 Shared-512 panel

- 所有 folds、protocols 和 matched methods 固定使用同一 ordered 512 Ensembl genes，不按 fold 重选。
- Panel：[`shared_panel_512_ensembl.txt`](diagnostics/dlpfc_151676_shared_panel_20260817/shared_panel_512_ensembl.txt)。
- Ordered-sequence SHA256：`72562d01005a5078a0d95b38a050824299fa906f4c9888ff989c2aba9a73a7ce`。
- 第一阶段排除 `MT-*`、`RPL/RPS*`、HB、IG、`SCGB*` genes；不使用额外 panel-UMI scalar input。

Panel provenance：该 panel 在 `151676` 的空间块 train/validation pilot 中开发，未使用该 pilot 的内部 test spots；但 `151676` 属于 Br8100，也是 `lodo_d3` 的 held-out donor。因此正式 LODO 只支持“固定任务 panel 下的模型训练泛化”，不能声称 panel 设计阶段完全未接触 Br8100。panel 在正式结果前冻结，所有方法一致使用，结果后不得重选。

### 3.2 Expression transform

- 对 panel counts 做 panel-only CP10K，再取 natural `log1p`：
  `x_sg = log(1 + 10000*c_sg / sum_{g in panel} c_sg)`。
- panel library `<=0` 时 fail closed。
- query counts/library 只用于 target；任务恢复 within-panel relative expression，不声称恢复 absolute counts 或 full-transcriptome library size。

### 3.3 Gene-wise scaling 与 detection weights

- 使用 train-only per-gene RMS、无中心化：`z_sg=x_sg/d_g`。
- 每张 train slice 先对全部 tissue spots 计算 gene-wise mean square，再对 slices 等权平均并开方。
- scale statistics 使用 CPU float64；模型使用 `d_g=max(float32(r_g),1e-6)`。
- `p_g` 由全部 train-role tissue spots计算：先逐 slice 计算 `x>0` detection rate，再对 slices 等权。
- positive element weight为 `clip(sqrt((1-p_g)/p_g),1,3)`，zero weight为1；`p_g=0` 或1时权重置1，不删除 gene、不做 smoothing。
- 同 fold 的 seeds 和 matched methods 复用同一 gene order、`d_g`、`p_g` 和 weight artifacts。
- 不做 per-spot RMSNorm、LayerNorm、L2 normalization、centering、clipping 或其他输入 rescaling。

### 3.4 Canonical direct state

- 不训练或加载 AE，不使用 AE checkpoint、latent statistics 或 latent target。
- `D_state=512`，`H0=z`；visible state形状为 `[N_visible,512]`。
- global/local branches 前不增加 shared trainable lifting；兼容 adapter 只能是无参数数值恒等映射。
- decoder 输出同序 `[N_query,512]`。
- 只训练 final gene-prediction loss；不使用 latent/local-state/consistency/contrastive/variance/routing 等辅助 loss。
- decoder 最后一层 bias 初始化为 train-role、slice-balanced 的 final-`z` gene mean；统计用 CPU float64 后转 float32，bias 随后可训练。

## 4. Full graph、geometry 与 batch

### 4.1 Canonical graph

- node universe 为 `in_tissue=1` 且 counts/positions 一一对应的 spots，按 `(array_row,array_col)` 升序编号。
- expression、barcode、coordinates、masks、predictions 和 diagnostics 使用同一顺序；缺失、重复或不一致时 fail closed。
- 六方向 neighbor deltas 固定为 `(0,-2),(0,+2),(-1,-1),(-1,+1),(+1,-1),(+1,+1)`，opposites 为 `1,0,5,4,3,2`。
- 缺失 tissue neighbor 的 slot 为 `-1`；禁止 KNN、Delaunay、距离阈值或跨 gap 补边。graph 不随 mask 改变。

### 4.2 Coordinates 与 native scale

- `full_xy` 使用 Space Ranger full-resolution pixel center，列顺序 `[pxl_col,pxl_row]`。
- 不做中心化、缩放、配准或 array-grid fallback；公开 query geometry 可进入 relative-bias/graph，但 absolute XY 不直接进入 decoder。
- 每张 slice 的 `native_scale` 为 full graph 全部 unique undirected native-edge pixel lengths 的 CPU-float64 median；在 mask 前固定并供全部 masks/methods 复用。
- 保留所有合法 components、boundary 和 isolated nodes，不桥接、不删除。

### 4.3 Batch 与 anti-leakage

- model input 只含 compact `visible_expression_z`、visible/query indices 和 full geometry。
- loss target 单独保存 query expression、同序 indices 及 positive/zero weights。
- visible/query 必须互斥并 partition 全部 nodes；更改 query truth 不得改变 forward prediction，target 必须 stop-gradient。

## 5. Progressive NORMST 拓扑

### 5.1 Fixed visible-only global branch

- 所有 queries 在 frontier propagation 前各读取一次 fixed global context `G_Q`，之后不更新。
- 使用8-head softmax cross-attention，`D_model=256`、每head 32维：
  `G_Q=MHA(Q=shared_mask_token, K=H0_visible, V=H0_visible; radial_bias)`。
- 使用独立 Q/K/V/O projections；不加入 query residual、FFN、query-query self-attention 或动态 query memory。
- K/V 输入直接为 `H0`，projection 前不做 per-token LayerNorm/RMSNorm。
- relative bias 只进入 attention logits。令 `r=||xy_q-xy_v||/native_scale`，输入 `[r,r^2]` 到 `Linear(2,32)->GELU->Linear(32,8)`；末层零初始化，不强制单调。
- global branch 不读取 signed direction，不向 token/value/decoder 注入绝对坐标。
- 输出使用无 affine 的 RMSNorm：`G_bar=RMSNorm(G_Q)`。

### 5.2 Synchronous local frontier

- Round `t` 的 frontier 只由上一轮 active/source snapshot 计算，整轮完成后统一 commit。
- original-visible source center为 `H0`；earlier-query source center只使用其 local state `L_Q`。fixed `G_Q` 和 decoder feature不向更深 query传播。
- 对 target `q` 的方向槽 `k`，取 `j=neighbor[q,k]`、`p=neighbor[j,k]` 构成 aligned `p->j->q` path。
- 合法两跳 candidate：
  `candidate_qk=h_j+lambda_qk*(h_j-h_p)`，`lambda_qk in [0,1]`。
- 缺少合法 `p` 时强制 `lambda_qk=0`，candidate退化为 `h_j`。
- 缺少合法 `p` 时 scorer 中的 `h_j-h_p`、`c_p` 和 `has_p` 分别为全零、`0` 和 `0`。
- scorer 输入 `[h_j,h_j-h_p,c_j,c_p,has_p,source_type]`，使用共享 `Linear(input,256)->GELU` trunk，再输出1个 bounded `lambda` 和8个 direction logits。
- scorer 不读取 direction ID、absolute direction vector、`G_Q`、absolute XY、query truth 或同轮 query state。

### 5.3 Direction routing 与 propagated state

- 使用8个 local directional heads；无效方向 logits 置 `-inf`。
- direction softmax加入 `log(path_reliability+eps)`。
- 学习全局共享 channel-to-head routing `A in R^(512x8)`，按 channel 对 heads 做 softmax；不依赖 gene/channel 顺序分组。
- 每个 channel 直接聚合 aligned candidates，得到 `L_q`；不增加 candidate value projection、post-aggregation MLP 或 recurrent refinement。
- local rule、path scorer、routing 和 `LocalProjection(512,256)` 在全部 rounds 共享，不使用 round/depth embedding。

### 5.4 Reliability、fusion 与 decoder

- original-visible confidence `c=1`。
- path reliability：
  `r_qk=c_j*((1-stopgrad(lambda_qk))+stopgrad(lambda_qk)*c_p)`。
- activated-query confidence：
  `c_q=0.95*sum_k stopgrad(alpha_bar_qk*r_qk)`；
  其中 `alpha_bar_qk` 是该方向对512个 channel-specific routing概率的算术均值。
  confidence metadata 全部停止梯度。
- coverage 为 previous-round 合法 source 数除以该 node 在完整 tissue graph 中的实际 degree；degree为0时coverage定义为0。
- local projection只对已激活 query计算：
  `U_bar=RMSNorm_no_affine(Linear_no_bias(L_Q))`。
- fusion gate `g_q=active_q*coverage_q*c_q`，最终 feature：
  `F_Q=[G_bar || g_q*U_bar]`。
- 未激活、Depth `>=5` 或 disconnected query严格使用 `[G_bar || 0_256]`。
- 不增加 fusion MLP、额外 projection、affine branch scale 或 convex global/local gate。
- shared decoder：
  `Linear(512,512)->GELU->Linear(512,512)`；
  输出层为 unconstrained linear，不在训练或 primary metrics 中 clipping。
- full BPTT 保留最多4轮；`L_Q` 不 detach，只有 confidence/reliability metadata stop-gradient。

### 5.5 初始化与固定数值

- shared mask token 和 source-type embedding 使用 `Normal(0,0.02)`；模型内无 dropout。
- non-affine RMSNorm `eps=1e-6`；path reliability `eps=1e-8`。
- lambda head small-random weight、zero bias，初始 `lambda≈0.5`。
- direction head使用 `Normal(0,1e-3)`/zero bias；channel routing近零小随机初始化。
- decoder hidden layer使用标准初始化；最后一层 weight使用 `Normal(0,1e-3)`，bias按3.4初始化。

## 6. Mask families 与 sampling

### 6.1 双任务和日程

- ordinary-random interpolation 与 spatial-gap 均用于 training、validation 和 test，并分别报告。
- 两个 families 的目标 query fraction均为50%，`N_target=floor(N/2)`。
- ordinary-random 必须精确抽取 `N_target`；gap 以其为上限，允许按6.4 underfill，禁止 overshoot。
- optimizer steps按 `ordinary -> gap` 确定性1:1交替；一个cycle含两个steps。
- `cycles_per_epoch=32`，即64 optimizer steps；每张 train slice 每epoch使用32个ordinary和32个gap masks。

### 6.2 Gap geometry

- standard core只使用完整 interior native-graph `r=3` 和 `r=4` balls，并保留一圈 original-visible protected buffer。
- standard holes不得 overlap 或相邻；按 accepted standard-core query mass追求 `r3:r4=1:1`。
- 不足部分依次由 `r=2` 和 random remainder填充。
- `r=2` 可被真实 tissue boundary/internal gap截断，但 actual eccentricity必须为2且 realized size `>=15`；仅当整个candidate与已有query和standard protected ring均不相交、且加入后不超过 `N_target` 时才整体接纳，否则整块拒绝；不裁剪冲突节点，不设自身 protected ring。
- random remainder无比例上限，但不得跳过仍可接纳的 standard 或 `r=2` candidate来主动增加 random 比例。
- random eligible set为全部 tissue nodes 去除 `Q_structured` 和 `P_standard`；从中 uniform without replacement。random nodes可彼此相邻或与 `r=2` 相邻。
- final components 和 depth基于完整 query union重新计算。

### 6.3 Candidate order 与 RNG

- `r=3/4/2` 各使用独立 seeded random permutation；顺序为 standard `r=3/4 -> r=2 -> random`，不回溯。
- standard阶段优先当前 accepted query mass较小的 radius；初始同量时由独立公平 bit决定先尝试 `r=3` 或 `r=4`。
- `base_mask_seed=2027`；不同 radius、ordinary、random fill 和 radius-start使用 domain-separated substreams。
- mask identity包含 schema、protocol、fold、role、slice、family、mask index和attempt index；不含 method、init seed或device。
- payload使用 sorted-key、无空白 UTF-8 JSON；SHA-256前8 bytes按 big-endian unsigned 64-bit seed解析。
- sampling统一使用 PyTorch CPU `Generator`，保存PyTorch版本；禁止CUDA RNG。

### 6.4 Resampling 与固定 banks

- training在每个 global family-cycle、每slice、每family生成新mask。
- validation/test各固定为每张slice × 每family 16个masks；matched methods和init repeats复用同一bank。
- 每个gap identity只运行 `attempt_index=0`，不得重试、换seed或按shortfall筛选。
- random eligible pool不足时选择全部eligible nodes并接受 `1<=N_query<N_target`；不设最低realized fraction，underfill不是失败。
- 每个mask必须保存 realized query fraction、hole sizes、provenance、depth和component信息。

## 7. Loss、指标与聚合

### 7.1 Training loss 与选模

- Gene SmoothL1 `beta=1.0`；prediction、target、weights和reduction均为float32。
- positive element使用 `w_g+`，zero element权重为1；每个gene先除以其query weight和，再对512 genes等权，最后对slices等权。
- NaN/Inf直接视为技术失败。
- 唯一 best-checkpoint criterion为上述 weighted SmoothL1 `z`-space validation loss；先按7.2聚合，再对ordinary/gap `1:1` 等权。
- raw-`x` scientific metrics不参与checkpoint选择。

### 7.2 Evaluation aggregation

- 对每个 `(fold,init,family)`：16个masks在slice内等权，再对该role实际包含的slices等权；LODO为4张，pilot validation/test为2张；query-pooled结果不是primary。
- ordinary与gap分别报告；需要综合值时两者 `1:1`。
- held-out donor/fold是生物学泛化单位：先在fold内汇总3个init，再跨3个fold汇总。
- masks、slices和init seeds均不得作为独立生物学重复。

### 7.3 Scientific metrics

- primary metrics在 inverse-scaled `x=log1p(panel-CP10K)` 空间使用 raw prediction。
- clipped-to-zero结果只作secondary；`z`-space只作optimization diagnostic。
- 必报：unweighted SmoothL1、MAE、RMSE、gene/spot Pearson、prediction/truth variance ratio、negative fraction、positive/zero error和paired IDW gain。
- 按mask family、actual depth、degree/component和gap provenance分层。
- IDW gain在同一mask内配对；error gain定义为 `IDW-NORMST`，correlation gain定义为 `NORMST-IDW`。
- 每个mask-stratum至少10个queries；slice层stratum至少8/16 masks有效。样本不足或Pearson/variance分母为零时记NA，并报告有效数量和coverage。
- gene Pearson为defined genes均值，spot Pearson为defined spots均值；per-gene variance ratio报告median和IQR。
- all-512为primary；另报告supported-gene结果：positive指标排除 `p_g=0`，zero指标排除 `p_g=1`，同时依赖两类时只用 `0<p_g<1`。

## 8. Optimization、checkpoint 与 diagnostics

### 8.1 Trainable scope 与 optimizer

- direct-expression adapter无参数；global branch、local scorer/routing、mask token、embeddings、projections和decoder从step 0起全部训练。gate/fusion严格由第5.4节公式确定，没有独立可训练参数。
- 主模型不 staged-unfreeze；冻结branch只能作为单独标记的ablation。
- AdamW：`lr=2e-5`、`betas=(0.9,0.999)`、`eps=1e-8`。
- Linear/MLP matrix weights使用 `weight_decay=1e-4`；bias、norm、mask token、embeddings和routing logits使用0。

### 8.2 Precision、gradient 与 determinism

- CUDA forward使用FP16 autocast和dynamic GradScaler；parameters、optimizer state及loss reduction保持FP32。
- validation/test沿用FP16 forward，metrics用FP32；非CUDA运行全FP32。
- 全部slice backward完成并 `unscale_` 后执行global L2 gradient clipping，`max_norm=1.0`；记录clip前norm和non-finite状态。
- 固定并分离所有RNG；关闭cuDNN benchmark和TF32。
- 允许CUDA nondeterministic kernels，不承诺bitwise复现；matched runs必须使用相同硬件、软件和determinism设置。
- checkpoint 回放与同一 checkpoint 的 round-invariance 中，indices/depth 等离散身份必须精确一致；浮点 prediction 固定使用 `rtol=2e-3, atol=2e-4`，并把容差、最大绝对/相对误差和 mismatch 数写入验收产物，不以 bitwise equality 误判允许的 CUDA 数值抖动。

### 8.3 Budget 与 checkpoint

- 最多50 epochs = 3200 optimizer steps。
- 前128 steps线性warmup至 `2e-5`，之后按step cosine衰减至 `2e-6`。
- 每个完整epoch后运行一次完整validation bank；epoch 0不参与选模。
- early stopping：`patience=10`、`min_delta=1e-5`；相同criterion保留更早epoch，只在完整cycle/epoch边界停止。
- 每个不可覆盖run目录保存完整 `best` 和 `last` checkpoint。
- checkpoint保存 model、optimizer、scheduler、scaler、RNG、epoch/step/cycle，并关联 split、gene/preprocessing、geometry、mask、model、loss和metric数值兼容 hash。
- 数值兼容 hash 只覆盖可能改变训练或锁定预测的字段；源码文件 hash、本文档 hash 和 runtime/version 作为 audit provenance 单独记录。只有前者不兼容时阻断 resume；仅 audit provenance 漂移时发出明确警告。
- 任何会改变 model/data/mask/loss/optimizer 数值语义的实现修改都必须显式提升 `NUMERICAL_IMPLEMENTATION_SCHEMA`；注释、格式、审计输出或 runtime 记录变化不提升。该版本是源码 hash 从兼容性判断中移除后的人工语义边界。
- resume只读取数值合同完全兼容的 `last`；正式prediction只使用锁定且与 `best.pt` 字节一致的 checkpoint。
- run目录必须非覆盖，并保存 config、manifest、history、predictions、diagnostics manifest和environment/version信息；gene order、preprocessing、geometry、mask或model schema不匹配时fail closed。
- epoch history中的validation只保存role级、family级完整聚合summary与best criterion，不递归嵌入每slice×mask明细；正式test仍保存完整per-mask metrics和raw predictions。

### 8.4 Diagnostics

- 在首个完整cycle、之后每5 epochs及最终best checkpoint记录 gradient norm、global/local contribution、gate/coverage/confidence、routing entropy/utilization、attention读取距离和depth/round activation。
- 记录branch pre/post RMS、gated-local/global norm ratio、local-state norm/variance/effective rank以及channel/head utilization。
- diagnostics复用正式forward/backward，不增加额外backward、不更新参数、不改变RNG。gradient norm只从正常训练backward的clip前统计缓存；最终best报告引用best epoch缓存值，best checkpoint回放只允许额外forward。

## 9. Pilot、对照与验收

### 9.1 实施门槛

1. 保留旧 prototype，新增独立 direct-512 `ProNORMST` real-data adapter，并生成 machine-readable config/manifest validator。
2. 在一张slice完成 anti-leakage、graph direction、depth/frontier、early-exit、CPU/GPU和AMP/non-AMP容差检查。
3. 运行1 epoch smoke，验证finite loss/gradient、全部应训练parameter groups获得finite nonzero gradient、full-BPTT、checkpoint round-trip和resume/predict-only。
4. 运行 `8/2/2`、一个initialization的pilot；通过9.2后才进入正式LODO。
5. validation完成选模并锁定candidate后才执行一次正式test。

### 9.2 Pilot

- 首次pilot只比较完整 Progressive NORMST 与 strict original-visible-only IDW，使用完全相同的 visible/query masks。
- IDW不得递归使用自身或NORMST prediction；one-shot和branch ablations只在pilot通过后运行。
- 必须无anti-leakage/geometry错误、OOM、NaN或Inf。
- ordinary和gap的median prediction/truth variance ratio均须在 `[1e-3,10]`；raw-`x` SmoothL1均不得比matched IDW差超过20%。
- pilot不要求击败IDW；通过只表示模型可进入正式比较。
- pilot validation health、gradient 与 final-loss BPTT gates 通过后、pilot test 前生成不可覆盖、自包含的 `candidate_lock.json`。其中只冻结 candidate signature、pilot identity、三个gate的可验证摘要，以及best checkpoint、contract manifest和split snapshot hash；不包含test metrics、test marker、run status、重复checks或完整gate JSON。正式 LODO 只验证该 lock 自身，不依赖原 pilot 绝对路径或原目录继续存在。

### 9.3 Formal matrix 与 control definitions

- 必跑：strict IDW、matched one-shot、progressive full、local-only和global-only。
- Round evidence只训练 `T=4` full模型，在同一best checkpoint和相同masks上导出round `1/2/4` early exits。
- 不为round evidence独立训练 progressive `T=1/2`；matched one-shot是唯一独立训练的 `T=1` control。
- matched one-shot使用相同 direct-512、preprocessing、宽度、decoder、loss和训练预算。
- local-only输入 `[0_global || gated_local]`；global-only输入 `[global || 0_local]`，均保持完整concat decoder宽度并独立训练。
- strict IDW使用full-resolution pixel距离、6个最近original-visible spots和 `distance^-2` 归一化权重；ties按canonical node order。
- 所有神经模型复用 folds、masks、init seeds、数据顺序、最大训练步数、validation criterion和early-stopping规则；各方法按同一规则独立选best并早停，实际停止步数可以不同。
- legacy one-shot、recursive IDW、`gamma=1`、dynamic global和额外routing controls不属于必跑矩阵，若运行必须单独标记。

### 9.4 Formal acceptance

- gap：full相对IDW和matched one-shot的overall raw-`x` SmoothL1 gain均为正，且两项比较都至少在2/3 held-out-donor folds方向一致。
- ordinary：full相对IDW和matched one-shot的overall raw-`x` SmoothL1劣化均不得超过1%。
- Depth1 prediction在round `1/2/4`间不变，Depth2在round `2/4`间不变；违反即实现错误。
- final loss必须对早期round共享local参数产生finite nonzero gradient。
- Depth2的 `round2-round1`、Depth3/4的 `round4-round2` raw-`x` SmoothL1 gain均须overall为正，且分别至少在2/3 folds方向一致。
- 任一主要条件失败则不接受主要假设；正式test后不得据此修改合同或调参，只能报告失败或启动新的实验版本。

### 9.5 Uncertainty 与报告

- 报告每个 `fold x init` paired delta。
- 先在fold内汇总3个init，再报告3个 donor-fold effects的mean、sample SD、range和逐fold值。
- 不计算显著性p-value，不把9个runs表述为9个donors。
