# ProNORMST Round12 pilot contract

状态：pilot启动前冻结
Round identity：`pro-v2-round-012`
Human contract：`pro-normst-human-v12`
Numerical implementation：`pro-normst-numerical-v12`

## 1. 假设与单因素结构修改

Round9正式对照与validation-only机制诊断共同表明：local分支有真实但较小、donor-dependent的边际价值；`gated_local/global norm ratio`约为`0.395`，因此local幅度不是主要瓶颈。降低global/local表征重叠后，共享concat decoder仍主要依赖global。Round10/11的Pearson loss只产生微小变化，未改善promotion criterion。

Round12的唯一研究假设是：共享decoder构成local信息到gene prediction的输出瓶颈；在现有预测上增加一个不读取global的local-to-gene residual head，可以让local progressive state形成直接、可靠性/深度感知的gene修正，同时保留Round9主路径。

公式：

```text
base = Round9SharedDecoder([global_normalized, gated_local])
delta = LocalGeneResidualHead(gated_local, activation_round,
                              detach(coverage), detach(confidence))
prediction = base + active_query * delta
```

## 2. 新head的冻结定义

- 输入维度：`256 + 8 + 1 + 1 = 266`。
- `activation_round in {1,2,3,4}`使用`Embedding(4,8)`，初始化`Normal(0,0.02)`。
- MLP：`Linear(266,256) -> GELU -> Linear(256,512)`。
- 最终projection的weight和bias均为zero-init；step 0必须与同seed Round9 prediction逐值相等。
- 输入只含已固定gate后的`gated_local`、round embedding及detached coverage/confidence；不读取global、absolute XY、query truth或test信息。
- 只对active query执行；inactive、unreachable、Depth `>=5`及padding residual严格为零。
- 不再次乘coverage/confidence，避免将Round9固定gate平方；可靠性通过`gated_local`幅度和两个detached scalar显式输入。
- 按首次`activation_round`分组调用同一head，维持FP16 early-exit row shape与round invariance。
- 模块注册在全部Round9模块之后；继承参数的seeded initialization必须逐值不变。
- full、matched one-shot和local-only训练该head；global-only冻结并跳过该head。

## 3. 继承、不变项与隔离

- 除第1--2节的新head外，模型、数据、Shared-512 panel、preprocessing、mask、四轮同步传播、gate、shared decoder、weighted SmoothL1、optimizer、learning-rate schedule、AMP、epoch预算、early stopping、validation criterion、scientific metrics及test lifecycle完整继承`Pro_contract.md`的v9语义。
- 不继承Round10/11 Pearson loss或schedule。
- Round12通过独立`round12`模块显式启用；默认v9入口、Round9--11源码入口及全部既有产物不被覆盖。
- 输出只写入新的`save/pro_normst/pro-v2-round-012/`目录；任何已有run目录拒绝覆盖。
- 不复用Round9--11 checkpoint、candidate lock或run checkpoint lock。

## 4. 等价runtime优化

Round12允许使用已验证不改变评估结果的两项runtime优化：

1. 完全相同的query/gene selector只计算一次scientific metrics；必须保持完整评估JSON字节级一致。
2. strict IDW先用`argpartition`确定k-th边界，再对全部边界内候选执行原distance/canonical-index排序；输出必须与原full-sort实现逐元素相等。
3. matched runs可使用显式`--idw-cache-dir`共享内容寻址IDW prediction。key必须覆盖expression、full XY、visible/query indices、`k=6`与`power=2`；缓存损坏、identity不符或非finite时fail closed。

这些优化不构成研究因素，不得改变model prediction、IDW、metric、criterion或promotion判定。缓存路径不进入数值兼容hash。

## 5. 实施与smoke门

- 全部基础CPU/CUDA合同测试、Round10/11隔离测试及Round12新增测试必须通过。
- 新增测试必须覆盖：继承参数逐值相等、step-0 CPU/CUDA/AMP prediction等价、zero-init、variant trainable scope、inactive residual为零、metadata detach、head gradient、checkpoint round-trip及round `1/2/4` invariance。
- 1-epoch real-data smoke必须无OOM/NaN/Inf；所有requires-grad参数累计finite/nonzero；final loss对4个适用local rounds的BPTT gate通过；checkpoint/resume prediction通过原容差。
- smoke不生成candidate lock或test结果，不作为性能证据。

## 6. 冻结pilot

- Manifest：`pre-train/manifests/random_pair_8_2_2_seed2027_server_absolute.json`。
- Variant/seed：`full` / `2027`。
- Round reason：`dedicated local gene residual should bypass shared decoder underuse`。
- 训练预算、early stopping、mask banks、data order和checkpoint criterion全部保持v9。
- 同round不追加head宽度、embedding、输入、gate或loss变体。
- candidate lock仍只表示内置健康门和数值签名通过，不替代第7节外部performance promotion。
- pilot test不得进入promotion、选择或本轮调参。

## 7. 预声明validation-only promotion gate

Round9同manifest/seed baseline：

- criterion：`0.2232179318089038`
- ordinary gene/spot Pearson：`0.2517102840356529 / 0.4523640898987651`
- gap gene/spot Pearson：`0.24396102223545313 / 0.4538486395031214`
- ordinary RMSE/MAE：`1.5880189426243305 / 1.379087194800377`
- gap RMSE/MAE：`1.5861568786203861 / 1.3827247470617294`

只有同时满足以下条件才可进入formal LODO：

1. pilot、gradient、final-loss BPTT及新增head health全部通过。
2. criterion达到正式候选线`<=0.22244`。
3. ordinary/gap的四项gene/spot Pearson任一项相对Round9下降不得超过`0.001`。
4. ordinary/gap raw-x RMSE与MAE任一项相对恶化不得超过`0.25%`。
5. ordinary/gap median variance ratio均在`[1e-3,10]`。
6. locked-best固定validation gap mask上，local gene residual必须finite且非零；全部head trainable参数累计获得finite/nonzero gradient。

Round10/11的“gene Pearson平均提升至少0.001”是Pearson-loss机制门，不适用于Round12。任一条件失败则冻结Round12、不运行formal matrix、不以test结果挽救。

## 8. 时间记录与比较

- 保存setup、每epoch train/validation/total、cold/hit IDW cache状态、test及总壁钟。
- 分别比较Round9 pilot的首epoch和warm epochs：Round9首epoch约`122.48s`，warm epoch约`77--78s`，其中train约`38.5s`、validation约`37--38s`。
- runtime收益不构成performance promotion；必须与第7节判定分开报告。

## 9. 回退

Round12为独立opt-in模块和目录。任何结构、工程、等价或promotion gate失败时，停止Round12并保留现场；默认v9入口与Round9正式模型不变，不需要删除、reset或覆盖任何旧产物。
