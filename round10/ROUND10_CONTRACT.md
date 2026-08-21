# ProNORMST Round10 pilot contract

状态：pilot启动前冻结
Round identity：`pro-v2-round-010`
Human contract：`pro-normst-human-v10`
Numerical implementation：`pro-normst-numerical-v10`

## 1. 假设与证据

Round9 validation-only variance audit表明：输出标准差约为truth的`0.27x`，但prediction-on-truth slope仅约`0.064--0.073`；oracle方差恢复会令RMSE恶化约`25--27%`。因此不直接匹配方差。

Round10的唯一假设是：在原weighted SmoothL1之外加入小权重、gene-wise Pearson alignment loss，可以提高预测与真实空间变化方向的一致性，同时由原loss保持幅度、RMSE和MAE稳定。

## 2. 继承与隔离

- 除本文件明确修改的训练loss外，完整继承`Pro_contract.md`的Round8模型语义与Round9 batch执行语义。
- 不修改`Pro_contract.md`及Round9源码/产物。Round10通过独立`round10`模块显式启用，默认训练入口保持v9。
- 数据、512-gene panel、preprocessing、split manifest、mask identities、model/forward、variant definitions、optimizer、learning-rate schedule、epoch预算、early stopping、AMP、validation criterion、scientific metrics、IDW和test lifecycle均不变。
- 所有输出写入新的`save/pro_normst/pro-v2-round-010/`目录；任何已存在目录均拒绝覆盖。

## 3. Round10训练loss

对batch内每个slice `i` 独立计算。设有效query集合为`Q_i`，基因为`g`，prediction与detached target为`p`和`t`。

1. `B_i`：v9的gene-equal、target-positive-weighted SmoothL1，`beta=1`。
2. 在`Q_i`内分别中心化`p`和`t`，计算每个gene的centered energy `E_p`、`E_t`和cross energy `C`。
3. 仅`E_t > 1e-6`的gene为defined。
4. `rho = clamp(C / sqrt(max(E_p, 1e-6) * E_t), -1, 1)`。
5. `P_i = mean_g(1-rho)`，padding query不参与任何统计。
6. 最终每slice loss：`L_i = B_i + 0.01 * P_i`。
7. optimizer step仍对slice等权：`sum_i(L_i) / number_of_train_slices`。

计算固定为float32；target detach；gene scale不进入该项，因为Pearson对每基因正比例缩放不变。validation criterion仍只使用原weighted z-SmoothL1，Pearson辅助项不参与checkpoint selection。

## 4. 实施与smoke门

- 原66项CPU/CUDA合同测试必须保持通过。
- Round10新增测试必须覆盖：已知相关性、padding不变性、正affine不变性、常量prediction的finite/nonzero gradient、batch item独立性、loss权重与contract manifest、opt-in隔离。
- 1-epoch real-data smoke必须无OOM/NaN/Inf，全部requires-grad参数累计finite/nonzero，final-loss对适用local round states的BPTT gate通过，checkpoint/resume prediction round-trip通过。
- smoke不得生成candidate lock或test结果，不作为性能证据。

## 5. 冻结pilot

- Manifest：`pre-train/manifests/random_pair_8_2_2_seed2027_server_absolute.json`。
- Variant/seed：`full` / `2027`。
- 训练预算：50 epochs，原early stopping与固定validation banks。
- Round reason：`gene-wise Pearson alignment should improve spatial direction before variance amplitude`。
- 运行中不改变weight、阈值、seed、mask、预算或其他超参数，不追加同round trial。
- checkpoint仅按原validation weighted-z SmoothL1 criterion选择。
- pilot test即使由冻结lifecycle在lock后生成，也不得用于本轮选择、解释或后续调参。

## 6. 预声明promotion gate

Round9同manifest/seed baseline：

- criterion：`0.2232179318089038`
- ordinary gene/spot Pearson：`0.2517102840356529 / 0.4523640898987651`
- gap gene/spot Pearson：`0.24396102223545313 / 0.4538486395031214`
- ordinary RMSE/MAE：`1.5880189426243305 / 1.379087194800377`
- gap RMSE/MAE：`1.5861568786203861 / 1.3827247470617294`

Round10只有同时满足下列条件才可进入formal LODO：

1. 原pilot health、gradient和final-loss BPTT gates全部通过。
2. criterion达到既有正式候选线`<=0.22244`。
3. ordinary与gap gene Pearson均严格高于Round9，且二者平均绝对提升至少`0.001`，以支持主要机制假设。
4. 四项gene/spot Pearson任一项相对Round9下降不得超过`0.001`。
5. ordinary/gap的raw-x RMSE与MAE任一项相对恶化不得超过`0.25%`。
6. variance ratio只执行既有`[1e-3,10]`健康门，不要求提高，避免把噪声放大误认为成功。

任一项失败则Round10不进入formal，不覆盖、不补参、不以test结果挽救；保存诊断后以新round identity提出下一假设。
