# ProNORMST Round11 pilot contract

状态：pilot启动前冻结
Round identity：`pro-v2-round-011`
Human contract：`pro-normst-human-v11`
Numerical implementation：`pro-normst-numerical-v11`

## 1. 单一假设

Round10固定`0.01` Pearson辅助项令ordinary/gap gene Pearson分别提高`0.00055963/0.00032213`，四项Pearson均上升、RMSE微降，但criterion比Round9差`0.00002762`、MAE轻微恶化。训练轨迹显示早期方向对齐有效，后期固定辅助项没有继续扩大Pearson收益。

Round11的唯一假设是：Pearson只作为warm-start，随后衰减到0，可以保留早期方向引导，并让后期完全回到原weighted SmoothL1以优化criterion、RMSE和MAE。

## 2. 继承与隔离

- 完整继承Round9的模型、forward、数据、panel、preprocessing、mask、IDW、optimizer、LR schedule、epoch预算、AMP、validation criterion、scientific metrics和test lifecycle。
- 完整继承Round10的gene-wise Pearson公式；只改变其epoch权重。
- 不修改Round9、Round10或默认v9入口。Round11通过独立`round11`模块显式启用。
- 输出仅写入`save/pro_normst/pro-v2-round-011/`，存在即拒绝覆盖。
- epoch权重由checkpoint中可恢复的zero-based epoch纯函数确定，不新增隐式状态。

## 3. 冻结loss schedule

每slice的base loss `B_i`与Round9完全相同；Pearson penalty `P_i`与Round10完全相同。`L_i = B_i + lambda(epoch) * P_i`。

| Human epoch | Pearson weight |
| ---: | ---: |
| 1--5 | 0.010 |
| 6 | 0.008 |
| 7 | 0.006 |
| 8 | 0.004 |
| 9 | 0.002 |
| 10及以后 | 0 |

validation和checkpoint selection始终只使用原weighted-z SmoothL1，不包含Pearson项。batch内仍逐slice独立计算后等权。

## 4. 实施与smoke门

- 原66项合同测试、Round10依赖测试和Round11新增测试必须通过。
- 新增测试必须覆盖精确epoch schedule、epoch上下文fail-closed、epoch 10后与v9 base loss逐值相同、上下文异常恢复、resume确定性和opt-in隔离。
- 1-epoch real-data smoke必须通过finite/nonzero gradient、4-round final-loss BPTT和checkpoint round-trip，无OOM/NaN/Inf。
- smoke不生成candidate lock/test，不作为性能证据。

## 5. Pilot

- Manifest：`pre-train/manifests/random_pair_8_2_2_seed2027_server_absolute.json`。
- Variant/seed：`full` / `2027`。
- 最多50 epochs，原patience 10、fixed validation banks和data order。
- Round reason：`decay Pearson warm-start to zero before SmoothL1-only refinement`。
- 运行中不改变schedule、seed、mask、预算或任何其他参数，不追加同round trial。
- test只允许按冻结lifecycle在candidate lock后执行一次，不得读取其指标用于本轮选择或解释。

## 6. 预声明promotion gate

正式性能基线仍是Round9同manifest/seed best checkpoint：criterion `0.2232179318089038`；ordinary gene/spot Pearson `0.2517102840356529/0.4523640898987651`；gap gene/spot Pearson `0.24396102223545313/0.4538486395031214`；ordinary RMSE/MAE `1.5880189426243305/1.379087194800377`；gap RMSE/MAE `1.5861568786203861/1.3827247470617294`。

只有全部满足才进入formal LODO：

1. pilot、gradient和final-loss BPTT health全部通过。
2. criterion `<=0.22244`。
3. ordinary/gap gene Pearson均严格高于Round9，二者平均绝对提升至少`0.001`。
4. 四项gene/spot Pearson任一项相对Round9下降不超过`0.001`。
5. ordinary/gap raw-x RMSE与MAE任一项相对恶化不超过`0.25%`。
6. variance ratio位于既有`[1e-3,10]`健康范围。

Round10固定权重结果只用于解释schedule效应，不替代Round9正式基线或上述gate。任一条件失败则不启动formal，冻结本轮并以新round提出下一假设。

## 7. 条件formal

若且仅若第6节全部通过，使用Round11 candidate lock启动与第9.3节相同的`4 variants x 3 folds x 3 seeds = 36` runs和strict IDW；所有run必须使用Round11入口和同一loss schedule，不得复用Round9/Round10 checkpoint或lock。
