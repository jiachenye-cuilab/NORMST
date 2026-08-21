# ProNORMST loss-alignment audit

状态：执行前冻结
证据层级：validation-only diagnostic
模型结构变更：否
训练：否
测试集使用：否

## 目标

Round13 slice-context audit显示train-bias/context-affine可小幅改善部分raw-x MAE或SmoothL1，却在3/3 folds恶化合同weighted-z criterion。本诊断定位这种目标错位来自哪些基因、train detection-rate层和target-z符号，不重新选择或调优correction。

## 冻结输入与计算

- 复用Round9 `full / seed2027`的`lodo_d1/lodo_d2/lodo_d3` validation-selected `best.pt`。
- train correction严格复用已冻结slice-context audit：`context-audit-train` role，每slice/family 16 masks，只拟合train-bias和context-affine。
- evaluation严格复用`val` fixed banks；不构造test masks，不读取test指标或预测。
- 每mask逐gene重构合同criterion：target-z正元素使用fold-specific positive weight，非正元素权重为1；先按query元素加权归一，再gene等权。
- 同时计算逐gene raw-x SmoothL1和MAE，用于判断方向是否与合同criterion一致。

## 预声明分层与输出

Train detection-rate strata：

1. `undetected_train`: `d == 0`
2. `very_sparse_weight3`: `0 < d <= 0.1`
3. `sparse_weight1to3`: `0.1 < d < 0.5`
4. `common_weight1`: `0.5 <= d < 1`
5. `always_detected_train`: `d == 1`

每fold及三折等权汇总：

- weighted-z总delta、positive-target贡献delta、nonpositive-target贡献delta；
- raw-x SmoothL1/MAE delta；
- 各detection strata的上述delta；
- per-gene退化在top 1/5/10/25/50 genes的正退化质量占比；
- 每gene三折delta符号稳定性、fold间相关性及top退化gene；
- `positive_weight == 3`、`1 < weight < 3`和`weight == 1`基因组的净贡献。

## 完整性门

- 每fold逐gene重构baseline criterion与Round9锁定值绝对误差必须`<=1e-5`。
- 重构context/train-bias criterion与冻结slice-context audit相应值绝对误差必须`<=1e-6`。
- 任一完整性门失败则fail closed，不解释分层结果。

该诊断只提供下一步研究决策证据，不改变当前promotion gate或技术合同。
