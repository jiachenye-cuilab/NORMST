# ProNORMST Round13 slice-context feasibility audit

状态：执行前冻结
证据层级：validation-only diagnostic；不是训练轮次或candidate
测试集使用：否

## 目标

Round12证明专用local-to-gene residual head能够学习但未改善promotion criterion。Round9正式结果同时显示held-out donor差异远大于seed差异。本审计检验一个更窄的问题：仅从original-visible spots得到的slice context，是否包含可在train-role拟合、并泛化到held-out validation slices的系统性gene bias信息。

## 数据、checkpoint与信息边界

- 使用Round9 `full / seed2027`的`lodo_d1/lodo_d2/lodo_d3` validation-selected best checkpoints。
- 每fold只使用该fold train-role expression/targets拟合context correction，只在validation-role固定mask banks上判断效果。
- 不构造或评估test-role masks，不读取Round9 test metrics/predictions，不以test选择任何结论。
- context为每个mask的original-visible spots在fold-specific z-space中的per-gene算术均值；不读取query truth、absolute XY或test统计。
- train diagnostic masks使用独立role `context-audit-train`、每slice/family 16个固定mask；validation严格复用v9合同的role `val` fixed banks。

## 三条比较路径

对每个fold，将每个train mask的query mean residual定义为：

```text
r_g = mean_query(target_z_g - prediction_z_g)
c_g = mean_original_visible(z_g)
```

只在train-role masks上逐gene拟合：

1. baseline：不修正。
2. train-bias：`delta_g = mean_train(r_g)`。
3. context-affine：`delta_g = b_g + a_g * (c_g - mean_train(c_g))`，其中`a_g`为train-only最小二乘；context方差为零时`a_g=0`。

同一mask的`delta_g`加到每个query prediction-z。该审计是FiLM可行性的保守代理：只允许slice-level gene shift，不增加query truth或validation拟合。

## 指标与聚合

- 每mask计算合同weighted-z SmoothL1 criterion，以及raw-x SmoothL1、MAE、RMSE、gene Pearson、spot Pearson和variance ratio。
- masks在slice内等权、slices在family内等权，ordinary/gap综合criterion按1:1。
- 首先要求baseline validation replay与锁定best criterion绝对差`<=1e-5`，否则fail closed。
- 每fold报告baseline、train-bias、context-affine及paired gain：`criterion_baseline - criterion_method`。

## 预声明判断

只有同时满足以下条件，审计才支持创建Round13 FiLM结构合同：

1. context-affine overall validation criterion gain相对baseline大于0。
2. context-affine相对train-bias的overall gain大于0，证明收益来自context而非单纯train residual bias。
3. context-affine相对baseline至少在2/3 folds为正。
4. ordinary和gap综合结果均不恶化超过`1e-5`。
5. 四项ordinary/gap gene/spot Pearson相对baseline下降均不超过`0.001`。

不满足则不创建FiLM训练轮次；保存结果并重新评估研究方向。
