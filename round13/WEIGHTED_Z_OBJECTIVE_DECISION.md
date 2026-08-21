# ProNORMST metric objective decision

日期：2026-08-21
授权来源：用户明确确认

## 决定

- weighted-z SmoothL1继续作为唯一best-checkpoint与performance promotion主指标，保持`Pro_contract.md`当前数值语义不变。
- raw-x unweighted SmoothL1、MAE/RMSE与gene/spot Pearson作为科学指标护栏，不参与替代或重算主criterion。
- 后续round必须在执行前预声明raw-x与Pearson护栏阈值；不得观察validation/test结果后补设阈值。
- test仍不得用于选超参数、checkpoint或promotion。

## 直接后果

- Round13 context-affine虽然小幅改善raw-x MAE，但weighted-z退化，因此仍判定不promotion；不追溯改判。
- slice-level bias/context/FiLM分支终止，不继续围绕raw-x改善调参。
- 本决定不授权修改loss、模型结构或启动训练。若提出query-specific detection/expression等新结构，仍须先报告单因素假设、改动范围、兼容性、护栏、回退方案与独立round身份。
