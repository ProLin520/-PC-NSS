# PC-NSS Teacher 尺度置信只读诊断设计

日期：2026-08-10
状态：待用户书面审核
前置结果：Task 13 评价链审计、Task 14 近间隔机制诊断

## 1. 目标

在不训练、不加载 PC-NSS checkpoint、不执行神经模型前向、不访问 development 或
locked test 的前提下，解释近间隔样本的学生尺度分布为何接近均匀：

1. 固定尺度 teacher 本身是否缺乏区分度；
2. `tau_scale=0.10` 是否把有用的 score 差异压得过平；
3. teacher 主导尺度是否与四个固定 FBSS 中样本级最佳尺度具有足够一致性；
4. 若 teacher 有效但学生仍均匀，问题是否更可能位于尺度蒸馏或优化路径。

本诊断只形成可证伪机制结论。它不改善性能，不自动授权训练，也不把 validation
关联表述为因果证明。

## 2. 方案与边界

采用 CPU teacher-only 方案：确定性重建 Task 14 的 1270 个 `[2,4)` validation
样本，复用训练代码中的 `build_scale_teacher` 计算固定 FBSS teacher；学生分布直接
读取现有 schema-complete 诊断 CSV 的 `p_L4...p_L7`，不重新运行 PC-NSS。

冻结边界：

- 只读取已认证的 `audit_v4/validation_report`、Task 14 schema-complete 诊断输出和
  validation 确定性生成器；
- 设备固定为 CPU，batch size 固定为 128；
- 当前温度固定为 `0.10`，只读反事实温度固定为 `0.05`；
- 不加载 `best.pt`，不实例化 `MultiScalePCNSS`，不计算梯度；
- 不训练、不更新 checkpoint、不修改模型、损失、teacher 或超参数；
- 不运行完整 5000 样本 evaluator，不访问 development/locked test；
- 不覆盖 Task 13、Task 14 或任何历史输出；新输出目录默认拒绝覆盖并由 Git 忽略；
- 不提交 outputs、预测、权重、checkpoint 或生成数据。

## 3. 输入身份与完整性

启动正式诊断前必须验证：

- Task 13 报告为 validation schema v2，evaluator code SHA、四个来源文件 SHA 和
  checkpoint SHA 与 Task 14 manifest 一致；
- `pcnss_root_music` 与 `fbss_root_music_L4...L7` 各有 5000 个唯一
  `sample_seed`；五种算法的 seed 集合与真值、rho、SNR、snapshot、separation
  完全一致；
- Task 14 schema-complete CSV 恰有 1270 个唯一 seed，全部位于 `[2,4)`，且与
  audit_v4 的 PC-NSS/L7 近间隔集合一致；
- validation 索引由 `sample_seed - validation_split_seed` 恢复，位于
  `[0,4999]`；重建样本的角度和场景元数据与权威预测逐项一致；
- 学生 `p_L4...p_L7` 均有限、非负，行和在 `1e-6` 绝对容差内等于 1。

任何重复、缺失、非有限、集合差异或元数据不一致都显式失败，不静默丢样本。

## 4. 数据流

```text
audit_v4 predictions + Task 14 schema-complete CSV
  -> SHA / 5000×5 集合 / 1270 near seed 校验
  -> 确定性重建相同 validation 样本
  -> CPU、batch=128 构造 L4...L7 FBSS covariance
  -> build_scale_teacher(tau_scale=0.10)
  -> 同一 raw scores 计算 tau=0.05 反事实概率
  -> 连接学生 p_L、PC-NSS cohort 与固定尺度样本 RMSPE
  -> 样本级指标、固定分层、机制判定
  -> 独立 schema v1 报告
```

## 5. 样本级指标

每个样本记录：

- 四尺度 raw teacher score；
- `tau=0.10` 与 `tau=0.05` 的四尺度概率、归一化熵、最大概率和主导尺度；
- raw score 的 top1-top2 margin 及 `margin / 0.10`；
- 学生 `p_L4...p_L7`、归一化熵和主导尺度；
- `KL(teacher_0.10 || student)` 与 Jensen-Shannon divergence，数值下限固定
  `epsilon=1e-8`；
- 四个固定 FBSS 的权威 `sample_rmspe_deg`；
- 样本级 oracle 最佳尺度集合：与最小 RMSPE 相差不超过 `1e-6°` 的所有尺度；
- teacher top1 是否落入 oracle 集合；
- teacher 选择 regret：teacher 主导尺度 RMSPE 减去样本最小固定尺度 RMSPE；
- PC-NSS threshold cohort、rho、SNR、snapshot 和 separation。

温度变化不改变 raw score 与 top1 尺度；反事实只回答“固定降温能否显著提高置信”，
不等于训练效果预测。

## 6. 分层与汇总

分别按以下维度汇总，每个维度计数之和必须为 1270：

- rho：`0.8`、`0.9`、`0.99`、`1.0`；
- SNR：`[-5,0)`、`[0,5)`、`[5,10]`；
- snapshot：`8`、`20`、`50`；
- Task 14 的七个互斥 PC-NSS threshold cohort。

对 entropy、max probability、margin、KL、JS 和 regret 固定报告 count、mean、median、
p05、p95、min、max；对 teacher/oracle agreement 报告 count/rate；对主导尺度报告
L4...L7 完整计数。空组保留为显式零计数，不伪造统计量。

## 7. 冻结科研判定规则

### 7.1 允许后续考虑 `tau_scale: 0.10 -> 0.05`

必须同时满足：

1. `tau=0.10` teacher 总体中位归一化熵不低于 `0.90`；
2. `tau=0.05` 相对 `0.10` 的总体中位熵下降至少 `0.05`，且总体中位最大概率
   上升至少 `0.05`；
3. teacher top1 与样本 oracle 最佳尺度集合的一致率不低于 `0.40`；
4. teacher 选择 regret 的中位数不超过 `1.0°`；
5. rho、SNR、snapshot 三个维度中至少两个维度支持同一方向。一个维度“支持”定义
   为其中至少两个非空 bin 同时满足 entropy 中位数不低于 `0.90`、降温后的 entropy
   中位下降至少 `0.05`；
6. 全部完整性、数值和工程验证门通过。

达到这些条件只允许编写一份新的单因素训练预注册，不直接启动训练。唯一候选改变量
为 `tau_scale=0.10 -> 0.05`；不得同时增加熵损失、改变蒸馏权重、残差上限、网络结构
或正式 1°/50% 分辨定义。

### 7.2 不允许降低 teacher 温度的情况

- teacher entropy 已低于 `0.90` 而学生 entropy 仍高：优先提出“学生未学到 teacher”
  假设，后续只诊断蒸馏/优化；
- agreement 低于 `0.40` 或 regret 中位数大于 `1.0°`：teacher 可靠性不足，禁止把它
  进一步变尖；
- 降温后的 entropy 或最大概率变化未达到固定幅度：`0.05` 反事实不足以改变置信；
- 场景分层方向不一致：只记录异质性，不预注册训练。

## 8. 输出 schema

新目录固定写六个文件，schema version 为 1：

1. `diagnostic_config.json`；
2. `source_manifest.json`；
3. `teacher_sample_diagnostics.csv`；
4. `teacher_summary.json`；
5. `teacher_stratified_summary.csv`；
6. `decision.json`。

manifest 必须记录诊断 code SHA、audit 与 Task 14 输入 SHA、split seed、样本数、CPU、
batch 128、两个固定温度、`no_model_forward=true` 和 `training_performed=false`。
JSON 禁止 NaN/Infinity；CSV 必须能独立复算所有汇总与 decision。输出目录存在时抛出
`FileExistsError`。

## 9. 安全入口与测试

新增独立入口 `scripts/diagnose_pcnss_teacher_confidence.py`：默认
`stage=dry_run`、`dry_run=true`、`device=cpu`、`batch_size=128`、拒绝覆盖。正式 stage
只允许 `diagnose_validation_teacher`；非 CPU、非 128、development/locked、未知配置键
和已有输出都显式拒绝。

按 RED -> GREEN -> 重构覆盖：

- 5000×5 集合、1270 near seed、重复/缺失/元数据不一致；
- 概率、entropy、margin、KL/JS、oracle tie set、agreement 和 regret 的数值边界；
- `tau=0.10/0.05` 冻结与判定门边界；
- 四类分层计数完整性、空组和非有限值；
- schema、SHA、拒绝覆盖、CSV 独立复算；
- 默认 dry-run 不读取正式数据、不创建输出；
- 4 样本 CPU smoke 只使用临时 train 样本与合成权威标签；
- 目标 unittest、完整 unittest、`compileall`、默认 dry-run 和 4 样本 smoke。

获授权的正式 teacher 诊断只运行一次 1270 个冻结 validation 样本。不运行训练、模型
前向、development、locked test 或完整 5000 evaluator。

## 10. 代码合并标准

科研结果与工程合并分开判定。无论第 7 节是否支持降温，只要满足以下工程门，评价与
诊断基础设施即可合并：

- 正式输出通过 seed、SHA、计数、finite 和 CSV 独立复算审计；
- 目标/完整 unittest、`compileall`、两个默认 dry-run 和 4 样本 smoke 全通过；
- 最终 diff 审查无 Critical/Important；
- Git 不包含 outputs、checkpoint、权重、生成数据或未审查配置；
- 合并到 D 盘 `master` 后重新运行同一验证并通过，再非强制推送 `origin/master`。

D 盘 `master` 的提交 `d83e347` 保留在历史中，但其参数无参运行默认值为正式
`evaluate_validation + CUDA + audit_v3`，不满足项目安全入口规范。合并冲突解决时，
最终 `scripts/run_multiscale_pcnss.py` 必须采用最新分支的安全默认值：
`stage=dry_run`、`dry_run=true`、`device=cpu`、空 checkpoint、相对 output root；不得把
用户绝对路径或旧 audit_v3 设为合并后的默认运行配置。

合并成功只表示代码与诊断链达到工程标准，不表示模型性能、teacher 假设或论文门槛
已经通过。
