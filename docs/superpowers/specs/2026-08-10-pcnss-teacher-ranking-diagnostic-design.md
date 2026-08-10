# PC-NSS Teacher Score 排序与组成项只读诊断设计

日期：2026-08-10
状态：用户已批准，Task 16 按本规格验证
前置结果：Task 15 teacher 尺度置信诊断

## 1. 一句话目标

把 teacher 当作给 L4、L5、L6、L7 排名的“老师”，只读检查它是否经常把角度误差更小的
尺度排在前面，并解释当前分数为什么几乎拉不开；本 Task 不修改 teacher，不训练模型。

## 2. 已知问题

Task 15 已确认：

- teacher `tau=0.10` 的中位归一化熵为 `0.999763`，接近四尺度均匀；
- 降到 `tau=0.05` 后中位熵只下降 `0.000707`；
- teacher top1 与样本级最佳固定尺度一致率为 `35.4331%`；
- teacher/student KL 中位数只有 `0.001021`，学生实际上已经较好复现这个平坦 teacher；
- 因此不允许预注册 `tau_scale: 0.10 -> 0.05`。

Task 16 不再问“概率够不够尖”，而是回答两个更基础的问题：

1. teacher 分数的尺度排序是否仍含有可利用的信息；
2. 低区分度来自原始 MUSIC denominator 本身，还是来自真角项与中点项相减后的抵消。

## 3. 冻结边界

- 只使用 Task 15 已认证的 1270 个 `[2,4)` validation 样本及其固定 FBSS L4–L7
  failure-aware `sample_rmspe_deg`；
- CPU、batch size 128；只重建相同样本的四尺度 FBSS covariance；
- 只调用固定物理计算和 `normalized_music_denominator`；
- 不加载 checkpoint，不实例化或运行 `MultiScalePCNSS`，不计算梯度；
- 不修改 `build_scale_teacher`、teacher score、温度、模型、损失或超参数；
- 不训练，不运行完整 5000 样本 evaluator，不访问 development/locked test；
- 正式 1270 样本排序诊断只运行一次；新目录拒绝覆盖；
- 不提交 outputs、预测、权重、checkpoint、生成数据或临时正式配置。

这是一项机制诊断，不是选择新公式的 validation 搜索。Task 16 只决定后续应该研究“分数
标定”“teacher 公式”还是“训练集角误差 teacher”，不在本 Task 中试验这些改法。

## 4. 输入身份与重建校验

正式诊断必须认证：

- Task 15 六文件报告完整，schema version 为 1；
- Task 15 manifest 的 `sample_count=1270`、`device=cpu`、`batch_size=128`、
  `no_model_forward=true`、`training_performed=false`；
- Task 15 记录的 audit 与 Task 14 输入 SHA 仍与当前只读源文件一致；
- `teacher_sample_diagnostics.csv` 恰有 1270 个唯一、升序 sample seed；
- 每行四个 teacher raw score、三组尺度概率和 L4–L7 RMSPE 均有限；概率非负且在
  `1e-6` 绝对容差内和为 1；
- validation seed 映射位于 `[0,4999]`，确定性重建的真角、rho、SNR、snapshot、
  separation 与 Task 15 样本 CSV 逐项一致；
- 重新计算的当前 teacher score 与 Task 15 CSV 在 `rtol=1e-6, atol=1e-7` 内一致。

任何重复、缺失、非有限、SHA、集合或元数据差异都显式失败，不静默丢样本。

## 5. Teacher Score 拆解

当前每个尺度的 teacher score 为：

```text
truth_mean = (q(theta_1) + q(theta_2)) / 2
score       = q(midpoint) - truth_mean
```

其中 `q(angle)` 是归一化 MUSIC denominator。理想情况下，真角处 denominator 较小，
中点处 denominator 较大，因此 score 越大代表该尺度越可能分开两个真角。

Task 16 对每个样本、每个尺度记录：

- `q_true_1`、`q_true_2`；
- `q_truth_mean`；
- `q_midpoint`；
- `negative_truth_mean = -q_truth_mean`；
- `current_score = q_midpoint - q_truth_mean`；
- 固定尺度 failure-aware `sample_rmspe_deg`。

同时记录四尺度内部的 range 和标准差：

- `q_midpoint` 动态范围；
- `negative_truth_mean` 动态范围；
- `current_score` 动态范围；
- `score_range / (midpoint_range + truth_range)` 抵消比。分母为零时写 `null` 并单独计数，
  不写 NaN/Infinity。

这些组成项只用于解释当前 score，不构造或比较新的 teacher 公式。

## 6. 排序指标

“oracle”定义为该样本 L4–L7 中 failure-aware RMSPE 最小的尺度集合；与最小值相差不超过
`1e-6°` 的尺度全部保留为 tie，不强行选一个。

每个样本分别比较以下三个排序信号与 `-sample_rmspe_deg`：

1. 当前 `score`；
2. `q_midpoint`；
3. `negative_truth_mean`。

三个信号统一按“数值降序、L 升序”形成确定性尺度顺序：top1 为第一项，top2 为前两项；
top1 命中指 top1 落入 oracle 集合，top2 覆盖指 top2 集合与 oracle 集合交集非空。该固定
tie-break 只用于 top1/top2 和混淆矩阵；相关系数与 pairwise 统计仍保留真实 tie。

固定报告：

- 四尺度 Spearman rho，使用平均秩处理 tie；全常数导致未定义时写 `null` 并计数；
- Kendall tau-b，显式处理 teacher tie 与 RMSPE tie；
- 六个尺度对的 concordant、discordant、teacher tie、oracle tie 数量；
- 全体 pairwise concordance rate，分母只包含 oracle 非 tie 的尺度对，teacher tie 计为
  未命中而不是删除；
- 各信号 top1 是否落入 oracle 集合；
- 各信号固定 top2 是否覆盖 oracle；精确信号 tie 数量单独报告；
- teacher top1 regret 的 count/mean/median/p05/p95/min/max；
- regret `>1°`、`>3°`、`>10°` 的数量和比例。

混淆矩阵使用 teacher top1 为行、oracle scale 为列。若 oracle 有多个 tied scales，该样本对
对应列平均分配 `1/|oracle_set|` 权重；另行报告 oracle tie 样本数，保证矩阵权重总和为
1270，不隐藏或复制样本。

## 7. 固定分层

对当前 score、`q_midpoint`、`negative_truth_mean` 的排序指标分别按以下维度汇总：

- rho：`0.8`、`0.9`、`0.99`、`1.0`；
- SNR：`[-5,0)`、`[0,5)`、`[5,10]`；
- snapshot：`8`、`20`、`50`；
- Task 14 七个互斥 threshold cohort。

每个维度的样本计数之和必须为 1270。空组保留零计数；未定义相关系数只影响该统计量的
`defined_count`，不能从样本总数、top1/top2、pairwise 或 regret 中删除。

## 8. 冻结机制判定

### 8.1 当前 teacher 只有“标定问题”

必须同时满足：

1. 当前 score 总体 pairwise concordance rate 不低于 `0.60`；
2. 当前 score 样本级 Kendall tau-b 中位数不低于 `0.20`；
3. teacher top1/oracle 一致率不低于 `0.40`；
4. teacher top2/oracle 覆盖率不低于 `0.70`；
5. teacher regret 中位数不高于 `1.0°`；
6. rho、SNR、snapshot 至少两个维度支持。一个维度支持定义为其中至少两个非空 bin
   同时满足 pairwise concordance 不低于 `0.55`、top2 覆盖率不低于 `0.65`；
7. Task 15 中位 `margin/tau < 0.10` 且全部工程完整性门通过。

若全部通过，Task 16 只允许建议下一步预注册“teacher score 无超参数标准化/标定”方向；
仍不在本 Task 修改或训练。

### 8.2 当前组合公式可能抵消有用信息

仅当第 8.1 节不通过，并且 `q_midpoint` 或 `negative_truth_mean` 至少一个组成项同时满足：

- 总体 pairwise concordance 不低于 `0.60`；
- 相对当前 score 的 pairwise concordance 提高至少 `0.05`；
- top2/oracle 覆盖率不低于 `0.70`；
- rho、SNR、snapshot 至少两个维度按第 8.1 节的方式支持，但 pairwise 与 top2 指标使用
  被判定的组成项自身结果，而不是当前 score 的结果；

才允许建议下一步在 train split 上设计单因素 teacher 公式诊断。Task 16 不在 validation 上
枚举 normalized contrast、log contrast 或组合权重。

### 8.3 当前物理 teacher 排序无效

若第 8.1、8.2 节均不通过，则结论冻结为：当前物理 teacher 不适合通过简单放大或公式微调
继续使用。下一步只允许设计 train-only 的 failure-aware 固定尺度角误差 teacher：直接依据
L4–L7 匹配角 RMSPE 排名，失败保留 `60°` 罚值，目标聚焦近间隔角度精度，不加入间隔撑开
奖励。该方案仍需单独设计、预注册和批准，Task 16 不实现它。

## 9. 输出 schema

新目录固定写八个 schema-v1 文件：

1. `diagnostic_config.json`；
2. `source_manifest.json`；
3. `teacher_ranking_sample_diagnostics.csv`；
4. `teacher_ranking_summary.json`；
5. `teacher_component_summary.json`；
6. `teacher_ranking_stratified_summary.csv`；
7. `teacher_oracle_confusion.csv`；
8. `decision.json`。

manifest 必须记录 Task 16 code SHA、Task 15 六文件 SHA、Task 15 上游输入 SHA、validation
split seed、样本数、CPU、batch 128、`no_model_forward=true`、
`teacher_modified=false`、`training_performed=false`。JSON 禁止 NaN/Infinity；样本 CSV 必须
足以独立复算全部 summary、strata、confusion 和 decision。输出目录存在时抛出
`FileExistsError`。

## 10. 安全入口与测试

新增独立入口 `scripts/diagnose_pcnss_teacher_ranking.py`，默认：

```text
stage=dry_run
dry_run=true
split=validation
device=cpu
batch_size=128
overwrite=false
```

正式 stage 只允许 `diagnose_validation_teacher_ranking`。直接调用和 JSON 配置路径都拒绝：

- 非 CPU、非 128；
- 非 validation、development、locked test 或 `allow_locked_test=true`；
- 非 1270 正式样本；
- 未知配置键、非冻结容差、已有输出、`overwrite=true`。

按 RED -> GREEN -> 重构覆盖：

- Task 15 schema/SHA/1270 seed/finite/概率与固定 RMSPE 身份认证；
- 确定性重建元数据与 Task 15 score 数值一致性；
- Spearman tie、全常数、Kendall tau-b、pairwise 分母和 tie 边界；
- oracle tie、top1、固定 top2、regret 阈值和加权混淆矩阵；
- component range、抵消比分母为零；
- 17 个固定分层、defined count、零计数组；
- 三种机制结论的所有门槛边界；
- 八文件 schema、finite、CSV 独立复算和拒绝覆盖；
- 默认 dry-run 不读正式数据、不创建输出；
- 4 样本 CPU smoke 只使用 train 样本和内存合成 RMSPE 标签；
- 目标 unittest、完整 unittest、`compileall`、主程序默认 dry-run、Task 16 默认 dry-run
  和 4 样本 smoke。

## 11. 完成与合并标准

- 正式排序诊断只运行一次 1270 个冻结 validation 样本；
- 独立脚本从样本 CSV 复算所有排序、分层、混淆和 decision；
- 目标/完整 unittest、`compileall`、两个默认 dry-run 和 4 样本 smoke 全通过；
- 最终 diff 审查无 Critical/Important；
- Git 不包含 outputs、checkpoint、权重、生成数据或用户运行配置；
- Task 16 分支推送 origin；达到工程门后才合并 D 盘 `master`，合并后重新验证并非强制
  推送 `origin/master`；
- D 盘主脚本无参默认仍保持 `dry_run=true`、CPU、空 checkpoint、相对 output root。

合并成功只表示排序诊断基础设施达到工程标准，不表示 teacher 改进、模型性能或论文门槛
已经通过。
