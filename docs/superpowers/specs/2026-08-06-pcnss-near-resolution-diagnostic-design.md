# PC-NSS 近间隔精确分辨诊断设计规格

日期：2026-08-06
状态：设计已批准，等待书面规格审阅
前置结果：Task 13 schema v2 validation 审计

## 1. 目标

本 Task 对冻结的 seed2026、epoch 35 checkpoint 做一次性只读机理诊断，解释
PC-NSS 在 separation `[2,4)` 上为何总体 RMSPE 和大误差尾部显著优于固定
FBSS `L=7`，但严格成功分辨率仍低于 `L=7`。

诊断聚焦两个问题：

1. 未分辨样本距离“两角误差均不超过 `1°`”门槛有多远，是否集中在门槛附近；
2. 严格精度失败是否与尺度权重不确定、lag 残差饱和或结构投影改变量相关。

诊断完成后，只允许根据冻结结果预注册一个单因素训练实验。该实验必须提高
近间隔角度精度，不再强化已经通过率为 `99.685%` 的间隔撑开条件。

## 2. 范围与禁止事项

- 只读取 Task 13 的 `audit_v4/validation_report`、原 `best.pt` 和确定性
  validation 生成器。
- 只重算 `audit_v4` 中 PC-NSS separation 位于 `[2,4)` 的 1270 个样本。
- 固定神经推理 batch size 为 `128`，模型保持 `eval()`，不计算梯度。
- 不训练模型，不更新 checkpoint，不修改模型结构、损失或超参数。
- 不运行新的完整 5000 样本正式评价。
- 不访问 development 或 locked test，不生成其 manifest。
- 不覆盖或修改 `audit_v4`；诊断写入新的、拒绝覆盖且被 Git 忽略的输出目录。
- 不把输出、checkpoint、权重或生成数据提交到 Git。
- 诊断关联只用于形成可证伪假设，不表述为因果证明。

## 3. 方案选择

采用独立诊断 runner，而不扩展正式 evaluator schema。

独立 runner 复用模型、确定性数据生成、结构投影和现有评价定义，但输出独立的
诊断 schema。这样既能读取模型内部量，又不会要求再次运行全部传统基线或改变
已经稳定的 schema v2 正式报告。

一次性临时脚本不采用，因为它不能形成可测试、可复现和可审计的研究记录。

## 4. 输入身份与样本连接

诊断启动前必须验证：

- `run_config.json` 的 stage 为 `evaluate_validation`、split 为 `validation`；
- `summary.json` 的 `report_schema_version` 为 `2`；
- `source_manifest.json` 的 evaluator code SHA 为
  `129c3ba3b9fc1919451eef5c67376f04b4b24680`；
- checkpoint SHA256 与 `source_manifest.json` 完全一致；
- PC-NSS 和 `fbss_root_music_L7` 各有 5000 个唯一 `sample_seed`；
- 两种算法的 `sample_seed` 集合、真值、rho、SNR、snapshot 和 separation 完全一致；
- PC-NSS `[2,4)` 子集恰有 1270 个唯一样本。

`sample_seed` 是唯一连接键。validation 样本索引由
`sample_seed - validation_split_seed` 恢复，必须位于 `[0,4999]`，且重新生成的
样本元数据必须与 `predictions.csv` 一致。任何重复、缺失、不一致或越界都显式
报错，禁止静默丢样本。

`audit_v4/predictions.csv` 中的匹配角误差、resolved 和间隔条件是本诊断的权威
结果标签。内部量重算不重新定义或替换正式评价结果。

## 5. 数据流

```text
audit_v4 predictions + source manifest + frozen best.pt
  -> 身份、SHA、样本集合和元数据校验
  -> 筛选 PC-NSS [2,4) 的 1270 个 sample_seed
  -> 确定性重建相同 validation 样本
  -> batch=128 冻结前向
  -> scale weights / lag residual / candidate covariance / projected covariance
  -> 评价期 Dykstra 投影
  -> 按 sample_seed 连接 audit_v4 权威误差标签
  -> 样本级 CSV、阈值汇总、分层汇总和机制汇总
  -> 只读结论
  -> 预注册唯一单因素训练实验
```

## 6. 一度阈值附近的误差诊断

对 PC-NSS 与固定 FBSS `L=7` 分别定义：

```text
max_angle_error_deg = max(absolute_error_1_deg, absolute_error_2_deg)
threshold_margin_deg = max_angle_error_deg - 1.0
```

报告以下累计通过率：

```text
max error <= 0.50°, 0.75°, 1.00°, 1.25°, 1.50°, 2.00°
```

PC-NSS 样本按互斥顺序分组：

1. `estimation_failure`：正式评价标记失败；
2. `separation_failure`：估计成功，但估计间隔小于真实间隔的 50%；
3. `resolved`：两角误差均不超过 `1°` 且间隔条件通过；
4. `near_miss_1_1p25`：间隔条件通过，最大角误差位于 `(1.00,1.25]°`；
5. `near_miss_1p25_1p5`：位于 `(1.25,1.50]°`；
6. `near_miss_1p5_2`：位于 `(1.50,2.00]°`；
7. `far_miss_gt_2`：大于 `2.00°`。

阈值比较还报告 PC-NSS 相对 `L=7` 的样本级 win/tie/loss。tie 继续使用正式评价
链的 `1e-6°` RMSPE 容差，不另建更宽松定义。

## 7. 分层定义

所有阈值指标和 PC-NSS 机制指标按以下维度分别汇总：

- rho：`0.8`、`0.9`、`0.99`、`1.0`；
- SNR：`[-5,0)`、`[0,5)`、`[5,10]`；
- snapshot：`8`、`20`、`50`；
- PC-NSS 阈值组：第 6 节的七个互斥组。

每个分层表必须包含样本数。每一维各分组样本数之和必须等于 1270；连续值越界
或离散值不在冻结集合中时显式失败。

## 8. 尺度权重诊断

样本级四尺度分布沿用训练损失中的可靠性聚合：

```text
reliability = effective_counts * valid_mask
mass_L = sum_lag(scale_weight[L,lag] * reliability[L,lag])
p_L = mass_L / sum_L(mass_L)
```

报告：

- `p_L4`、`p_L5`、`p_L6`、`p_L7`；
- 最大尺度权重及主导尺度；
- 归一化样本级熵
  `H_scale = -sum_L(p_L log p_L) / log(4)`，范围 `[0,1]`；
- 每个 lag、每个有效尺度的平均权重；
- 每 lag 归一化熵。若该 lag 有 `m>=2` 个有效尺度，除以 `log(m)`；只有一个或
  没有有效尺度时记为空值，不把确定性 lag 误报为低熵选择。

重点比较 `resolved`、三个 near-miss 组和 `far_miss_gt_2`，并观察差异是否在至少
两个 rho/SNR/snapshot 分层中方向一致。

## 9. 残差饱和诊断

模型残差硬上限为冻结的 `residual_fraction=0.10`。对每样本、每 lag 计算：

```text
residual_magnitude = sqrt(real^2 + imag^2)
residual_ratio = residual_magnitude / 0.10
saturated = residual_ratio >= 0.95
```

报告每样本残差幅值的 p50、p95、最大值、饱和 lag 数与饱和 lag 比例，并按 lag
报告均值、p95 和饱和率。`0.095` 是本诊断唯一饱和阈值，不根据结果调整。

## 10. 结构投影变化

定义：

- `C0`：模型输出的 `candidate_covariance`；
- `C1`：模型固定迭代可微投影后的 `covariance`；
- `C2`：评价期 `dykstra_structured_projection(C1)` 的最终矩阵。

对每个样本计算：

```text
train_projection_change = ||C1-C0||F / max(||C0||F, 1e-12)
eval_projection_change  = ||C2-C1||F / max(||C1||F, 1e-12)
total_projection_change = ||C2-C0||F / max(||C0||F, 1e-12)
```

同时报告 Dykstra 收敛标志、迭代数、最终 Hermitian/Toeplitz/trace 误差和最小
特征值。任一 Dykstra 不收敛都保留该样本并显式记录，不删除或回填。

## 11. 结构化输出

独立诊断报告 schema version 固定为 `1`，包含：

```text
diagnostic_config.json
source_manifest.json
near_sample_diagnostics.csv
threshold_summary.json
stratified_summary.csv
mechanism_summary.json
```

`near_sample_diagnostics.csv` 每个 PC-NSS 样本一行，包含连接键、场景字段、正式
误差标签、阈值组、四尺度分布、熵、残差统计和三段投影变化。聚合 JSON/CSV 必须
能从该样本级 CSV 独立重算。

报告记录诊断代码 SHA、checkpoint SHA、来源报告路径与 SHA256、split seed、样本数、
batch size、device、残差上限和饱和阈值。输出目录已存在时默认抛出
`FileExistsError`。

## 12. 诊断判读与单因素选择

判读优先级固定如下：

1. 若 near-miss 相对 resolved 显示更高残差饱和，并且更大的投影变化与残差幅值
   同向，优先把“残差修正被边界或投影限制”作为候选机制；
2. 否则，若 near-miss 显示更高归一化尺度熵或持续接近均匀的尺度分布，且该方向
   在至少两个场景分层中复现，优先把“尺度置信不足”作为候选机制；
3. 否则，若失败主要集中在 `(1.00,1.50]°` 且间隔条件仍通过，选择“局部角度精度
   代理不足”作为候选机制；
4. 若证据相互矛盾或主要为 `>2°` 远失配，本轮只记录诊断，不预注册训练，先提出
   新的机理假设。

以上顺序只确定下一项要验证的假设，不凭 validation 诊断直接声称因果成立。

## 13. 单因素训练实验预注册要求

诊断完成后，新建独立预注册文档，且必须在任何新训练启动前提交。文档固定包含：

- 诊断证据及其来源文件 SHA；
- 一个可证伪机制假设；
- 唯一改变量及其唯一预设值或实现；
- 保持不变项：数据、split、model seed、网络宽度、子阵集合、学习率、batch size、
  epoch、checkpoint 规则、Root-MUSIC、1°和50%门槛、失败60°罚值；
- 主指标：`[2,4)` PC-NSS resolution rate；
- 守门指标：总体 failure-aware RMSPE、总体 resolution rate、failure count；
- 对照：原 epoch 35 checkpoint 和全局最佳固定 FBSS `L=7`；
- 停止规则：单次训练和一次冻结 validation 评价后判定，不围绕同一 validation
  调整第二个因素；未通过则停止，不访问 development/locked test；
- 进入 development 的条件仍需另行批准。

预注册不得选择新的间隔撑开损失，也不得改变正式成功分辨定义。

## 14. 测试与验证

严格按 RED → GREEN → 重构实施，至少覆盖：

- sample_seed 连接、重复/缺失/元数据不一致和越界失败；
- 1°阈值边界及七个互斥组；
- rho/SNR/snapshot 分组完全覆盖且不重复；
- 归一化尺度熵、单有效尺度空值和可靠性聚合；
- 残差 `0.095` 饱和边界；
- 三段投影变化公式、Dykstra 不收敛保留；
- 输出拒绝覆盖、schema 与来源 SHA；
- locked test 明确拒绝；
- 4 样本 CPU smoke 不读取正式 validation；
- 目标单测、完整 `unittest`、`compileall` 和默认 dry-run。

获授权的正式诊断只运行一次 1270 个冻结 validation 近间隔样本，不运行训练或
完整5000样本评价。完成后审查 Git diff，只提交源码、测试、规格和预注册文档。
