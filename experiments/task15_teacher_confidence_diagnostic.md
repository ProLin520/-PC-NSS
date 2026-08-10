# Task 15：PC-NSS Teacher 尺度置信只读诊断

日期：2026-08-10

## 诊断边界

- 输入：冻结的 audit-v4 validation schema-v2 报告，以及 Task 14 schema-complete 的
  1270 个 `[2,4)` 样本。
- 运行：CPU、batch size 128、teacher 当前温度 `0.10`、只读反事实温度 `0.05`。
- 未读取 checkpoint，未实例化或运行 PC-NSS，未训练，未访问 development/locked test，
  未运行完整 5000 样本 evaluator。
- 正式诊断只运行一次。输出保存在 Git 忽略目录
  `outputs/pcnss_teacher_confidence_seed2026_audit_v4`，不提交输出。
- 正式输出记录的诊断代码 SHA 为 `7ffd8cfd82393f37593f680de1fa1716776eeb0c`。随后最终审查
  仅增加了冻结 evaluator code SHA 的输入身份拒绝门（提交 `2e1ae55`），未改变样本重建、
  teacher 数学、汇总或 decision；最终 loader 已对同一输入只读复验为 1270 个 seed，按
  “正式诊断只运行一次”纪律未覆盖或重跑输出。

## 完整性审计

- audit 中 PC-NSS 与 FBSS L4–L7 各有 5000 个唯一 seed；五算法 seed 集合及真值、rho、
  SNR、snapshot、separation 完全一致。
- Task 14 输入有 1270 个唯一近间隔 seed，与 audit 的 `[2,4)` 集合完全一致。
- audit/Task 14 输入 SHA 与输出 manifest 逐项一致；checkpoint SHA 只在两个 manifest 间
  比较，未读取 checkpoint 文件。
- 样本 CSV 有 1270 行、1270 个唯一 seed，无 NaN/Infinity；三组四尺度概率均有限、非负，
  行和在 `1e-6` 内等于 1。
- 独立标准库脚本从样本 CSV 复算 entropy、pmax、margin、KL、JS、oracle、regret、总体
  分布、17 个固定分层计数及 decision，全部与结构化报告一致。
- 输出目录恰有设计规定的六个文件，manifest 明确
  `no_model_forward=true`、`training_performed=false`。

## 主要结果

| 指标 | 结果 | 冻结门 | 判定 |
|---|---:|---:|---|
| teacher `tau=0.10` 中位归一化熵 | 0.999763 | ≥ 0.90 | 通过：teacher 很平 |
| `tau=0.05` 中位归一化熵 | 0.999056 | — | — |
| 中位熵下降 | 0.000707 | ≥ 0.05 | 失败 |
| teacher 中位 pmax，`tau=0.10` | 0.257783 | — | — |
| teacher 中位 pmax，`tau=0.05` | 0.265639 | — | — |
| 中位 pmax 上升 | 0.007856 | ≥ 0.05 | 失败 |
| teacher top1/oracle 一致率 | 35.4331% | ≥ 40% | 失败 |
| teacher 选择 regret 中位数 | 0.787290° | ≤ 1.0° | 通过 |
| rho/SNR/snapshot 支持维度数 | 0/3 | ≥ 2/3 | 失败 |
| 学生中位归一化熵 | 0.998280 | — | 接近均匀 |
| teacher/student KL 中位数 | 0.001021 | — | 两者非常接近 |
| teacher/student JS 中位数 | 0.000255 | — | 两者非常接近 |
| raw score top1-top2 margin 中位数 | 0.001451 | — | 远小于 `tau=0.10` |
| margin/tau 中位数 | 0.014507 | — | 温度尺度不匹配明显 |

teacher 主导尺度计数为 L4/L5/L6/L7 = `823/143/120/184`；学生主导尺度计数为
`924/121/32/193`。teacher 并非严格等概率，但 raw score 差异很小，且 top1 与样本级
最佳固定尺度的一致性不足。

所有 rho、SNR 和 snapshot 分层的 teacher 中位熵均高于 0.9995 左右；降温后的中位熵
下降仅约 `0.00014–0.00128`，没有任何一个维度达到“至少两个 bin 下降 0.05”的支持定义。
oracle 一致率在 rho 分层约为 `33.99%–36.91%`，SNR 分层约为 `34.32%–37.10%`，snapshot
分层约为 `29.80%–38.19%`，未出现能支持简单降温的稳定子场景。

## 冻结结论

`allow_tau_preregistration=false`，且 `training_authorized=false`。

本次证据表明，学生尺度分布接近均匀主要与 teacher 自身极平一致，而不是“teacher 已经
尖锐、学生没有学到”。把 `tau_scale` 从 0.10 单独降到 0.05 几乎不能改变 teacher
置信；更重要的是，teacher top1/oracle 一致率低于冻结门，继续把当前 teacher 变尖会放大
一部分错误尺度选择。因此不创建 `tau_scale: 0.10 -> 0.05` 单因素训练预注册，也不启动
训练。

## 下一步建议

下一步仍应只读诊断 teacher 的“排序有效性”，而不是立即调整温度或训练：

1. 按样本计算四尺度 raw score 与 `-sample_rmspe_deg` 的 Spearman/Kendall 排序一致性，
   并形成 teacher top1 对 oracle scale 的 4×4 混淆矩阵。
2. 分别拆解 teacher score 中“真角 denominator”和“中点 denominator”的贡献，检查
   低 margin 是两项同时饱和，还是相减后动态范围被抵消。
3. 按 Task 14 七 cohort、rho、SNR、snapshot 比较排序一致性与 regret 尾部；只有找到
   在冻结场景中稳定关联 oracle 的 teacher score 定义后，才能设计新的单因素 teacher
   实验。

这三项仍属于机制诊断，不授权修改 teacher、损失、网络或启动训练。评价与诊断基础设施
是否合并，只依据独立的工程验证门，不因本次科研门失败而否决。
