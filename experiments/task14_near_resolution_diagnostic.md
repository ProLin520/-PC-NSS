# Task 14：近间隔严格分辨诊断结论（只读）

## 结论

本次不创建 `pcnss_near_angle_precision_preregistration.md`，也不启动训练。冻结保护门的理由高于任何候选机制：在 `[2,4)` 的 1270 个 validation 样本中，PC-NSS 的 `far_miss_gt_2` 为 1002/1270（78.90%）。设计规格第 12 节第 4 条规定：失败主要为大于 2° 的远失配时，本轮只能记录诊断、不得预注册训练。故不能把问题误述为仅在 1° 门限附近的局部精度问题，也不能为了产出预注册而选择 teacher temperature。

这是一份关联诊断，而非因果证明。下列尺度熵、残差和投影量只能用于提出后续可检验问题；它们不能证明某项改动会改善严格分辨率。

## 输入身份与完整性

诊断使用冻结的 validation 子集；未读取、未运行 development 或 locked test，未训练、未重跑正式诊断。Task 6 文档阶段未新增、修改或覆盖 `outputs/`；Task 5 已生成并保留下文列出的六个只读诊断输出。

| 项目 | 值 |
| --- | --- |
| 源报告 | `D:\Python\Project\doa_estimation\MultiSource_DOA\.worktrees\pcnss-foundation\scripts\outputs\multiscale_pcnss_snap20_seed2026_audit_v4\validation_report` |
| split / 样本 | validation / `[2,4)` 共 1270；validation split seed `202708040` |
| checkpoint SHA256 | `9f28a924cbe4fd8c3335e6d6b78b98cedac55c69859b8b3d9b60d368dbaf6787` |
| evaluator code SHA | `129c3ba3b9fc1919451eef5c67376f04b4b24680` |
| diagnostic code SHA | `106e83a7ff263e91e759bce2127d04ac9c3e8891` |
| residual limit / 饱和阈值 | `0.10` / `0.095` |
| Dykstra 未收敛数 | `0/1270`（全部记录保留） |

源报告文件 SHA256：`run_config.json` `bbebfc5a0238bfb593c8806bb8aa3dff67c391bd93b5f760c1a6180e88bb27b5`；`summary.json` `adcfc37ef2ef315af35ccaa76f2025d05cd0e8a9877c3485bf95e585766fb724`；`source_manifest.json` `eecf03d1a7a561814057831aad96d667d592993f749966175b25e24a2b6d5e46`；`predictions.csv` `eb91eb1bba8f9b26f06f2dac921b4ed208745187d83ad02aaa667c3aa379eb90`。

本次六个诊断输出 SHA256：`diagnostic_config.json` `70f32aaa3a0b1abb3955b4e0127a32ec2e58de72c7c4211a99e7546d73058355`；`source_manifest.json` `fe0674cc24f1c31bf21dd7660c41a48d617cc3754094090cd71ba223e3ed9131`；`near_sample_diagnostics.csv` `e1ed4bd0571dbb31923262e18409fc5cf00b1c425f2a2520b88caf9cc2bb8d54`；`threshold_summary.json` `c9bdde41b47bd0ccbca6b5db0cd9bf3d51655f586af2e333bb90febe10f2a9c3`；`stratified_summary.csv` `8286bd48ae58cb4fca83457c68c1faa9f2361ddc1a6933a1f6301cd5d0e9ab62`；`mechanism_summary.json` `b44854798309c23c6dfb1e5b8beed1f97d7af4adb11b02242842bf5e1af54602`。

## 最大角误差累计通过率

严格精度按两角中最大绝对误差计。PC-NSS 与固定 `FBSS L=7` 的同一 1270 个 `sample_seed` 的累计通过数/率如下。

| 阈值（°） | PC-NSS | FBSS L=7 |
| --- | ---: | ---: |
| 0.50 | 9 / 0.71% | 28 / 2.20% |
| 0.75 | 23 / 1.81% | 60 / 4.72% |
| 1.00 | 49 / 3.86% | 91 / 7.17% |
| 1.25 | 95 / 7.48% | 125 / 9.84% |
| 1.50 | 136 / 10.71% | 158 / 12.44% |
| 2.00 | 267 / 21.02% | 232 / 18.27% |

因此 PC-NSS 在更宽的 2° 门限下较高（+2.76 个百分点），但在本任务主要的 1° 严格门限低 3.31 个百分点；不能用宽门限收益替代严格 resolution 的结论。按样本 RMSPE（`1e-6°` tie 容差）比较，PC-NSS 相对 L=7 为 win/tie/loss = 919/0/351；这同样不等于严格分辨率的改善。

## 互斥 cohort 与远失配主体

七组按照正式标签的互斥顺序统计，合计 1270。`estimation_failure` 为空组，按要求记 0。

| cohort | 数量 | 比例 |
| --- | ---: | ---: |
| estimation_failure | 0 | 0.00% |
| separation_failure | 4 | 0.31% |
| resolved（两角均 ≤1° 且间隔通过） | 49 | 3.86% |
| near_miss_1_1p25 | 45 | 3.54% |
| near_miss_1p25_1p5 | 40 | 3.15% |
| near_miss_1p5_2 | 130 | 10.24% |
| far_miss_gt_2 | 1002 | 78.90% |

三类 near-miss 合计 215（16.93%），远失配数量是其 4.66 倍。这直接触发规格第 12 节第 4 条，故本轮拒绝所有单因素训练预注册。

## rho、SNR、snapshot 分层

下表是最相关的 1° 严格门限，列为 PC-NSS / L=7，通过率差为前者减后者。所有分层各自覆盖 1270 样本。

| 维度 | bin（n） | PC-NSS / L=7 | 差异 |
| --- | --- | ---: | ---: |
| rho | 0.8（343） | 3.50% / 6.41% | -2.92 pp |
| rho | 0.9（317） | 3.47% / 8.83% | -5.36 pp |
| rho | 0.99（306） | 4.58% / 7.84% | -3.27 pp |
| rho | 1.0（304） | 3.95% / 5.59% | -1.64 pp |
| SNR | [-5,0)（405） | 1.48% / 1.98% | -0.49 pp |
| SNR | [0,5)（407） | 2.70% / 4.18% | -1.47 pp |
| SNR | [5,10]（458） | 6.99% / 14.41% | -7.42 pp |
| snapshots | 8（406） | 2.22% / 2.22% | +0.00 pp |
| snapshots | 20（445） | 4.49% / 5.62% | -1.12 pp |
| snapshots | 50（419） | 4.77% / 13.60% | -8.83 pp |

最明显且跨维度同向的阈值差异是严格区间（0.50–1.25°）中 PC-NSS 相对 L=7 的劣势，尤其是 SNR `[5,10]` 和 50 snapshots。到 2° 时若干 bin 转为 PC-NSS 较高，说明它不是单调一致的严格精度增益；不支持把结果解释为局部 1° 边界的微调问题。

## 机制读数：尺度、残差、投影

`p_L` 为可靠性聚合后的四尺度质量；投影列依次为 train / eval Dykstra / total 的 Frobenius 相对变化均值。`—` 表示空组不可计算。

| cohort | p_L4 / p_L5 / p_L6 / p_L7 | 尺度熵 | 主导尺度计数 L4/L5/L6/L7 | 饱和 lag 率 | train / eval / total 投影变化 |
| --- | --- | ---: | --- | ---: | --- |
| estimation_failure（0） | — | — | 0/0/0/0 | — | — |
| separation_failure（4） | .2732/.2539/.2432/.2297 | .99830 | 4/0/0/0 | .71875 | .07308 / 8.23e-16 / .07308 |
| resolved（49） | .2758/.2573/.2404/.2265 | .99680 | 42/3/0/4 | .74235 | .09717 / 4.48e-09 / .09717 |
| near 1–1.25°（45） | .2708/.2572/.2419/.2301 | .99755 | 33/5/0/7 | .71389 | .09547 / 8.42e-09 / .09547 |
| near 1.25–1.5°（40） | .2717/.2568/.2415/.2299 | .99744 | 31/1/0/8 | .70313 | .09389 / 4.32e-09 / .09389 |
| near 1.5–2°（130） | .2689/.2563/.2415/.2333 | .99769 | 93/14/2/21 | .67885 | .09065 / 2.91e-09 / .09065 |
| far >2°（1002） | .2730/.2563/.2390/.2317 | .99663 | 721/98/30/153 | .72143 | .08264 / 1.65e-09 / .08264 |

残差幅值最大值几乎贴住上限（各关键 cohort 的均值均约 `0.09996–0.09999`），但这是共同现象，不能单独识别失败机制。更关键的是，本次 resolved 的饱和 lag 率 `0.74235` 高于三组 near-miss 的 `0.71389/0.70313/0.67885`，且 resolved 的 train/total 投影变化 `0.09717` 也高于三组 near-miss 的 `0.09547/0.09389/0.09065`。eval Dykstra 变化量都接近零，且无不收敛样本。因此优先级 1“残差被边界或投影限制”没有得到所需的同向证据；不得放宽 residual 或 projection。

尺度熵全部接近 1。三组 near-miss 的均值 `0.99755/0.99744/0.99769` 略高于 resolved 的 `0.99680`；其分布更接近均匀。合并 near-miss 相对 resolved 的熵在 rho `0.8/0.99/1.0`、全部三个 SNR bin、snapshot `20/50` 中均更高（例如 SNR `[5,10]`：`.998652` 对 `.998331`；snapshot 50：`.998115` 对 `.997927`）。这满足“尺度置信不足”作为**后续只读机制验证候选**的关联条件，但不能越过远失配冻结保护门。

## 固定优先级判读与下一步

1. 残差/投影限制：拒绝。near-miss 没有比 resolved 更高的饱和或投影变化，方向相反。
2. 尺度置信不足：观察到可重复的关联（near-miss 熵略高且近均匀）；`build_scale_teacher(..., tau_scale=0.1)` 当前通过 `softmax(scale_scores / tau_scale)` 生成固定 teacher 分布，`pcnss_loss` 再以 KL 蒸馏该分布。若没有冻结保护门，单一、可证伪且不改结构的候选才会是只把该固定 `tau_scale` 从 `0.10` 改为 `0.05`，不同时增加熵项、不改损失权重或网络结构。这一候选比新增间隔撑开损失更直接对应尺度置信，也不改正式 resolution 定义；但本轮不得执行或预注册。
3. 局部角度精度代理不足：不进入。失败主体不是 `(1,1.5]°`，而是 `>2°` 远失配。
4. 冻结保护门：触发，最终决定为“证据不足，不启动训练”。

下一项建议仅做只读机制验证：在不训练、不改变参数、不访问 development 或 locked test 的前提下，审计冻结 checkpoint 的 teacher 分数间隔及 `tau=0.10` 下 teacher 概率熵，按本报告 cohort 和三类场景分层比较。目的只是检验“学生尺度质量近均匀”是否与“teacher 本身不够尖锐”一致；结果仍不能替代新的独立训练证据。

## 研究纪律

本文不声称性能或因果关系已得到证明。任何以后可能的训练都须在单独批准、独立预注册后进行；本文件不构成训练授权，并继续禁止访问 development 与 locked test。
