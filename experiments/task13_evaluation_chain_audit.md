# Task 13：评价链审计记录

## 范围

本 Task 只修正评价链，不训练模型、不修改模型结构或损失、不访问
development/locked test，也不由 Agent 重跑正式 5000 样本 validation。
既有正式输出只读用于定位，未复制到仓库。

## 7.2503 与 7.2640 的来源

只读证据来自 seed2026 的 epoch 35 checkpoint 和既有 validation 报告：

- checkpoint validation：failure-aware RMSPE `7.2502939419439665°`；
- 既有正式报告：failure-aware RMSPE `7.264029072527954°`；
- 两者使用同一 code SHA `8c781360a3eb738c24d1c87d776094c3a7deb632`、
  同一 checkpoint SHA256
  `9f28a924cbe4fd8c3335e6d6b78b98cedac55c69859b8b3d9b60d368dbaf6787`、
  同一 5000 个 validation 样本；failure count 都为 0，resolution rate 都为
  `0.162`；
- MAE 只差约 `1.15e-7°`，但 RMSPE 相差约 `0.013735°`，累计平方误差相差
  约 `1993.56 deg²`，最大绝对误差相差约 `0.8232°`。

代码路径审计定位到两个数值差异源：

1. checkpoint validation 由 DataLoader 按冻结的 `batch_size=128` 推理；旧版
   正式 evaluator 把全部 5000 个样本一次性送入模型。CUDA 的 batched linear、
   eigendecomposition 等算子会随批形状产生微小浮点差异；Root-MUSIC 在候选根
   接近筛选边界时可把这种微扰放大为少数尾部样本的根选择变化。这是 RMSPE 和
   最大误差差异的主因。
2. checkpoint validation 的计分真值来自 batch 内的 float32 张量，正式报告使用
   仿真样本的 float64 真值；它解释 MAE 的 `1e-7°` 量级尾差，不足以解释
   `0.013735°` 的 RMSPE 差异。

修正后 evaluator 分批推理，默认并显式记录 `evaluation_batch_size=128`。CPU
批量一致性测试对输出协方差使用 `rtol=1e-5, atol=2e-6`；正式 CUDA 指标仍需
用户用原 checkpoint 在 schema v2 下重跑，才能得到新的权威聚合值。

## schema v2

原有七个报告文件与旧字段全部保留，`summary.json` 新增
`report_schema_version=2` 和 `near_separation_audit`。`predictions.csv` 新增：

- `both_angle_errors_within_1_deg`；
- `estimated_separation_at_least_half_true`。

近间隔固定为 separation bin `[2,4)`，分别报告两个条件、最终 resolved 的
数量/比例，以及样本级 RMSPE 严格大于 `10°`、`30°`、`60°` 的数量/比例。
连续 paired strata 固定为：

- separation：`[2,4)`、`[4,6)`、`[6,8)`、`[8,10]`；
- SNR：`[-5,0)`、`[0,5)`、`[5,10]`。

`rho` 与 `snapshot_count` 继续按离散值分组。连续值越界、paired sample ID
重复或左右 strata 不一致会显式报错，不会静默丢样本。

## 既有预测的只读近间隔重算

这组数字来自旧版 full-batch `predictions.csv`，只用于验证聚合定义，不替代
schema v2 的 batch-128 正式重跑：

| algorithm | n | 两角误差均≤1° | 间隔≥真值50% | resolved | RMSPE>10° | >30° | >60° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PC-NSS | 1270 | 49 (3.8583%) | 1266 (99.6850%) | 49 (3.8583%) | 67 | 28 | 6 |
| FBSS L=7 | 1270 | 91 (7.1654%) | 1235 (97.2441%) | 86 (6.7717%) | 767 | 386 | 44 |

该拆解表明旧报告中 PC-NSS 的近间隔退化来自“两角都在 1° 内”条件，而不是
估计间隔不足；同时 PC-NSS 的大误差尾部显著少于 L=7。此结论仅解释既有
validation，不授权调参或访问 development/locked test。

## 用户重跑配置

在本 Task 分支的 `scripts/run_multiscale_pcnss.py` 顶部使用：

```python
{
    "stage": "evaluate_validation",
    "dry_run": False,
    "device": "cuda",
    "output_root": "<新的、不存在的审计输出根目录>",
    "checkpoint_path": "<原 seed2026 输出根目录>/best.pt",
    "selected_best_fbss_scale": None,
    "evaluation_batch_size": 128,
    "allow_locked_test": False,
    "overwrite": False,
}
```

随后运行：

```powershell
D:\Python\Python\python.exe scripts\run_multiscale_pcnss.py
```

必须选择新输出目录，保留旧报告作为审计对照。
