# 多尺度 PC-NSS 正式训练与分阶段验收协议

## 分工与冻结规则

- 用户运行正式 40,000 样本训练、后续正式 seed，以及模型完全冻结后的最终 locked test。
- Agent 只运行单元测试、`compileall`、默认 dry-run 和 4 样本 smoke，不代跑正式训练或 locked test。
- 第一轮不搜索损失权重、网络宽度、子阵集合、学习率或其他超参数。
- 每次 `stage` 只能填写一个完整字符串，不能写成 `"train evaluate_development"`。
- 输出目录必须使用新的明确路径，保持 `overwrite=False`，不得覆盖历史有效输出。

## 0. 环境和最小检查

在 PyCharm 中打开 `scripts/run_multiscale_pcnss.py`，Parameters 留空，使用已安装 PyTorch、NumPy 和 SciPy 的解释器。先保留默认：

```python
RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "sample_count": 4,
    "allow_locked_test": False,
    "overwrite": False,
    ...
}
```

默认 dry-run 应报告 `parameter_count=46916`、`physical_chain_finite=true`、`locked_test_access=false` 和 `output_created=false`。如需再次检查训练链，可单独运行 `scripts/smoke_multiscale_pcnss.py`；它只使用 4 个样本和 1 epoch，不产生正式 checkpoint。

## 1. seed2026 正式训练

只修改脚本顶部 `RUN_CONFIG`：

```python
{
    "stage": "train",
    "dry_run": False,
    "model_seed": 2026,
    "output_root": "outputs/multiscale_pcnss_snap20_seed2026",
    "allow_locked_test": False,
    "overwrite": False,
    "device": "cuda",  # 仅在本机 CUDA 可用时使用，否则填 "cpu"
}
```

数据量、epoch、batch size、学习率和两阶段损失来自冻结的 `ExperimentConfig`，不要在首轮为追求 validation 指标修改。训练结束后应保留：

- `train_manifest.json` 与 `validation_manifest.json`；
- `metrics.csv`；
- 按 validation failure-aware RMSPE 选择的 `best.pt`；
- checkpoint 内的配置、model seed、validation split seed、代码 SHA 和优化器状态。

## 2. validation 评估与最佳固定 FBSS 尺度

训练完成后单独运行：

```python
{
    "stage": "evaluate_validation",
    "dry_run": False,
    "output_root": "outputs/multiscale_pcnss_snap20_seed2026",
    "checkpoint_path": "",  # 空字符串表示使用 output_root/best.pt
    "selected_best_fbss_scale": None,
    "evaluation_batch_size": 128,  # 与 checkpoint validation 保持一致
    "allow_locked_test": False,
    "overwrite": False,
}
```

validation 报告同时包含原始 Root-MUSIC、各固定尺度 FBSS Root-MUSIC、oracle 固定尺度上界、PC-NSS 和尚未接入的外部深度基线状态。根据整套 validation 全局选择唯一的 `best_fixed_fbss_scale`；不得逐样本选择，也不得按间隔、SNR 或结果好坏切换尺度。

评价报告 schema v2 保留原有七个文件和旧字段，并增加连续变量区间化的
paired comparison、`[2,4)` 近间隔分辨条件拆解和样本 RMSPE 尾部统计。
已有 `validation_report` 不得覆盖；审计重跑时使用新的 `output_root`，并把
`checkpoint_path` 显式指向原实验的 `best.pt`。

## 3. development 评估

把 validation 报告给出的唯一尺度填入：

```python
{
    "stage": "evaluate_development",
    "dry_run": False,
    "output_root": "outputs/multiscale_pcnss_snap20_seed2026",
    "checkpoint_path": "",
    "selected_best_fbss_scale": 6,  # 示例；必须替换为 validation 实际选出的尺度
    "allow_locked_test": False,
    "overwrite": False,
}
```

development 只用于确认冻结选择的外推表现，不得回到同一 development 结果反复搜索权重、宽度、尺度或阈值。

## 4. seed2026 稳健论文门槛

PC-NSS 必须同时满足：

1. failure-aware RMSPE 严格优于原始 Root-MUSIC；
2. failure-aware RMSPE 严格优于 validation 全局选择的最佳固定尺度 FBSS Root-MUSIC；
3. 近间隔分辨率率提高；
4. failure count 不增加；
5. 收益不是来自删除失败样本、不同裁剪或不同计分协议。

报告还需审计总体与分间隔性能、paired win/tie/loss、最大误差、投影前后最小特征值、尺度权重、lag 残差、子空间角、谱峰 margin/gap 和 train-validation gap。

## 5. 未通过时的无调参诊断

任一硬门槛失败，首轮先停止扩 seed 和 locked test，不在同一 validation 上做搜索。只诊断：

- 尺度权重是否塌缩到单一 `L`，且这种塌缩是否与间隔或样本质量有关；
- 残差是否长期贴近上界、接近零，或被结构投影基本抹除；
- 投影前后 PSD、Toeplitz 和 trace 误差是否说明网络输出不可实现；
- 预测协方差与 decorrelated teacher 的 signal-subspace angle 是否下降；
- 近间隔样本的谱峰 margin/gap 是否确实增加，而不是只降低协方差数值损失；
- train 与 validation 是否同时失败，还是出现明显泛化差距；
- paired 损失样本是否集中在某一间隔、SNR 或相关性条件。

上述诊断只解释失败机理，不自动授权修改模型。若需要第二版，先形成单独、可证伪且不依赖 locked test 的改进设计，再审批实施。

## 6. 后续审批顺序

1. seed2026 同时通过 validation 与 development 的冻结门槛后，单独审批 seed2027/2028；
2. 三 seed 结果稳定后，冻结模型、损失、尺度选择和验收阈值；
3. 冻结完成后再单独增加/批准 locked test 入口，由用户运行一次最终评估；
4. 基础路线成立后，再按同一数据和 failure-aware 协议接入 SubspaceNet、DA-MUSIC、DeepMUSIC 等外部基线；无法公平复现时明确记录原因，不填造结果。

当前基础框架没有 `evaluate_locked_test` stage，这是有意的安全边界，不应通过改 split 名称、直接实例化数据集或修改 `allow_locked_test` 绕过。
