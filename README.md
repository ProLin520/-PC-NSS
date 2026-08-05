# MultiSource_DOA

`MultiSource_DOA` 是独立的多源 DOA 研究项目，研究少快拍、强相关/相干、近间隔双源条件下的可解释深度学习增强传统子空间算法。

当前批准主线是**多尺度、结构保持、分辨率感知的 PC-NSS**：网络融合 `L={4,5,6,7}` 的 FBSS 物理视图，输出逐 lag 置信、受限复残差和受限对角加载；协方差经过 Hermitian/Toeplitz/PSD/trace 投影后，由固定 Root-MUSIC 产生两个 DOA。

本项目与同级 `DIO_DOA` 独立，不直接导入或修改其运行代码。

## 当前状态

基础框架包含确定性双源仿真、SCM/SPS/FBSS、结构投影、Root-MUSIC、传统基线、PC-NSS 网络、分辨率教师、两阶段损失、failure-aware 评价、训练/报告引擎和安全入口。

“基础框架可运行”不表示 PC-NSS 已超过 Root-MUSIC、最佳固定尺度 FBSS、SubspaceNet、DA-MUSIC 或 DeepMUSIC。正式性能结论必须来自用户运行的锁定训练协议；SubspaceNet 等外部深度基线留到投稿强度比较阶段。

## 环境

项目依赖记录在 `requirements.txt`：Python、NumPy、SciPy 和 PyTorch。Agent 在本机验证时使用：

```text
D:\Python\Python\python.exe
```

用户可在 PyCharm 选择自己已经配置好 PyTorch/NumPy/SciPy 的解释器。

## PyCharm 无参数运行

直接运行 `scripts/run_multiscale_pcnss.py`，Parameters 留空。脚本顶部 `RUN_CONFIG` 默认：

```python
{
    "stage": "dry_run",
    "dry_run": True,
    "sample_count": 4,
    "allow_locked_test": False,
    "overwrite": False,
}
```

建议顺序：

1. 保持默认值运行 `dry_run`；它只检查依赖、一个物理样本、模型参数量和结构输出，不创建正式目录。
2. 将 `stage` 改成 `smoke_train`，或运行 `scripts/smoke_multiscale_pcnss.py`；它只训练 4 个样本、1 个 epoch，不写正式 `best.pt`。
3. 审核基础框架后，用户将 `stage` 改成 `train`、`dry_run=False`，运行正式训练。
4. 正式训练完成后，用户分别运行 `evaluate_validation` 和 `evaluate_development`。

每次 `stage` 只能是一个完整字符串，不能写成 `"train evaluate_development"`。基础阶段没有 `evaluate_locked_test` 入口。

## 运行分工

- Agent：RED/GREEN、目标及完整工程单测、`compileall`、默认 dry-run、4 样本 smoke。
- 用户：正式 40,000 样本训练、多 seed 训练，以及冻结模型后的最终 locked test。

正式输出默认拒绝覆盖；`outputs/`、权重、生成数据和密钥不会提交到 Git。

设计与实施文档：

- `docs/superpowers/specs/2026-08-04-multiscale-pcnss-design.md`
- `docs/superpowers/plans/2026-08-04-multiscale-pcnss-implementation.md`
