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
    "evaluation_batch_size": 128,
    "allow_locked_test": False,
    "overwrite": False,
}
```

正式评价的神经推理默认固定为 `batch_size=128`，与 checkpoint validation
一致；该值会写入 `run_config.json` 和 `runtime_summary.json` 供审计。

建议顺序：

1. 保持默认值运行 `dry_run`；它只检查依赖、一个物理样本、模型参数量和结构输出，不创建正式目录。
2. 将 `stage` 改成 `smoke_train`，或运行 `scripts/smoke_multiscale_pcnss.py`；它只训练 4 个样本、1 个 epoch，不写正式 `best.pt`。
3. 审核基础框架后，用户将 `stage` 改成 `train`、`dry_run=False`，运行正式训练。
4. 正式训练完成后，用户分别运行 `evaluate_validation` 和 `evaluate_development`。

每次 `stage` 只能是一个完整字符串，不能写成 `"train evaluate_development"`。基础阶段没有 `evaluate_locked_test` 入口。

## 近间隔只读诊断

直接运行 `scripts/diagnose_pcnss_near_resolution.py` 默认只做 CPU `dry_run`，不会读取
checkpoint、`audit_v4` 或创建输出目录。正式诊断必须通过 `--config path.json` 显式设置
`stage="diagnose_validation_near"` 和 `dry_run=false`；它只接受既有 `audit_v4` 的
validation schema-v2 报告、对应冻结 checkpoint，以及由配对审计确定的近间隔样本。

神经推理 batch 固定为 `128`，不可通过 JSON 配置或直接调用覆盖。诊断输出写入新的目录并默认
拒绝覆盖，`outputs/` 和诊断结果不提交到 Git。此诊断只用于解释冻结结果，不构成训练、
重新训练或访问 development/locked test 的授权。

## Teacher 尺度置信只读诊断

直接运行 `scripts/diagnose_pcnss_teacher_confidence.py` 默认只做 CPU `dry_run`，不会读取
checkpoint、正式评价报告或 Task 14 输出，也不会创建目录。4 样本 smoke 只使用确定性的
train 样本和内存合成标签，不运行 PC-NSS 前向或训练。

正式入口只允许 `stage="diagnose_validation_teacher"`、validation、CPU、batch size 128，
并固定比较 `tau=0.10` 与只读反事实 `tau=0.05`。它认证 audit-v4 和 Task 14
schema-complete 输入后，仅重建冻结的 1270 个 `[2,4)` 样本并调用物理 teacher；不会加载
checkpoint、实例化 `MultiScalePCNSS`、访问 development/locked test 或运行完整 5000
样本 evaluator。

输出为六个新的 CSV/JSON 文件，目录存在时拒绝覆盖，且不得提交 `outputs/`。诊断结果只用于
判断是否允许另写 `tau_scale` 单因素训练预注册；无论结论如何，本入口都不授权训练。

## Teacher 排序有效性只读诊断

`scripts/diagnose_pcnss_teacher_ranking.py` 默认同样只做 CPU `dry_run`。正式入口固定为
`stage="diagnose_validation_teacher_ranking"`、validation、CPU、batch size 128，只认证并
重建 Task 15 的 1270 个近间隔样本，拆解当前 teacher score 并比较 L4–L7 的固定尺度
failure-aware RMSPE 排序。它不加载 checkpoint、不运行神经模型、不修改 teacher，也不训练。

4 样本 smoke 只使用 train split 和内存合成 RMSPE。正式输出为八个 schema-v1 CSV/JSON
文件，新目录存在时拒绝覆盖，且不得提交 `outputs/`。结论只决定下一项应研究标定、组成项抵消，
还是 train-only 角误差 teacher；Task 16 本身不授权任何训练。

## Train-only Failure-aware 角误差 Teacher Cache

`scripts/build_pcnss_failure_aware_teacher_cache.py` 默认只运行 1 个 train 样本的
CPU `dry_run`，不会创建输出、加载模型或读取 validation/development/locked test。
`smoke` 固定使用前 4 个 train 样本并写三文件 cache；正式
`build_train_teacher_cache` 固定为 train、CPU、batch size 128 和 40,000 样本，
由用户在实现与 smoke 审核后运行。输出目录存在时拒绝覆盖。

cache 记录 L4–L7 固定 FBSS + Root-MUSIC 的 failure-aware 匹配角 RMSPE，失败继续按
60 度罚值，最优尺度在 `1e-6` 度容差内并列时均分概率。cache 属于生成实验数据，
不得提交 Git；生成它不构成正式训练、development 或 locked test 授权。

## 运行分工

- Agent：RED/GREEN、目标及完整工程单测、`compileall`、默认 dry-run、4 样本 smoke。
- 用户：正式 40,000 样本训练、多 seed 训练，以及冻结模型后的最终 locked test。

正式输出默认拒绝覆盖；`outputs/`、权重、生成数据和密钥不会提交到 Git。

## 单因素身份审计

`scripts/audit_pcnss_teacher_single_factor.py` 在训练前只读认证物理 teacher 基线、其
validation 报告、Task 16 `ranking_invalid` 结论和 train-only teacher cache。默认
`dry_run` 不读取这些输入，也不创建目录；`smoke` 只写三文件合成审计报告。正式审计只允许
CPU，并拒绝覆盖或访问 locked test。只有所有数据、模型、checkpoint、评估协议和训练环境
身份门同时通过时才允许复用旧物理基线，否则结论固定为重新运行物理 teacher 对照组；审计
本身不授权或执行训练。

设计与实施文档：

- `docs/superpowers/specs/2026-08-04-multiscale-pcnss-design.md`
- `docs/superpowers/plans/2026-08-04-multiscale-pcnss-implementation.md`
