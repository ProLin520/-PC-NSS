# AGENTS.md

## 项目定位

`MultiSource_DOA` 是独立的多源 DOA 论文研究项目。当前主线是多尺度、结构保持、分辨率感知的 PC-NSS，不是 `DIO_DOA` 中单源 PALR 的版本升级。

项目使用 Python 3.10、PyTorch、NumPy、SciPy 和标准库 `unittest`。依赖统一记录在 `requirements.txt`，不得为简单功能随意增加依赖。

## 当前研究问题

研究 8 阵元半波长 ULA 在双源、少快拍、强相关或完全相干、近间隔条件下的 DOA 分辨。网络不直接输出角度，而是增强多尺度 FBSS 协方差，随后使用固定 Root-MUSIC。

```text
snapshots
  -> multi-scale FBSS views
  -> learned lag-wise fusion and bounded residual
  -> Hermitian/Toeplitz/PSD structure projection
  -> fixed Root-MUSIC
  -> two DOAs
```

## 创新边界

- 不把普通“神经网络修正协方差”本身写成创新。
- 不声称首次、首个或理论保证不退化，除非后续系统检索和证明支持。
- 相对 CNN+Toeplitz-prior 工作，必须突出多尺度 FBSS 物理视图、逐 lag 可信融合、相干源去相关目标和近间隔分辨损失。
- 相对 SubspaceNet，必须突出传统空间平滑锚点、低自由度 lag 修正和显式结构投影，而不是自由代理协方差。
- 相对 TransDOA，必须明确本模型不直接回归 DOA，不做源域/目标域特征迁移，最终角度来自固定 Root-MUSIC。
- 相对 CR-UNet，必须明确本模型不使用 Complex U-Net 从损坏 SCM 直接重建理想完整协方差，不以 data MSE + Toeplitz loss + noise-subspace loss 作为核心训练目标；本模型学习多尺度 FBSS 的逐 lag 融合，并以近间隔分辨能力和相对固定尺度 FBSS 的优势为训练信号。
- 相对 unit-nuclear-norm 单快拍方法，必须明确本项目是多快拍相干双源、多尺度 FBSS 融合，不预测 DOA/SNR 后重建协方差，也不复用其白化对称损失。

## 实验纪律

- 主场景：`N=8`、`K=2`、半波长 ULA、完全相干、`T=20`、`SNR=5 dB`、角度中心 `[-50, 50]`、间隔 `2-10` 度。
- train、validation、development test、locked test 使用独立 seed 和样本索引。
- locked test 在模型、损失和验收阈值冻结前禁止访问。
- 失败样本不得删除。缺失或无效角度按固定 `60` 度罚值进入 failure-aware RMSPE。
- 第一轮不搜索损失权重、网络宽度或子阵集合。
- 稳健门槛要求同时超过原始 Root-MUSIC 和最佳固定尺度 FBSS Root-MUSIC，并提高近间隔分辨率且不增加失败。
- SubspaceNet、DA-MUSIC、DeepMUSIC 用于投稿强度比较；无法按同一协议复现时必须记录原因。

## 工程规则

- 按 TDD 执行 RED -> GREEN -> 重构。
- Agent 负责目标单测、`compileall`、dry-run 和极小样本 smoke。
- 正式 40,000 样本训练须在实现和 smoke 审核后单独启动，不在脚手架创建时自动运行。
- 用户常跑脚本顶部提供 `RUN_CONFIG`，默认 `dry_run=True`，PyCharm Parameters 可留空。
- 结构化结果写 CSV/JSON，记录配置、seed、代码 SHA、checkpoint SHA、失败标志、参数量和运行时间。
- 不覆盖历史输出，不提交权重、生成数据、大型输出或敏感配置。
- 只修改当前任务路径，删除逐文件精确执行，禁止递归危险删除。
