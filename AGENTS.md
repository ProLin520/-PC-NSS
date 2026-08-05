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

## Agent 与用户运行分工

Agent 负责：

- 按 TDD 执行每项 RED -> GREEN -> 重构，并亲自确认 RED 原因正确；
- 运行目标单元测试、必要的完整 `unittest` 工程测试、`compileall`、默认 dry-run 和 4 样本 smoke；
- 在每个 Task 后审查代码、测试、配置、数据范围、防覆盖和 Git diff；
- 不运行正式模型训练，不访问或运行 locked test，不用 development/validation 反复搜索超参数；
- 每个 Task 完成后报告改动、验证证据、注意事项和下一步计划。

用户负责：

- 运行正式 40,000 样本模型训练和后续正式训练 seed；
- 在模型、损失和阈值冻结且单独批准后，运行最终 locked test；
- 审核正式训练结果并批准是否进入优化、多 seed 或投稿强度对比。

“最终测试由用户运行”特指最终 locked test/论文最终评价；不妨碍 Agent 运行实现所需的单元测试、完整工程测试、compileall、dry-run 和极小样本 smoke。

## Git 与远程同步

- 本项目远程仓库固定为 `https://github.com/ProLin520/-PC-NSS.git`，本地 remote 名称使用 `origin`。
- 每个完成并验证的 Task 使用独立、范围明确的提交，并同步到当前实现分支；不使用整目录无审查暂存。
- 推送前必须运行相应验证并检查 `git diff --cached --name-only`；禁止 force push，除非用户另行明确批准。
- 只同步本项目源码、测试、配置和文档；不得提交或推送 `outputs/`、checkpoint、权重、生成数据、`.env`、密钥、Token、Cookie 或证书。
- 不把 `DIO_DOA`、`Graduation` 或其他项目中的未提交改动带入本仓库。
- 实施计划优先在 `codex/` 前缀的隔离分支/工作树进行；完成基础框架并经用户审核后再决定如何合并到默认分支。

## 工程规则

- 按 TDD 执行 RED -> GREEN -> 重构。
- Agent 负责目标单测、`compileall`、dry-run 和极小样本 smoke。
- 正式 40,000 样本训练须在实现和 smoke 审核后单独启动，不在脚手架创建时自动运行。
- 用户常跑脚本顶部提供 `RUN_CONFIG`，默认 `dry_run=True`，PyCharm Parameters 可留空。
- 结构化结果写 CSV/JSON，记录配置、seed、代码 SHA、checkpoint SHA、失败标志、参数量和运行时间。
- 不覆盖历史输出，不提交权重、生成数据、大型输出或敏感配置。
- 只修改当前任务路径，删除逐文件精确执行，禁止递归危险删除。
