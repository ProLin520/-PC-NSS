# 多尺度 PC-NSS 基础框架审查

## 审查结论

基础框架已完成工程实现与验证，可以交给用户启动首轮正式训练。这个结论只表示代码链路、约束和审计接口可运行，不表示 PC-NSS 已经优于 Root-MUSIC、最佳固定尺度 FBSS Root-MUSIC 或外部深度学习基线。

- 实现分支：`codex/pcnss-foundation`
- 远端仓库：`https://github.com/ProLin520/-PC-NSS.git`
- 模型参数量：`46,916`
- Agent 验证解释器：`D:\Python\Python\python.exe`
- 验证环境：Python `3.14.0`、PyTorch `2.9.1+cpu`、NumPy `2.4.0`、SciPy `1.16.3`
- locked test：未访问，基础阶段也没有对应运行入口
- 正式训练：未由 Agent 运行

## 已完成范围

| Task | 内容 | 提交 |
| --- | --- | --- |
| 1 | 冻结实验协议与项目骨架 | `c55631c` |
| 2 | 确定性相干双源仿真器 | `36bd504` |
| 3 | 多尺度 FBSS lag 视图 | `2eebf6e` |
| 4 | Hermitian/Toeplitz/PSD/trace 结构投影 | `0445060` |
| 5 | 固定 Root-MUSIC 与传统基线 | `cbf310f` |
| 6 | failure-aware DOA 评价 | `a7c845e` |
| 7 | 有界多尺度 PC-NSS 模型 | `ef98ca2` |
| 8 | 分辨率感知两阶段损失 | `5f4d8ce` |
| 9 | 带配置、seed、代码和 checkpoint SHA 审计的训练引擎 | `c5a05e6` |
| 10 | 统一估计器评估与七类报告 | `5d318db` |
| 11 | 安全 dry-run、4 样本 smoke 和正式入口 | `22cbe26` |

各 Task 均按测试先行方式完成：先观察针对缺失实现或预期缺陷的 RED，再完成 GREEN；发现的结构投影末步破坏 PSD、模型反向传播原地修改、基线公共包装缺失等问题都在对应 Task 内修正并回归。

## 最终工程验证

Task 12 收尾时重新执行以下检查：

- `python -m unittest discover -s test_multisource -v`：`59/59` 通过；
- `python -m compileall multisource_doa scripts test_multisource`：通过；
- 默认 `dry_run`：物理链输出有限、结构投影后最小特征值为正、未创建正式输出；
- 独立 4 样本 smoke：完成 1 epoch 前向、反向、验证和临时报告，未写正式 `best.pt`；
- Git diff/输出审计：没有提交 `outputs/`、checkpoint、权重、生成数据或密钥。

4 样本 smoke 的 failure-aware RMSPE、分辨率等数值只用于证明链路能运行。样本极少且模型仅训练一个 epoch，严禁把这些数值当作论文性能或失败结论。

## 结构与安全审查

- 网络输入是 `L={4,5,6,7}` 的原始/FBSS lag 及有效性、计数和质量特征；不直接读取真实 DOA、SNR 或数据域标签。
- 网络只输出逐 lag 融合置信、受限复残差与受限对角加载；最终 DOA 仍由固定 Root-MUSIC 给出。
- 输出协方差经过显式 Hermitian、Toeplitz、PSD 和 trace 结构处理。
- 评估保留失败样本，并按固定 `60°` 惩罚进入 failure-aware RMSPE。
- validation、development 与 locked test 使用独立 split seed；locked test 在冻结前禁止访问。
- SubspaceNet、DA-MUSIC、DeepMUSIC 当前只记录为 `not_integrated`，没有伪造结果，也没有形成优越性主张。

## 尚未得到的研究结论

- 尚未运行 40,000 样本正式训练，不能判断模型是否收敛或是否存在尺度塌缩。
- 尚未证明 PC-NSS 同时超过原始 Root-MUSIC 与最佳固定尺度 FBSS Root-MUSIC。
- 尚未证明近间隔分辨率提高且 failure count 不增加。
- 尚未开展 seed 2027/2028、locked test 或与 SubspaceNet、DA-MUSIC、DeepMUSIC 的公平复现比较。
- 没有证据支持“首次”“首个”“理论保证”或“优于外部方法”等表述。

下一步由用户严格按 `formal_training_protocol.md` 运行首轮正式训练和 validation；只有达到稳健论文门槛后，才审批 development、多 seed 和最终 locked test。
