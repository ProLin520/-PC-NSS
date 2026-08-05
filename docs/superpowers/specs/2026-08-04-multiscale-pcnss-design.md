# MultiSource_DOA 多尺度 PC-NSS 设计规格

日期：2026-08-04
状态：已批准，进入基础框架实施；正式训练与 locked test 仍需分阶段审批
项目：`MultiSource_DOA`

## 1. 研究目标与边界

本项目研究少快拍、强相关或完全相干、近间隔双源条件下的 DOA 分辨。目标不是用神经网络替代传统估计器，而是学习传统空间平滑中固定的尺度和协方差构造规则，再使用固定 Root-MUSIC 输出最终角度。

首阶段成功标准是稳健论文门槛：多尺度 PC-NSS 在锁定主场景上必须同时优于原始 Root-MUSIC 和最佳固定尺度 FBSS Root-MUSIC，提高双源成功分辨率，且不增加失败。达到该门槛不等于已达到二区期刊投稿要求；投稿阶段还需要与 SubspaceNet、DA-MUSIC、DeepMUSIC 形成性能、复杂度或可解释性上的明确优势。

本项目与 `DIO_DOA` 独立。`DIO_DOA` 冻结保存单源 PALR、门控失败与负结果；本项目不直接导入其 Python 包，也不覆盖其输出。经过验证的仿真或物理代码只能逐模块迁移，并重新建立测试与来源记录。

## 2. 科学问题

传统前后向空间平滑可以恢复相干源条件下的协方差秩，但存在结构性权衡：较短子阵提供更多平滑样本、去相关更强，却损失有效孔径和近间隔分辨率；较长子阵保留孔径，却只有较少重叠子阵，有限快拍方差更大。

本项目检验以下命题：

> 在保持协方差物理可行性和固定 Root-MUSIC 后端的条件下，数据自适应地融合多个 FBSS 子阵尺度，能否改善相干近间隔双源的分辨率，并优于最佳固定尺度空间平滑？

该命题只在锁定实验协议和统一失败计分下成立或失败，不推广为所有阵列、信号或 SNR 条件的普遍结论。

## 3. 数据协议

### 3.1 阵列与信号

- 阵列：8 阵元 ULA，阵元间距为半波长。
- 信源数：固定 2 个，训练和主评价均假设源数已知。
- 信号：窄带远场复高斯源，不再比较 BPSK/Gaussian 数据域。
- 角度中心：均匀分布在 `[-50, 50]` 度。
- 双源间隔：均匀覆盖 `2-10` 度，并保证两个角度均在合法范围内。
- 相关信号生成：`s2 = rho * exp(j*phi) * s1 + sqrt(1-rho^2) * u2`，其中 `phi` 均匀随机，`u2` 与 `s1` 独立。
- 训练相关度：分层取 `rho in {0.8, 0.9, 0.99, 1.0}`。
- 训练 SNR：覆盖 `[-5, 10] dB`。
- 训练快拍数：分层取 `{8, 20, 50}`。

主验收场景固定为 `rho=1.0`、`T=20`、`SNR=5 dB`。`rho=0` 独立源只作额外泛化检查，不参与主门槛。

### 3.2 数据划分

| 划分 | 数量 | 用途 |
|---|---:|---|
| train | 40,000 | 更新模型参数 |
| validation | 5,000 | checkpoint 选择和一次性结构决策 |
| development test | 5,000 | 冻结前诊断，不作为最终论文结果 |
| locked test | 10,000 | 设计、损失和阈值冻结后的最终评价 |

每个划分使用独立 seed、独立样本索引和可审计 manifest。`locked test` 在模型、损失、阈值和停止规则冻结前不可访问。

### 3.3 最终扫描轴

- 间隔扫描：`delta in {2, 4, 6, 8, 10}` 度，固定 `rho=1.0, T=20, SNR=5 dB`。
- SNR 扫描：`{-10, -5, 0, 5, 10} dB`，固定 `rho=1.0, T=20, delta=4` 度。
- 快拍扫描：`T in {2, 5, 10, 20, 50, 100}`，固定 `rho=1.0, SNR=5 dB, delta=4` 度。
- 相关度检查：`rho=0.9`。
- 独立源泛化：`rho=0`，只报告，不参与主验收。

## 4. 算法架构

算法暂称 Multi-Scale Physics-Constrained Neural Spatial Smoothing，简称多尺度 PC-NSS。

```text
X in C^(8 x T)
  -> raw SCM and FBSS views for L={4,5,6,7}
  -> lag extraction, padding, validity masks, quality features
  -> shared small encoder
  -> per-lag multi-scale confidence weights
  -> bounded complex lag residual
  -> full-aperture lag reconstruction
  -> Hermitian/Toeplitz/PSD/trace projection
  -> fixed Root-MUSIC with K=2
  -> two DOAs
```

### 4.1 多尺度物理视图

对 `L={4,5,6,7}` 构造所有重叠子阵的 SCM，并执行前后向平均和空间平滑。每个 `L` 生成一个 FBSS 协方差视图。原始 8 阵元 SCM 作为完整孔径辅助视图，为高阶 lag 提供观测，不作为学习结果。

每个复数协方差视图转换为按对角线平均的 lag 表示；不存在的高阶 lag 用零填充并附加显式有效掩码，禁止网络把“缺失”误认为真实零值。

质量特征只来自观测本身，包括归一化 trace、特征值比、条件数的稳定变换、子阵间差异和每个 lag 的有效样本数。真实角度、SNR、快拍数、相关度和任何测试标签都不进入模型输入。

### 4.2 神经融合

一个共享的小型实值编码器处理各尺度 lag 和协方差视图。网络输出：

1. 每个 lag 上各有效尺度的置信 logits，经 masked softmax 变为非负且和为 1 的融合权重；
2. 一个有界复数 lag 残差，修正幅度按样本 trace 缩放并设硬上限；
3. 一个有界非负对角加载量。

模型不输出 DOA、空间谱、源数、SNR 或第二套全局候选。第一版参数预算上限为 80,000，目标自然落在 50,000 左右；参数预算用于限制黑箱容量，不要求人为凑足参数。

### 4.3 结构投影

融合后的 lag 构造成完整孔径 Hermitian Toeplitz 候选矩阵。随后执行固定结构投影：

- Hermitian 对称化；
- Toeplitz 对角一致化；
- PSD 投影；
- 最小特征值下限；
- trace 归一化。

若一次 PSD 投影破坏 Toeplitz 性，推理阶段使用固定容差的 Dykstra/交替投影收敛到 Hermitian Toeplitz 与 PSD 的交集；训练阶段使用固定次数的可微近似，并对最终结构残差施加审计。实现测试必须验证 Hermitian、Toeplitz、PSD 和 trace 误差均不超过规格容差。

### 4.4 固定 Root-MUSIC

结构投影后的矩阵进入固定 Root-MUSIC，源数固定为 2。网络不修改阵列流形、求根多项式、单位圆根筛选或角度映射。所有根筛选规则在实现前通过独立物理测试锁定。

## 5. 训练目标

CR-UNet 已经采用理想协方差 MSE、Toeplitz 对角一致性和噪声子空间对齐的复合损失。因此，本项目不把这三项的简单加权组合作为损失创新，也不训练一个从损坏 SCM 到理想完整协方差的通用映射。

PC-NSS 的核心监督来自同一样本下多个传统 FBSS 尺度的可分辨性差异。生成器中的真实角度只在训练阶段计算物理教师分数，推理时不进入网络。

### 5.1 多尺度分辨率教师

对每个固定尺度 `L in {4,5,6,7}`，从其 FBSS 协方差得到归一化 MUSIC 分母 `q_L(theta)`。对真实角度 `theta1, theta2`、中点 `theta_mid` 和固定 guard 位置，定义尺度分辨分数：

`g_L = q_L(theta_mid) - 0.5 * [q_L(theta1) + q_L(theta2)]`。

其中 `q_L(theta)=a(theta)^H Pn,L a(theta)/||a(theta)||^2`，数值位于 `[0,1]`。分数越大，表示真实角度处的投影能量更低、中点抑制更强，双峰越容易分开。训练时使用 `tau_scale=0.1`，令 `pi_scale=softmax(g_L/tau_scale)`。该教师只比较同一样本的传统物理尺度，不使用神经网络输出或测试集统计。

### 5.2 逐 lag 尺度蒸馏损失

网络为每个 lag 输出 masked scale weights。先按有效 lag 和可靠样本数聚合为样本级尺度分布 `w_scale`，再使用 `L_scale = KL(pi_scale || w_scale)` 监督网络学习何时相信短子阵的强去相关证据、何时相信长子阵的孔径证据。

逐 lag 权重仍允许同一样本的不同 lag 选择不同尺度；样本级蒸馏只提供方向性约束，不把网络退化为硬尺度分类器。

### 5.3 分辨优势损失

对结构投影后的预测协方差计算同样的分辨分数 `g_pred`。定义：

- `L_peak`：令 `q1,q2` 为两个真实角度的归一化 MUSIC 分母，`G` 包含双源中点以及两侧固定 guard 位置；使用 `0.5*(q1+q2) + mean_z relu(0.05 + max(q1,q2) - q(z))`，防止单峰塌缩；
- `L_dom`：令 `g_best=max_L g_L`，使用 `0.1*log(1+exp((g_best-g_pred)/0.1))` 作为平滑 dominance 惩罚。

`L_dom` 是相对传统物理尺度的训练目标，不代表逐样本理论保证。第一版不通过可微 Root-MUSIC 根反向传播，避免与 SubspaceNet 的训练链路重合并降低重根附近梯度不稳定风险；固定 Root-MUSIC 只用于 validation 选模和推理。

### 5.4 lag 重建与两阶段训练

生成器可构造去除相干交叉项的目标 lag：

`R_target = A(theta) diag(p1, p2) A(theta)^H + sigma^2 I`。

`L_lag` 只在归一化 lag 向量上使用 Smooth L1，作为训练稳定器，不承担完整协方差重建创新。Toeplitz、Hermitian 和 PSD 主要由输出参数化与结构投影硬实现，而不是复用 CR-UNet 的 Toeplitz variance loss。

- 残差幅度已有硬上限，再使用轻量 `L_residual` 抑制无必要修正。
- 前 10 epoch：`1.0*L_lag + 0.5*L_scale + 0.01*L_residual`。
- 后 40 epoch：`1.0*L_lag + 0.5*L_scale + 0.01*L_residual + 1.0*L_peak + 0.5*L_dom`。
- 上述温度、margin 和权重是第一轮锁定值，不围绕同一 validation 搜索。
- checkpoint 只按 validation failure-aware RMSPE 选择，所有其他指标同时记录但不参与选择。

## 6. 评价、匹配与失败

### 6.1 排列不变匹配

两个估计角度与两个真实角度使用 Hungarian 最小代价匹配。排序匹配只作为诊断，不作为正式计分。

### 6.2 失败处理

以下情况均标记为失败：输入或输出非有限、结构投影不收敛、Root-MUSIC 缺少两个有效根、重复根、角度映射越界。失败样本不得删除或用真实角度修复。

每个缺失角度使用固定 `60` 度罚值进入 failure-aware RMSPE。条件 RMSE 可以同时报告，但不能代替 failure-aware 主指标。

### 6.3 成功分辨定义

样本只有同时满足以下条件才算成功分辨：

- 返回两个有限且不同的角度；
- Hungarian 匹配后两个绝对误差均不超过 `1` 度；
- 估计角度间隔不小于真实间隔的 50%。

### 6.4 报告指标

- failure-aware RMSPE；
- 条件 RMSE、MAE、p95、p99、最大误差；
- 双源成功分辨率；
- 失败数量和失败原因；
- 按间隔、SNR、快拍数、相关度分层的 paired win/tie/loss；
- 参数量、单样本运行时间、峰值显存和主要算子复杂度。

## 7. 基线与验收

第一阶段统一实现和计分：MUSIC、Root-MUSIC、ESPRIT、固定尺度 SPS Root-MUSIC、固定尺度 FBSS Root-MUSIC、最佳固定尺度 FBSS Root-MUSIC、多尺度 PC-NSS Root-MUSIC。

投稿比较阶段增加 SubspaceNet、DA-MUSIC 和 DeepMUSIC。优先使用作者公开代码；如果无法按同一阵列流形、数据和计分协议复现，必须记录版本、修改范围和失败原因，禁止把自行复刻结果冒充官方结果。

### 7.1 稳健论文门槛

在主场景和三个独立训练 seed 上：

- PC-NSS failure-aware RMSPE 严格低于原始 Root-MUSIC；
- PC-NSS failure-aware RMSPE 严格低于最佳固定尺度 FBSS Root-MUSIC；
- 成功分辨率严格高于最佳固定尺度 FBSS；
- 失败数量不得增加；
- `delta=2-4` 度子集不能退化；
- 三个 seed 改善方向一致。

第一 seed 失败时先执行不调参机理诊断，检查尺度权重塌缩、lag 残差分布、结构投影改变量、预测子空间与目标子空间夹角、双峰中点 margin 和训练/验证差异。没有新设计审批，不围绕相同 validation 连续搜索网络宽度、损失权重或子阵集合。

## 8. 与紧密相关工作的区别

### 8.1 CNN with Toeplitz prior（Wu 等，IEEE SPL 2022）

该方法从样本协方差预测理想无噪 Toeplitz 协方差的首行，再用 Root-MUSIC，并以协方差向量平方误差训练。PC-NSS 不从单个 SCM 自由重建目标首行，而是以多尺度 FBSS 物理估计为显式视图，学习逐 lag 可信融合和受限残差；训练目标同时包含去相关协方差、子空间和近间隔分辨约束。

### 8.2 Neural covariance reconstruction（Barthelme 与 Utschick，IEEE SPL 2021）

该方法解决子阵采样下从子阵 SCM 恢复完整阵列协方差、进而估计多于射频链数量的信源。PC-NSS 不研究射频链缺失或欠采样阵列，而研究完整 8 阵元 ULA 上不同空间平滑尺度之间的去相干-孔径-方差权衡。

### 8.3 SubspaceNet（Shmuel 等，IEEE TVT 2025）

SubspaceNet 学习通用代理协方差，并通过可微 Root-MUSIC 的角度误差训练以适应相干、宽带、低 SNR、少快拍和阵列失配。PC-NSS 的范围更窄：固定窄带 ULA、已知双源；其结构差异在多尺度 FBSS 物理视图、masked lag 融合、受限残差和显式 Toeplitz/PSD 投影。

### 8.4 TransDOA（Zhou 等，arXiv:2504.13394）

TransDOA 将 SCM 分块送入 Transformer，直接回归多源 DOA，使用排列不变角度损失；阵列误差适配阶段使用 MSE 和余弦特征对齐。PC-NSS 不直接回归角度，不进行源域/目标域迁移，最终输出只来自固定 Root-MUSIC。

### 8.5 CR-UNet（Chen 等，厦门大学等）

CR-UNet 面向部分校准阵列的未知阵元增益和相位误差，使用 Complex U-Net 把损坏 SCM 直接映射为无校准误差的理想完整协方差；ULA 后端使用 MUSIC/ML，稀疏阵列后端使用 ANM+MUSIC。其复合损失由完整协方差 Frobenius MSE、Toeplitz 对角方差和噪声子空间相似度组成。

PC-NSS 不研究阵元校准误差，不使用 U-Net 或完整协方差到完整协方差的自由映射。它研究相干双源下不同 FBSS 尺度的去相关-孔径-方差权衡，网络只输出逐 lag 尺度置信和受限残差。训练核心是由多个固定 FBSS 视图产生的尺度分辨率教师、逐 lag 尺度蒸馏及相对最佳固定尺度的分辨优势损失；Toeplitz/PSD 通过硬结构投影实现。普通协方差 MSE、Toeplitz loss 和 subspace loss 不作为本项目的核心贡献。

### 8.6 Unit-nuclear-norm structured covariance prior（Pan 等，IEEE Sensors Journal 2026）

该方法面向单快拍，输出固定数量的 DOA/SNR 表示并据此构造 Toeplitz、正定、单位核范数协方差，使用白化拟合误差和投影乘法对称变换设计损失。PC-NSS 面向多快拍相干双源，不预测 DOA/SNR 后重建协方差，不复用单位核范数输出层或其损失；创新焦点是多尺度 FBSS 融合与近间隔分辨。

上述差异是结构和任务边界，不支持“首次”主张。正式论文写作前仍需执行系统检索。

## 9. 软件结构

```text
MultiSource_DOA/
  AGENTS.md
  README.md
  requirements.txt
  docs/superpowers/specs/
  docs/superpowers/plans/
  experiments/
  multisource_doa/
    data/
    physics/
    models/
    training/
    evaluation/
    baselines/
  scripts/
  test_multisource/
  outputs/                 # gitignored
```

模块边界如下：

- `data`：确定性双源仿真、split manifest 和样本审计；
- `physics`：SCM、SPS/FBSS、多尺度 lag、结构投影、Root-MUSIC；
- `models`：只包含 PC-NSS 网络与张量接口；
- `training`：损失、两阶段训练、checkpoint 和诊断；
- `evaluation`：Hungarian 匹配、失败计分、分辨率和分层统计；
- `baselines`：传统算法和外部深度基线适配；
- `scripts`：用户运行入口，顶部 `RUN_CONFIG` 默认 dry-run；
- `test_multisource`：独立的标准库 `unittest` 测试。

## 10. 测试与验证

按 TDD 逐任务执行：

1. steering convention、角度边界和相关信号生成；
2. SCM、FBSS 尺度、lag mask 和传统基线一致性；
3. Hermitian、Toeplitz、PSD、trace 投影不变量；
4. Root-MUSIC 正相位流形和双源根筛选；
5. masked softmax、残差上限和无标签泄漏；
6. 多尺度教师排序、masked scale 蒸馏、最佳固定尺度 dominance、塌缩单峰与正确双峰的损失相对关系；
7. Hungarian 匹配、60 度失败罚值和成功分辨定义；
8. split seed 不重叠、locked test 防访问和输出防覆盖；
9. `compileall`、默认 dry-run 和极小样本 smoke。

Agent 在实现阶段负责 RED/GREEN、目标单测、`compileall`、dry-run 和 smoke。正式 40,000 样本训练在实现与 smoke 审核后单独启动，不在脚手架创建时自动运行。

## 11. 复现与输出

每次训练和评价必须保存：完整配置、model/data/split seed、样本索引范围、steering convention、代码 SHA、checkpoint SHA、参数量、运行时间、所有失败标志和选模指标。输出目录按实验名和 seed 隔离并拒绝覆盖。

生成数据、权重、大型输出和敏感配置不进入 Git。论文图必须由独立脚本从 CSV/JSON 生成，并使用固定算法颜色表。

## 12. 参考工作

- Wu, Yang, Jia, Tian, “A Gridless DOA Estimation Method Based on Convolutional Neural Network With Toeplitz Prior,” IEEE Signal Processing Letters, 2022, DOI: `10.1109/LSP.2022.3176211`.
- Barthelme and Utschick, “DoA Estimation Using Neural Network-Based Covariance Matrix Reconstruction,” IEEE Signal Processing Letters, 2021, DOI: `10.1109/LSP.2021.3072564`.
- Shmuel et al., “SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation,” IEEE Transactions on Vehicular Technology, 2025, DOI: `10.1109/TVT.2024.3496119`.
- Merkofer et al., “DA-MUSIC: Data-Driven DoA Estimation via Deep Augmented MUSIC Algorithm,” arXiv:`2109.10581`.
- Zhou et al., “TransDOA: Calibrating Array Imperfections via Transformer-based Transfer Learning,” arXiv:`2504.13394`.
- Chen et al., “CR-UNet: Deep Covariance Reconstruction for DOA Estimation with Partially Calibrated Arrays,” manuscript inspected from the user-provided PDF on 2026-08-04.
- Pan, Lin, Yang, “Deep Learning-Based Single-Snapshot DOA Estimation With a Unit-Nuclear-Norm Structured Covariance Prior,” IEEE Sensors Journal, 2026, DOI: `10.1109/JSEN.2026.3650799`.
