# PC-NSS Train-only Failure-aware 固定尺度角误差 Teacher 设计

日期：2026-08-10
状态：交互设计已确认，待用户审核书面规格
前置版本：`master/origin/master=54a9fc2`
前置结论：Task 16 `ranking_invalid`

## 1. 一句话目标

保持数据、模型、训练过程、全部损失公式与验收协议不变，只把尺度 KL 蒸馏目标从排序无效的物理 teacher 概率替换为仅由 train split 固定 FBSS 角误差生成的 failure-aware、并列感知硬标签，检验它能否修复 `[2,4)` 近间隔分辨退化而不损害总体性能。

## 2. 诊断证据与设计动机

Task 16 已完成并合并到 `54a9fc2`。冻结诊断结论为：

- 当前物理 teacher 排序一致率为 `53.36%`；
- `q_midpoint` 排序一致率为 `55.00%`；
- 两者均未达到预注册门槛，最终判定为 `ranking_invalid`；
- 因此禁止继续降低 `tau_scale`，也不在 validation 上微调当前物理 score 公式；
- 下一步只允许预注册 train-only、failure-aware 固定尺度角误差 teacher。

Task 16 诊断实现提交为 `aca4238f24abbce0b968ca8b0743fd6066b2d481`，合并提交为 `54a9fc2`。正式 Task 16 输出按项目纪律未提交 Git，当前工作区没有其八文件目录，因此本文不伪造输出文件 SHA。正式训练前的单因素身份审计必须接收用户保留的 Task 16 八文件目录，计算并记录每个文件 SHA-256；缺失时拒绝启动训练。该身份检查属于训练前科研溯源，不得接入 train-only cache 生成链，也不得让 validation 信息参与标签计算。

## 3. 可证伪假设

当前尺度权重接近均匀的主要原因之一，是尺度 KL 蒸馏使用的物理 teacher 对 L4–L7 的样本级角误差排序信息不足。若只把 KL 目标换成 train split 上由固定 FBSS + Root-MUSIC 的 failure-aware 角误差直接确定的最佳尺度标签，则 seed 2026 的 `[2,4)` resolution 应恢复并至少达到固定 FBSS `L=7`，同时总体 RMSPE、总体 resolution 和 failure count 不退化。

如果一次冻结训练和一次冻结 validation 评价后任一硬门失败，该假设在当前实现下判为失败；不得在同一 validation 上调整标签形式、损失权重、温度、阈值或其他因素。

## 4. 唯一改变量与冻结项

### 4.1 唯一改变量

唯一改变量是尺度 KL 蒸馏目标的来源：

```text
physical teacher scale_probabilities
  -> train-only failure-aware angular-error scale_distillation_target
```

新目标只通过 `scale_distillation_target` 进入现有 KL 尺度损失。

### 4.2 保持不变

以下项目全部保持原 seed 2026 正式协议不变：

- 40,000 train、5,000 validation 的数据配置、split seed 和样本顺序；
- model seed、初始化方式、网络结构、宽度、参数量和 L4–L7 子阵集合；
- batch size `128`、学习率 `1e-3`、优化器和总计 `50` epochs；
- 两阶段训练时点和 lag、scale、residual、peak、dominance 的公式及权重；
- validation 频率及按 validation failure-aware RMSPE 选择 `best.pt` 的规则；
- 结构投影、Root-MUSIC、角度匹配和失败处理；
- 成功分辨的 `1°` 角误差与 `50%` 间隔双条件；
- 缺失或失败角度的 `60°` 罚值；
- 评价 batch size `128` 和当前统一 evaluator；
- 不访问 development/locked test，除非后续另行批准。

不增加近间隔采样、损失加权或间隔撑开奖励。`[2,4)` 只作为主评价分层，不改变训练样本权重。

## 5. 关键职责边界

当前 `ScaleTeacher` 同时提供：

1. `scale_probabilities`，用于 KL 尺度蒸馏；
2. `scale_scores.max()`，用于 `dominance_loss` 的物理分辨 score 基准。

角误差 RMSPE 的单位为度，不能替换 MUSIC denominator 构成的物理 score。否则 KL 目标和 dominance 基准会同时改变，成为双因素实验，并产生量纲错误。

因此采用最小接口方案：

```python
pcnss_loss(
    output,
    physical_teacher,
    ...,
    scale_distillation_target=None,
)
```

- `scale_distillation_target is None`：使用 `physical_teacher.scale_probabilities`，与原路径逐项一致；
- 新实验：传入缓存角误差标签，只替换 KL 输入；
- `dominance_loss` 始终使用 `physical_teacher.scale_scores.max()`；
- peak、lag、residual 和其他诊断量不受新标签影响。

不复制后再篡改 `ScaleTeacher` 对象，也不新增组合 teacher 大重构。

## 6. 整体数据流

```text
40,000 个冻结 train 样本
  -> 固定 FBSS L4/L5/L6/L7
  -> 固定 Root-MUSIC
  -> 与真实双角最佳排列匹配
  -> failure-aware sample RMSPE
  -> 并列感知硬标签
  -> 不可覆盖的 train-only cache
  -> 训练启动前完整认证
  -> 每个 batch 按 sample_seed 查找标签
  -> scale_distillation_target
  -> 仅进入现有 KL 尺度损失

同一训练 batch
  -> 现有 physical teacher score
  -> dominance_loss（保持不变）
```

缓存生成不加载 PC-NSS checkpoint、不实例化模型、不计算梯度、不训练，也不读取 validation、development 或 locked test。

## 7. 角误差 Teacher 数学定义

对每个 train 样本及每个 `L in {4,5,6,7}`，复用正式评价链计算：

```text
e_L = failure_aware_sample_rmspe(true_angles, fixed_FBSS_L_Root_MUSIC_estimate)
```

其中：

- 两个估计角与两个真实角先做最佳排列匹配；
- 非有限、重复、缺失或 Root-MUSIC 失败按现有评价函数处理；
- 失败角保留固定 `60°` 罚值；
- 不删除失败样本；
- 不加入间隔撑开分数、MUSIC midpoint 或其他代理量。

定义：

```text
e_min = min(e_4, e_5, e_6, e_7)
M = {L | e_L - e_min <= 1e-6 deg}
p_L = 1 / |M|,  L in M
p_L = 0,        L not in M
```

结果规则：

- 唯一最优尺度得到 one-hot 标签；
- `1e-6°` 内并列最优的尺度均分概率；
- 四尺度全部失败且均为 `60°` 时得到 `[0.25,0.25,0.25,0.25]`；
- 不引入 `tau_error`、softmax 温度、rank 权重或误差差值超参数；
- 保留现有 KL 实现中的固定数值下限，不为硬标签另改损失公式。

## 8. Train-only Cache

### 8.1 安全入口

新增独立入口：

```text
scripts/build_pcnss_failure_aware_teacher_cache.py
```

默认配置：

```text
stage=dry_run
dry_run=true
split=train
device=cpu
sample_count=1
batch_size=128
overwrite=false
```

允许阶段固定为：

- `dry_run`：重建 1 个 train 样本并在内存验证，不创建输出；
- `smoke`：固定 4 个 train 样本，写隔离 smoke 目录；
- `build_train_teacher_cache`：固定 train、CPU、40,000 样本、batch 128。

正式路径拒绝未知配置键、非 train split、非 CPU、非 40,000、非 128、`allow_locked_test=true`、已有输出目录和 `overwrite=true`。

### 8.2 固定输出

正式缓存目录恰好包含：

1. `teacher_cache_config.json`；
2. `teacher_cache_manifest.json`；
3. `train_teacher_labels.csv`。

JSON 禁止 NaN/Infinity。CSV 中失败估计角使用空字段并配套明确的 success/failure reason；RMSPE、匹配误差和 teacher 概率仍必须是可审计的有限值。

### 8.3 样本 CSV

CSV 按 sample seed 严格升序保存 40,000 行，至少包含：

- `sample_index`、`sample_seed`、两个真实角、separation、rho、SNR、snapshot；
- 每个 L 的 success、failure reason、两个估计角；
- 每个 L 的两个匹配误差和 failure-aware sample RMSPE；
- `teacher_p_L4`、`teacher_p_L5`、`teacher_p_L6`、`teacher_p_L7`；
- 最优尺度集合、是否并列、是否四尺度全失败。

### 8.4 Manifest 与认证

Manifest 固定记录：

- schema/算法版本；
- Git SHA 和关键源文件 SHA-256；
- 完整 ExperimentConfig；
- train split seed、seed 起止范围和样本数；
- Root-MUSIC、匹配、`60°` 罚值和 `1e-6°` tie 定义；
- CPU、batch size、运行时长；
- config JSON 与 CSV 的 SHA-256；
- `train_only=true`、`no_model_forward=true`、`training_performed=false`；
- `validation_accessed=false`、`development_accessed=false`、`locked_test_accessed=false`。

训练加载器在模型和优化器创建前验证：

- 恰好 40,000 个唯一、连续且升序的 train seed；
- seed 范围与冻结 train split 完全一致，不含其他 split；
- 由 train 数据确定性重建的角度、rho、SNR、snapshot、separation 逐项一致；
- L4–L7 行字段完整，失败状态与有限性合法；
- teacher 概率非负、有限且和在 `1e-6` 绝对容差内为 1；
- 最优集合和 teacher 概率可从四个 RMSPE 独立复算；
- config、CSV、代码和数据身份 SHA 一致；
- 输出目录和三个文件没有额外或缺失文件。

任一项失败均显式报错，不静默跳样本或退回物理 teacher。

## 9. 训练接入

正式训练配置新增两个显式字段：

```text
teacher_mode=physical | failure_aware_error
teacher_cache_path=
```

- 默认 `teacher_mode=physical` 且 cache path 为空，保证 PyCharm 无参数运行和既有流程安全；
- `failure_aware_error` 必须提供已认证 cache；
- `physical` 模式若提供 cache path 则拒绝，避免配置歧义；
- 正式新实验中，batch 的每个 sample seed 必须且只能命中一个缓存标签；
- 标签读取后转换为与模型分布相同的 device/dtype，但不进入模型输入、不计算梯度；
- 原物理 teacher 仍按现有代码计算，只有其 probabilities 在新模式下不作为 KL 目标；
- 训练不得在线重新运行 Root-MUSIC 标签，也不得修改缓存。

训练 manifest 额外记录：

- `teacher_mode`；
- cache manifest/config/CSV SHA；
- `scale_distillation_target_source=train_only_failure_aware_rmspe`；
- `dominance_target_source=physical_music_score`；
- 唯一最优、并列、全失败及 L4–L7 主导标签数量；
- 单因素身份审计文件 SHA。

## 10. 单因素对照认证

第一轮只运行 model seed 2026。原 epoch 35 checkpoint 可以作为 A 组，但必须先生成不可覆盖的 `single_factor_audit.json`，比较并记录：

- train/validation split、数量、seed 和数据配置；
- model seed、初始化和参数量；
- batch size、样本顺序及随机数设置；
- 优化器、学习率、epochs；
- 两阶段损失时点、公式和全部权重；
- checkpoint 选择规则和 validation 频率；
- 投影、Root-MUSIC、评价 batch 和计分协议；
- baseline checkpoint、train/validation manifest、当前评价报告 SHA；
- Task 16 八个正式诊断文件 SHA；
- 新 cache 三文件 SHA；
- 允许的代码差异只涉及本规格批准的 cache、认证和可选 KL 目标接入。

同时以自动回归测试证明 `teacher_mode=physical` 时 loss breakdown、梯度和训练步与旧接口在冻结容差内一致。

判定规则：

- 身份全部可证明：复用原 epoch 35、seed 2026 结果，只训练一次 B 组；
- 任一训练相关身份不一致或缺少证据：在启动 B 组前先按当前冻结环境重新运行物理 teacher A 组；
- 不允许先看 B 组 validation 再决定是否补跑 A 组。

A/B 必须使用相同 seed、初始化、batch 顺序、设备和软件环境。若旧 baseline manifest 无法证明这些身份，则自动进入重跑 A 组规则。

## 11. 冻结验收门

所有门使用来源 CSV/JSON 的精确计数与未舍入值，不用报告中显示的两位小数直接判定。B 组必须同时满足：

1. `[2,4)` resolution rate 严格高于原 epoch 35 PC-NSS；
2. `[2,4)` resolution rate 不低于固定 FBSS `L=7`；
3. 总体 failure-aware RMSPE 不高于原 epoch 35 PC-NSS；
4. 总体 resolution rate 不低于原 epoch 35 PC-NSS；
5. failure count 不超过原模型；当前对照为 0，因此 B 组也必须为 0；
6. 样本集合、评价算法、`1°/50%` 分辨定义和 `60°` 罚值身份完全一致。

当前已知显示值仅用于人工理解：原 PC-NSS `[2,4)` resolution 约 `3.86%`，L7 约 `6.77%`，原 PC-NSS 总体 RMSPE 约 `7.264°`、总体 resolution 约 `16.20%`、failure count 为 0。实际 gate 必须从已认证来源文件重算。

## 12. 固定报告与停止规则

预注册评价除硬门外还报告：

- B 对 A、B 对 L7 的逐样本 RMSPE win/tie/loss；
- `[2,4)` resolved/unresolved 状态转移；
- 配对统计和置信区间，但不作为 seed 2026 的额外硬门；
- separation、rho、SNR、snapshot 固定分层；
- `>10°`、`>30°`、`>60°` 离群数量与比例；
- teacher 主导尺度、并列和全失败标签数量；
- train/validation KL、尺度熵及原有训练诊断量；
- failure reason、最大误差及没有删样本的完整性证明。

停止规则：

- 任一硬门失败：结论固定为 `experiment_failed`，停止扩 seed；不访问 development/locked test，不在同一 validation 上调整标签、温度、损失、阈值或第二个因素；
- 全部门通过：只记为 `seed2026_gate_passed`，不直接声称路线成立；
- 是否运行 development 必须另行批准；
- development 通过后才单独审批 seed 2027/2028；
- locked test 保持冻结，直到模型、损失和验收阈值最终冻结并获得用户单独批准。

## 13. TDD 与验证

实现严格按 RED -> GREEN -> 重构，至少覆盖：

- failure-aware RMSPE、排列匹配、`60°` 罚值和 `1e-6°` tie 边界；
- 唯一最优、并列最优、部分失败、四尺度全失败；
- 40,000 train seed 连续性、唯一性、排序和 split 隔离；
- cache 重建元数据、SHA、schema、有限性、篡改、重复、缺失和拒绝覆盖；
- default dry-run 不创建输出、不读正式文件；
- development/locked/未知键/非 CPU/错误数量/错误 batch 被拒绝；
- `teacher_mode=physical` 的 loss 与梯度回归一致；
- 新目标只改变 scale KL，dominance、peak、lag、residual 输入及公式不变；
- batch sample seed 与 cache 标签一一对应，缺失或多余立即失败；
- 单因素审计失败时训练在模型/优化器创建前终止；
- manifest、三文件 cache 和不可覆盖契约；
- 目标 unittest、完整 unittest、`compileall`、默认 dry-run、4 样本 cache smoke 和 4 样本单 batch 训练 smoke。

不新增依赖。

## 14. Agent 与用户运行分工

Agent 负责：

- 设计、实施计划、TDD 实现和代码审查；
- 目标/完整 unittest、`compileall`、默认 dry-run；
- 4 样本 cache smoke 和 4 样本单 batch 训练 smoke；
- 审计 Git diff，提交并推送范围明确的 `codex/` 分支；
- 不生成正式 40,000 标签，不运行正式模型训练，不访问 development/locked test。

用户审核实现后依次负责：

1. 运行正式 40,000 train-only cache；
2. 审核 cache 独立复算与完整性报告；
3. 运行并审核单因素身份审计；
4. 运行一次 seed 2026 正式训练；
5. 运行一次冻结 validation 评价；
6. 根据冻结 gate 决定是否另行批准 development。

## 15. Git 与产物边界

- 规格、实施计划、源码、测试、配置模板和协议文档可以提交；
- teacher cache、single-factor audit 运行产物、outputs、checkpoint、权重、预测和临时正式配置不得提交；
- 实施优先在独立 `codex/` 分支和 worktree 完成；
- 不使用整目录无审查暂存，不 force push，不覆盖历史输出；
- 当前规格提交只表示设计边界已书面化，不授权实现或正式训练；用户审核本文件后才进入 implementation plan。

## 16. 已接受的限制

- 硬 teacher 只保留最佳尺度身份，不利用角误差差值大小；这是为避免新温度或权重搜索而接受的限制；
- train-only oracle 标签增加一次性离线计算和缓存成本，但避免每个 epoch 重复运行四尺度 Root-MUSIC；
- 物理 score 继续服务 dominance，因此本实验只检验尺度 KL 标签来源，不检验 dominance 物理代理；
- 单 seed 通过只允许进入下一审批阶段，不能替代多 seed、development 或最终 locked test 证据。
