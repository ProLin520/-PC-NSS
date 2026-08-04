# Multi-Scale PC-NSS 基础框架实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不运行正式 40,000 样本训练、不访问 locked test、也不要求立即超过 SubspaceNet 等深度基线的前提下，建立可审计、可测试、可由用户直接启动训练的多尺度 PC-NSS 基础框架。

**Architecture:** 先构造确定性的相干双源快拍和多尺度 `L={4,5,6,7}` FBSS 物理视图；网络只学习逐 lag 的尺度置信、受限复残差和受限对角加载；输出经过 Hermitian/Toeplitz/PSD/trace 投影后交给固定 Root-MUSIC。训练核心是固定尺度分辨率教师、尺度蒸馏、近间隔峰谷损失和相对最佳固定尺度的 dominance 损失；Root-MUSIC 不参与反向传播。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、标准库 `unittest`、JSON/CSV；依赖统一由 `requirements.txt` 管理。

## Global Constraints

- 全部命令从 `D:\Python\Project\doa_estimation\MultiSource_DOA` 执行。
- 严格按 RED → GREEN → 重构推进；每个 Task 完成后先审查 diff 和目标测试，再进入下一项。
- 第一阶段只以“基础框架数值正确、接口稳定、审计完整、tiny smoke 可运行”为完成条件。
- 第一轮模型超参数、损失权重、子阵集合固定，不在同一 validation 上搜索。
- 不访问 `locked_test`，不生成其 manifest，不运行正式 40,000 样本训练，不运行三个正式 seed。
- 不要求基础框架阶段超过 SubspaceNet、DA-MUSIC、DeepMUSIC；这些算法只保留后续统一协议适配位置。
- 第一轮正式结果仅与 raw Root-MUSIC 和最佳固定尺度 FBSS Root-MUSIC 执行稳健门槛；失败后先做不调参机理诊断。
- 禁止删除失败估计；缺失角度统一按 `60°` 罚值计入 failure-aware RMSPE。
- 用户常跑入口顶部必须有 `RUN_CONFIG`，默认 `dry_run=True`，拒绝覆盖已有输出。
- 结构化输出记录配置、seed、样本索引、steering convention、代码 SHA、checkpoint SHA、参数量、失败原因与耗时。
- 不得声称“首次”“首个”“理论保证不退化”或“已经优于 SubspaceNet”。

---

## Task 1：固化项目配置、依赖和 locked-test 防线

**Files:**

- Modify: `docs/superpowers/specs/2026-08-04-multiscale-pcnss-design.md`
- Create: `requirements.txt`
- Create: `multisource_doa/__init__.py`
- Create: `multisource_doa/config.py`
- Create: `test_multisource/__init__.py`
- Create: `test_multisource/test_config.py`

- [ ] **Step 1: 将规格状态改为已批准**

```text
状态：已批准，进入基础框架实施；正式训练与 locked test 仍需分阶段审批
```

- [ ] **Step 2: 写配置 RED 测试**

```python
import unittest

from multisource_doa.config import ExperimentConfig, SplitName


class ExperimentConfigTest(unittest.TestCase):
    def test_first_round_protocol_is_frozen(self):
        cfg = ExperimentConfig()
        self.assertEqual(cfg.array.sensor_count, 8)
        self.assertEqual(cfg.data.source_count, 2)
        self.assertEqual(cfg.physics.fbss_subarray_sizes, (4, 5, 6, 7))
        self.assertEqual(cfg.training.stage_one_epochs, 10)
        self.assertEqual(cfg.training.total_epochs, 50)
        self.assertTrue(cfg.runtime.dry_run)

    def test_locked_test_requires_explicit_permission(self):
        cfg = ExperimentConfig()
        with self.assertRaisesRegex(PermissionError, "locked_test"):
            cfg.split.require_access(SplitName.LOCKED_TEST)
        cfg.split.require_access(SplitName.TRAIN)
```

- [ ] **Step 3: 运行 RED**

Run: `python -m unittest test_multisource.test_config -v`

Expected: `ModuleNotFoundError` 或缺少 `ExperimentConfig`，失败原因只来自尚未实现的接口。

- [ ] **Step 4: 实现冻结配置**

`multisource_doa/config.py` 使用不可变 dataclass 和显式枚举：

```python
from dataclasses import dataclass, field
from enum import Enum


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    DEVELOPMENT = "development"
    LOCKED_TEST = "locked_test"


@dataclass(frozen=True)
class ArrayConfig:
    sensor_count: int = 8
    spacing_wavelengths: float = 0.5
    angle_limits_deg: tuple[float, float] = (-60.0, 60.0)


@dataclass(frozen=True)
class DataConfig:
    source_count: int = 2
    train_rhos: tuple[float, ...] = (0.8, 0.9, 0.99, 1.0)
    train_snr_db: tuple[float, float] = (-5.0, 10.0)
    train_snapshot_counts: tuple[int, ...] = (8, 20, 50)
    center_limits_deg: tuple[float, float] = (-50.0, 50.0)
    separation_limits_deg: tuple[float, float] = (2.0, 10.0)


@dataclass(frozen=True)
class PhysicsConfig:
    fbss_subarray_sizes: tuple[int, ...] = (4, 5, 6, 7)
    projection_iterations_train: int = 4
    projection_max_iterations_eval: int = 100
    projection_tolerance: float = 1e-7
    eigenvalue_floor: float = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    stage_one_epochs: int = 10
    total_epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 128
    tau_scale: float = 0.1
    peak_margin: float = 0.05
    residual_fraction: float = 0.10
    loading_fraction: float = 0.05


@dataclass(frozen=True)
class SplitConfig:
    sizes: dict[SplitName, int] = field(default_factory=lambda: {
        SplitName.TRAIN: 40_000,
        SplitName.VALIDATION: 5_000,
        SplitName.DEVELOPMENT: 5_000,
        SplitName.LOCKED_TEST: 10_000,
    })
    seeds: dict[SplitName, int] = field(default_factory=lambda: {
        SplitName.TRAIN: 202_608_040,
        SplitName.VALIDATION: 202_708_040,
        SplitName.DEVELOPMENT: 202_808_040,
        SplitName.LOCKED_TEST: 202_908_040,
    })
    allow_locked_test: bool = False

    def require_access(self, split: SplitName) -> None:
        if split is SplitName.LOCKED_TEST and not self.allow_locked_test:
            raise PermissionError("locked_test is frozen until explicit approval")


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool = True
    output_root: str = "outputs/multiscale_pcnss"
    refuse_overwrite: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    array: ArrayConfig = field(default_factory=ArrayConfig)
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
```

`requirements.txt`：

```text
numpy>=1.24,<3
scipy>=1.10,<2
torch>=2.1
```

- [ ] **Step 5: 运行 GREEN 并审查**

Run: `python -m unittest test_multisource.test_config -v`

Expected: 运行 2 个测试且结果为 `OK`。确认未引入配置搜索库或额外框架。

- [ ] **Step 6: 精确提交**

```powershell
git add requirements.txt multisource_doa/__init__.py multisource_doa/config.py test_multisource/__init__.py test_multisource/test_config.py docs/superpowers/specs/2026-08-04-multiscale-pcnss-design.md
git diff --cached --name-only
git commit -m "build: freeze PC-NSS experiment protocol"
```

---

## Task 2：实现确定性相干双源仿真和 split manifest

**Files:**

- Create: `multisource_doa/data/__init__.py`
- Create: `multisource_doa/data/simulator.py`
- Create: `multisource_doa/data/dataset.py`
- Create: `multisource_doa/data/manifest.py`
- Create: `test_multisource/test_simulator.py`
- Create: `test_multisource/test_splits.py`

- [ ] **Step 1: 写 steering、相关度、确定性和越权 RED 测试**

```python
sample_a = generate_two_source_sample(config, split_seed=1234, index=7)
sample_b = generate_two_source_sample(config, split_seed=1234, index=7)
np.testing.assert_array_equal(sample_a.snapshots, sample_b.snapshots)
np.testing.assert_allclose(np.abs(sample_a.source_correlation), 1.0, atol=1e-10)
self.assertEqual(sample_a.sample_seed, 1241)
self.assertTrue(np.all(np.diff(sample_a.angles_deg) > 0.0))
```

另验证 `PCNSSDataset(SplitName.LOCKED_TEST, ExperimentConfig())` 抛出 `PermissionError`，train/validation/development seed 区间不相交。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_simulator test_multisource.test_splits -v`

Expected: 因 `multisource_doa.data` 尚不存在而失败。

- [ ] **Step 3: 实现统一正相位阵列流形**

```python
def steering_vector(angles_deg, sensor_count=8, spacing_wavelengths=0.5):
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
    sensors = np.arange(sensor_count, dtype=np.float64)[:, None]
    phase = 2.0 * np.pi * spacing_wavelengths * sensors * np.sin(
        np.deg2rad(angles)
    )[None, :]
    return np.exp(1j * phase)
```

- [ ] **Step 4: 实现样本生成器**

`DOASample` 至少包含 `snapshots [N,T]`、排序角度、rho、SNR、T、sample_seed、经验相关度、噪声功率和去相干目标协方差。相关源严格使用：

```python
s1 = complex_normal(rng, (snapshot_count,))
u2 = complex_normal(rng, (snapshot_count,))
phase = rng.uniform(-np.pi, np.pi)
s2 = rho * np.exp(1j * phase) * s1 + np.sqrt(max(0.0, 1.0 - rho**2)) * u2
signals = np.stack([s1, s2], axis=0)
clean = steering @ signals
signal_power = float(np.mean(np.abs(clean) ** 2))
noise_power = signal_power / (10.0 ** (snr_db / 10.0))
snapshots = clean + np.sqrt(noise_power) * complex_normal(rng, clean.shape)
```

`complex_normal` 每个复样本总方差为 1。`target_covariance` 使用两个源的经验功率对角阵加 `noise_power*I`，不保留相干交叉项。

- [ ] **Step 5: 实现 Dataset 和 manifest**

`PCNSSDataset.__getitem__` 采用 `sample_seed = split_seed + index` 按需生成。`write_split_manifest` 只接收已通过 `require_access` 的 split，JSON 保存 config、split、seed、索引范围和生成器版本。

- [ ] **Step 6: 运行 GREEN 和统计审计**

Run: `python -m unittest test_multisource.test_simulator test_multisource.test_splits -v`

Expected: 全部通过；256 个 `rho=0.9` 样本的平均经验相关幅度位于 `[0.80,0.98]`。

- [ ] **Step 7: 精确提交**

```powershell
git add multisource_doa/data test_multisource/test_simulator.py test_multisource/test_splits.py
git diff --cached --name-only
git commit -m "feat: add deterministic coherent-source simulator"
```

---

## Task 3：实现 SCM、SPS/FBSS 和多尺度 lag 物理视图

**Files:**

- Create: `multisource_doa/physics/__init__.py`
- Create: `multisource_doa/physics/covariance.py`
- Create: `multisource_doa/physics/spatial_smoothing.py`
- Create: `multisource_doa/physics/lags.py`
- Create: `test_multisource/test_covariance_views.py`

- [ ] **Step 1: 写 RED 测试**

验证 `sample_covariance(X) == X @ X.conj().T / T`；FBSS 与显式循环参考一致；lag 固定为 `r[k]=mean(diag(R,-k))`；L=4/7 的 mask 分别只有前 4/7 项有效；raw SCM 只作全孔径 anchor，不进入四尺度教师分布。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_covariance_views -v`

Expected: physics 模块缺失而失败。

- [ ] **Step 3: 实现 SCM 和 FBSS**

```python
def sample_covariance(snapshots):
    return snapshots @ snapshots.conj().T / snapshots.shape[1]


def fbss_covariance(snapshots, subarray_size):
    sensor_count, snapshot_count = snapshots.shape
    subarray_count = sensor_count - subarray_size + 1
    forward = sum(
        snapshots[start:start + subarray_size]
        @ snapshots[start:start + subarray_size].conj().T
        / snapshot_count
        for start in range(subarray_count)
    ) / subarray_count
    reversal = np.fliplr(np.eye(subarray_size, dtype=np.complex128))
    return 0.5 * (forward + reversal @ forward.conj() @ reversal)
```

同时实现仅前向 `sps_covariance`。

- [ ] **Step 4: 实现 lag 视图**

`MultiScaleViews` 固定字段：

```python
raw_covariance: np.ndarray
raw_lags: np.ndarray
fbss_covariances: dict[int, np.ndarray]
fbss_lags: np.ndarray
valid_mask: np.ndarray
effective_counts: np.ndarray
quality_features: np.ndarray
```

6 个质量特征固定为归一化 trace、`log1p` 条件数、信号/噪声特征值比、前后向改变量、子阵 SCM 离散度、有效子阵数量归一值；全部用 `eps=1e-8` 防非有限。

- [ ] **Step 5: 运行 GREEN**

Run: `python -m unittest test_multisource.test_covariance_views -v`

Expected: 全部通过，测试名称明确包含 `positive_phase` 和 `first_column_lag`。

- [ ] **Step 6: 精确提交**

```powershell
git add multisource_doa/physics test_multisource/test_covariance_views.py
git diff --cached --name-only
git commit -m "feat: add multiscale FBSS lag views"
```

---

## Task 4：实现 Hermitian/Toeplitz/PSD/trace 结构投影

**Files:**

- Create: `multisource_doa/physics/projection.py`
- Create: `test_multisource/test_projection.py`

- [ ] **Step 1: 写不变量 RED 测试**

```python
self.assertLess(result.hermitian_error, 1e-7)
self.assertLess(result.toeplitz_error, 1e-7)
self.assertGreaterEqual(result.min_eigenvalue, 1e-6 - 1e-8)
self.assertAlmostEqual(np.trace(result.matrix).real, 8.0, places=6)
self.assertTrue(result.converged)
```

另验证 PyTorch 固定 4 次训练近似有有限梯度。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_projection -v`

Expected: 缺少 `projection.py` 而失败。

- [ ] **Step 3: 实现精确推理投影**

实现 `project_hermitian`、`project_toeplitz`、`project_psd`、`normalize_trace` 和 `dykstra_structured_projection`。Dykstra 的两个集合为 Hermitian Toeplitz 与 PSD；`ProjectionResult` 保存收敛标志、迭代数、结构残差和最小特征值。未收敛不得静默成功。

- [ ] **Step 4: 实现训练期固定迭代版本**

```python
def structured_projection_torch(covariance, target_trace=8.0,
                                iterations=4, eigenvalue_floor=1e-6):
    projected = covariance
    for _ in range(iterations):
        projected = hermitian_toeplitz_projection_torch(projected)
        values, vectors = torch.linalg.eigh(projected)
        values = values.clamp_min(eigenvalue_floor)
        projected = (
            vectors
            @ torch.diag_embed(values).to(projected.dtype)
            @ vectors.mH
        )
    projected = hermitian_toeplitz_projection_torch(projected)
    trace = projected.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-8)
    return projected * (target_trace / trace)[..., None, None]
```

训练版最终 Toeplitz 后允许极小 PSD 残差，但必须审计；评价改用收敛版，二者不得写成同一保证。

- [ ] **Step 5: 运行 GREEN**

Run: `python -m unittest test_multisource.test_projection -v`

Expected: 全部通过，含小矩阵 `complex128` 梯度检查。

- [ ] **Step 6: 精确提交**

```powershell
git add multisource_doa/physics/projection.py test_multisource/test_projection.py
git diff --cached --name-only
git commit -m "feat: add structured covariance projection"
```

---

## Task 5：锁定固定 Root-MUSIC 和传统基线

**Files:**

- Create: `multisource_doa/physics/root_music.py`
- Create: `multisource_doa/baselines/__init__.py`
- Create: `multisource_doa/baselines/classical.py`
- Create: `test_multisource/test_root_music.py`
- Create: `test_multisource/test_classical_baselines.py`

- [ ] **Step 1: 写 Root-MUSIC RED 测试**

无噪理论协方差、角度 `[-8°,7°]` 的估计误差均小于 `0.05°`。覆盖非有限、非方阵、`K>=N`、不足两根、重复角和越界。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_root_music -v`

Expected: Root-MUSIC 实现缺失而失败。

- [ ] **Step 3: 实现固定 Root-MUSIC**

`RootMusicResult` 字段：

```python
angles_deg: np.ndarray
success: bool
failure_reason: str | None
selected_roots: np.ndarray
candidate_count: int
minimum_root_separation: float
```

使用 `N-K` 维噪声子空间、对角线和多项式、单位圆内最近根、共轭互反簇去重，并由 `sin(theta)=angle(root)/(2*pi*d)` 映射。符号由测试锁定；失败不能回填谱峰或真值。

- [ ] **Step 4: 实现统一传统基线接口**

提供 `music_scan`、`esprit`、`root_music_raw`、`sps_root_music(L)`、`fbss_root_music(L)` 和 `evaluate_fixed_scale_family`，均返回 `DOAEstimate`。最佳固定 L 只能在 validation 聚合指标上选一个全局 L，禁止逐样本 oracle 选择。

- [ ] **Step 5: 运行 GREEN**

Run: `python -m unittest test_multisource.test_root_music test_multisource.test_classical_baselines -v`

Expected: 全部通过，固定尺度家族恰含 L=4/5/6/7。

- [ ] **Step 6: 精确提交**

```powershell
git add multisource_doa/physics/root_music.py multisource_doa/baselines test_multisource/test_root_music.py test_multisource/test_classical_baselines.py
git diff --cached --name-only
git commit -m "feat: lock Root-MUSIC and classical baselines"
```

---

## Task 6：实现排列不变、failure-aware 的统一评价

**Files:**

- Create: `multisource_doa/evaluation/__init__.py`
- Create: `multisource_doa/evaluation/matching.py`
- Create: `multisource_doa/evaluation/metrics.py`
- Create: `test_multisource/test_evaluation.py`

- [ ] **Step 1: 写 RED 测试**

覆盖反序估计、缺一个角度时另一个误差为 60、NaN/Inf 失败、重复根不成功分辨，以及真实 `[-2,2]`、估计 `[-1.5,1.5]` 满足误差和 50% 间隔规则。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_evaluation -v`

Expected: evaluation 模块缺失而失败。

- [ ] **Step 3: 实现匹配和分辨规则**

```python
FAILURE_PENALTY_DEG = 60.0


def hungarian_match(true_angles, estimated_angles, failure_penalty_deg=60.0):
    true_angles = np.asarray(true_angles, dtype=np.float64)
    estimated = np.full(2, np.nan, dtype=np.float64)
    supplied = np.asarray(estimated_angles, dtype=np.float64).reshape(-1)[:2]
    estimated[:supplied.size] = supplied
    valid = np.isfinite(estimated)
    cost = np.full((2, 2), failure_penalty_deg, dtype=np.float64)
    cost[:, valid] = np.abs(
        true_angles[:, None] - estimated[valid][None, :]
    )
    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    matched = np.full(2, np.nan, dtype=np.float64)
    errors = np.full(2, failure_penalty_deg, dtype=np.float64)
    for row, col in zip(rows, cols):
        if valid[col]:
            matched[row] = estimated[col]
            errors[row] = cost[row, col]
    return MatchResult(
        true_angles_deg=true_angles,
        estimated_angles_deg=matched,
        absolute_errors_deg=errors,
        success=bool(valid.all()),
    )


def is_resolved(match, true_angles):
    return (
        match.success
        and np.all(match.absolute_errors_deg <= 1.0)
        and np.diff(np.sort(match.estimated_angles_deg))[0]
        >= 0.5 * np.diff(np.sort(true_angles))[0]
    )
```

`cost` 始终构造成 2×2，无效/缺失列代价直接为 60。

- [ ] **Step 4: 实现聚合与 paired 审计**

`aggregate_metrics` 输出 failure-aware RMSPE、条件 RMSE、MAE、p95、p99、最大误差、resolution rate、failure count/reasons。`paired_comparison` 的 tie 容差固定 `1e-6°`，支持 separation/SNR/T/rho 分层。

- [ ] **Step 5: 运行 GREEN**

Run: `python -m unittest test_multisource.test_evaluation -v`

Expected: 全部通过；failure-aware 分母包含失败样本。

- [ ] **Step 6: 精确提交**

```powershell
git add multisource_doa/evaluation test_multisource/test_evaluation.py
git diff --cached --name-only
git commit -m "feat: add failure-aware DOA evaluation"
```

---

## Task 7：实现约 5 万参数的 PC-NSS 网络骨架

**Files:**

- Create: `multisource_doa/models/__init__.py`
- Create: `multisource_doa/models/pc_nss.py`
- Create: `test_multisource/test_pc_nss_model.py`

- [ ] **Step 1: 写模型接口 RED 测试**

输入固定为：

```python
raw_lags_ri       # [B,8,2]
fbss_lags_ri      # [B,4,8,2]
valid_mask        # [B,4,8]
effective_counts  # [B,4,8]
quality_features  # [B,4,6]
```

验证有效 lag 的权重和为 1、无效权重为 0；lag 7 无 FBSS 时取 raw anchor；复残差模长≤0.10；加载≤0.05；输出 `[B,8,8]`；参数量 `[30_000,80_000]`；forward 不接收角度、SNR、rho、T 或 domain 标签。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_pc_nss_model -v`

Expected: models 模块缺失而失败。

- [ ] **Step 3: 实现安全 masked softmax 和残差边界**

```python
def masked_softmax(logits, mask, dim):
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def bounded_complex_vector(raw_ri, max_magnitude):
    norm = raw_ri.norm(dim=-1, keepdim=True)
    direction = raw_ri / norm.clamp_min(1e-12)
    magnitude = torch.tanh(norm) * max_magnitude
    return direction * magnitude
```

全 mask 的 lag 7 不调用 softmax，权重置零后使用 raw anchor。

- [ ] **Step 4: 实现固定网络结构**

- cell encoder：`7 -> 64 -> 64`，GELU；
- quality encoder：`6 -> 24 -> 24`；
- 8 个 16 维 lag embedding；
- global context：cell 均值 64 + quality 均值 24；
- logit head：`192 -> 96 -> 32 -> 1`；
- residual head：`108 -> 96 -> 64 -> 2`；
- loading head：`88 -> 32 -> 1`。

融合 lag 加受限残差，构造 Toeplitz covariance，加非负有界加载，再调用训练期结构投影。

- [ ] **Step 5: 运行 GREEN、反传和参数审计**

Run: `python -m unittest test_multisource.test_pc_nss_model -v`

Expected: 全部通过；随机 batch 反传后梯度有限。参数范围失败时只修尺寸实现，不搜索宽度。

- [ ] **Step 6: 精确提交**

```powershell
git add multisource_doa/models test_multisource/test_pc_nss_model.py
git diff --cached --name-only
git commit -m "feat: add bounded multiscale PC-NSS model"
```

---

## Task 8：实现多尺度教师和两阶段损失

**Files:**

- Create: `multisource_doa/training/__init__.py`
- Create: `multisource_doa/training/teacher.py`
- Create: `multisource_doa/training/losses.py`
- Create: `test_multisource/test_teacher.py`
- Create: `test_multisource/test_losses.py`

- [ ] **Step 1: 写教师 RED 测试**

人工构造四尺度，使 `g_6` 最大，验证 `pi_scale.argmax()==2`、概率和为 1、温度为 0.1；教师不能接收神经输出或 split 聚合统计。

- [ ] **Step 2: 写损失 RED 测试**

验证正确尺度权重比错误塌缩的 `L_scale` 小；正确双谷比单峰的 `L_peak` 小；`g_pred>g_best` 时 `L_dom` 小；epoch 0-9 不含 peak/dom，10-49 加入；归一化 lag loss 对整体幅度缩放稳定。

- [ ] **Step 3: 运行 RED**

Run: `python -m unittest test_multisource.test_teacher test_multisource.test_losses -v`

Expected: training 模块缺失而失败。

- [ ] **Step 4: 实现归一化 MUSIC 分母和教师**

```python
def normalized_music_denominator(covariance, angles_deg, source_count=2):
    values, vectors = torch.linalg.eigh(covariance)
    noise = vectors[..., :covariance.shape[-1] - source_count]
    steering = steering_vector_torch(angles_deg, covariance.shape[-1])
    projection = noise @ noise.mH
    numerator = torch.einsum(
        "...na,...nm,...ma->...a",
        steering.conj(), projection, steering
    ).real
    denominator = steering.abs().square().sum(dim=-2).clamp_min(1e-8)
    return (numerator / denominator).clamp(0.0, 1.0)
```

各尺度使用自己的阵元数；`g_L=q(mid)-0.5*(q(theta1)+q(theta2))`；`pi=softmax(g/0.1)`。

- [ ] **Step 5: 实现精确损失**

guard 固定为 `[theta1-0.5°, midpoint,theta2+0.5°]` 并裁剪。尺度权重按 `effective_counts*valid_mask` 聚合。

```python
stage_one = 1.0 * lag + 0.5 * scale + 0.01 * residual
stage_two = stage_one + 1.0 * peak + 0.5 * dominance
```

`peak=0.5*(q1+q2)+mean(relu(0.05+max(q1,q2)-q_guard))`；
`dominance=0.1*softplus((g_best-g_pred)/0.1)`；
`residual=mean(|delta_lag|^2)+mean(loading^2)`。

返回含所有未加权项、加权项和 total 的 `LossBreakdown`。

- [ ] **Step 6: 运行 GREEN**

Run: `python -m unittest test_multisource.test_teacher test_multisource.test_losses -v`

Expected: 全部通过；退化 covariance 无 NaN，梯度回到预测 covariance 和尺度 logits。

- [ ] **Step 7: 精确提交**

```powershell
git add multisource_doa/training test_multisource/test_teacher.py test_multisource/test_losses.py
git diff --cached --name-only
git commit -m "feat: add resolution-aware PC-NSS losses"
```

---

## Task 9：实现训练引擎、checkpoint 和诊断审计

**Files:**

- Create: `multisource_doa/training/engine.py`
- Create: `multisource_doa/training/artifacts.py`
- Create: `test_multisource/test_training_engine.py`
- Create: `test_multisource/test_artifacts.py`

- [ ] **Step 1: 写 RED 测试**

4 个确定性样本验证一个 epoch 可反传；epoch 9/10 权重切换正确；checkpoint 只在 validation failure-aware RMSPE 严格变小时更新；已有输出且拒绝覆盖时抛 `FileExistsError`；locked test 不能传入训练或选模。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_training_engine test_multisource.test_artifacts -v`

Expected: engine/artifacts 缺失而失败。

- [ ] **Step 3: 实现 batch 与单 epoch**

`collate_samples` 在 CPU 构造多尺度视图和 target lag。`train_one_epoch` 不调用 Root-MUSIC，只返回损失与诊断；真角度/rho/SNR/T/sample_seed 仅供监督与审计，不进模型输入。

- [ ] **Step 4: 实现 validation 和 checkpoint**

validation 使用收敛投影和固定 Root-MUSIC。checkpoint 固定包含：

```python
model_state_dict
optimizer_state_dict
epoch
selection_metric_name
selection_metric_value
experiment_config
model_seed
data_split_seed
parameter_count
code_sha
```

保存后生成 checkpoint SHA256 旁路 JSON；唯一选模指标为 `failure_aware_rmspe_deg`。

- [ ] **Step 5: 实现不调参诊断字段**

每 epoch 记录四尺度平均权重/熵、每 lag 权重、残差模长分位数、loading 分位数、投影改变量、结构误差、`g_best-g_pred`、peak margin、train/validation loss 分项和 Root-MUSIC failure reasons。

- [ ] **Step 6: 运行 GREEN**

Run: `python -m unittest test_multisource.test_training_engine test_multisource.test_artifacts -v`

Expected: 全部通过；测试只写 `TemporaryDirectory`。

- [ ] **Step 7: 精确提交**

```powershell
git add multisource_doa/training/engine.py multisource_doa/training/artifacts.py test_multisource/test_training_engine.py test_multisource/test_artifacts.py
git diff --cached --name-only
git commit -m "feat: add audited PC-NSS training engine"
```

---

## Task 10：实现统一评价 runner 和对比报告

**Files:**

- Create: `multisource_doa/evaluation/runner.py`
- Create: `multisource_doa/evaluation/reporting.py`
- Create: `multisource_doa/baselines/registry.py`
- Create: `test_multisource/test_evaluation_runner.py`

- [ ] **Step 1: 写 RED 测试**

4 个内存样本验证 runner 输出 raw Root-MUSIC、四个固定 FBSS 和 PC-NSS；CSV 保留失败；summary 只能在 validation 上选择单个全局最佳 L；development 不更新 checkpoint；locked test 拒绝。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_evaluation_runner -v`

Expected: runner 模块缺失而失败。

- [ ] **Step 3: 实现 registry**

注册 `music`、`root_music`、`esprit`、`sps_root_music_L4..L7`、`fbss_root_music_L4..L7`、`pcnss_root_music`。外部深度基线只定义 `NOT_INTEGRATED/AVAILABLE/FAILED_REPRODUCTION`，不伪造结果。

- [ ] **Step 4: 实现报告**

```text
run_config.json
source_manifest.json
predictions.csv
summary.json
paired_comparisons.csv
failure_reasons.csv
runtime_summary.json
```

summary 区分 `framework_validation` 与 `research_acceptance`。基础框架阶段前者可为 true，后者必须为 `not_run`。

- [ ] **Step 5: 运行 GREEN**

Run: `python -m unittest test_multisource.test_evaluation_runner -v`

Expected: 全部通过，schema 不依赖 Pandas。

- [ ] **Step 6: 精确提交**

```powershell
git add multisource_doa/evaluation/runner.py multisource_doa/evaluation/reporting.py multisource_doa/baselines/registry.py test_multisource/test_evaluation_runner.py
git diff --cached --name-only
git commit -m "feat: add unified estimator evaluation reports"
```

---

## Task 11：实现 PyCharm 无参数入口、dry-run 和 4 样本 smoke

**Files:**

- Create: `scripts/run_multiscale_pcnss.py`
- Create: `scripts/smoke_multiscale_pcnss.py`
- Create: `test_multisource/test_entrypoints.py`
- Modify: `README.md`

- [ ] **Step 1: 写入口 RED 测试**

入口顶部固定：

```python
RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "model_seed": 2026,
    "split": "train",
    "sample_count": 4,
    "output_root": "outputs/multiscale_pcnss_snap20",
    "allow_locked_test": False,
    "overwrite": False,
}
```

允许 stage：`dry_run`、`smoke_train`、`train`、`evaluate_validation`、`evaluate_development`；不得有 `evaluate_locked_test`。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest test_multisource.test_entrypoints -v`

Expected: scripts 缺失而失败。

- [ ] **Step 3: 实现单 stage 入口**

stage 必须是单字符串；`"train evaluate_development"` 应报错并提示每次只运行一个阶段。`dry_run` 只检查依赖、配置、参数量、一个样本物理链路和输出路径，不创建正式输出。

- [ ] **Step 4: 实现 smoke**

仅 4 个 train 样本、batch 2、1 epoch、CPU、独立 `_smoke` 输出；执行 forward/backward、validation 物理推理和报告。smoke checkpoint 不得名为正式 `best.pt`。

- [ ] **Step 5: 更新 README**

说明 PyCharm Parameters 留空；先 dry-run 再 smoke；正式 train 由用户审查后改 RUN_CONFIG；当前没有 SubspaceNet 等结果，框架可运行不等于论文门槛通过。

- [ ] **Step 6: 运行 GREEN 和 Agent 验证**

```powershell
python -m unittest test_multisource.test_entrypoints -v
python -m compileall multisource_doa scripts test_multisource
python scripts/run_multiscale_pcnss.py
python scripts/smoke_multiscale_pcnss.py
```

Expected: 测试通过；compileall 无错误；dry-run 不建正式 checkpoint；4 样本 smoke 的 loss、梯度和结构审计有限。

- [ ] **Step 7: 精确提交**

```powershell
git add scripts/run_multiscale_pcnss.py scripts/smoke_multiscale_pcnss.py test_multisource/test_entrypoints.py README.md
git diff --cached --name-only
git commit -m "feat: add safe PC-NSS run entrypoints"
```

---

## Task 12：基础框架总审查与正式训练交接

**Files:**

- Create: `experiments/foundation_framework_review.md`
- Create: `experiments/formal_training_protocol.md`

- [ ] **Step 1: 完整基础框架测试**

Run: `python -m unittest discover -s test_multisource -v`

Expected: 全部通过。不得通过跳过测试、放宽容差或删除失败样本解决失败。

- [ ] **Step 2: 重跑 compileall、dry-run 和 smoke**

```powershell
python -m compileall multisource_doa scripts test_multisource
python scripts/run_multiscale_pcnss.py
python scripts/smoke_multiscale_pcnss.py
```

Expected: exit code 0；未访问/生成 locked test；无正式 40k 输出；不覆盖目录。

- [ ] **Step 3: 审计研究边界**

Run: `rg -n "首次|首个|理论保证|超过SubspaceNet|优于SubspaceNet|locked_test" README.md docs experiments multisource_doa scripts test_multisource`

Expected: 禁用性能主张无匹配；`locked_test` 只用于禁用说明、防线和测试。

- [ ] **Step 4: 写基础框架审查记录**

记录各 Task 提交、测试命令、通过数量、参数量、smoke 结果和限制。结论只能是“基础框架通过/未通过”，不能写模型性能达标。

- [ ] **Step 5: 写正式训练协议但不执行**

固定用户后续顺序：

1. seed2026 正式 train；
2. validation 选 checkpoint 和全局最佳固定 L；
3. development 只作不调参诊断；
4. 首 seed 未过 raw Root-MUSIC 与最佳固定 FBSS 时停止扩 seed，检查权重塌缩、lag 残差、投影改变量、子空间夹角、peak margin 和 train/validation gap；
5. 新设计获批后才优化；
6. seed2026 方向成立后再审批 seed2027/2028；
7. 模型、损失、阈值冻结后才审批 locked test；
8. SubspaceNet、DA-MUSIC、DeepMUSIC 在投稿阶段接入，不阻塞基础框架。

- [ ] **Step 6: 检查最终 diff 和提交**

```powershell
git status --short
git diff --check
git add experiments/foundation_framework_review.md experiments/formal_training_protocol.md
git diff --cached --name-only
git commit -m "docs: record PC-NSS framework verification"
```

Expected: 只暂存两个审查文档；没有 outputs、权重、生成数据或 DIO_DOA 文件。

---

## 基础框架完成后的停止点

完成 Task 12 后必须停止，不自动运行正式训练。此时只允许声称：

- 数据、物理、模型、损失、评价和入口按规格建立；
- 单元测试、compileall、默认 dry-run 和 4 样本 smoke 的实际结果；
- 参数量和结构不变量审计结果。

此时禁止声称：

- PC-NSS 已优于 raw Root-MUSIC 或最佳固定尺度 FBSS；
- PC-NSS 已达到稳健论文门槛；
- PC-NSS 已优于 SubspaceNet、DA-MUSIC、DeepMUSIC 或 CR-UNet；
- locked test 已通过。

正式训练由用户在基础框架审查后运行。首个正式 seed 的结果决定是进入不调参机理诊断、提出新设计，还是扩展到多 seed 和投稿强度比较。
