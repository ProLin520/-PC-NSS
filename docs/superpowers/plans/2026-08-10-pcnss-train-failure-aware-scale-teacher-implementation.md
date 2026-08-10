# PC-NSS Train-only Failure-aware Scale Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可审计的 train-only 固定尺度角误差 teacher 缓存，只用它替换 KL 尺度蒸馏目标，并以一次 seed 2026 单因素训练和冻结 validation gate 判定是否修复近间隔退化。

**Architecture:** 新的 `error_teacher.py` 只负责 L4–L7 failure-aware 标签数学；`teacher_cache.py` 只负责三文件 cache 写入、认证和按 seed 查询；训练损失增加可选 `scale_distillation_target`，但 dominance 继续使用原物理 score。单因素身份审计和实验结果审计各自使用独立的只读模块与安全脚本，正式训练入口只消费已经通过的 cache 与 audit，不自行访问 Task 16 或偷偷退回物理标签。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、标准库 `csv/json/hashlib/unittest`，沿用项目现有 `requirements.txt`，不新增依赖。

## Global Constraints

- 唯一实验变量固定为 `physical teacher scale_probabilities -> train-only failure-aware angular-error scale_distillation_target`。
- 新 target 只进入现有 KL；`dominance_loss` 始终使用原 `physical_teacher.scale_scores.max()`。
- train 固定 40,000、validation 固定 5,000、batch size 128、learning rate `1e-3`、50 epochs、model seed 2026。
- 数据、split seed、模型、优化器、两阶段时点、全部损失公式/权重、checkpoint 规则、Root-MUSIC、投影和评价协议保持不变。
- 标签 tie 容差固定 `1e-6°`；失败角保留 `60°` 罚值；四尺度全失败输出均匀分布。
- cache 正式阶段只允许 train、CPU、40,000 样本、batch 128；不读取 validation/development/locked test。
- 默认入口保持 `dry_run=true`，已有输出目录拒绝覆盖，不允许 `overwrite=true`。
- Agent 不生成正式 40,000 cache、不运行正式训练、不访问 development/locked test。
- 正式产物、cache、checkpoint、outputs 和临时配置不提交 Git。
- 每个 Task 严格执行 RED -> GREEN -> 重构、目标测试、diff 审查和范围明确的提交。

---

## File Map

### 新建文件

- `multisource_doa/training/error_teacher.py`：固定尺度估计、failure-aware RMSPE 行和硬标签数学。
- `multisource_doa/training/teacher_cache.py`：cache schema-v1、三文件写入、认证、重建校验和标签查询。
- `multisource_doa/training/single_factor_audit.py`：A/B 身份、Task 16 结论、cache 与 baseline 证据认证。
- `multisource_doa/training/single_factor_reporting.py`：三文件单因素审计报告和拒绝覆盖。
- `multisource_doa/evaluation/teacher_experiment.py`：读取两份既有 validation 报告并复算冻结 gate、配对和分层。
- `multisource_doa/evaluation/teacher_experiment_reporting.py`：五文件结果审计报告。
- `scripts/build_pcnss_failure_aware_teacher_cache.py`：cache dry-run/smoke/formal 安全入口。
- `scripts/audit_pcnss_teacher_single_factor.py`：身份审计 dry-run/smoke/formal 安全入口。
- `scripts/audit_pcnss_teacher_experiment.py`：只读结果审计 dry-run/smoke/formal 安全入口。
- `test_multisource/test_error_teacher.py`：标签数学与固定尺度估计测试。
- `test_multisource/test_teacher_cache.py`：cache schema、SHA、seed 和篡改测试。
- `test_multisource/test_single_factor_audit.py`：身份 gate 与报告测试。
- `test_multisource/test_teacher_experiment.py`：结果 gate、配对、分层和停止规则测试。

### 修改文件

- `multisource_doa/training/losses.py`：增加可选 KL target，不改变其余 loss。
- `multisource_doa/training/engine.py`：按 batch sample seed 构造 cached target。
- `multisource_doa/training/artifacts.py`：checkpoint 可选记录 teacher/audit 身份。
- `multisource_doa/data/manifest.py`：split manifest 支持可选、JSON-safe 的额外训练元数据。
- `multisource_doa/evaluation/reporting.py`：评价 manifest 记录 checkpoint teacher metadata。
- `scripts/run_multiscale_pcnss.py`：安全配置、cache/audit 预检、训练与评价元数据接入。
- `test_multisource/test_losses.py`、`test_training_engine.py`、`test_artifacts.py`、`test_entrypoints.py`、`test_evaluation_runner.py`：回归和入口契约。
- `README.md`、`experiments/formal_training_protocol.md`：用户运行顺序、停止规则和禁止项。

---

### Task 1: Failure-aware 角误差 Teacher 数学

**Files:**
- Create: `multisource_doa/training/error_teacher.py`
- Create: `test_multisource/test_error_teacher.py`
- Reference: `multisource_doa/baselines/classical.py`
- Reference: `multisource_doa/evaluation/metrics.py`

**Interfaces:**
- Consumes: `DOASample`、`DOAEstimate`、`evaluate_fixed_scale_family()`、`score_doa_sample()`。
- Produces: `SCALE_SIZES`、`ERROR_TIE_TOLERANCE_DEG`、`teacher_probabilities_from_rmspe()`、`build_error_teacher_row()`。

- [ ] **Step 1: 写概率数学的 RED 测试**

```python
class ErrorTeacherProbabilityTest(unittest.TestCase):
    def test_unique_best_is_one_hot_and_ties_share_mass(self):
        unique = teacher_probabilities_from_rmspe({4: 2.0, 5: 1.0, 6: 3.0, 7: 4.0})
        self.assertEqual(unique, (0.0, 1.0, 0.0, 0.0))

        tied = teacher_probabilities_from_rmspe(
            {4: 1.0, 5: 1.0 + 5e-7, 6: 2.0, 7: 60.0}
        )
        self.assertEqual(tied, (0.5, 0.5, 0.0, 0.0))

    def test_all_failed_is_uniform_and_nonfinite_is_rejected(self):
        self.assertEqual(
            teacher_probabilities_from_rmspe({4: 60.0, 5: 60.0, 6: 60.0, 7: 60.0}),
            (0.25, 0.25, 0.25, 0.25),
        )
        with self.assertRaises(ValueError):
            teacher_probabilities_from_rmspe({4: 1.0, 5: math.nan, 6: 2.0, 7: 3.0})
```

- [ ] **Step 2: 运行 RED 并确认缺失模块**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_error_teacher.ErrorTeacherProbabilityTest -v`
Expected: `ModuleNotFoundError: multisource_doa.training.error_teacher`。

- [ ] **Step 3: 实现冻结概率函数**

```python
SCALE_SIZES = (4, 5, 6, 7)
ERROR_TIE_TOLERANCE_DEG = 1e-6

def teacher_probabilities_from_rmspe(
    rmspe_by_scale: Mapping[int, float],
) -> tuple[float, float, float, float]:
    if set(rmspe_by_scale) != set(SCALE_SIZES):
        raise ValueError("rmspe_by_scale must contain L4-L7")
    values = tuple(float(rmspe_by_scale[size]) for size in SCALE_SIZES)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("fixed-scale RMSPE values must be finite and non-negative")
    minimum = min(values)
    winners = tuple(
        index for index, value in enumerate(values)
        if value - minimum <= ERROR_TIE_TOLERANCE_DEG
    )
    mass = 1.0 / len(winners)
    return tuple(mass if index in winners else 0.0 for index in range(4))
```

- [ ] **Step 4: 写样本行 RED 测试，锁定排列匹配和失败罚值**

```python
def _estimate(size, angles, success=True, reason=None):
    return DOAEstimate(
        algorithm=f"fbss_root_music_L{size}",
        angles_deg=np.asarray(angles, dtype=np.float64),
        success=success,
        failure_reason=reason,
        runtime_seconds=0.0,
        metadata={"subarray_size": size},
    )

def test_row_uses_failure_aware_matching_and_keeps_failed_scale(self):
    sample = generate_two_source_sample(
        ExperimentConfig(), split_seed=901, index=0,
        rho=1.0, snr_db=5.0, snapshot_count=20,
        center_deg=0.0, separation_deg=4.0,
    )
    estimates = {
        4: _estimate(4, sample.angles_deg[::-1]),
        5: _estimate(5, [], False, "no_valid_roots"),
        6: _estimate(6, sample.angles_deg + 1.0),
        7: _estimate(7, sample.angles_deg + 2.0),
    }
    row = build_error_teacher_row(
        sample, sample_index=0, estimates_by_scale=estimates
    )
    self.assertAlmostEqual(row["sample_rmspe_deg_L4"], 0.0)
    self.assertEqual(row["sample_rmspe_deg_L5"], 60.0)
    self.assertEqual(row["teacher_probabilities"], (1.0, 0.0, 0.0, 0.0))
    self.assertEqual(row["failure_reason_L5"], "no_valid_roots")
```

- [ ] **Step 5: 实现固定尺度样本行**

```python
def build_error_teacher_row(
    sample: DOASample,
    *,
    sample_index: int,
    estimates_by_scale: Mapping[int, DOAEstimate] | None = None,
) -> dict[str, Any]:
    estimates = estimates_by_scale
    if estimates is None:
        family = evaluate_fixed_scale_family(
            sample.snapshots, subarray_sizes=SCALE_SIZES, source_count=2
        )
        estimates = {
            size: family[f"fbss_root_music_L{size}"] for size in SCALE_SIZES
        }
    if set(estimates) != set(SCALE_SIZES):
        raise ValueError("estimates_by_scale must contain L4-L7")
    row: dict[str, Any] = {
        "sample_index": int(sample_index),
        "sample_seed": int(sample.sample_seed),
        "true_angle_1_deg": float(sample.angles_deg[0]),
        "true_angle_2_deg": float(sample.angles_deg[1]),
        "separation_deg": float(abs(np.diff(sample.angles_deg)[0])),
        "rho": float(sample.rho),
        "snr_db": float(sample.snr_db),
        "snapshot_count": int(sample.snapshot_count),
    }
    rmspe: dict[int, float] = {}
    for size in SCALE_SIZES:
        estimate = estimates[size]
        score = score_doa_sample(
            sample.sample_seed,
            sample.angles_deg,
            estimate.angles_deg,
            estimate_success=estimate.success,
            failure_reason=estimate.failure_reason,
        )
        rmspe[size] = score.sample_rmspe_deg
        row[f"success_L{size}"] = score.success
        row[f"failure_reason_L{size}"] = score.failure_reason or ""
        for index in range(2):
            estimate_value = score.match.estimated_angles_deg[index]
            row[f"estimated_angle_{index + 1}_deg_L{size}"] = (
                float(estimate_value) if np.isfinite(estimate_value) else None
            )
            row[f"absolute_error_{index + 1}_deg_L{size}"] = float(
                score.match.absolute_errors_deg[index]
            )
        row[f"sample_rmspe_deg_L{size}"] = score.sample_rmspe_deg
    probabilities = teacher_probabilities_from_rmspe(rmspe)
    winners = tuple(size for size, value in zip(SCALE_SIZES, probabilities) if value > 0.0)
    row["teacher_probabilities"] = probabilities
    row["best_scales"] = winners
    row["has_tied_best"] = len(winners) > 1
    row["all_scales_failed"] = all(not bool(row[f"success_L{size}"]) for size in SCALE_SIZES)
    return row
```

- [ ] **Step 6: 运行 GREEN 与原评价测试**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_error_teacher test_multisource.test_evaluation test_multisource.test_classical_baselines -v`
Expected: 全部通过，失败尺度仍计 `60°`，没有删样本。

- [ ] **Step 7: 审查并提交 Task 1**

```powershell
git diff --check
git add -- multisource_doa/training/error_teacher.py test_multisource/test_error_teacher.py
git diff --cached --name-only
git commit -m "feat: compute failure-aware scale labels"
```

---

### Task 2: 三文件 Teacher Cache 与严格认证

**Files:**
- Create: `multisource_doa/training/teacher_cache.py`
- Create: `test_multisource/test_teacher_cache.py`
- Modify: `multisource_doa/training/__init__.py`

**Interfaces:**
- Consumes: Task 1 `SCALE_SIZES`、`build_error_teacher_row()`、`teacher_probabilities_from_rmspe()`，以及 `PCNSSDataset(TRAIN)`。
- Produces: `TEACHER_CACHE_SCHEMA_VERSION`、`TeacherCache`、`write_teacher_cache()`、`load_teacher_cache()`、`sha256_file()`。

- [ ] **Step 1: 写三文件和拒绝覆盖 RED 测试**

```python
class TeacherCacheWriterTest(unittest.TestCase):
    def test_writer_creates_exact_three_files_and_refuses_overwrite(self):
        rows = [_teacher_row(index) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cache"
            write_teacher_cache(
                rows, output,
                experiment_config=ExperimentConfig(),
                run_config={"stage": "smoke", "split": "train", "batch_size": 128},
                code_sha="abc123", source_sha256={"error_teacher.py": "f" * 64},
                expected_count=4,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"teacher_cache_config.json", "teacher_cache_manifest.json", "train_teacher_labels.csv"},
            )
            with self.assertRaises(FileExistsError):
                write_teacher_cache(
                    rows, output,
                    experiment_config=ExperimentConfig(),
                    run_config={"stage": "smoke"}, code_sha="abc123",
                    source_sha256={"error_teacher.py": "f" * 64}, expected_count=4,
                )
```

- [ ] **Step 2: 运行 RED**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_cache.TeacherCacheWriterTest -v`
Expected: import 失败，指向 `teacher_cache` 尚未创建。

- [ ] **Step 3: 实现固定 schema、有限性和原子写出**

```python
TEACHER_CACHE_SCHEMA_VERSION = 1
CACHE_FILENAMES = (
    "teacher_cache_config.json",
    "teacher_cache_manifest.json",
    "train_teacher_labels.csv",
)

@dataclass(frozen=True)
class TeacherCache:
    labels_by_seed: Mapping[int, tuple[float, float, float, float]]
    manifest: Mapping[str, Any]
    file_sha256: Mapping[str, str]

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

`write_teacher_cache()` 在创建目标目录前完成：行数、连续 `sample_index`、升序唯一 seed、L4–L7 finite RMSPE、概率复算、JSON finite 和字段集合校验。CSV 将 `teacher_probabilities` 拆成四列，将 `best_scales` 写成 JSON 数组；写完 CSV 后计算 SHA，再写 manifest。manifest 必须显式写 `train_only=true`、`no_model_forward=true`、`training_performed=false`、`validation_accessed=false`、`development_accessed=false`、`locked_test_accessed=false`。目标已存在时直接 `FileExistsError`，不接受 overwrite 参数。

- [ ] **Step 4: 写 loader 的 RED 测试**

```python
class TeacherCacheLoaderTest(unittest.TestCase):
    def test_loader_rebuilds_train_metadata_and_returns_seed_lookup(self):
        cache_dir = self._write_four_sample_cache()
        loaded = load_teacher_cache(
            cache_dir, ExperimentConfig(), expected_count=4, regenerate_metadata=True
        )
        start = ExperimentConfig().split.seeds[SplitName.TRAIN]
        self.assertEqual(tuple(loaded.labels_by_seed), tuple(start + i for i in range(4)))
        self.assertEqual(sum(loaded.labels_by_seed[start]), 1.0)

    def test_loader_rejects_duplicate_wrong_split_sha_and_probability_tamper(self):
        for mutation in ("duplicate_seed", "validation_seed", "csv_sha", "probability"):
            cache_dir = self._mutated_cache(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_teacher_cache(cache_dir, ExperimentConfig(), expected_count=4)
```

- [ ] **Step 5: 实现 loader 身份认证**

`load_teacher_cache(directory, config, expected_count, regenerate_metadata=True)` 必须按以下顺序失败：

```python
paths = {name: directory / name for name in CACHE_FILENAMES}
if set(path.name for path in directory.iterdir()) != set(CACHE_FILENAMES):
    raise ValueError("teacher cache must contain exactly three files")
manifest = _read_json(paths["teacher_cache_manifest.json"])
if manifest["teacher_cache_schema_version"] != TEACHER_CACHE_SCHEMA_VERSION:
    raise ValueError("teacher cache schema mismatch")
if manifest["split"] != SplitName.TRAIN.value:
    raise PermissionError("teacher cache must be train-only")
if sha256_file(paths["train_teacher_labels.csv"]) != manifest["csv_sha256"]:
    raise ValueError("teacher cache CSV SHA mismatch")
```

随后验证 expected count、连续 seed、配置、算法常量、概率复算和 label counts。`regenerate_metadata=True` 时按 `sample_seed - train_seed` 访问 `PCNSSDataset(TRAIN)`，逐项比较角度、rho、SNR、snapshot 和 separation；任何差异立即失败。

- [ ] **Step 6: 运行 GREEN、split 与 manifest 回归测试**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_cache test_multisource.test_splits -v`
Expected: 全部通过；validation/development/locked seed 均被拒绝。

- [ ] **Step 7: 审查并提交 Task 2**

```powershell
git add -- multisource_doa/training/error_teacher.py multisource_doa/training/teacher_cache.py multisource_doa/training/__init__.py test_multisource/test_teacher_cache.py
git diff --cached --check
git commit -m "feat: write audited train teacher cache"
```

---

### Task 3: Cache 安全入口、dry-run 与 4 样本 smoke

**Files:**
- Create: `scripts/build_pcnss_failure_aware_teacher_cache.py`
- Modify: `test_multisource/test_entrypoints.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1/2 cache API、`PCNSSDataset(TRAIN)`、`ExperimentConfig`。
- Produces: `RUN_CONFIG`、`validate_stage()`、`load_config()`、`run_dry_run()`、`run_smoke()`、`run_formal_cache()`。

- [ ] **Step 1: 写入口安全 RED 测试**

```python
class ErrorTeacherCacheEntrypointTest(unittest.TestCase):
    def test_default_is_cpu_train_dry_run_and_creates_nothing(self):
        namespace = runpy.run_path(str(ERROR_TEACHER_CACHE_SCRIPT))
        result = namespace["run_stage"](dict(namespace["RUN_CONFIG"]))
        self.assertEqual(result["stage"], "dry_run")
        self.assertFalse(result["output_created"])
        self.assertTrue(result["train_only"])
        self.assertFalse(result["training_performed"])

    def test_formal_guards_reject_non_train_non_cpu_wrong_count_and_overwrite(self):
        namespace = runpy.run_path(str(ERROR_TEACHER_CACHE_SCRIPT))
        base = dict(namespace["RUN_CONFIG"], stage="build_train_teacher_cache", dry_run=False)
        for key, value in (
            ("split", "validation"), ("device", "cuda"),
            ("sample_count", 39999), ("batch_size", 64),
            ("overwrite", True), ("allow_locked_test", True),
        ):
            with self.subTest(key=key), self.assertRaises((ValueError, PermissionError)):
                namespace["run_stage"]({**base, key: value})
```

- [ ] **Step 2: 运行 RED**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints.ErrorTeacherCacheEntrypointTest -v`
Expected: 脚本路径不存在。

- [ ] **Step 3: 实现 RUN_CONFIG 和冻结 guards**

```python
RUN_CONFIG = {
    "stage": "dry_run", "dry_run": True, "split": "train",
    "output_root": "outputs/pcnss_failure_aware_teacher_cache",
    "device": "cpu", "batch_size": 128, "sample_count": 1,
    "allow_locked_test": False, "overwrite": False,
}
STAGES = ("dry_run", "smoke", "build_train_teacher_cache")
FORMAL_SAMPLE_COUNT = 40_000
FROZEN_BATCH_SIZE = 128
```

`load_config()` 只接受 `RUN_CONFIG` 已知键；formal guard 固定 train/CPU/40,000/128/false overwrite；`dry_run` 只在内存计算 index 0；smoke 固定前 4 个 train 样本并写三文件；formal 遍历完整 `PCNSSDataset(TRAIN)`，但不由 Agent 运行。

- [ ] **Step 4: 写并运行 4 样本 smoke 测试**

```python
def test_four_sample_smoke_writes_authenticated_three_file_cache(self):
    namespace = runpy.run_path(str(ERROR_TEACHER_CACHE_SCRIPT))
    with tempfile.TemporaryDirectory() as temporary:
        values = dict(
            namespace["RUN_CONFIG"], stage="smoke", sample_count=4,
            output_root=str(Path(temporary) / "cache"),
        )
        result = namespace["run_stage"](values)
        loaded = load_teacher_cache(result["cache"], ExperimentConfig(), expected_count=4)
        self.assertEqual(len(loaded.labels_by_seed), 4)
```

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints.ErrorTeacherCacheEntrypointTest test_multisource.test_teacher_cache -v`
Expected: 全部通过。

- [ ] **Step 5: 更新 README 的安全运行说明**

写明三个 stage、PyCharm Parameters 可留空、formal cache 由用户运行、cache 不提交、formal 阶段不读取 validation/development/locked test。

- [ ] **Step 6: dry-run 实证与提交**

Run: `D:\Python\Python\python.exe scripts\build_pcnss_failure_aware_teacher_cache.py`
Expected: JSON 显示 `stage=dry_run`、`output_created=false`、`train_only=true`、`training_performed=false`。

```powershell
git add -- scripts/build_pcnss_failure_aware_teacher_cache.py test_multisource/test_entrypoints.py README.md
git commit -m "feat: add safe teacher cache entrypoint"
```

---

### Task 4: KL Target 最小注入，保持 Dominance 物理基准

**Files:**
- Modify: `multisource_doa/training/losses.py`
- Modify: `test_multisource/test_losses.py`

**Interfaces:**
- Consumes: 原 `ScaleTeacher` 和可选 `[batch,4]` target。
- Produces: `pcnss_loss(..., scale_distillation_target: torch.Tensor | None = None)`。

- [ ] **Step 1: 写 physical 回归与职责隔离 RED 测试**

```python
def test_optional_scale_target_changes_only_scale_branch(self):
    physical = self._teacher(probabilities=(0.7, 0.1, 0.1, 0.1), scores=(0.1, 0.2, 0.3, 0.4))
    fallback = pcnss_loss(output, physical, target_lags, angles, mask, counts, epoch=10)
    explicit_physical = pcnss_loss(
        output, physical, target_lags, angles, mask, counts, epoch=10,
        scale_distillation_target=physical.scale_probabilities,
    )
    hard = pcnss_loss(
        output, physical, target_lags, angles, mask, counts, epoch=10,
        scale_distillation_target=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    self.assertEqual(fallback.scale.item(), explicit_physical.scale.item())
    for field in ("lag", "residual", "peak", "dominance"):
        self.assertEqual(getattr(fallback, field).item(), getattr(hard, field).item())
    self.assertNotEqual(fallback.scale.item(), hard.scale.item())
```

- [ ] **Step 2: 运行 RED**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_losses.ResolutionAwareLossTest.test_optional_scale_target_changes_only_scale_branch -v`
Expected: `pcnss_loss()` 不接受 `scale_distillation_target`。

- [ ] **Step 3: 实现严格 target 验证与 fallback**

```python
def _scale_distillation_target(
    physical: torch.Tensor,
    override: torch.Tensor | None,
) -> torch.Tensor:
    target = physical if override is None else override
    target = target.to(device=physical.device, dtype=physical.dtype)
    if target.shape != physical.shape:
        raise ValueError("scale_distillation_target must have shape [batch,4]")
    if not torch.isfinite(target).all() or (target < 0.0).any():
        raise ValueError("scale_distillation_target must be finite and non-negative")
    if not torch.allclose(
        target.sum(dim=-1), torch.ones(target.shape[0], device=target.device),
        atol=1e-6, rtol=0.0,
    ):
        raise ValueError("scale_distillation_target rows must sum to one")
    return target.detach()
```

在 `pcnss_loss()` 中只替换：

```python
scale_target = _scale_distillation_target(
    teacher.scale_probabilities.to(distribution.device),
    scale_distillation_target,
)
scale = scale_distillation_loss(scale_target, distribution)
best_score = teacher.scale_scores.to(predicted_score.device).max(dim=-1).values
```

- [ ] **Step 4: 增加错误 shape/nonfinite/negative/not-normalized 测试并运行 GREEN**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_losses test_multisource.test_teacher -v`
Expected: 全部通过；原 teacher 测试不变。

- [ ] **Step 5: 提交 Task 4**

```powershell
git add -- multisource_doa/training/losses.py test_multisource/test_losses.py
git diff --cached --check
git commit -m "feat: isolate scale distillation target"
```

---

### Task 5: Training Engine 按 Sample Seed 注入 Cache 标签

**Files:**
- Modify: `multisource_doa/training/engine.py`
- Modify: `test_multisource/test_training_engine.py`

**Interfaces:**
- Consumes: `Mapping[int, tuple[float,float,float,float]]`。
- Produces: `train_one_epoch(..., scale_targets_by_seed: Mapping[...] | None = None)` 和 `_batch_scale_target()`。

- [ ] **Step 1: 写 batch lookup RED 测试**

```python
def test_cached_targets_follow_batch_seed_order_and_missing_seed_fails(self):
    batch = collate_samples(self.samples)
    reversed_lookup = {
        seed: tuple(float(index == position) for index in range(4))
        for position, seed in enumerate(reversed(batch.sample_seeds))
    }
    target = _batch_scale_target(batch, reversed_lookup, torch.device("cpu"))
    self.assertEqual(target.shape, (len(self.samples), 4))
    self.assertEqual(tuple(target[0].tolist()), reversed_lookup[batch.sample_seeds[0]])
    with self.assertRaises(KeyError):
        _batch_scale_target(batch, {}, torch.device("cpu"))
```

- [ ] **Step 2: 运行 RED**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_training_engine.TrainingEngineTest.test_cached_targets_follow_batch_seed_order_and_missing_seed_fails -v`
Expected: `_batch_scale_target` 尚未定义。

- [ ] **Step 3: 实现 lookup 和可选训练参数**

```python
ScaleTargetLookup = Mapping[int, tuple[float, float, float, float]]

def _batch_scale_target(
    batch: PCNSSBatch,
    lookup: ScaleTargetLookup,
    device: torch.device,
) -> torch.Tensor:
    missing = [seed for seed in batch.sample_seeds if seed not in lookup]
    if missing:
        raise KeyError(f"teacher cache missing sample seeds: {missing[:4]}")
    return torch.tensor(
        [lookup[seed] for seed in batch.sample_seeds],
        dtype=torch.float32, device=device,
    )
```

`train_one_epoch()` 新增 keyword-only 参数；每个 batch 在 `pcnss_loss()` 前构造 target，None 时不构造也不查询。

- [ ] **Step 4: 写 physical path 同初始化同 batch 的一步等价测试**

使用两个相同 state dict 的模型、相同 Adam、相同 batch；一个不传 mapping，另一个显式传物理 teacher probabilities 的 mapping。固定 `torch.manual_seed` 后比较所有 loss 指标和更新后参数在 `atol=1e-7, rtol=1e-6` 内一致。

- [ ] **Step 5: 运行 GREEN**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_training_engine test_multisource.test_losses -v`
Expected: 全部通过，locked split 仍被拒绝。

- [ ] **Step 6: 提交 Task 5**

```powershell
git add -- multisource_doa/training/engine.py test_multisource/test_training_engine.py
git commit -m "feat: inject cached scale targets by seed"
```

---

### Task 6: 单因素身份审计与三文件只读报告

**Files:**
- Create: `multisource_doa/training/single_factor_audit.py`
- Create: `multisource_doa/training/single_factor_reporting.py`
- Create: `test_multisource/test_single_factor_audit.py`

**Interfaces:**
- Consumes: baseline training directory、baseline validation report、Task 16 八文件目录、Task 2 cache、当前 frozen config。
- Produces: `SingleFactorAuditResult`、`audit_single_factor_inputs()`、`write_single_factor_audit_report()`。

- [ ] **Step 1: 写 Task 16 与 baseline 身份 RED 测试**

```python
def test_audit_requires_ranking_invalid_and_matching_frozen_controls(self):
    inputs = self._valid_inputs()
    result = audit_single_factor_inputs(**inputs)
    self.assertTrue(result.baseline_reuse_allowed)
    self.assertTrue(all(result.gates.values()))

    self._write_json(
        Path(inputs["task16_directory"]) / "decision.json",
        {"mechanism_conclusion": "calibration_only", "training_authorized": False},
    )
    with self.assertRaises(ValueError):
        audit_single_factor_inputs(**inputs)
```

- [ ] **Step 2: 运行 RED**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_single_factor_audit -v`
Expected: `single_factor_audit` 尚未创建。

- [ ] **Step 3: 实现输入文件集合与 gate**

```python
TASK16_FILES = (
    "diagnostic_config.json", "source_manifest.json",
    "teacher_ranking_sample_diagnostics.csv", "teacher_ranking_summary.json",
    "teacher_component_summary.json", "teacher_ranking_stratified_summary.csv",
    "teacher_oracle_confusion.csv", "decision.json",
)
BASELINE_TRAINING_FILES = (
    "train_manifest.json", "validation_manifest.json", "metrics.csv",
    "best.pt", "best.pt.sha256.json",
)
EVALUATION_REPORT_FILES = (
    "run_config.json", "source_manifest.json", "predictions.csv", "summary.json",
    "paired_comparisons.csv", "failure_reasons.csv", "runtime_summary.json",
)

@dataclass(frozen=True)
class SingleFactorAuditResult:
    baseline_reuse_allowed: bool
    gates: Mapping[str, bool]
    evidence: Mapping[str, Any]
    source_sha256: Mapping[str, Mapping[str, str]]
```

`audit_single_factor_inputs()` 必须校验：精确文件集合、每文件 SHA、Task 16 schema-v1、`mechanism_conclusion=ranking_invalid`、`training_authorized=false`、current/q_midpoint pairwise 值与用户冻结结论一致；checkpoint sidecar SHA 与 `best.pt` 一致；checkpoint 的 config/model seed/validation seed/parameter count/selection metric 与 baseline manifests 和当前 config 一致；baseline report checkpoint SHA 与 checkpoint 一致；cache 三文件已通过 loader；teacher 唯一变量和 current code 允许差异说明存在。

不能证明相同设备、初始化、batch 顺序或训练关键代码身份时不抛弃报告，而是设置对应 gate false、`baseline_reuse_allowed=false`，从而要求先重跑 physical A。输入损坏、SHA 不一致、错误 split 或 Task 16 不是 ranking_invalid 则直接 ValueError。

- [ ] **Step 4: 写审计报告 RED 测试**

```python
def test_writer_emits_exact_three_files_even_when_reuse_is_denied(self):
    result = SingleFactorAuditResult(
        baseline_reuse_allowed=False,
        gates={"same_model_seed": True, "same_training_environment": False},
        evidence={"required_action": "rerun_physical_control"},
        source_sha256={"task16": {"decision.json": "a" * 64}},
    )
    write_single_factor_audit_report(result, output, run_config={"stage": "audit_single_factor"})
    self.assertEqual(
        {path.name for path in output.iterdir()},
        {"audit_config.json", "source_manifest.json", "single_factor_audit.json"},
    )
```

- [ ] **Step 5: 实现 finite JSON、拒绝覆盖和决定字段**

`single_factor_audit.json` 固定包含 `baseline_reuse_allowed`、所有 gates、失败原因、`required_action`（`reuse_baseline` 或 `rerun_physical_control`）、`training_authorized=false`。身份审计只决定能否复用 A，不授权 B 训练。

- [ ] **Step 6: 运行 GREEN 并提交**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_single_factor_audit test_multisource.test_artifacts -v`
Expected: 全部通过。

```powershell
git add -- multisource_doa/training/single_factor_audit.py multisource_doa/training/single_factor_reporting.py test_multisource/test_single_factor_audit.py
git commit -m "feat: audit single-factor teacher identity"
```

---

### Task 7: 单因素审计安全入口

**Files:**
- Create: `scripts/audit_pcnss_teacher_single_factor.py`
- Modify: `test_multisource/test_entrypoints.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 6 audit API。
- Produces: `dry_run`、`smoke`、`audit_single_factor` 三个 stage。

- [ ] **Step 1: 写入口 RED 测试**

```python
class SingleFactorAuditEntrypointTest(unittest.TestCase):
    def test_default_reads_nothing_and_creates_nothing(self):
        namespace = runpy.run_path(str(SINGLE_FACTOR_AUDIT_SCRIPT))
        result = namespace["run_stage"](dict(namespace["RUN_CONFIG"]))
        self.assertEqual(result["stage"], "dry_run")
        self.assertFalse(result["output_created"])
        self.assertFalse(result["training_performed"])

    def test_formal_requires_all_four_authenticated_sources(self):
        namespace = runpy.run_path(str(SINGLE_FACTOR_AUDIT_SCRIPT))
        values = dict(namespace["RUN_CONFIG"], stage="audit_single_factor", dry_run=False)
        with self.assertRaises(ValueError):
            namespace["run_stage"](values)
```

- [ ] **Step 2: 实现安全 RUN_CONFIG**

```python
RUN_CONFIG = {
    "stage": "dry_run", "dry_run": True,
    "baseline_training_directory": "", "baseline_validation_directory": "",
    "task16_directory": "", "teacher_cache_directory": "",
    "output_root": "outputs/pcnss_teacher_single_factor_audit",
    "device": "cpu", "allow_locked_test": False, "overwrite": False,
}
STAGES = ("dry_run", "smoke", "audit_single_factor")
```

formal 只读四个源目录并写新的三文件目录；目录存在拒绝；unknown key、non-CPU、locked、overwrite 被拒绝。smoke 使用临时合成 manifests/checkpoint/report/cache，不读取正式 validation。

- [ ] **Step 3: 运行目标测试和默认 dry-run**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints.SingleFactorAuditEntrypointTest test_multisource.test_single_factor_audit -v`
Expected: 全部通过。

Run: `D:\Python\Python\python.exe scripts\audit_pcnss_teacher_single_factor.py`
Expected: `output_created=false`、`training_performed=false`。

- [ ] **Step 4: 更新 README 并提交**

README 明确 audit false 时先重跑 physical A，禁止先看 B validation 再补 A。

```powershell
git add -- scripts/audit_pcnss_teacher_single_factor.py test_multisource/test_entrypoints.py README.md
git commit -m "feat: add safe single-factor audit entrypoint"
```

---

### Task 8: 正式训练入口、Manifest 与 Checkpoint 身份接入

**Files:**
- Modify: `multisource_doa/data/manifest.py`
- Modify: `multisource_doa/training/artifacts.py`
- Modify: `scripts/run_multiscale_pcnss.py`
- Modify: `multisource_doa/evaluation/reporting.py`
- Modify: `test_multisource/test_artifacts.py`
- Modify: `test_multisource/test_entrypoints.py`
- Modify: `test_multisource/test_evaluation_runner.py`

**Interfaces:**
- Consumes: Task 2 `TeacherCache`、Task 6 三文件 audit、Task 5 engine mapping。
- Produces: `teacher_mode`/`teacher_cache_path`/`single_factor_audit_path` 正式训练配置和 checkpoint `training_metadata`。

- [ ] **Step 1: 写 manifest/checkpoint 可选元数据 RED 测试**

```python
def test_checkpoint_records_teacher_identity_without_changing_default_api(self):
    manager.update(
        metric_value=1.0, epoch=0, model=model, optimizer=optimizer,
        experiment_config=ExperimentConfig(), model_seed=2026,
        data_split_seed=202_708_040, code_sha="abc", split=SplitName.VALIDATION,
        training_metadata={
            "teacher_mode": "failure_aware_error",
            "scale_distillation_target_source": "train_only_failure_aware_rmspe",
            "dominance_target_source": "physical_music_score",
            "teacher_cache_sha256": "a" * 64,
            "single_factor_audit_sha256": "b" * 64,
        },
    )
    payload = torch.load(output / "best.pt", weights_only=False)
    self.assertEqual(payload["training_metadata"]["teacher_mode"], "failure_aware_error")
```

- [ ] **Step 2: 实现 JSON-safe extra metadata**

`write_split_manifest(..., extra_metadata: Mapping[str, Any] | None = None)` 仅在非 None 时追加 `training_metadata`；`CheckpointManager.update(..., training_metadata=None)` 同理。现有调用不传时保持原字段集合。

- [ ] **Step 3: 写训练入口 guard RED 测试**

```python
def test_failure_aware_training_requires_cache_and_passing_audit_before_model(self):
    namespace = runpy.run_path(str(MAIN_SCRIPT))
    values = dict(
        namespace["RUN_CONFIG"], stage="train", dry_run=False,
        teacher_mode="failure_aware_error", teacher_cache_path="",
        single_factor_audit_path="", output_root="unused",
    )
    with mock.patch.object(namespace["MultiScalePCNSS"], "__init__", side_effect=AssertionError):
        with self.assertRaises(ValueError):
            namespace["run_stage"](values)
```

- [ ] **Step 4: 扩展 RUN_CONFIG 并在模型创建前认证**

```python
RUN_CONFIG.update({
    "teacher_mode": "physical",
    "teacher_cache_path": "",
    "single_factor_audit_path": "",
})
TEACHER_MODES = ("physical", "failure_aware_error")
```

规则：physical 模式拒绝非空 cache/audit；failure-aware 模式要求两者存在，先 `load_teacher_cache(..., expected_count=40_000)`，再认证 audit 三文件 SHA、`baseline_reuse_allowed=true`、cache SHA 相同，之后才创建 output、model、optimizer。若 audit 要求重跑 A，抛出 `PermissionError("rerun physical control before candidate training")`。

`run_formal_train()` 将 `cache.labels_by_seed` 传入每个 `train_one_epoch()`；validation 不读取标签。写入 train manifest 和 checkpoint 的 metadata 固定为：

```python
training_metadata = {
    "teacher_mode": values["teacher_mode"],
    "scale_distillation_target_source": (
        "physical_music_score" if values["teacher_mode"] == "physical"
        else "train_only_failure_aware_rmspe"
    ),
    "dominance_target_source": "physical_music_score",
    "teacher_cache_sha256": cache_manifest_sha_or_none,
    "single_factor_audit_sha256": audit_decision_sha_or_none,
    "teacher_label_counts": cache_manifest_counts_or_none,
}
```

- [ ] **Step 5: 为 smoke 增加 4 样本 failure-aware 路径**

smoke 在 `teacher_mode=failure_aware_error` 时只接受 4-row smoke cache 和 passing synthetic audit；运行一个 batch/epoch，不写 formal checkpoint。默认 smoke 仍为 physical。

- [ ] **Step 6: 评价 manifest 透传 checkpoint teacher metadata**

`run_formal_evaluation()` 从 checkpoint payload 读取 `training_metadata`，传入 `write_evaluation_report()`；`source_manifest.json` 记录该对象。旧 checkpoint 缺少字段时写 `training_metadata=None`，但 Task 17 正式结果审计会拒绝将其当作 B。

- [ ] **Step 7: 运行 GREEN 与完整入口回归**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_artifacts test_multisource.test_training_engine test_multisource.test_entrypoints test_multisource.test_evaluation_runner -v`
Expected: 全部通过；默认 physical dry-run 参数量仍为 46916，locked 无入口。

- [ ] **Step 8: 提交 Task 8**

```powershell
git add -- multisource_doa/data/manifest.py multisource_doa/training/artifacts.py multisource_doa/evaluation/reporting.py scripts/run_multiscale_pcnss.py test_multisource/test_artifacts.py test_multisource/test_entrypoints.py test_multisource/test_evaluation_runner.py
git diff --cached --check
git commit -m "feat: wire audited error teacher training"
```

---

### Task 9: 冻结 Validation Gate、配对统计与五文件结果审计

**Files:**
- Create: `multisource_doa/evaluation/teacher_experiment.py`
- Create: `multisource_doa/evaluation/teacher_experiment_reporting.py`
- Create: `scripts/audit_pcnss_teacher_experiment.py`
- Create: `test_multisource/test_teacher_experiment.py`
- Modify: `test_multisource/test_entrypoints.py`

**Interfaces:**
- Consumes: A/B 两份 schema-v2 validation report、B checkpoint metadata、passing single-factor audit 和 cache identity。
- Produces: `TeacherExperimentResult`、`audit_teacher_experiment()`、`write_teacher_experiment_report()`、冻结 decision。

- [ ] **Step 1: 写六个 conjunctive gate 的 RED 测试**

```python
def test_decision_requires_every_frozen_gate(self):
    result = audit_teacher_experiment(self.baseline_report, self.candidate_report)
    self.assertEqual(result.decision["conclusion"], "seed2026_gate_passed")
    self.assertTrue(all(result.decision["gates"].values()))

    for gate, mutation in self._gate_failure_mutations().items():
        with self.subTest(gate=gate):
            failed = audit_teacher_experiment(
                self.baseline_report, self._mutated_candidate(mutation)
            )
            self.assertFalse(failed.decision["gates"][gate])
            self.assertEqual(failed.decision["conclusion"], "experiment_failed")
            self.assertFalse(failed.decision["development_authorized"])
```

六个 gate 名固定为：

```python
gates = {
    "near_resolution_improves_over_original": candidate_near > original_near,
    "near_resolution_not_below_fbss_L7": candidate_near >= l7_near,
    "overall_rmspe_not_worse": candidate_rmspe <= original_rmspe,
    "overall_resolution_not_worse": candidate_resolution >= original_resolution,
    "failure_count_not_worse": candidate_failures <= original_failures,
    "protocol_identity": protocol_identity,
}
```

- [ ] **Step 2: 运行 RED**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_experiment -v`
Expected: `teacher_experiment` 尚未创建。

- [ ] **Step 3: 实现 report 认证和逐 seed join**

要求 A/B 各自恰有 schema-v2 七文件；读取 predictions 后按 `(algorithm, sample_seed)` 唯一索引。两边必须都是 validation、恰好 5,000 seeds、样本元数据一致、B 的 `training_metadata.teacher_mode=failure_aware_error`、single-factor audit/cache SHA 相同；两边 L7 预测必须逐 seed 一致。任何重复、缺失、非有限、不同计分字段或不是 best L7 立即失败。

- [ ] **Step 4: 实现配对、状态转移、离群和固定分层**

`TeacherExperimentResult` 固定包含：

```python
@dataclass(frozen=True)
class TeacherExperimentResult:
    decision: Mapping[str, Any]
    paired_rows: tuple[Mapping[str, Any], ...]
    transition_rows: tuple[Mapping[str, Any], ...]
    stratified_rows: tuple[Mapping[str, Any], ...]
    source_sha256: Mapping[str, Mapping[str, str]]
```

配对 RMSPE tie 容差 `1e-6°`；near resolved 状态记录 `A0_B0/A0_B1/A1_B0/A1_B1`。使用 `scipy.stats.binomtest(A0_B1, A0_B1 + A1_B0, 0.5)` 报告 McNemar exact p 值及 discordant improvement fraction 的 95% exact CI，但这些不进入 gate。分层固定 separation/SNR/rho/snapshot；每组报告 count、A/B/L7 RMSPE、resolution、failure 和 B-vs-A win/tie/loss。总体及 near 报告 sample RMSPE `>10/>30/>60°`。

- [ ] **Step 5: 写五文件 writer 和拒绝覆盖测试**

输出恰好：

1. `experiment_audit_config.json`；
2. `source_manifest.json`；
3. `paired_and_transitions.csv`；
4. `stratified_summary.csv`；
5. `decision.json`。

`decision.json` 固定包含 gates、精确来源值、`conclusion`、`development_authorized=false`、`multi_seed_authorized=false`、`locked_test_authorized=false` 和失败后的 `required_action=stop_without_tuning`。即使 seed2026 通过，也只写 `next_action=request_development_approval`。

- [ ] **Step 6: 实现安全脚本**

```python
RUN_CONFIG = {
    "stage": "dry_run", "dry_run": True,
    "baseline_validation_directory": "",
    "candidate_validation_directory": "",
    "single_factor_audit_directory": "",
    "teacher_cache_directory": "",
    "output_root": "outputs/pcnss_teacher_experiment_audit",
    "split": "validation", "device": "cpu",
    "allow_locked_test": False, "overwrite": False,
}
STAGES = ("dry_run", "smoke", "audit_validation_teacher_experiment")
```

formal 只读既有报告，不加载模型、不运行 evaluator、不训练；default dry-run 不读路径、不创建输出；smoke 使用内存/临时 4 seed 报告。

- [ ] **Step 7: 运行 GREEN 与 evaluator 回归**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_experiment test_multisource.test_entrypoints test_multisource.test_evaluation test_multisource.test_evaluation_runner -v`
Expected: 全部通过。

Run: `D:\Python\Python\python.exe scripts\audit_pcnss_teacher_experiment.py`
Expected: `stage=dry_run`、`no_model_forward=true`、`training_performed=false`、`output_created=false`。

- [ ] **Step 8: 提交 Task 9**

```powershell
git add -- multisource_doa/evaluation/teacher_experiment.py multisource_doa/evaluation/teacher_experiment_reporting.py scripts/audit_pcnss_teacher_experiment.py test_multisource/test_teacher_experiment.py test_multisource/test_entrypoints.py
git commit -m "feat: audit frozen teacher experiment gates"
```

---

### Task 10: 协议文档、完整验证、代码审查与用户交接

**Files:**
- Modify: `README.md`
- Modify: `experiments/formal_training_protocol.md`
- Review: all Task 17 source/test files

**Interfaces:**
- Consumes: Tasks 1–9 全部入口。
- Produces: 用户可逐步执行且默认安全的正式运行协议；不产生正式实验结果。

- [ ] **Step 1: 写完整用户运行顺序**

在 `experiments/formal_training_protocol.md` 增加 Task 17 章节，固定顺序：

1. `build_pcnss_failure_aware_teacher_cache.py` 默认 dry-run；
2. 4 样本 cache smoke；
3. 用户创建未跟踪 formal JSON，运行一次 40,000 cache；
4. 独立复算 cache SHA、40,000 seeds、概率和 label counts；
5. `audit_pcnss_teacher_single_factor.py`；
6. audit 不通过则先重跑 physical A，并重新审计；
7. audit 通过后一次 seed 2026 B 训练；
8. 一次 `evaluate_validation`；
9. `audit_pcnss_teacher_experiment.py` 只读判门；
10. 失败停止，成功仅申请 development 审批。

每个正式 JSON 示例都使用新 output path、`overwrite=false`、`allow_locked_test=false`，并注明文件不提交 Git。不得在文档中给出 development/locked 绕过命令。

- [ ] **Step 2: 运行目标测试集合**

Run:

```powershell
D:\Python\Python\python.exe -m unittest `
  test_multisource.test_error_teacher `
  test_multisource.test_teacher_cache `
  test_multisource.test_single_factor_audit `
  test_multisource.test_teacher_experiment `
  test_multisource.test_losses `
  test_multisource.test_training_engine `
  test_multisource.test_entrypoints -v
```

Expected: 全部通过，0 failures/errors。

- [ ] **Step 3: 运行完整工程测试与 compileall**

Run: `D:\Python\Python\python.exe -m unittest discover -s test_multisource -v`
Expected: 全部通过。

Run: `D:\Python\Python\python.exe -m compileall -q multisource_doa scripts test_multisource`
Expected: exit code 0，无输出。

- [ ] **Step 4: 运行所有默认 dry-run**

```powershell
D:\Python\Python\python.exe scripts\run_multiscale_pcnss.py
D:\Python\Python\python.exe scripts\build_pcnss_failure_aware_teacher_cache.py
D:\Python\Python\python.exe scripts\audit_pcnss_teacher_single_factor.py
D:\Python\Python\python.exe scripts\audit_pcnss_teacher_experiment.py
```

Expected: 均不创建正式输出；主程序仍报告 `parameter_count=46916`；三个 Task 17 入口均报告无训练、无 locked access。

- [ ] **Step 5: 运行 4 样本 cache、训练、审计 smoke**

为每个 smoke 使用新的临时 output path。验证 cache 恰好三文件、single-factor audit 恰好三文件、结果 audit 恰好五文件；训练 smoke 只运行 1 epoch/4 样本且不写 formal checkpoint。smoke 完成后逐文件检查 manifest 的 split、SHA、teacher source 和 `training_performed`。

- [ ] **Step 6: 完成科研与安全审查**

逐项确认：

- validation/development/locked 数据没有进入 cache；
- KL 以外的 loss 数值和梯度回归不变；
- failure-aware `60°` 和失败样本保留；
- formal output 不可覆盖；
- 没有 API key、Token、绝对用户缓存路径或正式输出被追踪；
- B 训练在 cache/audit 失败时早于 model/optimizer 创建终止；
- result audit 不运行第二次 evaluator；
- gate 使用精确值且全部 conjunctive；
- 通过不自动授权 development/multi-seed/locked。

- [ ] **Step 7: 审查 Git diff 和提交文档**

```powershell
git status --short
git diff --check
git diff --stat master...HEAD
git add -- README.md experiments/formal_training_protocol.md
git diff --cached --name-only
git commit -m "docs: add failure-aware teacher protocol"
```

- [ ] **Step 8: 请求代码审查并推送分支**

先使用 `superpowers:requesting-code-review` 审查规格覆盖、单因素边界、split 安全、cache 身份、测试证据和 Git diff；处理 Critical/Important 后重新运行完整验证。随后推送当前 `codex/` 分支，不创建或合并 PR，除非用户另行要求。

```powershell
git push -u origin codex/task-17-failure-aware-teacher-spec
```

- [ ] **Step 9: 用户交接**

报告每个提交 SHA、改动文件、测试数量、dry-run/smoke 证据和未运行项目。明确下一步只由用户审核实现；未获批准前不得生成 40,000 cache 或启动 seed 2026 正式训练。
