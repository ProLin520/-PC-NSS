# PC-NSS Teacher 尺度置信只读诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个只在 CPU 上重建冻结 validation 近间隔样本、计算固定尺度 teacher 置信与可靠性，并依据预注册门输出结构化结论的独立诊断入口。

**Architecture:** 新的 `teacher_confidence.py` 负责认证两个既有报告、计算 teacher/student/oracle 样本指标；新的 `teacher_reporting.py` 负责固定分层、科研门判定和六文件 schema-v1 输出；独立脚本只负责编排安全配置、确定性重建样本和调用上述接口。实现不加载 checkpoint、不实例化 PC-NSS、不执行模型前向，也不修改训练或评价代码。

**Tech Stack:** Python 3.10、PyTorch、NumPy、标准库 `csv/json/hashlib/pathlib/unittest`。

## Global Constraints

- 只允许 `validation`，禁止 development 与 locked test。
- 正式诊断固定 `device="cpu"`、`batch_size=128`、`tau_current=0.10`、`tau_counterfactual=0.05`。
- 正式集合固定为 5000 个 source seed、1270 个 `[2,4)` near seed和 `L={4,5,6,7}`。
- 不读取 checkpoint 文件；只比较 manifest 中已经记录的 checkpoint SHA。
- 不实例化 `MultiScalePCNSS`，不运行神经模型前向，不计算梯度，不训练。
- 输出目录存在即拒绝，JSON 禁止 NaN/Infinity，CSV 必须能独立复算汇总与 decision。
- 不提交 `outputs/`、checkpoint、权重、生成数据、临时正式配置或 `.superpowers/`。
- 保留用户现有 `scripts/run_multiscale_pcnss.py` 工作区改动，不暂存、不覆盖。
- 所有实现步骤在当前隔离 worktree 内单人执行，不使用子智能体。

---

## File Structure

- Create `multisource_doa/diagnostics/teacher_confidence.py`: 输入认证、样本重建校验、teacher/student/oracle 数学指标和 CPU 批处理。
- Create `multisource_doa/diagnostics/teacher_reporting.py`: 完整分层、总体汇总、科研门判定、六文件不可覆盖写出。
- Create `scripts/diagnose_pcnss_teacher_confidence.py`: 默认安全 dry-run 与正式只读编排入口。
- Create `test_multisource/test_teacher_confidence_diagnostic.py`: 输入认证、概率指标、oracle、批处理与无模型前向契约。
- Create `test_multisource/test_teacher_confidence_reporting.py`: 分层、判定边界、schema、finite、拒绝覆盖和 CSV 复算。
- Modify `test_multisource/test_entrypoints.py`: 新入口的安全配置、未知键、CPU/128/validation 限制和 4 样本 smoke。
- Modify `README.md`: 增加 teacher 诊断用途、安全边界、默认运行和输出说明。
- Create `experiments/task15_teacher_confidence_diagnostic.md`: 正式只读结果、门判定和下一步结论；只在唯一正式运行及独立复算后写入。

---

### Task 1: 冻结输入认证

**Files:**
- Create: `multisource_doa/diagnostics/teacher_confidence.py`
- Test: `test_multisource/test_teacher_confidence_diagnostic.py`

**Interfaces:**
- Consumes: audit-v4 `run_config.json/summary.json/source_manifest.json/predictions.csv` 与 Task 14 `source_manifest.json/near_sample_diagnostics.csv`。
- Produces: `TeacherDiagnosticInputs`, `TeacherAuthorityLabel`, `load_teacher_diagnostic_inputs(report_directory, task14_directory, expected_source_count=5000, expected_near_count=1270)`。

- [ ] **Step 1: 写失败测试，固定五算法集合和两个报告的身份契约**

```python
class TeacherInputAuthenticationTest(unittest.TestCase):
    def test_loads_five_complete_algorithms_and_exact_near_seed_set(self):
        audit, task14 = self._write_authenticated_inputs(source_count=2)
        loaded = load_teacher_diagnostic_inputs(
            audit, task14, expected_source_count=2, expected_near_count=1
        )
        self.assertEqual(tuple(loaded.labels_by_seed), (202708040,))
        self.assertEqual(
            set(loaded.labels_by_seed[202708040].fixed_rmspe_deg),
            {4, 5, 6, 7},
        )

    def test_rejects_duplicate_missing_or_metadata_mismatched_rows(self):
        for mutation, message in (
            ("duplicate_seed", "duplicate sample_seed"),
            ("missing_algorithm_row", "sample_seed sets"),
            ("metadata_mismatch", "metadata mismatch"),
        ):
            audit, task14 = self._write_authenticated_inputs(mutation=mutation)
            with self.assertRaisesRegex(ValueError, message):
                load_teacher_diagnostic_inputs(
                    audit, task14, expected_source_count=2, expected_near_count=1
                )

    def test_rejects_manifest_hash_checkpoint_and_student_probability_mismatches(self):
        for mutation, message in (
            ("audit_hash", "audit input SHA"),
            ("checkpoint_sha", "checkpoint SHA"),
            ("near_seed_set", "near sample_seed set"),
            ("student_probability", "student probabilities"),
        ):
            audit, task14 = self._write_authenticated_inputs(mutation=mutation)
            with self.assertRaisesRegex(ValueError, message):
                load_teacher_diagnostic_inputs(
                    audit, task14, expected_source_count=2, expected_near_count=1
                )
```

- [ ] **Step 2: 运行 RED 并确认因缺少新模块失败**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_diagnostic.TeacherInputAuthenticationTest -v`

Expected: `ImportError` 或 `ModuleNotFoundError` 指向 `multisource_doa.diagnostics.teacher_confidence`，而不是测试夹具错误。

- [ ] **Step 3: 实现最小认证数据结构与 loader**

```python
SCALE_SIZES = (4, 5, 6, 7)
ALGORITHMS = ("pcnss_root_music",) + tuple(
    f"fbss_root_music_L{size}" for size in SCALE_SIZES
)
METADATA_FIELDS = (
    "true_angle_1_deg", "true_angle_2_deg", "rho", "snr_db",
    "snapshot_count", "separation_deg",
)

@dataclass(frozen=True)
class TeacherAuthorityLabel:
    sample_seed: int
    true_angles_deg: tuple[float, float]
    rho: float
    snr_db: float
    snapshot_count: int
    separation_deg: float
    threshold_cohort: str
    student_probabilities: tuple[float, float, float, float]
    fixed_rmspe_deg: dict[int, float]

@dataclass(frozen=True)
class TeacherDiagnosticInputs:
    labels_by_seed: dict[int, TeacherAuthorityLabel]
    source_manifest: dict[str, Any]
    input_sha256: dict[str, str]

def load_teacher_diagnostic_inputs(
    report_directory: str | Path,
    task14_directory: str | Path,
    *,
    expected_source_count: int = 5000,
    expected_near_count: int = 1270,
) -> TeacherDiagnosticInputs:
    audit = Path(report_directory)
    task14 = Path(task14_directory)
    run_config = _read_json(audit / "run_config.json")
    summary = _read_json(audit / "summary.json")
    audit_manifest = _read_json(audit / "source_manifest.json")
    task14_manifest = _read_json(task14 / "source_manifest.json")
    _require_validation_schema_v2(run_config, summary)
    audit_hashes = {
        name: _sha256(audit / name)
        for name in ("run_config.json", "summary.json", "source_manifest.json", "predictions.csv")
    }
    if task14_manifest.get("audit_input_sha256") != audit_hashes:
        raise ValueError("audit input SHA mismatch")
    if task14_manifest.get("checkpoint_sha") != audit_manifest.get("checkpoint_sha"):
        raise ValueError("checkpoint SHA mismatch")
    algorithm_rows = _read_and_validate_algorithms(
        audit / "predictions.csv", expected_source_count
    )
    student_rows = _read_and_validate_task14_rows(
        task14 / "near_sample_diagnostics.csv", expected_near_count
    )
    labels = _join_near_authority(algorithm_rows, student_rows)
    return TeacherDiagnosticInputs(
        labels_by_seed={label.sample_seed: label for label in labels},
        source_manifest={"audit": audit_manifest, "task14": task14_manifest},
        input_sha256={
            **{f"audit/{key}": value for key, value in audit_hashes.items()},
            "task14/source_manifest.json": _sha256(task14 / "source_manifest.json"),
            "task14/near_sample_diagnostics.csv": _sha256(task14 / "near_sample_diagnostics.csv"),
        },
    )
```

`_read_and_validate_algorithms` 必须按 algorithm 建立唯一 seed 索引，要求每组计数等于 `expected_source_count`、五组 seed 集合相同、`split == "validation"`，并逐 seed 比较 `METADATA_FIELDS`。`_read_and_validate_task14_rows` 必须要求 schema version 1、行数与唯一 seed 数均等于 `expected_near_count`、`2 <= separation_deg < 4`，且四个学生概率有限、非负、绝对误差 `1e-6` 内和为 1。`_join_near_authority` 必须要求 Task 14 seed 集合恰好等于 audit PC-NSS 的 `[2,4)` 集合，并从 L4–L7 行提取 `sample_rmspe_deg`。

- [ ] **Step 4: 运行 GREEN 与已有近间隔 loader 回归**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_diagnostic.TeacherInputAuthenticationTest test_multisource.test_near_resolution_diagnostic.NearAuditLoadTest -v`

Expected: 全部通过；旧 Task 14 loader 行为不变。

- [ ] **Step 5: 精确提交 Task 1**

```powershell
git add -- multisource_doa/diagnostics/teacher_confidence.py test_multisource/test_teacher_confidence_diagnostic.py
git diff --cached --check
git commit -m "feat: authenticate teacher diagnostic inputs"
```

---

### Task 2: Teacher/student/oracle 样本指标与 CPU 批处理

**Files:**
- Modify: `multisource_doa/diagnostics/teacher_confidence.py`
- Modify: `test_multisource/test_teacher_confidence_diagnostic.py`

**Interfaces:**
- Consumes: Task 1 `TeacherAuthorityLabel`、`DOASample`、`collate_samples()`、`build_scale_teacher()`。
- Produces: `TeacherDiagnosticResult`, `distribution_metrics()`, `build_teacher_sample_row()`, `diagnose_teacher_samples()`。

- [ ] **Step 1: 写概率、oracle tie、regret 和顺序保持的失败测试**

```python
class TeacherMetricTest(unittest.TestCase):
    def test_uniform_entropy_divergence_oracle_tie_and_regret(self):
        row = build_teacher_sample_row(
            label=self._label(
                student=(0.25, 0.25, 0.25, 0.25),
                rmspe={4: 1.0, 5: 1.0 + 5e-7, 6: 2.0, 7: 3.0},
            ),
            scores=torch.tensor([0.4, 0.3, 0.2, 0.1]),
            probabilities_current=torch.tensor([0.4, 0.3, 0.2, 0.1]),
            probabilities_counterfactual=torch.tensor([0.7, 0.2, 0.08, 0.02]),
        )
        self.assertAlmostEqual(row["student_entropy_normalized"], 1.0)
        self.assertEqual(row["oracle_best_scales"], (4, 5))
        self.assertTrue(row["teacher_oracle_agreement"])
        self.assertAlmostEqual(row["teacher_regret_deg"], 0.0)
        self.assertGreaterEqual(row["teacher_student_kl"], 0.0)
        self.assertGreaterEqual(row["teacher_student_js"], 0.0)

    def test_cpu_batches_preserve_seed_order_and_use_no_model(self):
        result = diagnose_teacher_samples(
            self.samples,
            self.labels_by_seed,
            batch_size=2,
            tau_current=0.10,
            tau_counterfactual=0.05,
        )
        self.assertEqual(
            [row["sample_seed"] for row in result.sample_rows],
            [sample.sample_seed for sample in self.samples],
        )
        self.assertEqual(len(result.sample_rows), 4)
```

- [ ] **Step 2: 运行 RED 并确认缺少指标接口**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_diagnostic.TeacherMetricTest -v`

Expected: FAIL 指向 `build_teacher_sample_row` 或 `diagnose_teacher_samples` 尚未定义。

- [ ] **Step 3: 实现冻结数学定义和 teacher-only 批处理**

```python
CURRENT_TAU = 0.10
COUNTERFACTUAL_TAU = 0.05
ORACLE_TIE_TOLERANCE_DEG = 1e-6
DIVERGENCE_EPSILON = 1e-8

@dataclass(frozen=True)
class TeacherDiagnosticResult:
    sample_rows: tuple[dict[str, Any], ...]

def distribution_metrics(probabilities: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    _validate_probability_vector(values)
    entropy = -np.sum(values * np.log(np.clip(values, DIVERGENCE_EPSILON, None)))
    dominant = int(np.argmax(values))
    return {
        "entropy_normalized": float(entropy / math.log(len(SCALE_SIZES))),
        "max_probability": float(values[dominant]),
        "dominant_scale": SCALE_SIZES[dominant],
    }

def build_teacher_sample_row(
    label: TeacherAuthorityLabel,
    scores: torch.Tensor,
    probabilities_current: torch.Tensor,
    probabilities_counterfactual: torch.Tensor,
    *,
    tau_current: float,
) -> dict[str, Any]:
    raw = np.asarray(scores.detach().cpu(), dtype=np.float64)
    current = np.asarray(probabilities_current.detach().cpu(), dtype=np.float64)
    counterfactual = np.asarray(
        probabilities_counterfactual.detach().cpu(), dtype=np.float64
    )
    student = np.asarray(label.student_probabilities, dtype=np.float64)
    current_metrics = distribution_metrics(current)
    counterfactual_metrics = distribution_metrics(counterfactual)
    student_metrics = distribution_metrics(student)
    ordered_scores = np.sort(raw)
    oracle_min = min(label.fixed_rmspe_deg.values())
    oracle = tuple(
        size for size in SCALE_SIZES
        if label.fixed_rmspe_deg[size] - oracle_min <= ORACLE_TIE_TOLERANCE_DEG
    )
    teacher_scale = int(current_metrics["dominant_scale"])
    teacher_safe = np.clip(current, DIVERGENCE_EPSILON, None)
    student_safe = np.clip(student, DIVERGENCE_EPSILON, None)
    midpoint = 0.5 * (teacher_safe + student_safe)
    return {
        "sample_seed": label.sample_seed,
        "rho": label.rho,
        "snr_db": label.snr_db,
        "snapshot_count": label.snapshot_count,
        "separation_deg": label.separation_deg,
        "threshold_cohort": label.threshold_cohort,
        **{f"teacher_score_L{size}": float(raw[index]) for index, size in enumerate(SCALE_SIZES)},
        **{f"teacher_p_current_L{size}": float(current[index]) for index, size in enumerate(SCALE_SIZES)},
        **{f"teacher_p_counterfactual_L{size}": float(counterfactual[index]) for index, size in enumerate(SCALE_SIZES)},
        **{f"student_p_L{size}": float(student[index]) for index, size in enumerate(SCALE_SIZES)},
        "teacher_entropy_current": current_metrics["entropy_normalized"],
        "teacher_entropy_counterfactual": counterfactual_metrics["entropy_normalized"],
        "teacher_max_probability_current": current_metrics["max_probability"],
        "teacher_max_probability_counterfactual": counterfactual_metrics["max_probability"],
        "teacher_dominant_scale": teacher_scale,
        "student_entropy_normalized": student_metrics["entropy_normalized"],
        "student_dominant_scale": student_metrics["dominant_scale"],
        "teacher_score_margin": float(ordered_scores[-1] - ordered_scores[-2]),
        "teacher_score_margin_over_tau": float((ordered_scores[-1] - ordered_scores[-2]) / tau_current),
        "teacher_student_kl": float(np.sum(teacher_safe * np.log(teacher_safe / student_safe))),
        "teacher_student_js": float(0.5 * np.sum(teacher_safe * np.log(teacher_safe / midpoint)) + 0.5 * np.sum(student_safe * np.log(student_safe / midpoint))),
        "oracle_best_scales": oracle,
        "teacher_oracle_agreement": teacher_scale in oracle,
        "teacher_regret_deg": float(label.fixed_rmspe_deg[teacher_scale] - oracle_min),
        **{f"fbss_L{size}_sample_rmspe_deg": label.fixed_rmspe_deg[size] for size in SCALE_SIZES},
    }

def diagnose_teacher_samples(
    samples: Sequence[DOASample],
    labels_by_seed: Mapping[int, TeacherAuthorityLabel],
    *,
    batch_size: int = 128,
    tau_current: float = 0.10,
    tau_counterfactual: float = 0.05,
) -> TeacherDiagnosticResult:
    _require_frozen_runtime(batch_size, tau_current, tau_counterfactual)
    _require_unique_sample_seeds(samples)
    rows = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = list(samples[start:start + batch_size])
            batch = collate_samples(batch_samples)
            teacher = build_scale_teacher(
                batch.fbss_covariances,
                batch.true_angles_deg,
                tau_scale=tau_current,
            )
            counterfactual = scale_probabilities_from_scores(
                teacher.scale_scores, tau_scale=tau_counterfactual
            )
            for sample, scores, current, colder in zip(
                batch_samples, teacher.scale_scores,
                teacher.scale_probabilities, counterfactual, strict=True,
            ):
                label = labels_by_seed[sample.sample_seed]
                _validate_regenerated_metadata(sample, label)
                rows.append(build_teacher_sample_row(
                    label, scores, current, colder, tau_current=tau_current
                ))
    return TeacherDiagnosticResult(sample_rows=tuple(rows))
```

实现时 `teacher_score_margin_over_tau` 使用调用方已经验证过的 `tau_current`，不要硬编码常量；所有概率、scores、RMSPE 与 divergence 在返回前必须检查有限性，KL/JS 允许浮点容差内的极小负值归零。

- [ ] **Step 4: 运行 GREEN、teacher 原有测试及 4 样本批量一致性**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_diagnostic.TeacherMetricTest test_multisource.test_teacher.ScaleTeacherTest -v`

Expected: 全部通过；batch 1、2、4 的 scores/probabilities 在 `rtol=1e-6, atol=1e-7` 内一致。

- [ ] **Step 5: 精确提交 Task 2**

```powershell
git add -- multisource_doa/diagnostics/teacher_confidence.py test_multisource/test_teacher_confidence_diagnostic.py
git diff --cached --check
git commit -m "feat: compute teacher confidence diagnostics"
```

---

### Task 3: 分层汇总、科研门与六文件报告

**Files:**
- Create: `multisource_doa/diagnostics/teacher_reporting.py`
- Create: `test_multisource/test_teacher_confidence_reporting.py`

**Interfaces:**
- Consumes: `TeacherDiagnosticResult.sample_rows`。
- Produces: `build_teacher_summary()`, `build_teacher_stratified_summary()`, `build_teacher_decision()`, `write_teacher_diagnostic_report()`。

- [ ] **Step 1: 写失败测试，冻结空组、门边界、finite 与 schema**

```python
class TeacherDecisionTest(unittest.TestCase):
    def test_all_six_scientific_gates_are_conjunctive(self):
        summary = self._passing_summary()
        decision = build_teacher_decision(summary, self._supporting_strata())
        self.assertTrue(decision["allow_tau_preregistration"])
        for gate in decision["gates"]:
            failed = copy.deepcopy(summary)
            self._make_gate_fail(failed, gate)
            self.assertFalse(
                build_teacher_decision(failed, self._supporting_strata())["allow_tau_preregistration"]
            )

class TeacherReportingTest(unittest.TestCase):
    def test_strata_include_all_fixed_bins_and_each_dimension_sums_to_sample_count(self):
        rows = build_teacher_stratified_summary(self.sample_rows)
        expected = {"rho": 4, "snr_db": 3, "snapshot_count": 3, "threshold_cohort": 7}
        for dimension, bin_count in expected.items():
            selected = [row for row in rows if row["dimension"] == dimension]
            self.assertEqual(len(selected), bin_count)
            self.assertEqual(sum(row["sample_count"] for row in selected), len(self.sample_rows))

    def test_writer_creates_exact_schema_and_refuses_overwrite(self):
        result = TeacherDiagnosticResult(sample_rows=tuple(self.sample_rows))
        config = {"stage": "diagnose_validation_teacher", "device": "cpu"}
        manifest = {
            "sample_count": len(self.sample_rows),
            "no_model_forward": True,
            "training_performed": False,
        }
        output = write_teacher_diagnostic_report(
            result,
            self.output_directory,
            diagnostic_config=config,
            source_manifest=manifest,
        )
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"diagnostic_config.json", "source_manifest.json",
             "teacher_sample_diagnostics.csv", "teacher_summary.json",
             "teacher_stratified_summary.csv", "decision.json"},
        )
        with self.assertRaises(FileExistsError):
            write_teacher_diagnostic_report(
                result,
                self.output_directory,
                diagnostic_config=config,
                source_manifest=manifest,
            )
```

- [ ] **Step 2: 运行 RED 并确认报告模块尚不存在**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_reporting -v`

Expected: `ModuleNotFoundError` 指向 `teacher_reporting`。

- [ ] **Step 3: 实现完整固定分层和科研判定**

```python
TEACHER_SCHEMA_VERSION = 1
RHO_BINS = (("0.8", 0.8), ("0.9", 0.9), ("0.99", 0.99), ("1.0", 1.0))
SNR_BINS = (("[-5,0)", -5.0, 0.0, False), ("[0,5)", 0.0, 5.0, False), ("[5,10]", 5.0, 10.0, True))
SNAPSHOT_BINS = (("8", 8), ("20", 20), ("50", 50))
COHORT_BINS = tuple((name, name) for name in THRESHOLD_COHORTS)
SUMMARY_METRICS = (
    "teacher_entropy_current", "teacher_entropy_counterfactual",
    "teacher_max_probability_current", "teacher_max_probability_counterfactual",
    "teacher_score_margin", "teacher_score_margin_over_tau",
    "student_entropy_normalized", "teacher_student_kl",
    "teacher_student_js", "teacher_regret_deg",
)

def build_teacher_decision(
    summary: Mapping[str, Any],
    stratified_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    overall = summary["metrics"]
    entropy_drop = (
        overall["teacher_entropy_current"]["median"]
        - overall["teacher_entropy_counterfactual"]["median"]
    )
    pmax_rise = (
        overall["teacher_max_probability_counterfactual"]["median"]
        - overall["teacher_max_probability_current"]["median"]
    )
    dimension_support = {
        dimension: sum(
            row["sample_count"] > 0
            and row["teacher_entropy_current_median"] >= 0.90
            and row["teacher_entropy_drop_median"] >= 0.05
            for row in stratified_rows if row["dimension"] == dimension
        ) >= 2
        for dimension in ("rho", "snr_db", "snapshot_count")
    }
    gates = {
        "teacher_entropy_high": overall["teacher_entropy_current"]["median"] >= 0.90,
        "counterfactual_entropy_drop": entropy_drop >= 0.05,
        "counterfactual_pmax_rise": pmax_rise >= 0.05,
        "oracle_agreement": summary["teacher_oracle_agreement_rate"] >= 0.40,
        "median_regret": overall["teacher_regret_deg"]["median"] <= 1.0,
        "stratified_support": sum(dimension_support.values()) >= 2,
        "engineering_integrity": bool(summary["engineering_integrity"]),
    }
    allowed = all(gates.values())
    return {
        "allow_tau_preregistration": allowed,
        "candidate_change": "tau_scale: 0.10 -> 0.05" if allowed else None,
        "training_authorized": False,
        "gates": gates,
        "dimension_support": dimension_support,
        "reason": "all frozen gates passed" if allowed else "one or more frozen gates failed",
    }
```

`build_teacher_summary` 使用与 Task 14 相同的线性插值 quantile，报告 count/mean/median/p05/p95/min/max、agreement count/rate、L4–L7 teacher/student dominant 完整计数。`build_teacher_stratified_summary` 必须枚举全部 17 个固定 bin，空组写 `sample_count=0` 且统计字段为 `None`。`write_teacher_diagnostic_report` 先在内存完成所有 finite/schema 校验，再创建目录，写六个精确文件；manifest 增加 `teacher_diagnostic_schema_version=1`、`no_model_forward=true`、`training_performed=false`。

- [ ] **Step 4: 运行 GREEN 与 CSV 独立复算测试**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_reporting test_multisource.test_diagnostic_reporting -v`

Expected: 全部通过；从 `teacher_sample_diagnostics.csv` 重读后可逐项复算 summary、strata 和 decision。

- [ ] **Step 5: 精确提交 Task 3**

```powershell
git add -- multisource_doa/diagnostics/teacher_reporting.py test_multisource/test_teacher_confidence_reporting.py
git diff --cached --check
git commit -m "feat: report teacher confidence decision"
```

---

### Task 4: 安全入口、4 样本 smoke 与 README

**Files:**
- Create: `scripts/diagnose_pcnss_teacher_confidence.py`
- Modify: `test_multisource/test_entrypoints.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1–3 的 loader、诊断和 writer。
- Produces: `RUN_CONFIG`, `run_dry_run(values)`, `run_diagnostic(values)`, `run_stage(values)`, `load_config(path)`, `main(argv=None)`。

- [ ] **Step 1: 写入口 RED 测试**

```python
class TeacherDiagnosticEntrypointTest(unittest.TestCase):
    def test_default_dry_run_is_cpu_only_and_creates_nothing(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        result = namespace["run_stage"](dict(namespace["RUN_CONFIG"]))
        self.assertEqual(result["stage"], "dry_run")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["batch_size"], 128)
        self.assertFalse(result["output_created"])
        self.assertTrue(result["no_model_forward"])

    def test_rejects_cuda_non128_nonvalidation_unknown_keys_and_existing_output(self):
        namespace = runpy.run_path(str(TEACHER_DIAGNOSTIC_SCRIPT))
        base = dict(namespace["RUN_CONFIG"])
        for update, message in (
            ({"device": "cuda"}, "CPU"),
            ({"batch_size": 4}, "128"),
            ({"split": "development"}, "validation"),
            ({"allow_locked_test": True}, "locked_test"),
        ):
            with self.assertRaisesRegex((ValueError, PermissionError), message):
                namespace["run_stage"]({**base, **update})

    def test_four_sample_smoke_uses_train_samples_and_synthetic_authority(self):
        result = namespace["run_smoke"]({**base, "stage": "smoke", "sample_count": 4})
        self.assertEqual(result["sample_count"], 4)
        self.assertFalse(result["training_performed"])
        self.assertTrue(result["no_model_forward"])
```

- [ ] **Step 2: 运行 RED 并确认入口文件缺失**

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints.TeacherDiagnosticEntrypointTest -v`

Expected: FAIL 指向新入口不存在。

- [ ] **Step 3: 实现默认安全配置与正式编排**

```python
RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "split": "validation",
    "report_directory": "",
    "task14_directory": "",
    "output_root": "outputs/pcnss_teacher_confidence_diagnostic",
    "device": "cpu",
    "batch_size": 128,
    "expected_source_count": 5000,
    "expected_near_count": 1270,
    "tau_current": 0.10,
    "tau_counterfactual": 0.05,
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "smoke", "diagnose_validation_teacher")

def run_dry_run(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values, formal=False)
    return {
        "stage": "dry_run", "locked_test_access": False,
        "output_created": False, "device": "cpu", "batch_size": 128,
        "no_model_forward": True, "training_performed": False,
    }

def run_diagnostic(values: dict[str, Any]) -> dict[str, Any]:
    _validate_safe_config(values, formal=True)
    if values["dry_run"]:
        raise ValueError("正式诊断前必须把 dry_run 改为 False")
    inputs = load_teacher_diagnostic_inputs(
        values["report_directory"], values["task14_directory"],
        expected_source_count=values["expected_source_count"],
        expected_near_count=values["expected_near_count"],
    )
    dataset = PCNSSDataset(SplitName.VALIDATION, ExperimentConfig())
    split_seed = dataset.split_seed
    samples = []
    for seed, label in inputs.labels_by_seed.items():
        index = seed - split_seed
        if not 0 <= index < len(dataset):
            raise ValueError("sample_seed maps outside validation")
        sample = dataset[index]
        validate_regenerated_metadata(sample, label)
        samples.append(sample)
    result = diagnose_teacher_samples(samples, inputs.labels_by_seed)
    output = write_teacher_diagnostic_report(
        result, values["output_root"], diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "input_sha256": inputs.input_sha256,
            "validation_split_seed": split_seed,
            "sample_count": len(samples), "device": "cpu", "batch_size": 128,
            "tau_current": 0.10, "tau_counterfactual": 0.05,
            "no_model_forward": True, "training_performed": False,
        },
    )
    return {"stage": values["stage"], "sample_count": len(samples), "report": str(output)}
```

`_validate_safe_config` 同时被直接调用路径和 JSON 路径使用，严格拒绝未知键、非 CPU、非 128、非 validation、locked-test 开关、非冻结温度和正式目录覆盖。`run_smoke` 使用四个确定性 train sample 和内存合成 authority，不读取正式报告、不创建正式输出、不训练。

- [ ] **Step 4: 更新 README 并运行入口 GREEN**

README 新增“Teacher 尺度置信只读诊断”段落，明确默认命令：

```powershell
D:\Python\Python\python.exe scripts\diagnose_pcnss_teacher_confidence.py
```

说明默认只 dry-run；正式 stage 只接受 validation、CPU、batch 128；不读取 checkpoint、不运行 PC-NSS；六个输出文件不会提交 Git。

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints.TeacherDiagnosticEntrypointTest -v`

Expected: 全部通过。

- [ ] **Step 5: 精确提交 Task 4**

```powershell
git add -- scripts/diagnose_pcnss_teacher_confidence.py test_multisource/test_entrypoints.py README.md
git diff --cached --check
git commit -m "feat: add safe teacher diagnostic entrypoint"
```

---

### Task 5: 工程验证、唯一正式诊断与独立审计

**Files:**
- Create after formal run: `experiments/task15_teacher_confidence_diagnostic.md`
- Do not commit: `.superpowers/sdd/task15_teacher_formal_config.json`
- Do not commit: `outputs/pcnss_teacher_confidence_seed2026_audit_v4/`

**Interfaces:**
- Consumes: Tasks 1–4 完整实现与冻结 audit-v4/Task14 产物。
- Produces: 一次正式六文件报告、独立复算结论和 Task 15 研究记录。

- [ ] **Step 1: 运行目标测试、完整 unittest 和 compileall**

当前 worktree 的 `scripts/run_multiscale_pcnss.py` 含用户明确要求保留的未提交运行参数，
其中三项安全默认值测试会有意失败。不得为测试覆盖或暂存该文件。先用本地 clone 建立只含
HEAD 提交的忽略验证副本，再在该副本运行完整工程验证：

```powershell
git clone --no-hardlinks C:\Users\16420\.codex\worktrees\7989\MultiSource_DOA .superpowers\sdd\task15-clean-verification
```

Expected: clone 的 HEAD 等于当前分支 HEAD；副本不含用户未提交配置和 outputs。

Run: `D:\Python\Python\python.exe -m unittest test_multisource.test_teacher_confidence_diagnostic test_multisource.test_teacher_confidence_reporting test_multisource.test_entrypoints.TeacherDiagnosticEntrypointTest -v`

Expected: 全部通过。

Run from `.superpowers\sdd\task15-clean-verification`: `D:\Python\Python\python.exe -m unittest discover -s test_multisource -v`

Expected: 全部通过，零失败、零错误。

Run from `.superpowers\sdd\task15-clean-verification`: `D:\Python\Python\python.exe -m compileall -q multisource_doa scripts test_multisource`

Expected: exit code 0，无语法错误。

- [ ] **Step 2: 运行两个默认 dry-run 与 4 样本 smoke**

Run from `.superpowers\sdd\task15-clean-verification`: `D:\Python\Python\python.exe scripts\run_multiscale_pcnss.py`

Expected: `stage=dry_run`、`locked_test_access=false`、`output_created=false`。

Run: `D:\Python\Python\python.exe scripts\diagnose_pcnss_teacher_confidence.py`

Expected: `stage=dry_run`、`device=cpu`、`batch_size=128`、`no_model_forward=true`、`output_created=false`。

Run: `D:\Python\Python\python.exe scripts\diagnose_pcnss_teacher_confidence.py --config .superpowers\sdd\task15_teacher_smoke_config.json`

Expected: 4 个 train sample，`training_performed=false`、`no_model_forward=true`，不读取 validation 正式报告。

- [ ] **Step 3: 在运行前审计正式配置并只运行一次 1270 样本**

正式配置只写未跟踪的 `.superpowers/sdd/task15_teacher_formal_config.json`：

```json
{
  "stage": "diagnose_validation_teacher",
  "dry_run": false,
  "split": "validation",
  "report_directory": "D:\\Python\\Project\\doa_estimation\\MultiSource_DOA\\.worktrees\\pcnss-foundation\\scripts\\outputs\\multiscale_pcnss_snap20_seed2026_audit_v4\\validation_report",
  "task14_directory": "C:\\Users\\16420\\.codex\\worktrees\\7989\\MultiSource_DOA\\outputs\\pcnss_near_resolution_seed2026_audit_v4_schema_complete",
  "output_root": "outputs/pcnss_teacher_confidence_seed2026_audit_v4",
  "device": "cpu",
  "batch_size": 128,
  "expected_source_count": 5000,
  "expected_near_count": 1270,
  "tau_current": 0.1,
  "tau_counterfactual": 0.05,
  "allow_locked_test": false,
  "overwrite": false
}
```

先确认目标不存在，再且仅运行一次：

Run: `D:\Python\Python\python.exe scripts\diagnose_pcnss_teacher_confidence.py --config .superpowers\sdd\task15_teacher_formal_config.json`

Expected: `sample_count=1270`，生成六个新文件；无 checkpoint 读取、无模型前向、无训练。

- [ ] **Step 4: 独立重读 CSV 审计全部输出和科研门**

审计脚本从 `teacher_sample_diagnostics.csv` 独立检查：1270 个唯一 seed；四组概率分别有限、非负且和为 1；所有 RMSPE/divergence/regret 有限；每个维度计数和为 1270；summary/strata/decision 可逐项复算；manifest 的输入 SHA 与当前源文件一致；目录恰有六文件。将实际总体 entropy、pmax 变化、agreement、regret、三个维度支持和最终 allow/deny 写入 `experiments/task15_teacher_confidence_diagnostic.md`，不得把关联写成因果。

- [ ] **Step 5: 最终 diff 审查并提交 Task 15 文档**

```powershell
git status --short
git diff --check
git diff --stat 18156a2..HEAD
git add -- experiments/task15_teacher_confidence_diagnostic.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: record teacher confidence diagnosis"
```

Expected: staged 仅 Task 15 文档；`outputs/`、`.superpowers/` 和用户 `scripts/run_multiscale_pcnss.py` 改动不在提交中。

---

### Task 6: 合并到 D 盘主项目并复验

**Files:**
- Merge target: `D:\Python\Project\doa_estimation\MultiSource_DOA`
- Resolve only if needed: `scripts/run_multiscale_pcnss.py`

**Interfaces:**
- Consumes: 当前 codex 分支全部已验证提交和 D 盘 `master` 的已提交配置变更 `d83e347`。
- Produces: 通过工程门的 D 盘 `master` 与非强制更新后的 `origin/master`。

- [ ] **Step 1: 合并前只读审计两边状态和提交图**

```powershell
git -C C:\Users\16420\.codex\worktrees\7989\MultiSource_DOA status --short
git -C D:\Python\Project\doa_estimation\MultiSource_DOA status --short
git -C D:\Python\Project\doa_estimation\MultiSource_DOA log --oneline --decorate --graph --all -20
```

Expected: D 盘 `master` 工作树干净；codex 工作树仅保留事先识别的用户脚本修改和 `.superpowers/`，所有 Task 提交可追踪。

- [ ] **Step 2: 推送 codex 分支并在 D 盘非破坏合并**

```powershell
git push origin codex/task-13-evaluation-audit
git -C D:\Python\Project\doa_estimation\MultiSource_DOA merge --no-ff codex/task-13-evaluation-audit
```

若 `scripts/run_multiscale_pcnss.py` 冲突，最终内容必须采用分支安全默认值：`stage="dry_run"`、`dry_run=True`、`device="cpu"`、空 checkpoint、相对 output root，同时保留 D 盘提交 `d83e347` 在历史中。不得使用 reset、checkout 丢弃或 force push。

- [ ] **Step 3: 在 D 盘运行同一工程验证**

Run: `D:\Python\Python\python.exe -m unittest discover -s test_multisource -v`

Expected: 全部通过。

Run: `D:\Python\Python\python.exe -m compileall -q multisource_doa scripts test_multisource`

Expected: exit code 0。

Run: `D:\Python\Python\python.exe scripts\run_multiscale_pcnss.py`

Expected: 安全 `dry_run`，不创建输出。

Run: `D:\Python\Python\python.exe scripts\diagnose_pcnss_teacher_confidence.py`

Expected: 安全 CPU `dry_run`，不创建输出。

- [ ] **Step 4: 检查 merge diff、敏感产物和安全默认值**

```powershell
git -C D:\Python\Project\doa_estimation\MultiSource_DOA status --short
git -C D:\Python\Project\doa_estimation\MultiSource_DOA show --check --stat HEAD
git -C D:\Python\Project\doa_estimation\MultiSource_DOA ls-files | rg "(^|/)(outputs|checkpoints?)/|\.(pt|pth|ckpt)$"
```

Expected: 工作树干净；merge 无 whitespace error；敏感产物查询无输出；脚本无参运行仍是 dry-run/CPU。

- [ ] **Step 5: 非强制推送主分支并记录 SHA**

```powershell
git -C D:\Python\Project\doa_estimation\MultiSource_DOA push origin master
git -C D:\Python\Project\doa_estimation\MultiSource_DOA rev-parse HEAD
```

Expected: 推送成功且无 force；最终报告同时给出 codex 分支 SHA、merge SHA、远程分支、正式诊断科研结论及是否允许下一轮单因素预注册。
