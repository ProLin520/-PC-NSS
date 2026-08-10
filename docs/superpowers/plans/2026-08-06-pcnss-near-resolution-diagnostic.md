# PC-NSS Near-Resolution Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并运行一次可审计的只读诊断，解释冻结 PC-NSS checkpoint 在 `[2,4)` validation 样本上的严格 `1°` 精度不足，并据结果预注册唯一一个下一轮训练改变量。

**Architecture:** 独立 diagnostics 模块读取 Task 13 schema v2 报告，以 `sample_seed` 连接 PC-NSS 与固定 FBSS `L=7` 的权威误差标签；随后只重建 1270 个近间隔 validation 样本，按 batch 128 冻结前向并提取尺度权重、lag 残差和三段投影变化。独立 reporting 模块写 schema v1 诊断文件，不修改正式 evaluator；结果审核后再写单因素训练预注册。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、标准库 `csv/json/hashlib/argparse/unittest`；不新增依赖。

## Global Constraints

- 工作目录固定为隔离 worktree `C:\Users\16420\.codex\worktrees\7989\MultiSource_DOA`，分支以 `codex/` 开头。
- 严格执行 RED → GREEN → 重构；每个生产接口必须先有失败测试并亲自确认失败原因。
- 只读使用 Task 13 `audit_v4/validation_report`、原 epoch 35 `best.pt` 和 validation 生成器。
- 只重算 `[2,4)` 的 1270 个 validation 样本；固定 `batch_size=128`、`model.eval()` 和 `torch.no_grad()`。
- 不训练、不更新 checkpoint、不运行新的完整5000样本评价、不访问 development/locked test。
- `audit_v4` 是误差、resolved 和间隔条件的权威标签；诊断不得重定义正式结果。
- 失败、重复、缺失、样本不一致和 Dykstra 不收敛必须保留或显式报错，不得静默删样本。
- 诊断输出写入新的 gitignored 目录并拒绝覆盖；outputs、checkpoint、权重和生成数据不得提交。
- 用户已有 `scripts/run_multiscale_pcnss.py` 配置改动始终保持未暂存；每次只精确暂存本 Task 文件。
- 预注册只允许一个改变量；数据、seed、网络宽度、子阵集合、学习率、batch size、epoch、选模、Root-MUSIC 和评价门槛保持冻结。
- 禁止新增间隔撑开损失，禁止在同一 validation 上连续搜索，禁止在预注册前启动新训练。

---

### Task 1: 锁定 audit_v4 身份、样本连接和一度阈值分组

**Files:**
- Create: `multisource_doa/diagnostics/__init__.py`
- Create: `multisource_doa/diagnostics/near_resolution.py`
- Create: `test_multisource/test_near_resolution_diagnostic.py`

**Interfaces:**
- Consumes: Task 13 `run_config.json`、`summary.json`、`source_manifest.json`、`predictions.csv` 和冻结 checkpoint 路径。
- Produces: `NearAuditLabel`、`NearAuditSelection`、`load_near_audit(...)`、`classify_threshold_cohort(...)`、`build_threshold_summary(...)`。

- [ ] **Step 1: 写 audit 连接和阈值边界 RED 测试**

在 `test_multisource/test_near_resolution_diagnostic.py` 写入：

```python
import unittest

from multisource_doa.diagnostics.near_resolution import (
    build_threshold_summary,
    classify_threshold_cohort,
)


class NearResolutionThresholdTest(unittest.TestCase):
    def test_one_degree_boundaries_are_mutually_exclusive(self):
        self.assertEqual(classify_threshold_cohort(True, True, 1.0), "resolved")
        self.assertEqual(
            classify_threshold_cohort(True, True, 1.000001),
            "near_miss_1_1p25",
        )
        self.assertEqual(classify_threshold_cohort(True, True, 1.25), "near_miss_1_1p25")
        self.assertEqual(
            classify_threshold_cohort(True, True, 1.5),
            "near_miss_1p25_1p5",
        )
        self.assertEqual(classify_threshold_cohort(True, True, 2.0), "near_miss_1p5_2")
        self.assertEqual(classify_threshold_cohort(True, True, 2.000001), "far_miss_gt_2")
        self.assertEqual(classify_threshold_cohort(True, False, 0.5), "separation_failure")
        self.assertEqual(classify_threshold_cohort(False, False, 60.0), "estimation_failure")

    def test_threshold_summary_uses_maximum_matched_angle_error(self):
        rows = [
            {"algorithm": "pcnss_root_music", "absolute_error_1_deg": 0.4,
             "absolute_error_2_deg": 0.75},
            {"algorithm": "pcnss_root_music", "absolute_error_1_deg": 0.8,
             "absolute_error_2_deg": 1.25},
        ]
        summary = build_threshold_summary(rows, "pcnss_root_music")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["max_error_le_0p75_deg"]["count"], 1)
        self.assertEqual(summary["max_error_le_1p25_deg"]["count"], 2)
```

再用 `TemporaryDirectory` 构造最小 schema v2 报告，覆盖：code SHA 不符、checkpoint SHA 不符、sample_seed 重复、PC-NSS/L7 样本集合不一致、元数据不一致、近间隔数量不等于传入的 `expected_near_count`。

- [ ] **Step 2: 运行 RED 并确认接口缺失**

Run:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic -v
```

Expected: FAIL，原因是 `multisource_doa.diagnostics` 或上述函数尚不存在，不是测试夹具错误。

- [ ] **Step 3: 实现不可变标签和阈值分组**

在 `near_resolution.py` 实现：

```python
ERROR_THRESHOLDS_DEG = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
EXPECTED_EVALUATOR_CODE_SHA = "129c3ba3b9fc1919451eef5c67376f04b4b24680"


@dataclass(frozen=True)
class NearAuditLabel:
    sample_seed: int
    rho: float
    snr_db: float
    snapshot_count: int
    separation_deg: float
    pcnss_row: dict[str, Any]
    fbss_l7_row: dict[str, Any]
    threshold_cohort: str


@dataclass(frozen=True)
class NearAuditSelection:
    labels: tuple[NearAuditLabel, ...]
    source_manifest: dict[str, Any]
    input_sha256: dict[str, str]


def classify_threshold_cohort(
    estimate_success: bool,
    separation_pass: bool,
    max_angle_error_deg: float,
) -> str:
    if not estimate_success:
        return "estimation_failure"
    if not separation_pass:
        return "separation_failure"
    if max_angle_error_deg <= 1.0:
        return "resolved"
    if max_angle_error_deg <= 1.25:
        return "near_miss_1_1p25"
    if max_angle_error_deg <= 1.5:
        return "near_miss_1p25_1p5"
    if max_angle_error_deg <= 2.0:
        return "near_miss_1p5_2"
    return "far_miss_gt_2"
```

`build_threshold_summary` 对每个阈值输出 `{count, rate}`，分母固定为传入算法的全部近间隔行。

- [ ] **Step 4: 实现 schema v2 与 SHA 强校验加载器**

实现：

```python
def load_near_audit(
    report_directory: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_code_sha: str = EXPECTED_EVALUATOR_CODE_SHA,
    expected_near_count: int = 1270,
) -> NearAuditSelection:
    report = Path(report_directory)
    checkpoint = Path(checkpoint_path)
    run_config = json.loads((report / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((report / "source_manifest.json").read_text(encoding="utf-8"))
    if run_config.get("stage") != "evaluate_validation":
        raise ValueError("diagnostic source must be evaluate_validation")
    if run_config.get("split") != "validation" or summary.get("split") != "validation":
        raise ValueError("diagnostic source must be validation")
    if summary.get("report_schema_version") != 2:
        raise ValueError("diagnostic source must use report schema v2")
    if manifest.get("code_sha") != expected_code_sha:
        raise ValueError("unexpected evaluator code SHA")
    if _sha256(checkpoint) != manifest.get("checkpoint_sha"):
        raise ValueError("checkpoint SHA does not match source manifest")
    prediction_path = report / "predictions.csv"
    pcnss_rows, fbss_rows = _read_algorithm_rows(prediction_path)
    labels = _pair_and_validate_near_rows(
        pcnss_rows,
        fbss_rows,
        expected_near_count=expected_near_count,
    )
    return NearAuditSelection(
        labels=tuple(labels),
        source_manifest=manifest,
        input_sha256={
            name: _sha256(report / name)
            for name in (
                "run_config.json",
                "summary.json",
                "source_manifest.json",
                "predictions.csv",
            )
        },
    )
```

实现必须使用标准库 `csv.DictReader`、`json.loads` 和流式 SHA256；只选择
`pcnss_root_music` 与 `fbss_root_music_L7`，以 `sample_seed` 建立两个唯一字典，
逐字段核对真角、rho、SNR、snapshot、separation。筛选规则严格为 `2.0 <= separation < 4.0`。
同一步实现私有 `_sha256(path)`、`_read_algorithm_rows(path)` 和
`_pair_and_validate_near_rows(...)`；前者每次读取 `1024*1024` bytes，后两者负责
类型转换、唯一性、集合一致性、元数据一致性和1270计数，不承担任何聚合。

- [ ] **Step 5: 运行 GREEN、重构并复跑**

Run:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic -v
```

Expected: 新增测试全部通过；错误输入均显式抛出 `ValueError` 或 `FileNotFoundError`。

- [ ] **Step 6: 精确提交 Task 1**

```powershell
git add multisource_doa/diagnostics/__init__.py multisource_doa/diagnostics/near_resolution.py test_multisource/test_near_resolution_diagnostic.py
git diff --cached --name-only
git commit -m "feat: validate near-resolution audit inputs"
```

Expected staged files: 仅上述三个文件。

---

### Task 2: 提取尺度熵、残差饱和和三段投影变化

**Files:**
- Modify: `multisource_doa/diagnostics/near_resolution.py`
- Modify: `test_multisource/test_near_resolution_diagnostic.py`

**Interfaces:**
- Consumes: `PCNSSForward`、`PCNSSBatch`、冻结残差上限 `0.10` 和 Dykstra 投影。
- Produces: `scale_weight_diagnostics(...)`、`residual_diagnostics(...)`、`projection_diagnostics(...)`、`diagnose_near_samples(...)`、`NearDiagnosticResult`。

- [ ] **Step 1: 写尺度熵 RED 测试**

构造一个样本、四尺度、三个 lag：第一个 lag 权重完全均匀，第二个 lag 单尺度有效，第三个 lag 权重集中。断言：

```python
metrics = scale_weight_diagnostics(weights, valid_mask, effective_counts)
self.assertAlmostEqual(metrics[0]["scale_entropy_normalized"], 1.0, places=6)
self.assertIsNone(metrics[0]["lag_entropy_normalized"][1])
self.assertEqual(metrics[0]["dominant_scale"], 4)
```

另用不等 effective counts 验证样本级 `p_L` 与 `training.losses.aggregate_scale_weights` 完全一致。

- [ ] **Step 2: 运行尺度 RED**

Run:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic.NearResolutionMechanismTest.test_scale_entropy_normalization -v
```

Expected: FAIL，原因是 `scale_weight_diagnostics` 缺失。

- [ ] **Step 3: 最小实现尺度诊断**

实现时复用 `aggregate_scale_weights`，样本级熵除以 `log(4)`；每 lag 只在有效尺度数 `m>=2` 时除以 `log(m)`。相同最大权重时 `dominant_scale` 按 `(4,5,6,7)` 顺序选较小尺度，保证确定性。

- [ ] **Step 4: 写残差 `0.095` 边界 RED 测试**

```python
residual = torch.tensor([[[0.094999, 0.0], [0.095, 0.0], [0.10, 0.0]]])
metrics = residual_diagnostics(residual, residual_limit=0.10)
self.assertEqual(metrics[0]["saturated_lag_count"], 2)
self.assertAlmostEqual(metrics[0]["saturated_lag_rate"], 2 / 3)
```

- [ ] **Step 5: 运行残差 RED、实现并运行 GREEN**

Run RED then GREEN:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic.NearResolutionMechanismTest.test_residual_saturation_boundary -v
```

Expected after implementation: PASS；饱和比较固定使用 `magnitude / limit >= 0.95`。

- [ ] **Step 6: 写三段投影 RED 测试**

用已知 `C0`、`C1` 和替代 `projection_fn` 返回的 `C2`，验证三个相对 Frobenius 公式分别使用自己的分母；再让替代投影返回 `converged=False`，断言样本行仍存在且带 `dykstra_converged=False`。

- [ ] **Step 7: 运行投影 RED、实现并运行 GREEN**

实现：

```python
def projection_diagnostics(
    candidate_covariances: np.ndarray,
    train_projected_covariances: np.ndarray,
    *,
    projection_fn: Callable[[np.ndarray], ProjectionResult] = dykstra_structured_projection,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for candidate, train_projected in zip(
        candidate_covariances,
        train_projected_covariances,
        strict=True,
    ):
        final = projection_fn(train_projected)
        candidate_norm = max(float(np.linalg.norm(candidate, ord="fro")), 1e-12)
        train_norm = max(float(np.linalg.norm(train_projected, ord="fro")), 1e-12)
        rows.append({
            "train_projection_change": float(
                np.linalg.norm(train_projected - candidate, ord="fro") / candidate_norm
            ),
            "eval_projection_change": float(
                np.linalg.norm(final.matrix - train_projected, ord="fro") / train_norm
            ),
            "total_projection_change": float(
                np.linalg.norm(final.matrix - candidate, ord="fro") / candidate_norm
            ),
            "dykstra_converged": bool(final.converged),
            "dykstra_iterations": int(final.iterations),
            "final_hermitian_error": float(final.hermitian_error),
            "final_toeplitz_error": float(final.toeplitz_error),
            "final_trace_error": float(final.trace_error),
            "final_min_eigenvalue": float(final.min_eigenvalue),
        })
    return tuple(rows)
```

Run:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic -v
```

Expected: 全部机制测试通过，未收敛样本未被过滤。

- [ ] **Step 8: 写并实现冻结批量诊断 RED/GREEN**

先写测试，用4个确定性 train 样本和随机初始化模型调用：

```python
result = diagnose_near_samples(
    samples,
    labels_by_seed,
    model,
    device=torch.device("cpu"),
    batch_size=2,
)
self.assertEqual(len(result.sample_rows), 4)
self.assertEqual({row["sample_seed"] for row in result.sample_rows}, set(labels_by_seed))
```

实现 `NearDiagnosticResult` 与 `diagnose_near_samples`。函数必须保持输入顺序、调用
`model.eval()`、使用 `torch.no_grad()`，每批调用 `collate_samples`，并按 sample_seed
连接正式标签。真实角度、rho、SNR、snapshot 只进入诊断行，不进入模型 forward。

- [ ] **Step 9: 精确提交 Task 2**

```powershell
git add multisource_doa/diagnostics/near_resolution.py test_multisource/test_near_resolution_diagnostic.py
git diff --cached --name-only
git commit -m "feat: compute PC-NSS mechanism diagnostics"
```

---

### Task 3: 聚合分层结果并写独立 schema v1 报告

**Files:**
- Create: `multisource_doa/diagnostics/reporting.py`
- Create: `test_multisource/test_diagnostic_reporting.py`
- Modify: `multisource_doa/diagnostics/__init__.py`

**Interfaces:**
- Consumes: `NearDiagnosticResult`、`NearAuditSelection` 和诊断配置。
- Produces: `build_stratified_summary(...)`、`build_mechanism_summary(...)`、`write_near_diagnostic_report(...)`。

- [ ] **Step 1: 写分层完整性 RED 测试**

构造覆盖 rho、三个 SNR bins、snapshot 和阈值组的样本行，断言每个维度的 count
之和等于总样本数。加入 `snr_db=10.1`、`rho=0.95`、`snapshot=16`，分别断言显式
`ValueError`。

- [ ] **Step 2: 运行 RED**

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_diagnostic_reporting -v
```

Expected: FAIL，原因是 reporting 模块尚不存在。

- [ ] **Step 3: 实现固定分层与机制聚合**

固定常量：

```python
SNR_BINS = ((-5.0, 0.0, "[-5,0)"), (0.0, 5.0, "[0,5)"), (5.0, 10.0, "[5,10]"))
RHO_VALUES = (0.8, 0.9, 0.99, 1.0)
SNAPSHOT_VALUES = (8, 20, 50)
DIAGNOSTIC_SCHEMA_VERSION = 1
```

`build_stratified_summary` 每行至少包含 `dimension`、`bin`、`sample_count`、六个
误差阈值通过率、resolved rate、四尺度均值、归一化熵均值/中位数、残差饱和率和
三段投影变化均值/中位数。空组不写；非空组总数必须对账。

同一步实现机制汇总：

```python
MECHANISM_METRICS = (
    "scale_entropy_normalized",
    "residual_magnitude_p50",
    "residual_magnitude_p95",
    "residual_magnitude_max",
    "saturated_lag_rate",
    "train_projection_change",
    "eval_projection_change",
    "total_projection_change",
)


def build_mechanism_summary(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cohorts = {}
    for cohort in THRESHOLD_COHORTS:
        rows = [row for row in sample_rows if row["threshold_cohort"] == cohort]
        if not rows:
            continue
        cohorts[cohort] = {
            "sample_count": len(rows),
            "metrics": {
                metric: _finite_distribution_summary([row[metric] for row in rows])
                for metric in MECHANISM_METRICS
            },
            "dominant_scale_counts": {
                str(size): sum(row["dominant_scale"] == size for row in rows)
                for size in (4, 5, 6, 7)
            },
        }
    return {"sample_count": len(sample_rows), "cohorts": cohorts}
```

`_finite_distribution_summary` 固定返回 count、mean、median、p05、p95、min、max；
空输入不调用该函数，非有限输入显式报错。

- [ ] **Step 4: 写报告拒绝覆盖和可重算 RED 测试**

使用 `TemporaryDirectory` 写4样本报告，断言六个文件恰好存在；再次写同一目录抛
`FileExistsError`。从 `near_sample_diagnostics.csv` 重新计算一个阈值 count、一个
尺度熵均值和一个饱和率，必须与 JSON/CSV 汇总一致。

- [ ] **Step 5: 实现报告写入并运行 GREEN**

实现：

```python
def write_near_diagnostic_report(
    result: NearDiagnosticResult,
    output_directory: str | Path,
    *,
    diagnostic_config: dict[str, Any],
    source_manifest: dict[str, Any],
    refuse_overwrite: bool = True,
) -> Path:
    output = prepare_run_directory(
        output_directory,
        refuse_overwrite=refuse_overwrite,
    )
    threshold_summary = result.threshold_summary
    stratified_rows = list(result.stratified_rows)
    mechanism_summary = result.mechanism_summary
    _require_finite_or_null(result.sample_rows)
    _require_finite_or_null(threshold_summary)
    _require_finite_or_null(stratified_rows)
    _require_finite_or_null(mechanism_summary)
    _write_json(output / "diagnostic_config.json", diagnostic_config)
    _write_json(
        output / "source_manifest.json",
        {"diagnostic_schema_version": 1, **source_manifest},
    )
    _write_csv(
        output / "near_sample_diagnostics.csv",
        list(result.sample_rows),
    )
    _write_json(output / "threshold_summary.json", threshold_summary)
    _write_csv(output / "stratified_summary.csv", stratified_rows)
    _write_json(output / "mechanism_summary.json", mechanism_summary)
    return output
```

只使用标准库 CSV/JSON；JSON 使用 `indent=2`、`sort_keys=True`，非有限值在写入前
显式拒绝。同一步实现 `_require_finite_or_null` 的递归数值检查、`_write_json` 和
`_write_csv`；CSV 无行时显式报错。输出固定为设计规格第11节的六个文件。

Run:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_diagnostic_reporting -v
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic test_multisource.test_diagnostic_reporting -v
```

Expected: 两组测试全部通过。

- [ ] **Step 6: 精确提交 Task 3**

```powershell
git add multisource_doa/diagnostics/__init__.py multisource_doa/diagnostics/reporting.py test_multisource/test_diagnostic_reporting.py
git diff --cached --name-only
git commit -m "feat: report stratified near-resolution diagnostics"
```

---

### Task 4: 增加安全入口、dry-run 和4样本 smoke

**Files:**
- Create: `scripts/diagnose_pcnss_near_resolution.py`
- Modify: `test_multisource/test_entrypoints.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1–3 diagnostics API、checkpoint 和 schema v2 报告目录。
- Produces: `RUN_CONFIG`、`run_stage(values)`、`run_dry_run(values)`、`run_diagnostic(values)` 和可选 `--config` JSON 入口。

- [ ] **Step 1: 写入口安全 RED 测试**

在 `test_entrypoints.py` 动态加载新脚本并断言默认值：

```python
self.assertEqual(module.RUN_CONFIG["stage"], "dry_run")
self.assertTrue(module.RUN_CONFIG["dry_run"])
self.assertEqual(module.RUN_CONFIG["split"], "validation")
self.assertEqual(module.RUN_CONFIG["batch_size"], 128)
self.assertFalse(module.RUN_CONFIG["allow_locked_test"])
self.assertFalse(module.RUN_CONFIG["overwrite"])
self.assertNotIn("development", module.STAGES)
self.assertNotIn("locked_test", module.STAGES)
```

另断言正式 stage 在 `dry_run=True` 时拒绝、`allow_locked_test=True` 时拒绝、输出已存在
且 `overwrite=False` 时拒绝。

- [ ] **Step 2: 运行 RED**

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints -v
```

Expected: FAIL，原因是新入口不存在。

- [ ] **Step 3: 实现默认 dry-run 和外部 JSON 配置**

脚本顶部固定：

```python
RUN_CONFIG = {
    "stage": "dry_run",
    "dry_run": True,
    "split": "validation",
    "report_directory": "",
    "checkpoint_path": "",
    "output_root": "outputs/pcnss_near_resolution_diagnostic",
    "device": "cpu",
    "batch_size": 128,
    "expected_near_count": 1270,
    "allow_locked_test": False,
    "overwrite": False,
}
STAGES = ("dry_run", "diagnose_validation_near")
```

无参数时使用 `RUN_CONFIG`。`--config path.json` 时完整读取外部字典，但禁止未知键，
并把实际配置原样写入 `diagnostic_config.json`。dry-run 不读取正式报告或 checkpoint，
不创建输出目录，返回 `output_created=False`。

- [ ] **Step 4: 实现只读正式诊断 orchestration**

`run_diagnostic(values)` 固定按以下数据流实现：

```python
def run_diagnostic(values: dict[str, Any]) -> dict[str, Any]:
    if values.get("dry_run", True):
        raise ValueError("正式诊断前必须把 dry_run 改为 False")
    if values.get("split") != "validation":
        raise PermissionError("near-resolution diagnostic accepts validation only")
    if values.get("allow_locked_test", False):
        raise PermissionError("locked_test access is forbidden")
    selection = load_near_audit(
        values["report_directory"],
        values["checkpoint_path"],
        expected_near_count=int(values["expected_near_count"]),
    )
    config = ExperimentConfig()
    dataset = PCNSSDataset(SplitName.VALIDATION, config)
    split_seed = config.split.seeds[SplitName.VALIDATION]
    samples = []
    labels_by_seed = {}
    for label in selection.labels:
        index = label.sample_seed - split_seed
        if not 0 <= index < len(dataset):
            raise ValueError("sample_seed maps outside validation")
        sample = dataset[index]
        _validate_regenerated_metadata(sample, label)
        samples.append(sample)
        labels_by_seed[label.sample_seed] = label
    device = _device(values)
    model = MultiScalePCNSS().to(device)
    payload = torch.load(
        values["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    result = diagnose_near_samples(
        samples,
        labels_by_seed,
        model,
        device=device,
        batch_size=int(values["batch_size"]),
    )
    report = write_near_diagnostic_report(
        result,
        values["output_root"],
        diagnostic_config=values,
        source_manifest={
            "diagnostic_code_sha": _code_sha(),
            "checkpoint_sha": selection.source_manifest["checkpoint_sha"],
            "evaluator_code_sha": selection.source_manifest["code_sha"],
            "input_sha256": selection.input_sha256,
            "validation_split_seed": split_seed,
            "sample_count": len(selection.labels),
            "batch_size": int(values["batch_size"]),
        },
        refuse_overwrite=not bool(values.get("overwrite", False)),
    )
    return {"stage": values["stage"], "sample_count": len(samples), "report": str(report)}
```

同一步实现 `_device`、`_code_sha` 和 `_validate_regenerated_metadata`；元数据比较对
角度/separation 使用 `atol=1e-9`，rho/SNR/snapshot 要求冻结值精确一致。

- [ ] **Step 5: 写4样本 CPU smoke RED/GREEN**

测试不读取正式 validation：用 `TemporaryDirectory` 创建2个算法×4个相同 train
样本标签和临时 checkpoint，调用底层 `diagnose_near_samples` 与报告器，断言六个
输出存在、4个 sample_seed 完整、没有 development/locked 字段。

- [ ] **Step 6: 更新 README 并运行入口 GREEN**

README 新增“近间隔只读诊断”段落，明确：默认 dry-run、正式诊断只接受 validation、
固定 batch 128、读取既有 audit_v4、输出不提交、诊断不等于授权训练。

Run:

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_entrypoints -v
D:\Python\Python\python.exe scripts/diagnose_pcnss_near_resolution.py
```

Expected: 测试通过；dry-run 输出 `stage=dry_run`、`locked_test_access=false`、
`output_created=false`。

- [ ] **Step 7: 精确提交 Task 4**

```powershell
git add scripts/diagnose_pcnss_near_resolution.py test_multisource/test_entrypoints.py README.md
git diff --cached --name-only
git commit -m "feat: add safe near-resolution diagnostic entrypoint"
```

---

### Task 5: 完整工程验证并运行一次冻结近间隔诊断

**Files:**
- No tracked source changes expected before the run.
- Create untracked/gitignored: `outputs/pcnss_near_resolution_seed2026_audit_v4/`

**Interfaces:**
- Consumes: 已验证入口、`audit_v4/validation_report`、原 `best.pt`。
- Produces: 六个 schema v1 诊断文件，仅存在于 outputs。

- [ ] **Step 1: 运行目标与完整单测**

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic test_multisource.test_diagnostic_reporting test_multisource.test_entrypoints -v
D:\Python\Python\python.exe -m unittest discover -s test_multisource -v
```

Expected: exit code 0，0 failures，0 errors。

- [ ] **Step 2: 运行 compileall、默认 dry-run 和4样本 smoke**

```powershell
D:\Python\Python\python.exe -m compileall multisource_doa scripts test_multisource
D:\Python\Python\python.exe scripts/diagnose_pcnss_near_resolution.py
D:\Python\Python\python.exe scripts/smoke_multiscale_pcnss.py
```

Expected: compileall 无错误；诊断 dry-run 不创建输出；既有 smoke 仍为4样本且不访问
正式 validation/development/locked test。

- [ ] **Step 3: 在 outputs 创建一次性运行配置**

创建不提交的
`outputs/pcnss_near_resolution_seed2026_audit_v4_config.json`，内容固定为：

```json
{
  "stage": "diagnose_validation_near",
  "dry_run": false,
  "split": "validation",
  "report_directory": "D:\\Python\\Project\\doa_estimation\\MultiSource_DOA\\.worktrees\\pcnss-foundation\\scripts\\outputs\\multiscale_pcnss_snap20_seed2026_audit_v4\\validation_report",
  "checkpoint_path": "D:\\Python\\Project\\doa_estimation\\MultiSource_DOA\\.worktrees\\pcnss-foundation\\scripts\\outputs\\multiscale_pcnss_snap20_seed2026\\best.pt",
  "output_root": "outputs/pcnss_near_resolution_seed2026_audit_v4",
  "device": "cuda",
  "batch_size": 128,
  "expected_near_count": 1270,
  "allow_locked_test": false,
  "overwrite": false
}
```

若 `torch.cuda.is_available()` 为 false，停止并报告，不静默切换 CPU，因为正式诊断
应与 audit_v4 的 CUDA 推理环境一致。

- [ ] **Step 4: 运行唯一一次1270样本诊断**

```powershell
D:\Python\Python\python.exe scripts/diagnose_pcnss_near_resolution.py --config outputs/pcnss_near_resolution_seed2026_audit_v4_config.json
```

Expected: exit code 0；输出报告路径；`sample_count=1270`；没有训练、development、
locked test 或完整5000样本评价。

- [ ] **Step 5: 独立审计输出**

运行只读检查，确认：

```text
schema_version == 1
sample_count == 1270
unique sample_seed == 1270
all source SHA256 match
batch_size == 128
sum of each rho/SNR/snapshot/cohort partition == 1270
all Dykstra rows retained
all numeric aggregate values finite or explicitly null for single-valid-scale entropy
```

从 `near_sample_diagnostics.csv` 独立重算六个误差阈值、PC-NSS resolved 数、残差
饱和总数和三个投影均值，与结构化汇总逐项一致。

- [ ] **Step 6: 审查 Git 状态**

```powershell
git status --short
git diff --check
git check-ignore outputs/pcnss_near_resolution_seed2026_audit_v4/near_sample_diagnostics.csv
```

Expected: outputs 被忽略；没有 checkpoint、CSV/JSON 输出进入 Git；用户的
`scripts/run_multiscale_pcnss.py` 修改仍未暂存。

---

### Task 6: 根据冻结诊断写结论和单因素实验预注册

**Files:**
- Create: `experiments/task14_near_resolution_diagnostic.md`
- Create conditionally when evidence permits: `experiments/pcnss_near_angle_precision_preregistration.md`

**Interfaces:**
- Consumes: Task 5 六个输出文件及其 SHA256。
- Produces: 可审计诊断结论；若设计规格第12节的证据顺序能选出唯一机制，则产生训练前预注册。

- [ ] **Step 1: 写诊断记录**

`task14_near_resolution_diagnostic.md` 必须逐项记录：

- 输入报告、checkpoint、代码和诊断输出 SHA；
- PC-NSS 与 L=7 在 `0.5/0.75/1/1.25/1.5/2°` 的累计通过率；
- 七个互斥 cohort 数量；
- rho、SNR、snapshot 分层中最明显且方向一致的门槛差异；
- resolved、三个 near-miss、far-miss 的尺度熵、主导尺度、残差饱和和投影改变量；
- Dykstra 未收敛数量；
- 明确区分“观察到的关联”和“尚未证明的因果”。

- [ ] **Step 2: 按固定优先级选择或拒绝单因素假设**

严格按设计规格第12节依次判定：残差/投影限制 → 尺度置信不足 → 局部角度精度
代理不足。只有当一个方向在总体 cohort 对比和至少两个场景分层中一致时才进入
预注册；否则在 Task 14 记录“证据不足，不启动训练”，不创建伪预注册。

- [ ] **Step 3: 写唯一单因素预注册（仅证据允许时）**

`pcnss_near_angle_precision_preregistration.md` 必须在标题下直接写出：

```text
Hypothesis: 一个可证伪机制假设
Single change: 唯一代码/损失/边界变化及唯一固定值
Frozen controls: 数据、split、seed2026、网络宽度、L={4,5,6,7}、lr=1e-3、batch=128、50 epochs、checkpoint 规则、Root-MUSIC、1°/50%门槛、60°罚值
Primary metric: validation [2,4) resolution rate
Guardrails: overall failure-aware RMSPE、overall resolution、failure count
Baseline: epoch35 checkpoint 与 FBSS L=7
Stop rule: 一次训练 + 一次冻结 validation；失败即停止，不修改第二个因素，不访问 development/locked test
```

唯一改变量必须直接对应诊断中优先级最高且证据一致的机制；不得选择间隔撑开损失。

- [ ] **Step 4: 自审研究纪律和禁止主张**

```powershell
rg -n "首次|首个|理论保证|已经解决|locked_test|development" experiments/task14_near_resolution_diagnostic.md experiments/pcnss_near_angle_precision_preregistration.md
git diff --check -- experiments/task14_near_resolution_diagnostic.md experiments/pcnss_near_angle_precision_preregistration.md
```

Expected: 不包含未经证据支持的性能/因果主张；development/locked test 只出现在禁止
访问或后续审批说明中。

- [ ] **Step 5: 精确提交 Task 6**

若证据允许预注册：

```powershell
git add experiments/task14_near_resolution_diagnostic.md experiments/pcnss_near_angle_precision_preregistration.md
git diff --cached --name-only
git commit -m "docs: preregister near-angle precision experiment"
```

若证据不足：

```powershell
git add experiments/task14_near_resolution_diagnostic.md
git diff --cached --name-only
git commit -m "docs: record near-resolution diagnostic"
```

Expected: 只提交诊断结论和证据允许的预注册文档，不提交 outputs。

---

### Task 7: 最终验证、diff 审查和远程同步

**Files:**
- Review only: 本计划所有 tracked files。

**Interfaces:**
- Consumes: Tasks 1–6 的提交和验证证据。
- Produces: 完整 Task 交接、提交 SHA 和远程分支。

- [ ] **Step 1: 重新运行最终验证**

```powershell
D:\Python\Python\python.exe -m unittest test_multisource.test_near_resolution_diagnostic test_multisource.test_diagnostic_reporting test_multisource.test_entrypoints -v
D:\Python\Python\python.exe -m unittest discover -s test_multisource -v
D:\Python\Python\python.exe -m compileall multisource_doa scripts test_multisource
D:\Python\Python\python.exe scripts/diagnose_pcnss_near_resolution.py
```

Expected: 目标与完整测试 0 failures/0 errors；compileall 成功；默认 dry-run 不创建输出。

- [ ] **Step 2: 审查范围和 outputs 排除**

```powershell
git status --short
git diff --check 129c3ba3b9fc1919451eef5c67376f04b4b24680..HEAD
git log --oneline --decorate -8
git ls-files | rg "outputs|\.pt$|checkpoint|best\.pt"
```

Expected: tracked 变更只包含 diagnostics 源码、测试、入口、README、规格/计划和实验
文档；用户 RUN_CONFIG 修改未暂存；没有输出或权重被跟踪。

- [ ] **Step 3: 推送当前 codex 分支**

```powershell
git push origin codex/task-13-evaluation-audit
```

Expected: 非 force push 成功，远程分支包含本 Task 的范围明确提交。

- [ ] **Step 4: 最终报告**

向用户报告：

- 修改文件与各提交 SHA；
- 每条目标/完整测试、compileall、dry-run、smoke 和1270样本诊断命令及结果；
- 1°附近误差分布、rho/SNR/snapshot 分层结论；
- 尺度熵、残差饱和、结构投影变化的证据；
- 是否形成预注册、唯一改变量和停止规则；
- 明确未训练、未访问 development/locked test、未提交 outputs；
- 远程分支名称。
