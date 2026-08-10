import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.diagnostics.teacher_confidence import (
    load_teacher_diagnostic_inputs,
)


ALGORITHMS = (
    "pcnss_root_music",
    "fbss_root_music_L4",
    "fbss_root_music_L5",
    "fbss_root_music_L6",
    "fbss_root_music_L7",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class TeacherInputAuthenticationTest(unittest.TestCase):
    def _prediction_row(
        self,
        algorithm: str,
        sample_seed: int,
        *,
        separation_deg: float,
        rho: float = 1.0,
        sample_rmspe_deg: float = 0.5,
    ) -> dict[str, object]:
        return {
            "split": "validation",
            "sample_seed": sample_seed,
            "algorithm": algorithm,
            "true_angle_1_deg": -1.5,
            "true_angle_2_deg": -1.5 + separation_deg,
            "absolute_error_1_deg": 0.25,
            "absolute_error_2_deg": 0.75,
            "sample_rmspe_deg": sample_rmspe_deg,
            "success": True,
            "estimated_separation_at_least_half_true": True,
            "rho": rho,
            "snr_db": 5.0,
            "snapshot_count": 20,
            "separation_deg": separation_deg,
        }

    def _write_inputs(self, mutation: str | None = None) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        audit = root / "audit"
        task14 = root / "task14"
        audit.mkdir()
        task14.mkdir()
        (audit / "run_config.json").write_text(
            json.dumps({"stage": "evaluate_validation", "split": "validation"}),
            encoding="utf-8",
        )
        (audit / "summary.json").write_text(
            json.dumps({"split": "validation", "report_schema_version": 2}),
            encoding="utf-8",
        )
        (audit / "source_manifest.json").write_text(
            json.dumps({"code_sha": "evaluator", "checkpoint_sha": "checkpoint"}),
            encoding="utf-8",
        )
        rows: list[dict[str, object]] = []
        for algorithm in ALGORITHMS:
            for offset, separation in enumerate((3.0, 5.0)):
                rows.append(
                    self._prediction_row(
                        algorithm,
                        202708040 + offset,
                        separation_deg=separation,
                        sample_rmspe_deg=0.4 + 0.1 * ALGORITHMS.index(algorithm),
                    )
                )
        if mutation == "duplicate_seed":
            rows.append(dict(rows[0]))
        elif mutation == "missing_algorithm_row":
            rows = [
                row
                for row in rows
                if not (
                    row["algorithm"] == "fbss_root_music_L4"
                    and row["sample_seed"] == 202708041
                )
            ]
        elif mutation == "metadata_mismatch":
            next(
                row
                for row in rows
                if row["algorithm"] == "fbss_root_music_L6"
                and row["sample_seed"] == 202708040
            )["rho"] = 0.9
        elif mutation == "nonfinite_rmspe":
            next(
                row
                for row in rows
                if row["algorithm"] == "fbss_root_music_L5"
                and row["sample_seed"] == 202708040
            )["sample_rmspe_deg"] = "nan"
        with (audit / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        audit_hashes = {
            name: _sha256(audit / name)
            for name in (
                "run_config.json",
                "summary.json",
                "source_manifest.json",
                "predictions.csv",
            )
        }
        task14_manifest = {
            "diagnostic_schema_version": 1,
            "sample_count": 1,
            "checkpoint_sha": "checkpoint",
            "audit_input_sha256": audit_hashes,
            "no_model_forward": True,
        }
        if mutation == "audit_hash":
            task14_manifest["audit_input_sha256"]["predictions.csv"] = "wrong"
        elif mutation == "checkpoint_sha":
            task14_manifest["checkpoint_sha"] = "wrong"
        (task14 / "source_manifest.json").write_text(
            json.dumps(task14_manifest), encoding="utf-8"
        )
        task14_seed = 202708041 if mutation == "near_seed_set" else 202708040
        probabilities: tuple[object, ...] = (
            (0.4, 0.3, 0.2, 0.2)
            if mutation == "student_probability"
            else (0.4, 0.3, 0.2, 0.1)
        )
        task14_row = {
            "split": "validation",
            "sample_seed": task14_seed,
            "rho": 1.0,
            "snr_db": 5.0,
            "snapshot_count": 20,
            "separation_deg": 3.0,
            "threshold_cohort": "resolved",
            **{
                f"p_L{size}": probability
                for size, probability in zip((4, 5, 6, 7), probabilities, strict=True)
            },
        }
        with (task14 / "near_sample_diagnostics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(task14_row))
            writer.writeheader()
            writer.writerow(task14_row)
        return audit, task14, temporary

    def test_loads_five_complete_algorithms_and_exact_near_seed_set(self):
        audit, task14, temporary = self._write_inputs()
        self.addCleanup(temporary.cleanup)

        loaded = load_teacher_diagnostic_inputs(
            audit,
            task14,
            expected_source_count=2,
            expected_near_count=1,
        )

        self.assertEqual(tuple(loaded.labels_by_seed), (202708040,))
        label = loaded.labels_by_seed[202708040]
        self.assertEqual(set(label.fixed_rmspe_deg), {4, 5, 6, 7})
        self.assertEqual(label.student_probabilities, (0.4, 0.3, 0.2, 0.1))
        self.assertEqual(set(loaded.input_sha256), {
            "audit/run_config.json",
            "audit/summary.json",
            "audit/source_manifest.json",
            "audit/predictions.csv",
            "task14/source_manifest.json",
            "task14/near_sample_diagnostics.csv",
        })

    def test_rejects_duplicate_sample_seed(self):
        audit, task14, temporary = self._write_inputs("duplicate_seed")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "duplicate sample_seed"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_missing_algorithm_row(self):
        audit, task14, temporary = self._write_inputs("missing_algorithm_row")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "sample_seed sets|expected_source_count"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_algorithm_metadata_mismatch(self):
        audit, task14, temporary = self._write_inputs("metadata_mismatch")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_audit_input_sha_mismatch(self):
        audit, task14, temporary = self._write_inputs("audit_hash")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "audit input SHA"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_checkpoint_sha_mismatch(self):
        audit, task14, temporary = self._write_inputs("checkpoint_sha")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_near_seed_set_mismatch(self):
        audit, task14, temporary = self._write_inputs("near_seed_set")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "near sample_seed set"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_invalid_student_probability_vector(self):
        audit, task14, temporary = self._write_inputs("student_probability")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "student probabilities"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )

    def test_rejects_nonfinite_fixed_scale_rmspe(self):
        audit, task14, temporary = self._write_inputs("nonfinite_rmspe")
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, "sample_rmspe_deg"):
            load_teacher_diagnostic_inputs(
                audit, task14, expected_source_count=2, expected_near_count=1
            )


if __name__ == "__main__":
    unittest.main()
