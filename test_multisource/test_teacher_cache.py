import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.training.error_teacher import build_error_teacher_row
from multisource_doa.training.teacher_cache import (
    load_teacher_cache,
    write_teacher_cache,
)


class TeacherCacheTest(unittest.TestCase):
    def _rows(self):
        dataset = PCNSSDataset(SplitName.TRAIN, ExperimentConfig())
        return [
            build_error_teacher_row(dataset[index], sample_index=index)
            for index in range(4)
        ]

    def _write(self, root):
        output = Path(root) / "cache"
        write_teacher_cache(
            self._rows(),
            output,
            experiment_config=ExperimentConfig(),
            run_config={
                "stage": "smoke",
                "split": "train",
                "device": "cpu",
                "batch_size": 128,
            },
            code_sha="abc123",
            source_sha256={"error_teacher.py": "f" * 64},
            expected_count=4,
        )
        return output

    def test_writer_creates_exact_three_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._write(temporary)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "teacher_cache_config.json",
                    "teacher_cache_manifest.json",
                    "train_teacher_labels.csv",
                },
            )
            manifest = json.loads(
                (output / "teacher_cache_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["train_only"])
            self.assertFalse(manifest["validation_accessed"])
            with self.assertRaises(FileExistsError):
                write_teacher_cache(
                    self._rows(),
                    output,
                    experiment_config=ExperimentConfig(),
                    run_config={"stage": "smoke"},
                    code_sha="abc123",
                    source_sha256={"error_teacher.py": "f" * 64},
                    expected_count=4,
                )

    def test_loader_rebuilds_train_metadata_and_returns_seed_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._write(temporary)
            loaded = load_teacher_cache(
                output,
                ExperimentConfig(),
                expected_count=4,
                regenerate_metadata=True,
            )
            start = ExperimentConfig().split.seeds[SplitName.TRAIN]
            self.assertEqual(tuple(loaded.labels_by_seed), tuple(start + i for i in range(4)))
            self.assertAlmostEqual(sum(loaded.labels_by_seed[start]), 1.0)
            self.assertEqual(loaded.manifest["sample_count"], 4)

    def test_loader_rejects_extra_file_and_csv_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._write(temporary)
            (output / "extra.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_teacher_cache(output, ExperimentConfig(), expected_count=4)
            (output / "extra.txt").unlink()
            csv_path = output / "train_teacher_labels.csv"
            csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_teacher_cache(output, ExperimentConfig(), expected_count=4)

    def test_loader_rejects_duplicate_seed_even_with_updated_csv_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._write(temporary)
            csv_path = output / "train_teacher_labels.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[1]["sample_seed"] = rows[0]["sample_seed"]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = output / "teacher_cache_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_teacher_cache(output, ExperimentConfig(), expected_count=4)


if __name__ == "__main__":
    unittest.main()
