import json
import tempfile
import unittest
from pathlib import Path

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.dataset import PCNSSDataset
from multisource_doa.data.manifest import (
    assert_split_seed_ranges_disjoint,
    split_seed_interval,
    write_split_manifest,
)


class SplitAuditTest(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig()

    def test_locked_test_dataset_requires_explicit_permission(self):
        with self.assertRaisesRegex(PermissionError, "locked_test"):
            PCNSSDataset(SplitName.LOCKED_TEST, self.config)

    def test_train_validation_and_development_seed_ranges_are_disjoint(self):
        assert_split_seed_ranges_disjoint(self.config)

        intervals = [
            split_seed_interval(self.config, split)
            for split in (
                SplitName.TRAIN,
                SplitName.VALIDATION,
                SplitName.DEVELOPMENT,
            )
        ]
        for left, right in zip(intervals, intervals[1:]):
            self.assertLess(left[1], right[0])

    def test_dataset_uses_split_seed_plus_index(self):
        dataset = PCNSSDataset(SplitName.TRAIN, self.config)

        self.assertEqual(len(dataset), 40_000)
        self.assertEqual(dataset[11].sample_seed, self.config.split.seeds[SplitName.TRAIN] + 11)

    def test_manifest_is_auditable_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_manifest.json"
            write_split_manifest(path, self.config, SplitName.TRAIN)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["split"], "train")
            self.assertEqual(payload["size"], 40_000)
            self.assertEqual(payload["sample_seed_start"], 202_608_040)
            self.assertEqual(payload["sample_seed_end"], 202_648_039)
            self.assertEqual(payload["generator_version"], "pcnss-two-source-v1")
            with self.assertRaises(FileExistsError):
                write_split_manifest(path, self.config, SplitName.TRAIN)


if __name__ == "__main__":
    unittest.main()
