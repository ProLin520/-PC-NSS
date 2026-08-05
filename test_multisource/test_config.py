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


if __name__ == "__main__":
    unittest.main()
