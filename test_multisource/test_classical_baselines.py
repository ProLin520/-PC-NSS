import unittest

import numpy as np

from multisource_doa.baselines.classical import (
    DOAEstimate,
    esprit,
    evaluate_fixed_scale_family,
    fbss_root_music,
    music_scan,
    root_music_raw,
    sps_root_music,
)
from multisource_doa.config import ExperimentConfig
from multisource_doa.data.simulator import generate_two_source_sample
from multisource_doa.physics.covariance import sample_covariance


class ClassicalBaselineTest(unittest.TestCase):
    def setUp(self):
        config = ExperimentConfig()
        self.sample = generate_two_source_sample(
            config,
            split_seed=701,
            index=0,
            rho=0.9,
            snr_db=10.0,
            snapshot_count=100,
            center_deg=0.0,
            separation_deg=10.0,
        )

    def test_fixed_scale_family_contains_every_locked_scale(self):
        family = evaluate_fixed_scale_family(
            self.sample.snapshots,
            subarray_sizes=(4, 5, 6, 7),
            source_count=2,
        )

        self.assertEqual(
            set(family),
            {
                "sps_root_music_L4",
                "sps_root_music_L5",
                "sps_root_music_L6",
                "sps_root_music_L7",
                "fbss_root_music_L4",
                "fbss_root_music_L5",
                "fbss_root_music_L6",
                "fbss_root_music_L7",
            },
        )
        self.assertTrue(all(isinstance(item, DOAEstimate) for item in family.values()))

    def test_raw_root_music_music_and_esprit_share_result_interface(self):
        covariance = sample_covariance(self.sample.snapshots)
        estimates = [
            root_music_raw(covariance, source_count=2),
            music_scan(covariance, source_count=2),
            esprit(covariance, source_count=2),
        ]

        for estimate in estimates:
            self.assertIsInstance(estimate, DOAEstimate)
            self.assertIsInstance(estimate.algorithm, str)
            self.assertGreaterEqual(estimate.runtime_seconds, 0.0)
            self.assertEqual(estimate.angles_deg.ndim, 1)

    def test_family_does_not_expose_per_sample_oracle_selection(self):
        family = evaluate_fixed_scale_family(
            self.sample.snapshots,
            subarray_sizes=(4, 5, 6, 7),
            source_count=2,
        )

        self.assertFalse(any("best" in name or "oracle" in name for name in family))

    def test_public_fixed_scale_wrappers_report_their_scale(self):
        sps = sps_root_music(self.sample.snapshots, subarray_size=6, source_count=2)
        fbss = fbss_root_music(self.sample.snapshots, subarray_size=6, source_count=2)

        self.assertEqual(sps.algorithm, "sps_root_music_L6")
        self.assertEqual(fbss.algorithm, "fbss_root_music_L6")
        self.assertEqual(sps.metadata["subarray_size"], 6)
        self.assertEqual(fbss.metadata["subarray_size"], 6)


if __name__ == "__main__":
    unittest.main()
