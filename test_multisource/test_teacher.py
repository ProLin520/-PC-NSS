import unittest

import numpy as np
import torch

from multisource_doa.data.simulator import steering_vector
from multisource_doa.training.teacher import (
    build_scale_teacher,
    scale_probabilities_from_scores,
)


def _theoretical_covariance(sensor_count, angles_deg):
    steering = steering_vector(np.asarray(angles_deg), sensor_count=sensor_count)
    covariance = steering @ steering.conj().T + 0.05 * np.eye(sensor_count)
    return torch.from_numpy(covariance.astype(np.complex64)).unsqueeze(0)


class ScaleTeacherTest(unittest.TestCase):
    def test_soft_teacher_preserves_score_ranking(self):
        scores = torch.tensor([[0.1, 0.2, 0.8, 0.0]])

        probabilities = scale_probabilities_from_scores(scores, tau_scale=0.1)

        self.assertEqual(probabilities.argmax(dim=-1).item(), 2)
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))

    def test_physical_teacher_prefers_scale_six_when_only_it_resolves_truth(self):
        true_angles = torch.tensor([[-5.0, 5.0]])
        covariances = {
            4: _theoretical_covariance(4, [0.0, 30.0]),
            5: _theoretical_covariance(5, [0.0, 30.0]),
            6: _theoretical_covariance(6, [-5.0, 5.0]),
            7: _theoretical_covariance(7, [0.0, 30.0]),
        }

        teacher = build_scale_teacher(covariances, true_angles, tau_scale=0.1)

        self.assertEqual(teacher.scale_scores.shape, (1, 4))
        self.assertEqual(teacher.scale_probabilities.argmax(dim=-1).item(), 2)
        torch.testing.assert_close(
            teacher.scale_probabilities.sum(dim=-1),
            torch.ones(1),
        )
        self.assertFalse(teacher.scale_scores.requires_grad)


if __name__ == "__main__":
    unittest.main()
