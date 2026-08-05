import unittest

import numpy as np
import torch

from multisource_doa.physics.projection import (
    dykstra_structured_projection,
    structured_projection_torch,
    structure_errors,
)


class StructuredProjectionTest(unittest.TestCase):
    def test_dykstra_projection_satisfies_all_invariants(self):
        rng = np.random.default_rng(17)
        matrix = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        matrix[0, 7] += 20.0j
        matrix[3, 3] -= 12.0

        result = dykstra_structured_projection(
            matrix,
            target_trace=8.0,
            tolerance=1e-7,
            max_iterations=300,
            eigenvalue_floor=1e-6,
        )

        self.assertTrue(result.converged)
        self.assertLess(result.hermitian_error, 1e-7)
        self.assertLess(result.toeplitz_error, 1e-7)
        self.assertLess(result.trace_error, 1e-7)
        self.assertGreaterEqual(result.min_eigenvalue, 1e-6 - 1e-8)
        self.assertAlmostEqual(np.trace(result.matrix).real, 8.0, places=6)

    def test_structure_errors_detect_unstructured_matrix(self):
        matrix = np.eye(4, dtype=np.complex128)
        matrix[0, 1] = 2.0 + 3.0j

        errors = structure_errors(matrix, target_trace=4.0)

        self.assertGreater(errors.hermitian_error, 0.0)
        self.assertGreater(errors.toeplitz_error, 0.0)

    def test_torch_projection_has_finite_gradients_and_auditable_structure(self):
        generator = torch.Generator().manual_seed(9)
        real = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64)
        imag = torch.randn(2, 8, 8, generator=generator, dtype=torch.float64)
        real.requires_grad_()
        imag.requires_grad_()
        matrix = torch.complex(real, imag)

        projected = structured_projection_torch(
            matrix,
            target_trace=8.0,
            iterations=4,
            eigenvalue_floor=1e-6,
        )
        loss = projected.abs().square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.isfinite(real.grad).all())
        self.assertTrue(torch.isfinite(imag.grad).all())
        torch.testing.assert_close(projected, projected.mH, atol=1e-9, rtol=0.0)
        torch.testing.assert_close(
            projected.diagonal(dim1=-2, dim2=-1).real.sum(-1),
            torch.full((2,), 8.0, dtype=torch.float64),
            atol=1e-8,
            rtol=0.0,
        )
        for lag in range(-7, 8):
            diagonal = projected.diagonal(offset=lag, dim1=-2, dim2=-1)
            centered = diagonal - diagonal.mean(dim=-1, keepdim=True)
            self.assertLess(centered.abs().max().detach().item(), 1e-9)
        minimum = torch.linalg.eigvalsh(projected).min().detach().item()
        self.assertGreaterEqual(minimum, 1e-6 - 1e-8)


if __name__ == "__main__":
    unittest.main()
