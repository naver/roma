# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import torch
import roma
import unittest
from test import utils

class TestProcrustesDerivatives(unittest.TestCase):
    def test_derivatives(self):
        # Can only be done using float64
        device = torch.device(0) if torch.cuda.is_available() else None
        dtype=torch.float64
        batch_size = 10
        torch.manual_seed(666)
        d = 3

        M = torch.randn(batch_size, d, d, dtype=dtype, device=device)
        # Check derivatives
        eps = 1e-7
        eps2 = 1e-4
        for func in (lambda x : roma.procrustes(x),
                    lambda x : roma.special_procrustes(x),
                    lambda x: roma.procrustes_naive(x),
                    lambda x: roma.special_procrustes_naive(x),
                    lambda x: roma.procrustes(x, return_singular_values=True)[1],
                    lambda x: roma.special_procrustes(x, return_singular_values=True)[1],
                    lambda x: roma.procrustes_naive(x, return_singular_values=True)[1],
                    lambda x: roma.special_procrustes_naive(x, return_singular_values=True)[1],
                    ):
            # Numerical gradient
            num = utils.numerical_jacobian(func, M, eps)
            auto = utils.automatic_jacobian(func, M)
            self.assertTrue(utils.is_close(num, auto, eps2=eps2))

    def _test_convergence(self, random_initialization, regularization=0.0):
        r"""
        Try to solve an optimization problem using Special Procrustes on SO(3)
        """
        device=torch.device(0) if torch.cuda.is_available() else None
        torch.manual_seed(666)
        b = 5
        d = 3
        # Remark: does not converge to the true minimum  without enforcing rotation.
        force_rotation = True
        if random_initialization:
            # Random initialization
            M = torch.randn((b, d, d), requires_grad=True, device=device)
        else:
            # Harder degenerated case to test numerical stability
            M = torch.zeros((b, d, d), requires_grad=True, device=device)
            # If given a zero matrix as input, gradients are equal to 0 and nothing happens.
            M.data[:,0,0] = 1

        Rtarget = roma.random_rotmat(b, device=device)
        assert roma.is_rotation_matrix(Rtarget, 1e-5)

        optimizer = torch.optim.Adam([M], lr=0.1)

        # display_period = 100
        # print(f"Regularization: {regularization}")
        for iteration in range(2000):
            optimizer.zero_grad()

            R = roma.procrustes(M, force_rotation=force_rotation, regularization=regularization)
            if force_rotation:
                assert roma.is_rotation_matrix(R, 1e-5)
            else:
                assert roma.is_orthonormal_matrix(R, 1e-5)
            loss = torch.nn.functional.mse_loss(R, Rtarget)
            with torch.no_grad():
                unnormalized_loss = torch.nn.functional.mse_loss(R, M)
            # if iteration % display_period == display_period-1:
            #     print(f"{iteration}: loss {loss.item()} -- unnormalized_loss {unnormalized_loss.item()}")
            
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(),  1e-7)
        if regularization > 0:
            # M should be roughly equal to R after optimization due to the regularization
            self.assertLess(unnormalized_loss.item(),  1e-4)

    def test_convergence_random_initialization(self):
        self._test_convergence(True, 0.0)
        self._test_convergence(True, 1e-4)

    def test_convergence_degenerated_initialization(self):
        self._test_convergence(False)    
        self._test_convergence(False, 1e-4)

if __name__ == '__main__':
    unittest.main()