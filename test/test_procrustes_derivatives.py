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
        dtype = torch.float64
        batch_size = 10
        torch.manual_seed(666)
        d = 3

        M = torch.randn(batch_size, d, d, dtype=dtype, device=device, requires_grad=True)
        # Check derivatives
        for func in (
            lambda x: roma.procrustes(x),
            lambda x: roma.special_procrustes(x),
            lambda x: roma.procrustes_naive(x),
            lambda x: roma.special_procrustes_naive(x),
            lambda x: roma.procrustes(x, return_singular_values=True)[1],
            lambda x: roma.special_procrustes(x, return_singular_values=True)[1],
            lambda x: roma.procrustes_naive(x, return_singular_values=True)[1],
            lambda x: roma.special_procrustes_naive(x, return_singular_values=True)[1],
        ):
            self.assertTrue(torch.autograd.gradcheck(func, (M,), eps=1e-7, atol=1e-4))

    def _test_convergence(self, random_initialization, regularization=0.0):
        r"""
        Try to solve an optimization problem using Special Procrustes on SO(3)
        """
        device = torch.device(0) if torch.cuda.is_available() else None
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
            M.data[:, 0, 0] = 1

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
        self.assertLess(loss.item(), 1e-7)
        if regularization > 0:
            # M should be roughly equal to R after optimization due to the regularization
            self.assertLess(unnormalized_loss.item(), 1e-4)

    def test_convergence_random_initialization(self):
        self._test_convergence(True, 0.0)
        self._test_convergence(True, 1e-4)

    def test_convergence_degenerated_initialization(self):
        self._test_convergence(False)
        self._test_convergence(False, 1e-4)


class TestProcrustesForwardDerivatives(unittest.TestCase):
    r"""
    Tests of forward-mode differentiation (jvp) for procrustes and special_procrustes.
    """

    def setUp(self):
        self.device = torch.device(0) if torch.cuda.is_available() else None
        self.dtype = torch.float64
        torch.manual_seed(666)

    def _jvp_dual(self, func, M, dM):
        r"""
        Forward-mode directional derivative of func at M in the direction dM,
        using dual tensors.
        """
        with torch.autograd.forward_ad.dual_level():
            output = func(torch.autograd.forward_ad.make_dual(M, dM))
            return torch.autograd.forward_ad.unpack_dual(output).tangent

    def test_jvp_numerical(self):
        batch_size = 10
        d = 3
        M = torch.randn(batch_size, d, d, dtype=self.dtype, device=self.device)
        dM = torch.randn_like(M)
        eps = 1e-6
        eps2 = 1e-6
        for func in (
            lambda x: roma.procrustes(x),
            lambda x: roma.special_procrustes(x),
            lambda x: roma.procrustes(x, return_singular_values=True)[1],
            lambda x: roma.special_procrustes(x, return_singular_values=True)[1],
        ):
            num = utils.central_difference(func, M, dM, eps)
            fwd = self._jvp_dual(func, M, dM)
            self.assertTrue(utils.is_close(num, fwd, eps2=eps2))

    def test_jvp_vjp_consistency(self):
        r"""
        Test that the manually-written jvp (forward mode) and backward (backward mode) of procrustes
        describe the same Jacobian J, by checking the adjoint identity that relates them:
        <J^T g, dM> == <g, J dM> for arbitrary input tangent dM and output cotangent g = (gR, gDS),
        i.e. <vjp(gR, gDS), dM> == <gR, dR> + <gDS, dDS>.
        Both sides are computed independently (autograd backward vs dual tensors), so a transposition,
        sign or clamping mismatch between the two hand-written implementations would break the equality.
        In exact arithmetic the identity is exact, allowing a tolerance much tighter than
        finite differences.
        Note: this only holds with regularization=0, since the regularization term is added to the
        gradient during backpropagation only, and is deliberately ignored by the jvp.
        """
        batch_size = 10
        for d in (2, 3, 4):
            for force_rotation in (False, True):
                M = torch.randn(batch_size, d, d, dtype=self.dtype, device=self.device, requires_grad=True)
                dM = torch.randn_like(M)
                gR = torch.randn_like(M)
                gDS = torch.randn(batch_size, d, dtype=self.dtype, device=self.device)
                R, DS = roma.procrustes(M, force_rotation=force_rotation, return_singular_values=True)
                (grad_M,) = torch.autograd.grad((R * gR).sum() + (DS * gDS).sum(), M)
                lhs = (grad_M * dM).sum()
                dR = self._jvp_dual(lambda x: roma.procrustes(x, force_rotation=force_rotation), M.detach(), dM)
                dDS = self._jvp_dual(
                    lambda x: roma.procrustes(x, force_rotation=force_rotation, return_singular_values=True)[1],
                    M.detach(),
                    dM,
                )
                rhs = (dR * gR).sum() + (dDS * gDS).sum()
                self.assertLess(abs(lhs.item() - rhs.item()), 1e-9)

    def test_gradcheck(self):
        batch_size = 3
        d = 3
        for force_rotation in (False, True):
            M = torch.randn(batch_size, d, d, dtype=self.dtype, device=self.device, requires_grad=True)
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda M: roma.procrustes(M, force_rotation, 0.0, 1e-7, return_singular_values=True),
                    (M,),
                    check_forward_ad=True,
                    check_backward_ad=True,
                )
            )

    def test_torch_func_transforms(self):
        batch_size = 5
        d = 3
        M = torch.randn(batch_size, d, d, dtype=self.dtype, device=self.device)
        dM = torch.randn_like(M)
        func = roma.special_procrustes
        # torch.func.jvp
        _, dR = torch.func.jvp(func, (M,), (dM,))
        num = utils.central_difference(func, M, dM, 1e-6)
        self.assertTrue(utils.is_close(num, dR, eps2=1e-6))
        # torch.func.jacfwd
        jacobian_fwd = torch.func.jacfwd(func)(M)
        jacobian_bwd = torch.autograd.functional.jacobian(func, M)
        self.assertTrue(utils.is_close(jacobian_bwd, jacobian_fwd, eps2=1e-7))
        # torch.vmap
        R_vmap = torch.vmap(lambda m: func(m[None])[0])(M)
        self.assertTrue(utils.is_close(func(M), R_vmap, eps2=1e-7))
        # torch.func.grad
        g = torch.func.grad(lambda M: func(M).sum())(M)
        Mg = M.clone().requires_grad_(True)
        func(Mg).sum().backward()
        self.assertTrue(utils.is_close(Mg.grad, g, eps2=1e-7))

    def test_jvp_degenerated_input(self):
        r"""
        For a zero input matrix, the jvp should be clamped to finite values (0 in practice).
        """
        batch_size = 5
        d = 3
        M = torch.zeros(batch_size, d, d, dtype=self.dtype, device=self.device)
        dM = torch.randn_like(M)
        dR = self._jvp_dual(lambda x: roma.special_procrustes(x), M, dM)
        self.assertTrue(torch.all(torch.isfinite(dR)))

    def test_jvp_ignores_regularization(self):
        r"""
        The regularization parameter only affects backpropagation
        and should have no effect on forward-mode derivatives.
        """
        batch_size = 5
        d = 3
        M = torch.randn(batch_size, d, d, dtype=self.dtype, device=self.device)
        dM = torch.randn_like(M)
        dR0 = self._jvp_dual(lambda x: roma.special_procrustes(x, regularization=0.0), M, dM)
        dR1 = self._jvp_dual(lambda x: roma.special_procrustes(x, regularization=1e-2), M, dM)
        self.assertTrue(torch.all(dR0 == dR1))

    def test_jvp_torch_compile(self):
        r"""
        Regression test for a PyTorch bug: Dynamo silently drops the custom jvp of an autograd Function
        when torch.func.jvp is applied inside a torch.compile'd function.
        Uses an input with degenerate singular values, for which the default SVD forward-AD rule
        returns non-finite values whereas the custom clamped jvp does not.
        """
        batch_size = 5
        d = 3
        M = torch.eye(d, dtype=self.dtype, device=self.device).expand(batch_size, d, d).clone()
        dM = torch.randn_like(M)

        def func(M, dM):
            return torch.func.jvp(roma.special_procrustes, (M,), (dM,))[1]

        dR_eager = func(M, dM)
        torch._dynamo.reset()
        dR_compiled = torch.compile(func)(M, dM)
        self.assertTrue(torch.all(torch.isfinite(dR_compiled)))
        self.assertTrue(torch.allclose(dR_eager, dR_compiled))


if __name__ == "__main__":
    unittest.main()
