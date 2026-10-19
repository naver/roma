# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import unittest
import torch
import roma
import numpy as np
from test.utils import is_close

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


class TestMappings(unittest.TestCase):
    def test_sinc_real_domain(self):
        x = torch.tensor(
            [-7.0, -np.pi - 1e-4, -np.pi + 1e-4, -1.0, -1e-4, 0.0, 1e-4, 1.0, np.pi - 1e-4, np.pi + 1e-4, 7.0],
            dtype=torch.float64,
            device=device,
        )
        # torch.sinc uses the normalized convention sin(pi*x)/(pi*x).
        expected = torch.sinc(x / np.pi)
        torch.testing.assert_close(roma.sinc(x), expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(roma.inv_sinc(x), 1.0 / expected, atol=1e-10, rtol=1e-10)

    def test_sinc_derivatives(self):
        x = torch.tensor([-1.0, -1e-4, 0.0, 1e-4, 1.0, np.pi - 1e-3], dtype=torch.float64, device=device)
        x.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(roma.sinc, (x,)))
        self.assertTrue(torch.autograd.gradcheck(roma.inv_sinc, (x,), eps=1e-7, rtol=1e-4))
        for function, second_derivative in ((roma.sinc, -1.0 / 3.0), (roma.inv_sinc, 1.0 / 3.0)):
            zero = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
            first = torch.autograd.grad(function(zero), zero, create_graph=True)[0]
            second = torch.autograd.grad(first, zero)[0]
            torch.testing.assert_close(first, torch.zeros_like(first))
            torch.testing.assert_close(second, torch.full_like(second, second_derivative))

    def test_unitquat_to_rotvec_near_full_turn(self):
        axis = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 1.0]], device=device)
        for dtype in (torch.float32, torch.float64):
            axes = torch.nn.functional.normalize(axis.to(dtype), dim=-1)
            angles = 2 * np.pi - torch.tensor([1e-2, 1e-3, 1e-4, 1e-6], dtype=dtype, device=device)
            # Construct quaternions directly from their axis and angle.
            vector = torch.sin(angles[:, None, None] / 2) * axes[None]
            scalar = torch.cos(angles[:, None, None] / 2).expand(-1, len(axes), 1)
            q = torch.cat((vector, scalar), dim=-1)
            for shortest_arc in (False, True):
                with self.subTest(dtype=dtype, shortest_arc=shortest_arc):
                    expected_angle = angles - 2 * np.pi if shortest_arc else angles
                    expected = expected_angle[:, None, None] * axes[None]
                    actual = roma.unitquat_to_rotvec(q, shortest_arc=shortest_arc)
                    torch.testing.assert_close(
                        actual, expected, atol=2e-6 if dtype == torch.float32 else 1e-10, rtol=1e-6
                    )
                    torch.testing.assert_close(roma.rotvec_to_unitquat(actual), -q if shortest_arc else q)

    def test_unitquat_to_rotvec_near_full_turn_derivative(self):
        axis = torch.nn.functional.normalize(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64, device=device), dim=-1)
        for delta in (1e-3, 1e-4, 1e-6):
            with self.subTest(delta=delta):
                angle = torch.tensor(2 * np.pi - delta, dtype=torch.float64, device=device, requires_grad=True)
                q = torch.cat((torch.sin(angle / 2) * axis, torch.cos(angle / 2).reshape(1)))
                recovered = roma.unitquat_to_rotvec(q, shortest_arc=False)
                derivative = torch.autograd.grad((recovered * axis).sum(), angle)[0]
                torch.testing.assert_close(derivative, torch.ones_like(derivative), atol=1e-7, rtol=1e-7)

    def test_orthonormal(self):
        for dtype in (torch.float32, torch.float64):
            M = torch.eye(3, dtype=dtype, device=device).expand(10, 3, 3).contiguous()
            self.assertTrue(roma.is_orthonormal_matrix(M))
            M[:, :, -1] *= -1
            self.assertTrue(roma.is_orthonormal_matrix(M))
            torch.manual_seed(666)
            M = torch.randn((1, 3, 3))
            self.assertFalse(roma.is_orthonormal_matrix(M))

    def test_rotation(self):
        for dtype in (torch.float32, torch.float64):
            M = torch.eye(3, dtype=dtype, device=device).expand(10, 3, 3).contiguous()
            self.assertTrue(roma.is_rotation_matrix(M))
            M[:, :, -1] *= -1
            self.assertFalse(roma.is_rotation_matrix(M))
            torch.manual_seed(666)
            M = torch.randn((1, 3, 3))
            self.assertFalse(roma.is_rotation_matrix(M))

    def test_procrustes(self):
        torch.manual_seed(666)
        for dtype in (torch.float32, torch.float64):
            for i in range(10):
                M = torch.randn((10, 3, 3), dtype=dtype, device=device)
                R = roma.procrustes(M)
                self.assertTrue(roma.is_orthonormal_matrix(R, 1e-5))
                Rbis = roma.procrustes(R)
                self.assertTrue(is_close(R, Rbis))
                # Ensure consistency
                Rter = roma.procrustes_naive(M)
                self.assertTrue(is_close(R, Rter))

    def test_special_procrustes(self):
        torch.manual_seed(666)
        for dtype in (torch.float32, torch.float64):
            for i in range(10):
                M = torch.randn((10, 3, 3), dtype=dtype, device=device)
                R = roma.special_procrustes(M)
                self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
                Rbis = roma.special_procrustes(R)
                self.assertTrue(is_close(R, Rbis))
                # Ensure consistency
                Rter = roma.special_procrustes_naive(M)
                self.assertTrue(is_close(R, Rter))

    def test_special_procrustes_degenerate(self):
        """
        Inputs for which the nearest rotation is not unique or is ill-conditioned (closed-form 3D solution).
        """
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            R0 = roma.random_rotmat(batch_size, dtype=dtype, device=device)
            scales = torch.logspace(-3, 3, batch_size, dtype=dtype, device=device)[:, None, None]
            signed_permutation = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=dtype, device=device
            )
            cases = {
                "zero": torch.zeros((batch_size, 3, 3), dtype=dtype, device=device),
                "reflection": R0 * torch.tensor([1.0, 1.0, -1.0], dtype=dtype, device=device),
                "rank-1": R0 * torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device),
                "rank-2": R0 * torch.tensor([1.0, 1.0, 0.0], dtype=dtype, device=device),
                "ambiguous": R0 * torch.tensor([1.0, 0.5, -0.5], dtype=dtype, device=device),
                "nearly ambiguous": R0 * torch.tensor([1.0, 0.5, -0.5 + 1e-3], dtype=dtype, device=device),
                "tiny": 1e-30 * R0,
                "huge": 1e30 * R0,
                # Isotropic inputs: M^T M is exactly proportional to the identity.
                "scaled identity": scales * torch.eye(3, dtype=dtype, device=device),
                "axis aligned": scales * signed_permutation,
            }
            for name, M in cases.items():
                R = roma.special_procrustes(M)
                self.assertTrue(torch.all(torch.isfinite(R)), name)
                self.assertTrue(roma.is_rotation_matrix(R, 1e-5), name)
                # Optimality: tr(R^T M) should be as large as with the SVD-based solution (in float64).
                Rref = roma.special_procrustes_naive(M.double())
                objective = torch.sum(R.double() * M.double(), dim=(-1, -2))
                objective_ref = torch.sum(Rref * M.double(), dim=(-1, -2))
                scale = torch.linalg.matrix_norm(M.double()).clamp_min(1e-300)
                self.assertTrue(torch.all((objective_ref - objective) / scale < 1e-6), name)
                # Finite gradients (backward and forward mode)
                M = M.requires_grad_()
                R = roma.special_procrustes(M)
                torch.sum(R * torch.randn_like(R)).backward()
                self.assertTrue(torch.all(torch.isfinite(M.grad)))
                _, dR = torch.func.jvp(roma.special_procrustes, (M.detach(),), (torch.randn_like(M),))
                self.assertTrue(torch.all(torch.isfinite(dR)))
            # Zero input: identity output.
            self.assertTrue(
                torch.allclose(roma.special_procrustes(cases["zero"]), torch.eye(3, dtype=dtype, device=device))
            )

    def test_procrustes_rejects_integer_input(self):
        for d in (3, 4):
            M = torch.eye(d, dtype=torch.int64, device=device)[None]
            for force_rotation in (False, True):
                with self.assertRaises(AssertionError):
                    roma.procrustes(M, force_rotation=force_rotation)

    def test_special_procrustes_vs_symmatrix(self):
        """
        The 3D closed-form solution relies on Horn's symmetric matrix N(M), whose top eigenvector is the quaternion of the
        nearest rotation: roma.symmatrix_to_projective_point(-N(M)) should therefore give the same rotation.
        """
        torch.manual_seed(666)
        M = torch.randn((100, 3, 3), dtype=torch.float64, device=device)
        N = roma.mappings._horn_matrix(M)
        quat = roma.symmatrix_to_projective_point(-N)
        R = roma.unitquat_to_rotmat(quat)
        self.assertTrue(is_close(R, roma.special_procrustes(M), eps2=1e-10))
        # Horn's identity: tr(R^T M) = q^T N(M) q.
        q = roma.random_unitquat(100, dtype=torch.float64, device=device)
        lhs = torch.sum(roma.unitquat_to_rotmat(q) * M, dim=(-1, -2))
        rhs = torch.einsum("bi,bij,bj->b", q, N, q)
        self.assertTrue(torch.allclose(lhs, rhs))

    def test_procrustes_method(self):
        torch.manual_seed(666)
        for dtype in (torch.float32, torch.float64):
            M = torch.randn((10, 3, 3), dtype=dtype, device=device)
            R_auto = roma.special_procrustes(M)
            R_svd = roma.special_procrustes(M, method="svd")
            R_horn = roma.special_procrustes(M, method="horn")
            self.assertTrue(is_close(R_auto, R_svd))
            self.assertTrue(torch.equal(R_auto, R_horn))
            self.assertTrue(torch.equal(roma.procrustes(M, force_rotation=True, method="horn"), R_horn))
            # Singular values are only available with the SVD.
            R, DS = roma.special_procrustes(M, return_singular_values=True, method="svd")
            self.assertTrue(is_close(R, R_svd))
            with self.assertRaises(ValueError):
                roma.special_procrustes(M, return_singular_values=True, method="horn")
            # Horn's method only applies to 3D rotations.
            with self.assertRaises(ValueError):
                roma.procrustes(M, force_rotation=False, method="horn")
            with self.assertRaises(ValueError):
                roma.special_procrustes(torch.randn((10, 4, 4), dtype=dtype, device=device), method="horn")
            with self.assertRaises(ValueError):
                roma.special_procrustes(M, method="foo")
            # Gradients should be available with both methods.
            for method in ("svd", "horn"):
                Mg = M.clone().requires_grad_()
                torch.sum(roma.special_procrustes(Mg, method=method)).backward()
                self.assertTrue(torch.all(torch.isfinite(Mg.grad)))

    def test_procrustes_autocast(self):
        r"""
        Under autocast, both methods behave as if applied to the input cast to float32 without autocast:
        the output is float32, and the gradient (including the regularization term) is computed in float32
        and rounded once to the input dtype.
        """
        torch.manual_seed(666)
        autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        M = torch.randn((100, 3, 3), device=device).to(autocast_dtype)
        grad_R = torch.randn((100, 3, 3), device=device)
        for method in ("svd", "horn"):
            for regularization in (0.0, 0.1):
                M32 = M.float().requires_grad_()
                R32 = roma.special_procrustes(M32, regularization=regularization, method=method)
                R32.backward(grad_R)
                Mr = M.clone().requires_grad_()
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    R = roma.special_procrustes(Mr, regularization=regularization, method=method)
                R.backward(grad_R)
                self.assertEqual(R.dtype, torch.float32)
                self.assertTrue(torch.equal(R, R32))
                self.assertEqual(Mr.grad.dtype, autocast_dtype)
                self.assertTrue(torch.equal(Mr.grad, M32.grad.to(autocast_dtype)))

    def test_special_procrustes_other_dimensions(self):
        torch.manual_seed(666)
        for d in (2, 4):
            for dtype in (torch.float32, torch.float64):
                M = torch.randn((10, d, d), dtype=dtype, device=device)
                R = roma.special_procrustes(M)
                self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
                self.assertTrue(is_close(R, roma.special_procrustes_naive(M)))

    def test_procrustes_naive_singular_values_shape(self):
        torch.manual_seed(666)
        # Compare with procrustes for zero, one and two batch dimensions.
        for batch_shape in ((), (10,), (2, 5)):
            M = torch.randn(batch_shape + (3, 3), dtype=torch.float64, device=device)
            for force_rotation in (False, True):
                R, DS = roma.procrustes(M, force_rotation=force_rotation, return_singular_values=True)
                Rbis, DSbis = roma.procrustes_naive(M, force_rotation=force_rotation, return_singular_values=True)
                self.assertEqual(Rbis.shape, batch_shape + (3, 3))
                self.assertEqual(DSbis.shape, batch_shape + (3,))
                self.assertTrue(is_close(R, Rbis))
                self.assertTrue(is_close(DS, DSbis))

    def test_special_gramschmidt(self):
        torch.manual_seed(666)
        for dtype in (torch.float32, torch.float64):
            M = torch.randn((100, 3, 2), dtype=dtype, device=device)
            R = roma.special_gramschmidt(M)
            self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
            Rbis = roma.special_gramschmidt(R)
            self.assertTrue(is_close(R, Rbis))

    def test_rotvec_unitquat(self):
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            x = 10 * torch.randn((batch_size, 3), dtype=dtype, device=device)
            # Forward mapping.
            q = roma.rotvec_to_unitquat(x)
            # Ensure that the output is a valid unit quaternion.
            self.assertEqual(q.shape, (batch_size, 4))
            self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1) < 1e-6))

            # Backward mapping
            xbis = roma.unitquat_to_rotvec(q)
            qbis = roma.rotvec_to_unitquat(xbis)
            # Ensure cyclic consistency of the mappings
            self.assertTrue(torch.all(torch.min(torch.norm(qbis - q, dim=-1), torch.norm(qbis + q, dim=-1)) < 1e-6))
            xter = roma.unitquat_to_rotvec(qbis)
            self.assertTrue(is_close(xbis, xter))

    def test_unitquat_to_rotvec(self):
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float64, torch.float32):
            for small_angle in (True, False):
                if small_angle:
                    q = torch.randn((batch_size, 4), dtype=dtype, device=device)
                    q[..., 3] = 1000
                    q = q / torch.norm(q, dim=-1, keepdim=True)
                else:
                    q = roma.random_unitquat(batch_size, dtype=dtype, device=device)
                xp = roma.unitquat_to_rotvec(q, shortest_arc=True)
                xm = roma.unitquat_to_rotvec(-q, shortest_arc=True)
                self.assertEqual(xp.shape, (batch_size, 3))
                self.assertTrue(is_close(xp, xm))
                self.assertTrue(torch.all(torch.norm(xp, dim=-1) <= np.pi))

                # Mappings from q and -q should give different results in general when shortest_arc==False
                xpn = roma.unitquat_to_rotvec(q, shortest_arc=False)
                xmn = roma.unitquat_to_rotvec(-q, shortest_arc=False)
                self.assertFalse(any([is_close(xpn[i], xmn[i]) for i in range(batch_size)]))

                # However, the result should be similar when shortest_arc==True
                xpn = roma.unitquat_to_rotvec(q, shortest_arc=True)
                xmn = roma.unitquat_to_rotvec(-q, shortest_arc=True)
                self.assertTrue(is_close(xpn, xmn))

                # Ensure cyclic consistency of the mappings in any case
                for x in xp, xm, xpn, xmn:
                    qbis = roma.rotvec_to_unitquat(x)
                    self.assertTrue(
                        torch.all(torch.min(torch.norm(qbis - q, dim=-1), torch.norm(qbis + q, dim=-1)) < 5e-6)
                    )

    def test_rotvec_rotmat(self):
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            # Perform the test for large and small angles
            for scale in (10.0, 1e-7):
                x = scale * torch.randn((batch_size, 3), dtype=dtype, device=device)
                # Forward mapping
                R = roma.rotvec_to_rotmat(x)
                self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
                # Backward mapping
                xbis = roma.rotmat_to_rotvec(R)
                Rbis = roma.rotvec_to_rotmat(x)
                self.assertTrue(is_close(R, Rbis))
                xter = roma.rotmat_to_rotvec(Rbis)
                self.assertTrue(is_close(xbis, xter))

    def test_unitquat_rotmat(self):
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            q = roma.random_unitquat(batch_size, dtype=dtype, device=device)
            # Forward
            R = roma.unitquat_to_rotmat(q)
            self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
            # Backward
            qbis = roma.rotmat_to_unitquat(R)
            self.assertTrue(torch.all(torch.min(torch.norm(qbis - q, dim=-1), torch.norm(qbis + q, dim=-1)) < 1e-6))

    def test_symmatrix_to_unitquat(self):
        torch.manual_seed(668)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            x = torch.randn((batch_size, 10), dtype=dtype, device=device)
            q = roma.symmatrixvec_to_unitquat(x)
            # Ensure that the output is a unit quaternion.
            self.assertEqual(q.shape, (batch_size, 4))
            self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1) < 1e-6))

            # Same test using explicitly a symmetric matrix.
            A = torch.randn((batch_size, 4, 4), dtype=dtype, device=device)
            A = A + A.permute(0, 2, 1)
            q = roma.symmatrix_to_projective_point(A)
            # Ensure that the output is a unit quaternion.
            self.assertEqual(q.shape, (batch_size, 4))
            self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1) < 1e-6))

    @staticmethod
    def _symmatrix_from_vec(x):
        x00, x10, x20, x30, x11, x21, x31, x22, x32, x33 = x.unbind(dim=-1)
        # fmt: off
        return torch.stack((x00, x10, x20, x30,
                            x10, x11, x21, x31,
                            x20, x21, x22, x32,
                            x30, x31, x32, x33), dim=-1).reshape(-1, 4, 4)
        # fmt: on

    @staticmethod
    def _align_sign(q, ref):
        """Flip the sign of the unit vectors q to match ref (projective points)."""
        return q * torch.sign(torch.sum(q * ref, dim=-1, keepdim=True))

    def test_symmatrix_vs_eigh(self):
        """
        The closed-form eigenvector should match a batched eigendecomposition.
        """
        torch.manual_seed(666)
        batch_size = 100
        A = torch.randn((batch_size, 4, 4), dtype=torch.float64, device=device)
        A = A + A.transpose(-1, -2)
        q_ref = torch.linalg.eigh(A)[1][..., 0]
        q = roma.symmatrix_to_projective_point(A)
        self.assertTrue(torch.all(torch.norm(self._align_sign(q, q_ref) - q_ref, dim=-1) < 1e-10))
        # Single precision input: computations are performed in double precision internally.
        A = A.to(torch.float32)
        q_ref = torch.linalg.eigh(A.double())[1][..., 0]
        q = roma.symmatrix_to_projective_point(A)
        self.assertEqual(q.dtype, torch.float32)
        self.assertTrue(torch.all(torch.norm(self._align_sign(q.double(), q_ref) - q_ref, dim=-1) < 1e-6))

    def test_symmatrix_method(self):
        torch.manual_seed(666)
        batch_size = 100
        x = torch.randn((batch_size, 10), dtype=torch.float64, device=device)
        q_auto = roma.symmatrixvec_to_unitquat(x)
        q_eigh = roma.symmatrixvec_to_unitquat(x, method="eigh")
        q_horn = roma.symmatrixvec_to_unitquat(x, method="horn")
        self.assertTrue(torch.equal(q_auto, q_horn))
        self.assertTrue(torch.all(torch.norm(self._align_sign(q_eigh, q_horn) - q_horn, dim=-1) < 1e-10))
        A = self._symmatrix_from_vec(x)
        self.assertTrue(torch.equal(roma.symmatrix_to_projective_point(A, method="horn"), q_horn))
        self.assertTrue(torch.equal(roma.symmatrix_to_projective_point(A, method="eigh"), q_eigh))
        with self.assertRaises(ValueError):
            roma.symmatrix_to_projective_point(A, method="foo")
        # Gradients should be available with both methods.
        for method in ("eigh", "horn"):
            xg = x.clone().requires_grad_()
            torch.sum(roma.symmatrixvec_to_unitquat(xg, method=method)).backward()
            self.assertTrue(torch.all(torch.isfinite(xg.grad)))

    def test_symmatrix_non_symmetric_input(self):
        """
        For a non-symmetric input, "eigh" only considers the lower triangular part and "horn" the symmetric part,
        and the gradient with respect to the input is symmetric with both methods.
        """
        torch.manual_seed(666)
        batch_size = 100
        A = torch.randn((batch_size, 4, 4), dtype=torch.float64, device=device)
        lower = torch.tril(A) + torch.tril(A, -1).transpose(-1, -2)
        sym = (A + A.transpose(-1, -2)) / 2
        for method, A_ref in (("eigh", lower), ("horn", sym)):
            q = roma.symmatrix_to_projective_point(A, method=method)
            q_ref = torch.linalg.eigh(A_ref)[1][..., 0]
            self.assertTrue(torch.all(torch.norm(self._align_sign(q, q_ref) - q_ref, dim=-1) < 1e-10))
        # On symmetric inputs, both methods have the same symmetric gradient (up to the damping of "horn").
        G = torch.randn((batch_size, 4, 4), dtype=torch.float64, device=device)
        grads = []
        for method in ("eigh", "horn"):
            Ag = sym.clone().requires_grad_()
            q = roma.symmatrix_to_projective_point(Ag, method=method)
            # Loss defined on q q^T, as q is defined up to sign.
            torch.sum(G * q[..., :, None] * q[..., None, :]).backward()
            self.assertTrue(torch.allclose(Ag.grad, Ag.grad.transpose(-1, -2), atol=1e-12))
            grads.append(Ag.grad)
        self.assertTrue(torch.allclose(grads[0], grads[1], atol=1e-6))

    def test_symmatrix_autocast(self):
        r"""
        Under autocast, both methods behave as if applied to the input cast to float32 without autocast
        (as for special_procrustes): the output is float32, and the gradient is computed in float32 and rounded once
        to the input dtype. float64 inputs are not affected.
        """
        torch.manual_seed(666)
        autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        A = torch.randn((100, 4, 4), device=device)
        A = (A + A.transpose(-1, -2)).to(autocast_dtype)
        grad_q = torch.randn((100, 4), device=device)
        for method in ("eigh", "horn"):
            A32 = A.float().requires_grad_()
            q32 = roma.symmatrix_to_projective_point(A32, method=method)
            q32.backward(grad_q)
            Ar = A.clone().requires_grad_()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                q = roma.symmatrix_to_projective_point(Ar, method=method)
                q64 = roma.symmatrix_to_projective_point(A.double(), method=method)
            q.backward(grad_q)
            self.assertEqual(q.dtype, torch.float32)
            self.assertTrue(torch.equal(q, q32))
            self.assertEqual(Ar.grad.dtype, autocast_dtype)
            self.assertTrue(torch.equal(Ar.grad, A32.grad.to(autocast_dtype)))
            self.assertEqual(q64.dtype, torch.float64)

    def test_symmatrix_derivatives(self):
        torch.manual_seed(666)
        batch_size = 20
        x = torch.randn((batch_size, 10), dtype=torch.float64, device=device, requires_grad=True)
        # Backward and forward-mode derivatives should match those of a batched eigendecomposition, where well-defined.
        q = roma.symmatrixvec_to_unitquat(x)
        q_ref = torch.linalg.eigh(self._symmatrix_from_vec(x))[1][..., 0]
        sign = torch.sign(torch.sum(q * q_ref, dim=-1, keepdim=True)).detach()
        grad_output = torch.randn_like(q)
        (grad,) = torch.autograd.grad(torch.sum(q * grad_output), x)
        (grad_ref,) = torch.autograd.grad(torch.sum(sign * q_ref * grad_output), x)
        self.assertTrue(torch.allclose(grad, grad_ref, atol=1e-6))
        dx = torch.randn_like(x)
        _, dq = torch.func.jvp(roma.symmatrixvec_to_unitquat, (x.detach(),), (dx,))
        _, dq_ref = torch.func.jvp(
            lambda x: torch.linalg.eigh(self._symmatrix_from_vec(x))[1][..., 0], (x.detach(),), (dx,)
        )
        self.assertTrue(torch.allclose(dq, sign * dq_ref, atol=1e-6))
        # Numerical check of the derivatives.
        x = torch.randn((5, 10), dtype=torch.float64, device=device, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(roma.symmatrixvec_to_unitquat, (x,), eps=1e-7, atol=1e-5, check_forward_ad=True)
        )

    def test_symmatrix_degenerate(self):
        """
        Inputs for which the smallest eigenvalue is repeated or nearly repeated, or of extreme scale.
        """
        torch.manual_seed(666)
        batch_size = 100
        Q = torch.linalg.qr(torch.randn((batch_size, 4, 4), dtype=torch.float64, device=device))[0]
        cases = {
            "identity": (1.0, 1.0, 1.0, 1.0),
            "double": (0.0, 0.0, 1.0, 2.0),
            "triple": (0.0, 0.0, 0.0, 1.0),
            "nearly double": (0.0, 1e-9, 1.0, 2.0),
            "huge": (1e30, 2e30, 3e30, 4e30),
            "tiny": (1e-30, 2e-30, 3e-30, 4e-30),
        }
        for name, eigenvalues in cases.items():
            eigenvalues = torch.tensor(eigenvalues, dtype=torch.float64, device=device)
            A0 = (Q * eigenvalues) @ Q.transpose(-1, -2)
            for dtype in (torch.float32, torch.float64):
                A = A0.to(dtype).requires_grad_()
                q = roma.symmatrix_to_projective_point(A)
                self.assertTrue(torch.all(torch.isfinite(q)), name)
                self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1) < 1e-6), name)
                # Optimality: q^T A q should be equal to the smallest eigenvalue (relatively to the scale of A).
                objective = torch.einsum("bi,bij,bj->b", q.double(), A.double(), q.double())
                scale = torch.linalg.matrix_norm(A.double())
                self.assertTrue(torch.all((objective - eigenvalues[0]) / scale < 1e-6), name)
                # Finite gradients (backward and forward mode).
                torch.sum(q * torch.randn_like(q)).backward()
                self.assertTrue(torch.all(torch.isfinite(A.grad)), name)
                _, dq = torch.func.jvp(roma.symmatrix_to_projective_point, (A.detach(),), (torch.randn_like(A),))
                self.assertTrue(torch.all(torch.isfinite(dq)), name)

    def test_rotvec_rotmat_nan_issues(self):
        # Check that casting rotation vectors to rotation matrices does not lead to non finite values with 0 angles.
        rotvec = torch.zeros(3, requires_grad=True)
        R = roma.rotvec_to_rotmat(rotvec)
        loss = torch.sum(R)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(loss)))
        self.assertTrue(torch.all(torch.isfinite(rotvec.grad)))

    def test_rotvec_unitquat_nan_issues(self):
        r"""
        Check that casting using rotation vectors does not lead to non finite values with 0 angles.
        """
        rotvec = torch.zeros(3, requires_grad=True)
        q = roma.rotvec_to_unitquat(rotvec)
        loss = torch.sum(q)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(loss)))
        self.assertTrue(torch.all(torch.isfinite(rotvec.grad)))

    def test_quat_conventions(self):
        for batch_shape in [(), (10,), (23, 5)]:
            quat_xyzw = torch.randn(batch_shape + (4,))
            quat_wxyz = roma.mappings.quat_xyzw_to_wxyz(quat_xyzw)
            self.assertTrue(quat_xyzw.shape == quat_wxyz.shape)
            quat_xyzw_bis = roma.mappings.quat_wxyz_to_xyzw(quat_wxyz)
            self.assertTrue(torch.all(quat_xyzw == quat_xyzw_bis))


if __name__ == "__main__":
    unittest.main()
