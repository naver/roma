# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import unittest
import torch
import roma
import numpy as np
from test.utils import is_close

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

class TestMappings(unittest.TestCase):
    def test_orthonormal(self):
        for dtype in (torch.float32, torch.float64):
            M = torch.eye(3, dtype=dtype, device=device).expand(10, 3, 3).contiguous()
            self.assertTrue(roma.is_orthonormal_matrix(M))
            M[:,:,-1] *= -1
            self.assertTrue(roma.is_orthonormal_matrix(M))
            torch.manual_seed(666)
            M = torch.randn((1,3,3))
            self.assertFalse(roma.is_orthonormal_matrix(M))
        
    def test_rotation(self):
        for dtype in (torch.float32, torch.float64):
            M = torch.eye(3, dtype=dtype, device=device).expand(10, 3, 3).contiguous()
            self.assertTrue(roma.is_rotation_matrix(M))
            M[:,:,-1] *= -1
            self.assertFalse(roma.is_rotation_matrix(M))
            torch.manual_seed(666)
            M = torch.randn((1,3,3))
            self.assertFalse(roma.is_rotation_matrix(M))
        
    def test_procrustes(self):
        torch.manual_seed(666)
        for dtype in (torch.float32, torch.float64):
            for i in range(10):
                M = torch.randn((10,3,3), dtype=dtype, device=device)
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
                M = torch.randn((10,3,3), dtype=dtype, device=device)
                R = roma.special_procrustes(M)
                self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
                Rbis = roma.special_procrustes(R)
                self.assertTrue(is_close(R, Rbis))
                # Ensure consistency
                Rter = roma.special_procrustes_naive(M)
                self.assertTrue(is_close(R, Rter))
            
    def test_special_gramschmidt(self):
        torch.manual_seed(666)
        for dtype in (torch.float32, torch.float64):
            M = torch.randn((100,3,2), dtype=dtype, device=device)
            R = roma.special_gramschmidt(M)
            self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
            Rbis = roma.special_gramschmidt(R)
            self.assertTrue(is_close(R, Rbis))
        
    def test_rotvec_unitquat(self):
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            x = 10 * torch.randn((batch_size, 3), dtype=dtype, device=device)
            #Forward mapping.
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
        for dtype in (torch.float32, torch.float64):
            q = roma.random_unitquat(batch_size, dtype)
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
                self.assertTrue(torch.all(torch.min(torch.norm(qbis - q, dim=-1), torch.norm(qbis + q, dim=-1)) < 5e-6))

    def test_rotvec_rotmat(self):
        torch.manual_seed(666)
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            # Perform the test for large and small angles
            for scale in (10.0, 1e-7):
                x = scale * torch.randn((batch_size, 3), dtype=dtype, device=device)
                #Forward mapping
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
        # Remark: the eigenvalue decomposition may fail using float32 or depending on the random seed,
        # due to conditionning issues.
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
        for batch_shape in [(), (10,), (23,5)]:
            quat_xyzw = torch.randn(batch_shape + (4,))
            quat_wxyz = roma.mappings.quat_xyzw_to_wxyz(quat_xyzw)
            self.assertTrue(quat_xyzw.shape == quat_wxyz.shape)
            quat_xyzw_bis = roma.mappings.quat_wxyz_to_xyzw(quat_wxyz)
            self.assertTrue(torch.all(quat_xyzw == quat_xyzw_bis))

if __name__ == "__main__":
    unittest.main()