# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.

import unittest
import torch
import roma
from test.utils import is_close

def is_close(A, B, eps1 = 1e-5, eps2 = 1e-5):
    return torch.norm(A - B) / (torch.norm(torch.abs(A) + torch.abs(B)) + eps1) < eps2

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

class TestMappings(unittest.TestCase):
    def test_orthonormal(self):
        M = torch.eye(3, device=device).expand(10, 3, 3).contiguous()
        self.assertTrue(roma.is_orthonormal_matrix(M))
        M[:,:,-1] *= -1
        self.assertTrue(roma.is_orthonormal_matrix(M))
        torch.manual_seed(666)
        M = torch.randn((1,3,3))
        self.assertFalse(roma.is_orthonormal_matrix(M))
        
    def test_rotation(self):
        M = torch.eye(3, device=device).expand(10, 3, 3).contiguous()
        self.assertTrue(roma.is_rotation_matrix(M))
        M[:,:,-1] *= -1
        self.assertFalse(roma.is_rotation_matrix(M))
        torch.manual_seed(666)
        M = torch.randn((1,3,3))
        self.assertFalse(roma.is_rotation_matrix(M))
        
    def test_procrustes(self):
        torch.manual_seed(666)
        for i in range(10):
            M = torch.randn((10,3,3), device=device)
            R = roma.procrustes(M)
            self.assertTrue(roma.is_orthonormal_matrix(R, 1e-5))
            Rbis = roma.procrustes(R)
            self.assertTrue(is_close(R, Rbis))
            
    def test_special_procrustes(self):
        torch.manual_seed(666)
        for i in range(10):
            M = torch.randn((10,3,3), device=device)
            R = roma.special_procrustes(M)
            self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
            Rbis = roma.special_procrustes(R)
            self.assertTrue(is_close(R, Rbis))
            
    def test_special_gramschmidt(self):
        torch.manual_seed(666)
        M = torch.randn((100,3,2), device=device)
        R = roma.special_gramschmidt(M)
        self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
        Rbis = roma.special_gramschmidt(R)
        self.assertTrue(is_close(R, Rbis))
        
    def test_rotvec_unitquat(self):
        torch.manual_seed(666)
        batch_size = 100
        x = 10 * torch.randn((batch_size, 3), device=device)
        #Forward mapping
        q = roma.rotvec_to_unitquat(x)
        self.assertEqual(q.shape, (batch_size, 4))
        self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1) < 1e-6))
        # Backward mapping
        xbis = roma.unitquat_to_rotvec(q)
        qbis = roma.rotvec_to_unitquat(xbis)
        self.assertTrue(torch.all(torch.min(torch.norm(qbis - q, dim=-1), torch.norm(qbis + q, dim=-1)) < 1e-6))
        xter = roma.unitquat_to_rotvec(qbis)
        self.assertTrue(is_close(xbis, xter))
        
    def test_rotvec_rotmat(self):
        torch.manual_seed(666)
        batch_size = 100
        # Perform the test for large and small angles
        for scale in (10.0, 1e-7):
            x = scale * torch.randn((batch_size, 3), device=device)
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
        x = torch.randn((batch_size, 4), device=device)
        q = x / torch.norm(x, dim=-1, keepdim=True)
        # Forward
        R = roma.unitquat_to_rotmat(q)
        self.assertTrue(roma.is_rotation_matrix(R, 1e-5))
        # Backward
        qbis = roma.rotmat_to_unitquat(R)
        self.assertTrue(torch.all(torch.min(torch.norm(qbis - q, dim=-1), torch.norm(qbis + q, dim=-1)) < 1e-6))
      
    def test_symmatrix_to_unitquat(self):
        torch.manual_seed(666)
        batch_size = 100
        x = torch.randn((batch_size, 10), device=device)
        q = roma.symmatrixvec_to_unitquat(x)
        self.assertEqual(q.shape, (batch_size, 4))
        self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1) < 1e-6))

if __name__ == "__main__":
    unittest.main()