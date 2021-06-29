# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.

import unittest
import torch
import numpy as np
import roma
import roma.internal
from test.utils import is_close

class TestUtils(unittest.TestCase):
    def test_flatten(self):
        x = torch.randn((32, 24, 3, 4))
        xflat, batch_shape = roma.internal.flatten_batch_dims(x, end_dim=-3)
        self.assertEqual(batch_shape, (32, 24))
        self.assertTrue(xflat.shape == (32*24, 3, 4))
        xbis = roma.internal.unflatten_batch_dims(xflat, batch_shape)
        self.assertTrue(torch.all(xbis == x))

    def test_geodesic_distance(self):
        batch_size = 100
        axis = torch.nn.functional.normalize(torch.randn((batch_size,3)), dim=-1)
        alpha = (np.pi - 1e-5) * 2 * (torch.rand(batch_size)-0.5)
        x = alpha[:,None] * axis
        R = roma.rotvec_to_rotmat(x)
        
        cosine = roma.rotmat_cosine_angle(R)
        self.assertTrue(is_close(cosine, torch.cos(alpha)))
        
        I = torch.eye(3)
        M = roma.random_rotmat(batch_size)
        
        geo_dist = roma.rotmat_geodesic_distance(R, I[None,:,:])
        self.assertTrue(is_close(torch.abs(alpha), geo_dist))
        
        # Left-invariance of the metric
        geo_dist_bis = roma.rotmat_geodesic_distance(M @ R, M @ I[None,:,:])
        self.assertTrue(is_close(geo_dist_bis, geo_dist))
        
        # Right-invariance of the metric
        geo_dist_ter = roma.rotmat_geodesic_distance(R @ M, I[None,:,:] @ M)
        self.assertTrue(is_close(geo_dist_ter, geo_dist))
        
        geo_dist_naive = roma.rotmat_geodesic_distance_naive(M @ R, M @ I[None,:,:])
        self.assertTrue(is_close(torch.abs(alpha), geo_dist_naive))
        
    def test_unitquat(self):
        batch_size = 100
        q = roma.random_unitquat(batch_size)
        # Ensure that conjugation of a unit quaternion equates to its inverse.
        iq = roma.quat_conjugation(q)
        q_id = roma.rotvec_to_unitquat(torch.zeros(1,3))
        self.assertTrue(is_close(q_id, roma.quat_product(q, iq)))
        self.assertTrue(is_close(q_id, roma.quat_product(iq, q)))

    def test_quat(self):
        batch_size = 100
        q = roma.random_unitquat(batch_size)
        # Test quaternion inverse (hoping for non zero quaternions during the test).
        iq = roma.quat_inverse(q)
        q_id = roma.rotvec_to_unitquat(torch.zeros(1,3))
        self.assertTrue(is_close(q_id, roma.quat_product(q, iq)))
        self.assertTrue(is_close(q_id, roma.quat_product(iq, q)))
        
    def test_slerp(self):
        batch_size = 100
        
        # Identity
        x0 = torch.zeros(1, 3)
        # 'Small' rotation vector
        axis = torch.nn.functional.normalize(torch.randn((batch_size,3)), dim=-1)
        alpha = (np.pi-1e-10) * 2 * (torch.rand(batch_size)-0.5)
        x1 = alpha[:,None] * axis
        
        # Slerp
        steps = torch.linspace(0, 1, 11)
        x_true = steps[:,None,None] * x1[None,:,:]
        x_pred = roma.rotvec_slerp(x0.expand(batch_size, 3), x1, steps)
        self.assertTrue(is_close(x_true, x_pred))
        
        # Check left invariance of slerp
        q0 = roma.random_unitquat(batch_size)
        q1 = roma.random_unitquat(batch_size)
        q = roma.random_unitquat(batch_size)
        iq = roma.quat_conjugation(q)
        slerp1 = roma.unitquat_slerp(q0, q1, steps)
        slerp2 = roma.quat_product(iq.expand(len(steps), batch_size, 4),
                                   roma.unitquat_slerp(roma.quat_product(q, q0),
                                                           roma.quat_product(q, q1),
                                                           steps))
        self.assertTrue(is_close(slerp1, slerp2))       

    def test_composition(self):
        R_list = [roma.random_rotmat() for _ in range(3)]
        q_list = [roma.rotmat_to_unitquat(R) for R in R_list]
        rotvec_list = [roma.rotmat_to_rotvec(R) for R in R_list]
        
        # Test consistency of results between the different representations
        R = roma.rotmat_composition(R_list)
        q = roma.quat_composition(q_list)
        rotvec = roma.rotvec_composition(rotvec_list)
        self.assertTrue(is_close(roma.unitquat_to_rotmat(q), R))
        self.assertTrue(is_close(roma.rotvec_to_rotmat(rotvec), R))

        # Test consistency with respect to inverse
        self.assertTrue(is_close(roma.rotmat_cosine_angle(roma.rotmat_composition((R_list[0], roma.rotmat_inverse(R_list[0])))), torch.as_tensor(1.0)))
        self.assertTrue(is_close(roma.quat_composition((q_list[0], roma.quat_inverse(q_list[0]))), torch.as_tensor([0.0, 0.0, 0.0, 1])))
        self.assertTrue(is_close(roma.rotvec_composition((rotvec_list[0], roma.rotvec_inverse(rotvec_list[0]))), torch.zeros(3)))
        
if __name__ == "__main__":
    unittest.main()
    