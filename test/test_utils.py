# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import unittest
import torch
import numpy as np
import roma
import roma.internal
from test.utils import is_close

class TestUtils(unittest.TestCase):
    def test_flatten(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.randn((32, 24, 3, 4), dtype=dtype)
            xflat, batch_shape = roma.internal.flatten_batch_dims(x, end_dim=-3)
            self.assertEqual(batch_shape, (32, 24))
            self.assertTrue(xflat.shape == (32*24, 3, 4))
            xbis = roma.internal.unflatten_batch_dims(xflat, batch_shape)
            self.assertTrue(torch.all(xbis == x))

    def test_rotmat_geodesic_distance(self):
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            axis = torch.nn.functional.normalize(torch.randn((batch_size,3), dtype=dtype), dim=-1)
            alpha = (np.pi - 1e-5) * 2 * (torch.rand(batch_size, dtype=dtype)-0.5)
            x = alpha[:,None] * axis
            R = roma.rotvec_to_rotmat(x)
            
            cosine = roma.rotmat_cosine_angle(R)
            self.assertTrue(is_close(cosine, torch.cos(alpha)))
            
            I = torch.eye(3, dtype=dtype)
            M = roma.random_rotmat(batch_size, dtype=dtype)
            
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

    def test_other_geodesic_distance(self):
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            q1 = roma.random_unitquat(batch_size, dtype=dtype)
            q2 = roma.random_unitquat(batch_size, dtype=dtype)
            alpha_q = roma.unitquat_geodesic_distance(q1, q2)
            # Ensure consistency between functions
            R1 = roma.unitquat_to_rotmat(q1)
            R2 = roma.unitquat_to_rotmat(q2)
            alpha_R = roma.rotmat_geodesic_distance(R1, R2)
            self.assertTrue(is_close(alpha_q, alpha_R))
            rotvec1 = roma.unitquat_to_rotvec(q1)
            rotvec2 = roma.unitquat_to_rotvec(q2)
            alpha_rotvec = roma.rotvec_geodesic_distance(rotvec1, rotvec2)
            self.assertTrue(is_close(alpha_rotvec, alpha_q))

    def test_identity_quat(self):
        q = roma.identity_quat()
        self.assertTrue(q.shape == (4,))
        self.assertTrue(is_close(q, roma.quat_inverse(q)))

        q = roma.identity_quat(5)
        self.assertTrue(q.shape == (5,4))
        self.assertTrue(is_close(q, roma.quat_inverse(q)))

        q = roma.identity_quat((3,2))
        self.assertTrue(q.shape == (3,2,4))
        self.assertTrue(is_close(q, roma.quat_inverse(q)))

    def test_random_unitquat(self):
        q = roma.random_unitquat((3,5))
        self.assertTrue(q.shape == (3,5,4))

        batch_size = 100000
        repetitions = 10
        torch.manual_seed(565441)
        for _ in range(10):
            for dtype in (torch.float32, torch.float64):
                # Sample multiple times a batch of random quaternions
                q = torch.cat([roma.random_unitquat(batch_size, dtype=dtype) for _ in range(repetitions)], dim=0)
                size = repetitions * batch_size
                self.assertTrue(q.shape == (size, 4))

                # Ensure that the generated quaternions are of unit norm
                self.assertTrue(torch.all(torch.abs(torch.norm(q, dim=-1) - 1.0) < 1e-6))

                # If the distribution is uniform, the mean quaternion should be close to 0
                # (we use a constant random seed for this test, otherwise it could fail in some unlikely cases).
                mean = torch.mean(q, dim=0)
                self.assertTrue(torch.norm(mean) < 10.0 / np.sqrt(size))

                # There should be approximately as many quaternions on both sides of a random hyperplane going through 0.
                for _ in range(5):
                    axis = torch.randn((size, 4), dtype=dtype)
                    axis /= torch.norm(axis, dim=-1, keepdim=True)
                    count_difference = torch.sum(torch.sign(torch.sum(axis * q, dim=-1)))
                    self.assertTrue(torch.abs(count_difference) < 10.0 * np.sqrt(size))

    def test_unitquat(self):
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            q = roma.random_unitquat(batch_size, dtype=dtype)
            # Ensure that conjugation of a unit quaternion equates to its inverse.
            iq = roma.quat_conjugation(q)
            q_id = roma.rotvec_to_unitquat(torch.zeros(1,3))
            self.assertTrue(is_close(q_id, roma.quat_product(q, iq)))
            self.assertTrue(is_close(q_id, roma.quat_product(iq, q)))

    def test_quat(self):
        batch_size = 100
        for dtype in (torch.float32, torch.float64):
            q = roma.random_unitquat(batch_size, dtype=dtype)
            # Test quaternion inverse (hoping for non zero quaternions during the test).
            iq = roma.quat_inverse(q)
            q_id = roma.rotvec_to_unitquat(torch.zeros(1,3))
            self.assertTrue(is_close(q_id, roma.quat_product(q, iq)))
            self.assertTrue(is_close(q_id, roma.quat_product(iq, q)))
            nq = roma.quat_normalize(q)
            self.assertTrue(is_close(roma.internal.norm(nq, dim=-1), torch.ones(batch_size, dtype=dtype)))

    def test_quat_action(self):
        batch_size=100
        for dtype in (torch.float32, torch.float64):
            q = torch.randn(batch_size, 4)
            v = torch.randn(batch_size, 3)
            vrot0 = roma.quat_action(q, v)

            # Check that results are consistent with respect to quaternion normalization
            q_normalized = q / torch.norm(q, dim=-1, keepdim=True)
            vrot1 = roma.quat_action(q_normalized, v, is_normalized=True)
            vrot2 = roma.quat_action(q_normalized, v, is_normalized=False)

            self.assertTrue(is_close(vrot0, vrot1))
            self.assertTrue(is_close(vrot1, vrot2))

            # Check that results are consistent with rotation using a rotation matrix
            R = roma.unitquat_to_rotmat(q_normalized)
            vrotR = torch.einsum('bik, bk -> bi', R, v)
            self.assertTrue(is_close(vrot1, vrotR))
        
    def test_slerp(self):
        for dtype in (torch.float32, torch.float64):
            for batch_shape in [(100,), (24, 32), (3, 4, 7)]:
                # Identity
                x0 = torch.zeros(1, 3, dtype=dtype)
                # 'Small' rotation vector
                axis = torch.nn.functional.normalize(torch.randn(batch_shape + (3,), dtype=dtype), dim=-1)
                alpha = (np.pi-1e-10) * 2 * (torch.rand(batch_shape, dtype=dtype)-0.5)
                x1 = alpha[...,None] * axis
                
                # Rotvec slerp
                steps = torch.linspace(0, 1, 11, dtype=dtype)
                x_true = steps.reshape(-1, *([1] * (len(batch_shape) + 1))) * x1.unsqueeze(0)
                x_pred = roma.rotvec_slerp(x0.expand(batch_shape + (3,)), x1, steps)
                self.assertTrue(is_close(x_true, x_pred))
                
                # Check left invariance of unit quat slerp
                q0 = roma.random_unitquat(batch_shape, dtype=dtype)
                q1 = roma.random_unitquat(batch_shape, dtype=dtype)
                q = roma.random_unitquat(batch_shape, dtype=dtype)
                iq = roma.quat_conjugation(q)
                slerp1 = roma.unitquat_slerp(q0, q1, steps)
                slerp2 = roma.quat_product(iq.expand(len(steps), *batch_shape, 4),
                                        roma.unitquat_slerp(roma.quat_product(q, q0),
                                                                roma.quat_product(q, q1),
                                                                steps))
                self.assertTrue(is_close(slerp1, slerp2))

                # Check consistency between rotmat_slerp and unitquat_slerp
                R0 = roma.unitquat_to_rotmat(q0)
                R1 = roma.unitquat_to_rotmat(q1)
                R_slerp = roma.rotmat_slerp(R0, R1, steps)
                self.assertTrue(is_close(roma.unitquat_to_rotmat(slerp1), R_slerp))

                # Check that the first and last element of quaternion slerp are consistent with the inputs
                q = roma.unitquat_slerp(q0, q1, steps, shortest_arc=False)
                self.assertTrue(is_close(q[0], q0))
                self.assertTrue(is_close(q[-1], q1))

                # Similar test when shortest_arc==True
                q = roma.unitquat_slerp(q0, q1, steps, shortest_arc=True)
                self.assertTrue(is_close(q[0], q0))
                self.assertTrue(torch.all(torch.min(torch.norm(q[-1] - q1, dim=-1), torch.norm(q[-1] + q1, dim=-1)) < 5e-6))

    def test_unitquat_slerp(self):
        # Test of slerp on a specific example,
        # where enforcing -- or not -- interpolation along the shortest arc give different results.
        q0 = torch.as_tensor([1,0,0,0], dtype=torch.float)
        q1 = -torch.as_tensor([1,1,0,0], dtype=torch.float)
        q1 /= torch.norm(q1, keepdim=True)
        steps = torch.linspace(0, 1.0, 3)

        q05 = q0 + q1
        q05 /= torch.norm(q05, keepdim=True)
        q = roma.utils.unitquat_slerp(q0, q1, steps, shortest_arc=False)
        self.assertTrue(is_close(q, torch.stack((q0, q05, q1))))

        # Interpolation along the shortest arc
        q05m = q0 - q1
        q05m /= torch.norm(q05m, keepdim=True)
        q = roma.utils.unitquat_slerp(q0, q1, steps, shortest_arc=True)
        self.assertTrue(is_close(q, torch.stack((q0, q05m, -q1))))

    def test_slerp_consistency(self):
        # Test consistency between both slerp methods.
        for batch_shape in [(3,), (10, 20), (3, 14, 7)]:
            for steps_shape in [(6,), (4,2,), (3,2,1)]:
                for shortest_arc in (True, False):
                    q0 = roma.random_unitquat(batch_shape)
                    q1 = roma.random_unitquat(batch_shape)
                    steps = torch.rand(steps_shape)
                    q = roma.unitquat_slerp(q0, q1, steps=steps, shortest_arc=shortest_arc)
                    qbis = roma.unitquat_slerp_fast(q0, q1, steps=steps, shortest_arc=shortest_arc)
                    self.assertTrue(q.shape == steps_shape + batch_shape + (4,))
                    self.assertTrue(is_close(q, qbis))
                    # Same tests with nearby rotations
                    q1 = q0 + 1e-3 * torch.randn_like(q0)
                    q1 /= torch.norm(q1, dim=-1, keepdim=True)
                    q = roma.unitquat_slerp(q0, q1, steps=steps, shortest_arc=shortest_arc)
                    qbis = roma.unitquat_slerp_fast(q0, q1, steps=steps, shortest_arc=shortest_arc)
                    self.assertTrue(q.shape == steps_shape + batch_shape + (4,))
                    self.assertTrue(is_close(q, qbis))            

    def test_composition(self):
        for dtype in (torch.float32, torch.float64):
            R_list = [roma.random_rotmat(dtype=dtype) for _ in range(3)]
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

    def test_rigid_vectors_registration(self):
        batch_shape = (34, 16)
        n = 100
        for dtype in (torch.float32, torch.float64):
            for weights in [None, torch.rand(size=batch_shape + (n,), dtype=dtype)]:
                R_true = roma.random_rotmat(batch_shape, dtype=dtype)
                X = torch.randn(batch_shape + (n, 3,), dtype=dtype)
                Y = torch.einsum('...ik, ...jk -> ...ji', R_true, X)
                R = roma.rigid_vectors_registration(X, Y, weights)
                self.assertTrue(is_close(R, R_true))

    def test_rigid_vectors_registration_with_scale(self):
        batch_shape = (34, 16)
        n = 100
        for dtype in (torch.float32, torch.float64):
            for weights in [None, torch.rand(size=batch_shape + (n,), dtype=dtype)]:
                R_true = roma.random_rotmat(batch_shape, dtype=dtype)
                scale_true = torch.exp(torch.randn(size=batch_shape, dtype=dtype))
                X = torch.randn(batch_shape + (n, 3,), dtype=dtype)
                Y = scale_true[...,None, None] * torch.einsum('...ik, ...jk -> ...ji', R_true, X)
                R, scale = roma.rigid_vectors_registration(X, Y, weights, compute_scaling=True)
                self.assertTrue(is_close(R, R_true))
                self.assertTrue(is_close(scale, scale_true))

    def test_rigid_point_registration(self):
        batch_shape = (34, 16)
        n = 100
        for dtype in (torch.float32, torch.float64):
            for weights in [None, torch.rand(size=batch_shape + (n,), dtype=dtype)]:
                R_true = roma.random_rotmat(batch_shape, dtype=dtype)
                t_true = torch.randn(batch_shape + (3,), dtype=dtype)
                X = torch.randn(batch_shape + (n, 3,), dtype=dtype)
                Y = torch.einsum('...ik, ...jk -> ...ji', R_true, X) + t_true.unsqueeze(-2)
                R, t = roma.rigid_points_registration(X, Y, weights)

                self.assertTrue(is_close(R, R_true))
                self.assertTrue(is_close(t, t_true))

    def test_rigid_points_registration_with_scale(self):
        batch_shape = (34, 16)
        n = 100
        for dtype in (torch.float32, torch.float64):
            for weights in [None, torch.rand(size=batch_shape + (n,), dtype=dtype)]:
                R_true = roma.random_rotmat(batch_shape, dtype=dtype)
                t_true = torch.randn(batch_shape + (3,), dtype=dtype)
                scale_true = torch.exp(torch.randn(size=batch_shape, dtype=dtype))
                X = torch.randn(batch_shape + (n, 3,), dtype=dtype)
                Y = scale_true[...,None, None] * torch.einsum('...ik, ...jk -> ...ji', R_true, X) + t_true.unsqueeze(-2)
                R, t, scale = roma.rigid_points_registration(X, Y, weights, compute_scaling=True)
                self.assertTrue(is_close(R, R_true))
                self.assertTrue(is_close(t, t_true))
                self.assertTrue(is_close(scale, scale_true))
        
if __name__ == "__main__":
    unittest.main()
    