# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import unittest
import torch
import roma
import numpy as np
from test.utils import is_close
import itertools

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

class TestEuler(unittest.TestCase):
    def test_euler(self):
        batch_shape = torch.Size((3,2))
        x = torch.randn(batch_shape)
        y = torch.randn(batch_shape)
        q = roma.euler_to_unitquat('xy', (x, y))
        self.assertTrue(q.shape == batch_shape + (4,))

    def test_euler_unitquat_consistency(self):
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float64
        for degrees in (True, False):
            for batch_shape in [tuple(),
                                torch.Size((30,)),
                                torch.Size((50,60))]:
                for intrinsics in (True, False):
                    for convention in ["".join(permutation) for permutation in itertools.permutations('xyz')] + ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]:
                        if intrinsics:
                            convention = convention.upper()
                        q = roma.random_unitquat(batch_shape, device=device, dtype=dtype)
                        angles = roma.unitquat_to_euler(convention, q, degrees=degrees, as_tuple=True)
                        self.assertTrue(len(angles) == 3)
                        self.assertTrue(all([angle.shape == batch_shape for angle in angles]))
                        if degrees:
                            self.assertTrue(all([torch.all(angle > -180.) and torch.all(angle <= 180) for angle in angles]))
                        else:
                            self.assertTrue(all([torch.all(angle > -np.pi) and torch.all(angle <= np.pi) for angle in angles]))
                        q1 = roma.euler_to_unitquat(convention, angles, degrees=degrees)
                        self.assertTrue(torch.all(roma.unitquat_geodesic_distance(q, q1) < 1e-6))

    def test_euler_rotvec_consistency(self):
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float64
        for degrees in (True, False):
            for batch_shape in [tuple(),
                                torch.Size((30,)),
                                torch.Size((50,60))]:
                for intrinsics in (True, False):
                    for convention in ["".join(permutation) for permutation in itertools.permutations('xyz')] + ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]:
                        if intrinsics:
                            convention = convention.upper()
                        q = roma.random_rotvec(batch_shape, device=device, dtype=dtype)
                        angles = roma.rotvec_to_euler(convention, q, degrees=degrees, as_tuple=True)
                        self.assertTrue(len(angles) == 3)
                        self.assertTrue(all([angle.shape == batch_shape for angle in angles]))
                        if degrees:
                            self.assertTrue(all([torch.all(angle > -180.) and torch.all(angle <= 180) for angle in angles]))
                        else:
                            self.assertTrue(all([torch.all(angle > -np.pi) and torch.all(angle <= np.pi) for angle in angles]))
                        q1 = roma.euler_to_rotvec(convention, angles, degrees=degrees)
                        self.assertTrue(torch.all(roma.rotvec_geodesic_distance(q, q1) < 1e-6))

    def test_euler_rotmat_consistency(self):
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float64
        for degrees in (True, False):
            for batch_shape in [tuple(),
                                torch.Size((30,)),
                                torch.Size((50,60))]:
                for intrinsics in (True, False):
                    for convention in ["".join(permutation) for permutation in itertools.permutations('xyz')] + ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]:
                        if intrinsics:
                            convention = convention.upper()
                        q = roma.random_rotmat(batch_shape, device=device, dtype=dtype)
                        angles = roma.rotmat_to_euler(convention, q, degrees=degrees, as_tuple=True)
                        self.assertTrue(len(angles) == 3)
                        self.assertTrue(all([angle.shape == batch_shape for angle in angles]))
                        if degrees:
                            self.assertTrue(all([torch.all(angle > -180.) and torch.all(angle <= 180) for angle in angles]))
                        else:
                            self.assertTrue(all([torch.all(angle > -np.pi) and torch.all(angle <= np.pi) for angle in angles]))
                        q1 = roma.euler_to_rotmat(convention, angles, degrees=degrees)
                        self.assertTrue(torch.all(roma.rotmat_geodesic_distance(q, q1) < 1e-6))

    def test_euler_backward(self):
        for intrinsics in (True, False):
            for convention in ["".join(permutation) for permutation in itertools.permutations('xyz')] + ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]:
                if intrinsics:
                    convention = convention.upper()
                    rotvec = torch.randn(3, requires_grad=True)
                    angles = roma.rotvec_to_euler('xyz', rotvec)
                    sum(angles).backward()

    def test_euler_tensor(self):
        r"""
        Test that Euler conversion methods support both list and tensor inputs.
        """
        batch_shape = (10,34)
        dtype = torch.float64
        q = roma.random_unitquat(batch_shape, device=device, dtype=dtype)
        convention = 'xyz'
        angles = roma.unitquat_to_euler(convention, q, as_tuple=True)
        angles_tensor = roma.unitquat_to_euler(convention, q)
        assert type(angles) == tuple
        assert type(angles_tensor) == torch.Tensor
        q1 = roma.euler_to_unitquat(convention, angles)
        q2 = roma.euler_to_unitquat(convention, angles_tensor)
        self.assertTrue(torch.all(roma.rotmat_geodesic_distance(q1, q2) < 1e-6))


if __name__ == "__main__":
    unittest.main()