import torch
import roma
from roma.transforms import *
import unittest

class TestTransforms(unittest.TestCase):
    def test_linear_apply(self):
        batch_shape = (10,)
        dtype = torch.float64
        x = torch.randn(batch_shape + (3,), dtype=dtype)
        R = roma.random_rotmat(batch_shape, dtype=dtype)
        t = torch.randn(batch_shape + (3,), dtype=dtype)

        for transform in (Linear, Orthonormal, Rotation, RotationUnitQuat):
            if transform == RotationUnitQuat:
                T = transform(roma.rotmat_to_unitquat(R))
            else:
                T = transform(R)
            Tx = T.apply(x)
            Rx = T.linear_apply(x)
            self.assertTrue(torch.all(torch.isclose(Rx, torch.einsum("...ik, ...k -> ...i", R, x))))
            self.assertTrue(torch.all(torch.isclose(Rx, Tx)))

    def test_apply(self):
        batch_shape = (10,)
        x = torch.randn(batch_shape + (3,))
        R = roma.random_rotmat(batch_shape)
        t = torch.randn(batch_shape + (3,)) 

        for transform in Affine, Isometry, Rigid, RigidUnitQuat:
            if transform == RigidUnitQuat:
                T = transform(roma.rotmat_to_unitquat(R), t)
            else:
                T = transform(R, t)
            Tx = T.apply(x)
            Rx = T.linear_apply(x)
            self.assertTrue(torch.all(torch.isclose(Rx, torch.einsum("...ik, ...k -> ...i", R, x))))
            self.assertTrue(torch.all(torch.isclose(Tx, Rx + t)))

    def test_composition(self):
        dtype = torch.float64
        batch_shape = (10,5)
        x = torch.randn(batch_shape + (3,), dtype=dtype)
        R1 = roma.random_rotmat(batch_shape, dtype=dtype)
        t1 = torch.randn(batch_shape + (3,), dtype=dtype)

        R2 = roma.random_rotmat(batch_shape, dtype=dtype)
        t2 = torch.randn(batch_shape + (3,), dtype=dtype)

        # Linear transformations
        for transform in (Linear, Orthonormal, Rotation, RotationUnitQuat):
            if transform == RotationUnitQuat:
                T1 = transform(roma.rotmat_to_unitquat(R1))
                T2 = transform(roma.rotmat_to_unitquat(R2))
            else:
                T1 = transform(R1)
                T2 = transform(R2)
            T = T1 @ T2
            self.assertTrue(type(T) == type(T1))

            Tx = T.apply(x)
            Rx = T.linear_apply(x)

            Txbis = T1.apply(T2.apply(x))
            Rxbis = T1.linear_apply(T2.linear_apply(x))
            self.assertTrue(torch.all(torch.isclose(Txbis, Tx)))
            self.assertTrue(torch.all(torch.isclose(Rxbis, Rx)))

        # Affine transformations
        for transform in (Affine, Isometry, Rigid, RigidUnitQuat):
            if transform == RigidUnitQuat:
                T1 = transform(roma.rotmat_to_unitquat(R1), t1)
                T2 = transform(roma.rotmat_to_unitquat(R2), t2)
            else:
                T1 = transform(R1, t1)
                T2 = transform(R2, t2)
            T = T1 @ T2
            self.assertTrue(type(T) == type(T1))

            Tx = T.apply(x)
            Rx = T.linear_apply(x)

            Txbis = T1.apply(T2.apply(x))
            Rxbis = T1.linear_apply(T2.linear_apply(x))
            self.assertTrue(torch.all(torch.isclose(Txbis, Tx)))
            self.assertTrue(torch.all(torch.isclose(Rxbis, Rx)))

    def test_inverse(self):
        batch_shape = (10,)
        dtype = torch.float64
        x = torch.randn(batch_shape + (3,), dtype=dtype)
        R = roma.random_rotmat(batch_shape, dtype=dtype)
        t = torch.randn(batch_shape + (3,), dtype=dtype)

        Ridentity = torch.eye(3, dtype=dtype).reshape([1] * len(batch_shape) + [3,3]).repeat(batch_shape + (1,1))
        tidentity = torch.zeros(batch_shape + (3,), dtype=dtype)

        for transform in (Linear, Orthonormal, Rotation, RotationUnitQuat):
            if transform == RotationUnitQuat:
                T = transform(roma.rotmat_to_unitquat(R))
            else:
                T = transform(R)
            Tinv = T.inverse()
            identity1 = Tinv @ T
            identity2 = T @ Tinv

            if transform == RotationUnitQuat:
                self.assertTrue(torch.all(torch.isclose(roma.unitquat_to_rotmat(identity1.linear), Ridentity)))
                self.assertTrue(torch.all(torch.isclose(roma.unitquat_to_rotmat(identity2.linear), Ridentity)))
            else:
                self.assertTrue(torch.all(torch.isclose(identity1.linear, Ridentity)))
                self.assertTrue(torch.all(torch.isclose(identity2.linear, Ridentity)))

        for transform in (Affine, Isometry, Rigid, RigidUnitQuat):
            if transform == RigidUnitQuat:
                T = transform(roma.rotmat_to_unitquat(R), t)
            else:
                T = transform(R, t)
            Tinv = T.inverse()
            identity1 = Tinv @ T
            identity2 = T @ Tinv

            if transform == RigidUnitQuat:
                self.assertTrue(torch.all(torch.isclose(roma.unitquat_to_rotmat(identity1.linear), Ridentity)))
                self.assertTrue(torch.all(torch.isclose(roma.unitquat_to_rotmat(identity2.linear), Ridentity)))
            else:
                self.assertTrue(torch.all(torch.isclose(identity1.linear, Ridentity)))
                self.assertTrue(torch.all(torch.isclose(identity2.linear, Ridentity)))
            self.assertTrue(torch.all(torch.isclose(identity1.translation, tidentity)))
            self.assertTrue(torch.all(torch.isclose(identity2.translation, tidentity)))
