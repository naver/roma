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
        dtype = torch.float64
        batch_shape = (10,)
        x = torch.randn(batch_shape + (3,), dtype=dtype)
        R = roma.random_rotmat(batch_shape, dtype=dtype)
        t = torch.randn(batch_shape + (3,), dtype=dtype) 

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

    def test_affine_homogeneous_cast(self):
        batch_shape = (10,)
        for D in range(2,5):
            for Type in (Affine, Isometry, Rigid):
                transformation = Type(torch.randn(batch_shape + (D,D)), torch.randn(batch_shape + (D,))).normalize()
                self.assertTrue(type(transformation) == Type)

                homogeneous = transformation.to_homogeneous()
                self.assertTrue(homogeneous.shape == batch_shape + (D+1, D+1))
                self.assertTrue(torch.all(homogeneous[...,:D,:D] == transformation.linear))
                self.assertTrue(torch.all(homogeneous[...,:D,D] == transformation.translation))
                self.assertTrue(torch.all(homogeneous[...,D,:D] == 0.0))
                self.assertTrue(torch.all(homogeneous[...,D,D] == 1))

                transformation2 = Type.from_homogeneous(homogeneous)
                self.assertTrue(type(transformation2) == Type)
                self.assertTrue(torch.all(transformation.linear == transformation2.linear))
                self.assertTrue(torch.all(transformation.translation == transformation2.translation))

                # Re-use of an existing buffer
                homogeneous_bis = transformation.to_homogeneous(homogeneous)
                self.assertTrue(homogeneous_bis is homogeneous)

    def test_unitquat_homogeneous_cast(self):
        batch_shape = (10,)
        D = 3
        dtype = torch.float64
        Type = RigidUnitQuat
        transformation = Type(torch.randn(batch_shape + (4,), dtype=dtype), torch.randn(batch_shape + (D,), dtype=dtype)).normalize()
        self.assertTrue(type(transformation) == Type)

        homogeneous = transformation.to_homogeneous()
        self.assertTrue(homogeneous.shape == batch_shape + (D+1, D+1))
        self.assertTrue(roma.is_rotation_matrix(homogeneous[...,:D,:D]))
        self.assertTrue(torch.all(roma.unitquat_geodesic_distance(roma.rotmat_to_unitquat(homogeneous[...,:D,:D]), transformation.linear) < 1e-6))
        self.assertTrue(torch.all(homogeneous[...,:D,D] == transformation.translation))
        self.assertTrue(torch.all(homogeneous[...,D,:D] == 0.0))
        self.assertTrue(torch.all(homogeneous[...,D,D] == 1))

        transformation2 = Type.from_homogeneous(homogeneous)
        self.assertTrue(type(transformation2) == Type)
        self.assertTrue(torch.all(roma.unitquat_geodesic_distance(transformation.linear, transformation2.linear) < 1e-6))
        self.assertTrue(torch.all(transformation.translation == transformation2.translation))

        # Re-use of an existing buffer
        homogeneous_bis = transformation.to_homogeneous(homogeneous)
        self.assertTrue(homogeneous_bis is homogeneous)

    def test_orthonormalization(self):
        batch_shape = (10,4)
        D = 5
        dtype = torch.float64

        for _ in range(10):
            raw = torch.randn(batch_shape + (D,D), dtype=dtype)

            ortho1 = Orthonormal(raw).normalize()
            ortho2 = ortho1.normalize()

            self.assertTrue(roma.is_orthonormal_matrix(ortho1.linear))
            self.assertTrue(roma.is_orthonormal_matrix(ortho2.linear))
            self.assertTrue(torch.all(torch.isclose(ortho1.linear, ortho2.linear)))

            translation = torch.randn(batch_shape + (D,), dtype=dtype)
            iso = Isometry(raw, translation).normalize()
            self.assertTrue(torch.all(torch.isclose(iso.linear, ortho1.linear)))
            self.assertTrue(torch.all(torch.isclose(iso.translation, translation)))

    def test_rotation(self):
        batch_shape = (10,4)
        D = 5
        dtype = torch.float64

        for _ in range(10):
            raw = torch.randn(batch_shape + (D,D), dtype=dtype)
            translation = torch.randn(batch_shape + (D,), dtype=dtype)

            rot1 = Rotation(raw).normalize()
            rot2 = rot1.normalize()

            self.assertTrue(roma.is_rotation_matrix(rot1.linear))
            self.assertTrue(roma.is_rotation_matrix(rot2.linear))
            self.assertTrue(torch.all(torch.isclose(rot1.linear, rot2.linear)))

            rigid = Rigid(raw, translation).normalize()
            self.assertTrue(torch.all(torch.isclose(rigid.linear, rot1.linear)))
            self.assertTrue(torch.all(torch.isclose(rigid.translation, translation)))


    def test_rotation_unit_quat(self):
        batch_shape = (10,4)
        D = 3
        dtype = torch.float64

        for _ in range(10):
            raw = torch.randn(batch_shape + (D,D), dtype=dtype)
            translation = torch.randn(batch_shape + (D,), dtype=dtype)

            rot1 = Rotation(raw).normalize()
            quat = RotationUnitQuat(roma.rotmat_to_unitquat(rot1.linear))
            quat1 = quat.normalize()
            self.assertTrue(torch.all(torch.isclose(roma.unitquat_to_rotmat(quat1.linear), rot1.linear)))
            self.assertTrue(torch.all(torch.isclose(quat1.linear, quat.linear)))

            rigidq = RigidUnitQuat(roma.rotmat_to_unitquat(rot1.linear), translation)
            rigidq1 = rigidq.normalize()
            self.assertTrue(torch.all(torch.isclose(roma.unitquat_to_rotmat(rigidq1.linear), rot1.linear)))
            self.assertTrue(torch.all(torch.isclose(rigidq1.translation, translation)))
            self.assertTrue(torch.all(torch.isclose(rigidq.linear, rigidq1.linear)))