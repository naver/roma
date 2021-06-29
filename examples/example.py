# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.

import torch
import roma

# Arbitrary number of batch dimensions are supported, including none (batch_shape = tuple()).
batch_shape = (2, 3)

# Conversion between rotation representations
rotvec = torch.randn(batch_shape + (3,))
q = roma.rotvec_to_unitquat(rotvec)
R = roma.unitquat_to_rotmat(q)
Rbis = roma.rotvec_to_rotmat(rotvec)

# Regression of a rotation from an arbitrary input
# Special Procrustes orthonormalization of a 3x3 matrix
R1 = roma.special_procrustes(torch.randn(batch_shape + (3, 3)))
# from a 6D representation
R2 = roma.special_gramschmidt(torch.randn(batch_shape + (3, 2)))
# From coefficients of a 4x4 symmetric matrix.
q = roma.symmatrixvec_to_unitquat(torch.randn(batch_shape + (10,)))

# Rotation space metrics
R1, R2 = roma.random_rotmat(size=5), roma.random_rotmat(size=5)
theta = roma.utils.rotmat_geodesic_distance(R1, R2)
cos_theta = roma.utils.rotmat_cosine_angle(R1.transpose(-2, -1) @ R2)

# Operations on quaternions
q_identity = roma.quat_product(roma.quat_conjugation(q), q)

# Spherical interpolation:
# Between unit quaternions
q0, q1 = roma.random_unitquat(10), roma.random_unitquat(10)
steps = torch.linspace(0, 1.0, 5)
q_interpolated = roma.utils.unitquat_slerp(q0, q1, steps)
# Print interpolations for an arbitrary element of the batch
idx = 1
print('q0:\n', q0[idx])
print('q1:\n', q1[idx])
print('q_interpolated:\n', q_interpolated[:,idx])

# Spherical interpolation between rotation vectors (shortest path)
rotvec0, rotvec1 = torch.randn(batch_shape + (3,)), torch.randn(batch_shape + (3,))
rotvec_interpolated = roma.rotvec_slerp(rotvec0, rotvec1, steps)