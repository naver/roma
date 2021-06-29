# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.
"""
Various utility functions related to rotation representations.
"""

import torch
import numpy as np
import roma.internal
import roma.mappings

def is_orthonormal_matrix(R, epsilon=1e-7):
    """
    Test if matrices are orthonormal.

    Args:
        R (...xDxD tensor): batch of square matrices.
        epsilon: tolerance threshold.
    Returns:
        boolean tensor (shape ...).

    """
    R, batch_shape = roma.internal.flatten_batch_dims(R, end_dim=-3)
    assert (R.dim() == 3), "Input should be a BxDxD batch of matrices."
    B, D, D1 = R.shape
    assert D == D1, "Input should be a BxDxD batch of matrices."
    errors = torch.norm(R @ R.transpose(-1, -2) - torch.eye(D, device=R.device, dtype=R.dtype), dim=[-2,-1])
    return torch.all(errors < epsilon)
    
def is_rotation_matrix(R, epsilon=1e-7):
    """
    Test if matrices are rotation matrices.

    Args:
        R (...xDxD tensor): batch of square matrices.
        epsilon: tolerance threshold.
    Returns:
        boolean tensor (shape ...).
    """
    if not is_orthonormal_matrix(R, epsilon):
        return False
    return torch.all(torch.det(R) > 0)

def random_unitquat(size = tuple(), device=None):
    """
    Generates a batch of random unit quaternions, uniformly sampled according to the usual quaternion metric.

    Args:
        size (tuple or int): batch size. Use for example ``tuple()`` to generate a single element, and ``(5,2)`` to generate a 5x2 batch.
    Returns:
        batch of unit quaternions (size x 4 tensor).
    """
    if type(size) == int:
        size = (size,)
    quat = torch.randn(size + (4,), device=device)
    quat /= torch.norm(quat, dim=-1, keepdim=True)
    assert(torch.all( torch.abs(torch.norm(quat, dim=-1) - 1) < 1e-3 ))
    return quat

def random_rotmat(size  = tuple(), device=None):
    """
    Generates a batch of random 3x3 rotation matrices, uniformly sampled according to the usual rotation metric.

    Args:
        size (tuple or int): batch size. Use for example ``tuple()`` to generate a single element, and ``(5,2)`` to generate a 5x2 batch.
    Returns:
        batch of rotation matrices (size x 3x3 tensor).
    """
    quat = random_unitquat(size, device)
    R = roma.mappings.unitquat_to_rotmat(quat)
    return R

def random_rotvec(size = tuple(), device=None):
    """
    Generates a batch of random rotation vectors, uniformly sampled according to the usual rotation metric.

    Args:
        size (tuple or int): batch size. Use for example ``tuple()`` to generate a single element, and ``(5,2)`` to generate a 5x2 batch.
    Returns:
        batch of rotation vectors (size x 3 tensor).
    """
    quat = random_unitquat(size, device)
    return roma.mappings.unitquat_to_rotvec(quat)

def rotmat_cosine_angle(R):
    """
    Returns the cosine angle of the input 3x3 rotation matrix R.
    Based on the equality :math:`Trace(R) = 1 + 2 cos(alpha)`.

    Args:
        R (...x3x3 tensor): batch of 3w3 rotation matrices.
    Returns:
        batch of cosine angles (... tensor).
    """
    assert R.shape[-2:] == (3,3), "Expecting a ...x3x3 batch of rotation matrices"
    return  0.5 * (R[...,0,0] + R[...,1,1] + R[...,2,2] - 1.0)


_ONE_OVER_2SQRT2 = 1.0 / (2 * np.sqrt(2))
def rotmat_geodesic_distance(R1, R2):
    """
    Returns the angular distance alpha between a pair of rotation matrices.
    Based on the equality :math:`|R_2 - R_1|_F = 2 \sqrt{2} sin(alpha/2)`.

    Args:
        R1, R2 (...x3x3 tensor): batch of 3x3 rotation matrices.
    Returns:
        batch of angles in radian (... tensor).
    """
    return 2.0 * torch.asin(torch.clamp_max(torch.norm(R2 - R1, dim=[-1, -2]) * _ONE_OVER_2SQRT2, 1.0))

def rotmat_geodesic_distance_naive(R1, R2):
    """
    Returns the angular distance between a pair of rotation matrices.
    Based on :func:`~rotmat_cosine_angle` and less precise than :func:`~roma.utils.rotmat_geodesic_distance` for nearby rotations.
    
    Args:
        R1, R2 (...x3x3 tensor): batch of 3x3 rotation matrices.
    Returns:
        batch of angles in radian (... tensor).
    """
    R = R1.transpose(-1,-2) @ R2
    cos = rotmat_cosine_angle(R)
    return torch.acos(torch.clamp(cos, -1.0, 1.0))

def quat_conjugation(quat):
    """
    Returns the conjugation of input batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    Note:
        Conjugation of a unit quaternion is equal to its inverse.        
    """
    inv = quat.clone()
    inv[...,:3] *= -1
    return inv

def quat_inverse(quat):
    """
    Returns the inverse of a batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    Note:
        - Inverse of null quaternion is undefined.
        - For unit quaternions, consider using conjugation instead.        
    """
    return quat_conjugation(quat) / torch.sum(quat**2, dim=-1, keepdim=True)

def quat_product(p, q):
    """
    Returns the product of two quaternions.

    Args:
        p, q (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L153
    batch_shape = p.shape[:-1]
    assert q.shape[:-1] == batch_shape
    p = p.reshape(-1, 4)
    q = q.reshape(-1, 4)
    product = torch.empty_like(q)
    product[..., 3] = p[..., 3] * q[..., 3] - torch.sum(p[..., :3] * q[..., :3], axis=-1)
    product[..., :3] = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
                      torch.cross(p[..., :3], q[..., :3], dim=-1))
    return product.reshape(*batch_shape, 4)

def quat_composition(sequence, normalize = False):
    """
    Returns the product of a sequence of quaternions.

    Args:
        sequence (sequence of ...x4 tensors, XYZW convention): sequence of batches of quaternions.
        normalize (bool): it True, normalize the returned quaternion.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    assert len(sequence) > 1, "Requiring at least two inputs."
    res = sequence[0]
    for q in sequence[1:]:
        res = quat_product(res, q)
    if normalize:
        q = q / torch.norm(q, dim=-1, keepdim=True)
    return res

def rotvec_inverse(rotvec):
    """
    Returns the inverse of the input rotation expressed using rotation vector representation.

    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
    Returns:
        batch of rotation vectors (...x3 tensor).
    """
    return -rotvec

def rotvec_composition(sequence, normalize = False):
    """
    Returns a rotation vector corresponding to the composition of a sequence of rotations represented by rotation vectors.
    Composition is performed using an intermediary quaternion representation.

    Args:
        sequence (sequence of ...x3 tensors): sequence of batches of rotation vectors.    
        normalize (bool): if True, normalize intermediary representation to compensate for numerical errors.
    """
    assert len(sequence) > 1, "Requiring at least two inputs."
    quats = [roma.rotvec_to_unitquat(rotvec) for rotvec in sequence]
    q = quat_composition(quats, normalize=normalize)
    return roma.unitquat_to_rotvec(q)

def rotmat_inverse(R):
    """
    Returns the inverse of a rotation matrix.

    Args:
        R (...xNxN tensor): batch of rotation matrices.
    Returns:
        batch of inverted rotation matrices (...xNxN tensor).
    Warning:
        The function returns a transposed view of the input, therefore one should be careful with in-place operations.
    """
    return R.transpose(-1, -2)

def rotmat_composition(sequence, normalize = False):
    """
    Returns the product of a sequence of rotation matrices.

    Args:
        sequence (sequence of ...xNxN tensors): sequence of batches of rotation matrices.  
        normalize: if True, apply special Procrustes orthonormalization to compensate for numerical errors.
    Returns:
        batch of rotation matrices (...xNxN tensor).
    """    
    assert len(sequence) > 1, "Requiring at least two inputs."
    result = sequence[0]
    for R in sequence[1:]:
        result = result @ R
    if normalize:
        result = roma.mappings.special_procrustes(result)
    return result

def unitquat_slerp(q0, q1, steps):
    """
    Spherical linear interpolation between two unit quaternions.
    
    Args: 
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape B): interpolation steps, 0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).
    Returns: 
        batch of interpolated quaternions (BxAx4 tensor).
    Note:
        When considering quaternions as rotation representations,
        one should keep in mind that interpolation is not necessarily performed along the shortest arc,
        depending on the sign of ``torch.sum(q0*q1,dim=-1)``.
    """
    # Relative rotation
    rel_q = quat_product(quat_conjugation(q0), q1)
    rel_rotvec = roma.mappings.unitquat_to_rotvec(rel_q)
    # Relative rotations to apply
    rel_rotvecs = steps.reshape(steps.shape + (1,) * rel_rotvec.dim()) * rel_rotvec.reshape((1,) * steps.dim() + rel_rotvec.shape)
    rots = roma.mappings.rotvec_to_unitquat(rel_rotvecs.reshape(-1, 3)).reshape(*rel_rotvecs.shape[:-1], 4)
    interpolated_q = quat_product(q0.reshape((1,) * steps.dim() + q0.shape).repeat(steps.shape + (1,) * q0.dim()), rots)
    return interpolated_q
    
def rotvec_slerp(rotvec0, rotvec1, steps):
    """
    Spherical linear interpolation between two rotation vector representations.

    Args:
        rotvec0, rotvec1 (Ax3 tensor): batch of rotation vectors (A may contain multiple dimensions).
        steps (tensor of shape B):  interpolation steps, 0.0 corresponding to rotvec0 and 1.0 to rotvec1 (B may contain multiple dimensions).
    Returns: 
        batch of interpolated rotation vectors (BxAx3 tensor).
    """
    q0 = roma.mappings.rotvec_to_unitquat(rotvec0)
    q1 = roma.mappings.rotvec_to_unitquat(rotvec1)
    # Flip some quaternions to ensure the shortest path interpolation
    q1 = -torch.sign(torch.sum(q0*q1, dim=-1, keepdim=True)) * q1
    interpolated_q = unitquat_slerp(q0, q1, steps)
    interpolated_rotvec = roma.mappings.unitquat_to_rotvec(interpolated_q)
    return interpolated_rotvec.reshape(steps.shape + rotvec0.shape)