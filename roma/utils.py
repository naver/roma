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

def is_torch_batch_svd_available() -> bool:
    """
    Returns True if the module 'torch_batch_svd' has been loaded. Returns False otherwise.
    """
    return roma.internal._IS_TORCH_BATCH_SVD_AVAILABLE

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

def random_unitquat(size = tuple(), dtype=torch.float, device=None):
    """
    Generates a batch of random unit quaternions, uniformly sampled according to the usual quaternion metric.

    Args:
        size (tuple or int): batch size. Use for example ``tuple()`` to generate a single element, and ``(5,2)`` to generate a 5x2 batch.
    Returns:
        batch of unit quaternions (size x 4 tensor).

    Reference:
        K. Shoemake, “Uniform Random Rotations”, in Graphics Gems III (IBM Version), Elsevier, 1992, pp. 124–132. doi: 10.1016/B978-0-08-050755-2.50036-1.

    """
    if type(size) == int:
        size = (size,)

    x0 = torch.rand(size, dtype=dtype, device=device)
    theta1 = (2.0 * np.pi) * torch.rand(size, dtype=dtype, device=device)
    theta2 = (2.0 * np.pi) * torch.rand(size, dtype=dtype, device=device)
    r1 = torch.sqrt(1.0 - x0)
    r2 = torch.sqrt(x0)
    return torch.stack((r1 * torch.sin(theta1), r1 * torch.cos(theta1), r2 * torch.sin(theta2), r2 * torch.cos(theta2)), dim=-1)

def random_rotmat(size  = tuple(), dtype=torch.float, device=None):
    """
    Generates a batch of random 3x3 rotation matrices, uniformly sampled according to the usual rotation metric.

    Args:
        size (tuple or int): batch size. Use for example ``tuple()`` to generate a single element, and ``(5,2)`` to generate a 5x2 batch.
    Returns:
        batch of rotation matrices (size x 3x3 tensor).
    """
    quat = random_unitquat(size, dtype=dtype, device=device)
    R = roma.mappings.unitquat_to_rotmat(quat)
    return R

def random_rotvec(size = tuple(), dtype=torch.float, device=None):
    """
    Generates a batch of random rotation vectors, uniformly sampled according to the usual rotation metric.

    Args:
        size (tuple or int): batch size. Use for example ``tuple()`` to generate a single element, and ``(5,2)`` to generate a 5x2 batch.
    Returns:
        batch of rotation vectors (size x 3 tensor).
    """
    quat = random_unitquat(size, dtype=dtype, device=device)
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
def rotmat_geodesic_distance(R1, R2, clamping=1.0):
    """
    Returns the angular distance alpha between a pair of rotation matrices.
    Based on the equality :math:`|R_2 - R_1|_F = 2 \sqrt{2} sin(alpha/2)`.

    Args:
        R1, R2 (...x3x3 tensor): batch of 3x3 rotation matrices.
        clamping: clamping value applied to the input of :func:`torch.asin()`.
                Use 1.0 to ensure valid angular distances.
                Use a value strictly smaller than 1.0 to ensure finite gradients.
    Returns:
        batch of angles in radians (... tensor).
    """
    return 2.0 * torch.asin(torch.clamp_max(torch.norm(R2 - R1, dim=[-1, -2]) * _ONE_OVER_2SQRT2, clamping))

def rotmat_geodesic_distance_naive(R1, R2):
    """
    Returns the angular distance between a pair of rotation matrices.
    Based on :func:`~rotmat_cosine_angle` and less precise than :func:`~roma.utils.rotmat_geodesic_distance` for nearby rotations.
    
    Args:
        R1, R2 (...x3x3 tensor): batch of 3x3 rotation matrices.
    Returns:
        batch of angles in radians (... tensor).
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

def quat_action(q, v, is_normalized=False):
    """
    Rotate a 3D vector :math:`v=(x,y,z)` by a rotation represented by a quaternion `q`.

    Based on the action by conjugation :math:`q,v : q v q^{-1}`, considering the pure quaternion :math:`v=xi + yj +zk` by abuse of notation.

    Args:
        q (...x4 tensor, XYZW convention): batch of quaternions.
        v (...x3 tensor): batch of 3D vectors.
        is_normalized: use True if the input quaternions are already normalized, to avoid unnecessary computations.
    Returns:
        batch of rotated 3D vectors (...x3 tensor).
    Note:
        One should favor rotation matrix representation to rotate multiple vectors by the same rotation efficiently.

    """
    batch_shape = v.shape[:-1]
    iquat = quat_conjugation(q) if is_normalized else quat_inverse(q)
    pure = torch.cat((v, torch.zeros(batch_shape + (1,))), dim=-1)
    res = quat_product(q, quat_product(pure, iquat))
    return res[...,:3]

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

def unitquat_slerp(q0, q1, steps, shortest_arc=True):
    """
    Spherical linear interpolation between two unit quaternions.
    
    Args: 
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape B): interpolation steps, 0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.
    Returns: 
        batch of interpolated quaternions (BxAx4 tensor).
    Note:
        When considering quaternions as rotation representations,
        one should keep in mind that spherical interpolation is not necessarily performed along the shortest arc,
        depending on the sign of ``torch.sum(q0*q1,dim=-1)``.

        Behavior is undefined when using ``shortest_arc=False`` with antipodal quaternions.
    """
    # Relative rotation
    rel_q = quat_product(quat_conjugation(q0), q1)
    rel_rotvec = roma.mappings.unitquat_to_rotvec(rel_q, shortest_arc=shortest_arc)
    # Relative rotations to apply
    rel_rotvecs = steps.reshape(steps.shape + (1,) * rel_rotvec.dim()) * rel_rotvec.reshape((1,) * steps.dim() + rel_rotvec.shape)
    rots = roma.mappings.rotvec_to_unitquat(rel_rotvecs.reshape(-1, 3)).reshape(*rel_rotvecs.shape[:-1], 4)
    interpolated_q = quat_product(q0.reshape((1,) * steps.dim() + q0.shape).repeat(steps.shape + (1,) * q0.dim()), rots)
    return interpolated_q

def unitquat_slerp_fast(q0, q1, steps, shortest_arc=True):
    """
    Spherical linear interpolation between two unit quaternions.
    This function requires less computations than :func:`roma.utils.unitquat_slerp`,
    but is **unsuitable for extrapolation (i.e.** ``steps`` **must be within [0,1])**.

    Args: 
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape B): interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.
    Returns: 
        batch of interpolated quaternions (BxAx4 tensor).
    """
    q0, batch_shape = roma.internal.flatten_batch_dims(q0, end_dim=-2)
    q1, batch_shape1 = roma.internal.flatten_batch_dims(q1, end_dim=-2)
    assert batch_shape == batch_shape1
    # omega is the 'angle' between both quaternions
    cos_omega = torch.sum(q0 * q1, dim=-1)
    if shortest_arc:
        # Flip some quaternions to perform shortest arc interpolation.
        q1 = q1.clone()
        q1[cos_omega < 0,:] *= -1
        cos_omega = torch.abs(cos_omega)
    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)

    cos_omega = cos_omega.reshape((1,) * steps.dim() + (-1,1))
    s = steps.reshape(steps.shape + (1,1))
    # General approach    
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1-s)*omega)
    beta = torch.sin(s*omega)
    # Use linear interpolation for nearby quaternions
    alpha[..., nearby_quaternions, :] = 1 - s
    beta[..., nearby_quaternions, :] = s
    # Interpolation
    q = alpha * q0.reshape((1,) * steps.dim() + q0.shape) + beta * q1.reshape((1,) * steps.dim() + q1.shape)
    # Normalization of the output
    q /= torch.norm(q, dim=-1, keepdim=True)
    return q.reshape(steps.shape + batch_shape + (4,))
    
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
    interpolated_q = unitquat_slerp(q0, q1, steps, shortest_arc=True)
    return roma.mappings.unitquat_to_rotvec(interpolated_q)

def rotmat_slerp(R0, R1, steps):
    """
    Spherical linear interpolation between two rotation matrices.

    Args:
        R0, R1 (Ax3x3 tensor): batch of rotation matrices (A may contain multiple dimensions).
        steps (tensor of shape B):  interpolation steps, 0.0 corresponding to R0 and 1.0 to R1 (B may contain multiple dimensions).
    Returns: 
        batch of interpolated rotation matrices (BxAx3x3 tensor).
    """    
    q0 = roma.mappings.rotmat_to_unitquat(R0)
    q1 = roma.mappings.rotmat_to_unitquat(R1)
    interpolated_q = unitquat_slerp(q0, q1, steps, shortest_arc=True)
    return roma.mappings.unitquat_to_rotmat(interpolated_q)

def rigid_vectors_registration(x, y, weights=None):
    """
    Returns the rotation matrix :math:`R` that best aligns an input list of vectors :math:`(x_i)_{i=1...n}` to a target list of vectors :math:`(y_i)_{i=1...n}`
    by minimizing the sum of square distance :math:`\sum_i w_i \|R x_i - y_i\|^2`, where :math:`(w_i)_{i=1...n}` denotes optional positive weights.
    See :func:`~roma.utils.rigid_points_registration` for details.

    Args:
        x (...xNxD tensor): list of N vectors of dimension D.
        y (...xNxD tensor): list of corresponding target vectors.
        weights (None or ...xN tensor): optional list of weights associated to each vector.
    Returns:
        The rotation matrix :math:`R` (...xDxD tensor).
    """
    if weights is None:
        M = torch.einsum("...ki, ...kj -> ...ij", y, x)
    else:
        M = torch.einsum("...k, ...ki, ...kj -> ...ij", weights, y, x)
    R = roma.special_procrustes(M)
    return R

def rigid_points_registration(x, y, weights=None):
    """
    Returns the rigid transformation :math:`(R,t)` that best aligns an input list of points :math:`(x_i)_{i=1...n}` to a target list of points :math:`(y_i)_{i=1...n}`,
    by minimizing the sum of square distance :math:`\sum_i w_i \|R x_i + t - y_i\|^2`, where :math:`(w_i)_{i=1...n}` denotes optional positive weights.
    This is sometimes referred to as the Kabsch/Umeyama algorithm.

    Args:
        x (...xNxD tensor): list of N points of dimension D.
        y (...xNxD tensor): list of corresponding target points.
        weights (None or ...xN tensor): optional list of weights associated to each point.
    Returns:
        a tuple :math:`(R, t)` consisting of a rotation matrix :math:`R` (...xDxD tensor) and a translation vector :math:`t` (...xD tensor).

    References:
        S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE Transactions on pattern analysis and machine intelligence, vol. 13, no. 4, Art. no. 4, 1991.        

        W. Kabsch, "A solution for the best rotation to relate two sets of vectors". Acta Crystallographica, A32, 1976.
    """
    # Center data
    if weights is None:
        xmean = torch.mean(x, dim=-2, keepdim=True)
        ymean = torch.mean(y, dim=-2, keepdim=True)
    else:
        normalized_weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        xmean = torch.sum(normalized_weights[...,None] * x, dim=-2, keepdim=True)
        ymean = torch.sum(normalized_weights[...,None] * y, dim=-2, keepdim=True)
    xhat = x - xmean
    yhat = y - ymean
    
    # Solve the vectors registration problem
    R = rigid_vectors_registration(xhat, yhat, weights)
    t = (ymean - torch.einsum('...ik, ...jk -> ...ji', R, xmean)).squeeze(-2)
    return R, t

