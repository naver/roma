# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import torch
import roma
import numpy as np
import roma.internal

def _elementary_basis_index(axis):
    r"""
    Return the index corresponding to a given axis label.
    """
    if axis == 'x':
        return 0
    elif axis == 'y':
        return 1
    elif axis == 'z':
        return 2
    else:
        raise ValueError("Invalid axis.")

def euler_to_unitquat(convention: str, angles, degrees=False, normalize=True, dtype=None, device=None):
    r"""
    Convert Euler angles to unit quaternion representation.

    Args:
        convention (string): string defining a sequence of D rotation axes ('XYZ' or 'xzx' for example).
            The sequence of rotation is expressed either with respect to a global 'extrinsic' coordinate system (in which case axes are denoted in lowercase: 'x', 'y', or 'z'),
            or with respect to an 'intrinsic' coordinates system attached to the object under rotation (in which case axes are denoted in uppercase: 'X', 'Y', 'Z').
            Intrinsic and extrinsic conventions cannot be mixed.
        angles (...xD tensor, or tuple/list of D floats or ... tensors): a list of angles associated to each axis, expressed in radians by default.
        degrees (bool): if True, input angles are assumed to be expressed in degrees.
        normalize (bool): if True, normalize the returned quaternion to compensate potential numerical.
    
    Returns:
        A batch of unit quaternions (...x4 tensor, XYZW convention).

    Warning:
        Case is important: 'xyz' and 'XYZ' denote different conventions.
    """
    if type(angles) == torch.Tensor:
        angles = [t.squeeze(dim=-1) for t in torch.split(angles, split_size_or_sections=1, dim=-1)]

    assert len(convention) == len(angles)

    extrinsics = convention.islower()
    if extrinsics:
        # Cast from intrinsics to extrinsics convention
        convention = convention.upper()[::-1]
        angles = angles[::-1]

    unitquats = []
    for axis, angle in zip(convention, angles):
        angle = torch.as_tensor(angle, device=device, dtype=dtype)
        if degrees:
            angle = torch.deg2rad(angle)
        batch_shape = angle.shape
        rotvec = torch.zeros(batch_shape + torch.Size((3,)), device=angle.device, dtype=angle.dtype)
        if axis == 'X':
            rotvec[...,0] = angle
        elif axis == 'Y':
            rotvec[...,1] = angle
        elif axis == 'Z':
            rotvec[...,2] = angle
        else:
            raise ValueError("Invalid convention (expected format: 'xyz', 'zxz', 'XYZ', etc.).")
        q = roma.rotvec_to_unitquat(rotvec)
        unitquats.append(q)
    if len(unitquats) == 1:
        return unitquats[0]
    else:
        return roma.quat_composition(unitquats, normalize=normalize)

def euler_to_rotvec(convention: str, angles, degrees=False, dtype=None, device=None):
    r"""
    Convert Euler angles to rotation vector representation.

    Args:
        convention (string): 'xyz' for example. See :func:`~roma.euler.euler_to_unitquat()`.
        angles (...xD tensor, or tuple/list of D floats or ... tensors): a list of angles associated to each axis, expressed in radians by default.
        degrees (bool): if True, input angles are assumed to be expressed in degrees.

    Returns:
        a batch of rotation vectors (...x3 tensor).
    """
    return roma.unitquat_to_rotvec(euler_to_unitquat(convention=convention, angles=angles, degrees=degrees, dtype=dtype, device=device))

def euler_to_rotmat(convention: str, angles, degrees=False, dtype=None, device=None):
    r"""
    Convert Euler angles to rotation matrix representation.

    Args:
        convention (string): 'xyz' for example. See :func:`~roma.euler.euler_to_unitquat()`.
        angles (...xD tensor, or tuple/list of D floats or ... tensors): a list of angles associated to each axis, expressed in radians by default.
        degrees (bool): if True, input angles are assumed to be expressed in degrees.
    
    Returns:
        a batch of rotation matrices (...x3x3 tensor).
    """
    return roma.unitquat_to_rotmat(euler_to_unitquat(convention=convention, angles=angles, degrees=degrees, dtype=dtype, device=device))

def unitquat_to_euler(convention : str, quat, as_tuple=False, degrees=False, epsilon=1e-7):
    r"""
    Convert unit quaternion to Euler angles representation.

    Args:
        convention (str): string of 3 characters belonging to {'x', 'y', 'z'} for extrinsic rotations, or {'X', 'Y', 'Z'} for intrinsic rotations.
            Consecutive axes should not be identical.
        quat (...x4 tensor, XYZW convention): input batch of unit quaternion.
        as_tuple (boolean): if True, angles are not stacked but returned as a tuple of tensors.
        degrees (bool): if True, angles are returned in degrees.
        epsilon (float): a small value used to detect degenerate configurations.

    Returns:
        A stacked ...x3 tensor corresponding to Euler angles, expressed by default in radians.
        In case of gimbal lock, the third angle is arbitrarily set to 0.
    """
    # Code adapted from scipy.spatial.transform.Rotation.
    # Reference: https://github.com/scipy/scipy/blob/ac6bcaf00411286271f7cc21e495192c73168ae4/scipy/spatial/transform/_rotation.pyx#L325C12-L325C15
    assert len(convention) == 3

    pi = np.pi
    lamb = np.pi/2

    extrinsic = convention.islower()
    if not extrinsic:
        convention = convention.lower()[::-1]

    quat, batch_shape = roma.internal.flatten_batch_dims(quat, end_dim=-2)
    N = quat.shape[0]

    i = _elementary_basis_index(convention[0])
    j = _elementary_basis_index(convention[1])
    k = _elementary_basis_index(convention[2])
    assert i != j and j != k, "Consecutive axes should not be identical."

    symmetric = (i == k)

    if symmetric:
        # Get third axis
        k = 3 - i - j

    # Step 0
    # Check if permutation is even (+1) or odd (-1) 
    sign = (i - j) * (j - k) * (k - i) // 2

    # Step 1
    # Permutate quaternion elements            
    if symmetric:
        a = quat[:,3]
        b = quat[:,i]
        c = quat[:,j]
        d = quat[:,k] * sign
    else:
        a = quat[:,3] - quat[:,j]
        b = quat[:,i] + quat[:,k] * sign
        c = quat[:,j] + quat[:,3]
        d = quat[:,k] * sign - quat[:,i]

    
    # intrinsic/extrinsic conversion helpers
    if extrinsic:
        angle_first = 0
        angle_third = 2
    else:
        angle_first = 2
        angle_third = 0

    # Step 2
    # Compute second angle...
    angles = [torch.empty(N, device=quat.device, dtype=quat.dtype) for _ in range(3)]
    
    angles[1] = 2 * torch.atan2(roma.internal.hypot(c, d), roma.internal.hypot(a, b))

    # ... and check if equal to is 0 or pi, causing a singularity
    case1 = torch.abs(angles[1]) <= epsilon
    case2 = torch.abs(angles[1] - pi) <= epsilon
    case1or2 = torch.logical_or(case1, case2)
    # Step 3
    # compute first and third angles, according to case
    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)
    
    # no singularities
    angles[angle_first] = half_sum - half_diff
    angles[angle_third] = half_sum + half_diff
    
    # any degenerate case
    angles[2][case1or2] = 0
    angles[0][case1] = 2 * half_sum[case1]
    angles[0][case2] = 2 * (-1 if extrinsic else 1) * half_diff[case2]
            
    # for Tait-Bryan/asymmetric sequences
    if not symmetric:
        angles[angle_third] *= sign
        angles[1] -= lamb

    for idx in range(3):
        foo = angles[idx]
        foo[foo < -pi] += 2 * pi
        foo[foo > pi] -= 2 * pi
        if degrees:
            foo = torch.rad2deg(foo)
        angles[idx] = roma.internal.unflatten_batch_dims(foo, batch_shape)

    if as_tuple:
        return tuple(angles)
    else:
        return torch.stack(angles, dim=-1)

def rotvec_to_euler(convention : str, rotvec, as_tuple=False, degrees=False, epsilon=1e-7):
    r"""
    Convert rotation vector to Euler angles representation.

    Args:
        convention (str): string of 3 characters belonging to {'x', 'y', 'z'} for extrinsic rotations, or {'X', 'Y', 'Z'} for intrinsic rotations.
            Consecutive axes should not be identical.
        rotvec (...x3 tensor): input batch of rotation vectors.
        as_tuple (boolean): if True, angles are not stacked but returned as a tuple of tensors.
        degrees (bool): if True, angles are returned in degrees.
        epsilon (float): a small value used to detect degenerate configurations.

    Returns:
        A stacked ...x3 tensor corresponding to Euler angles, expressed by default in radians.
        In case of gimbal lock, the third angle is arbitrarily set to 0.
    """
    return unitquat_to_euler(convention, roma.rotvec_to_unitquat(rotvec), as_tuple=as_tuple, degrees=degrees, epsilon=epsilon)

def rotmat_to_euler(convention : str, rotmat, as_tuple=False, degrees=False, epsilon=1e-7):
    r"""
    Convert rotation matrix to Euler angles representation.

    Args:
        convention (str): string of 3 characters belonging to {'x', 'y', 'z'} for extrinsic rotations, or {'X', 'Y', 'Z'} for intrinsic rotations.
            Consecutive axes should not be identical.
        rotmat (...x3x3 tensor): input batch of rotation matrices.
        as_tuple (boolean): if True, angles are not stacked but returned as a tuple of tensors.
        degrees (bool): if True, angles are returned in degrees.
        epsilon (float): a small value used to detect degenerate configurations.

    Returns:
        A stacked ...x3 tensor corresponding to Euler angles, expressed by default in radians.
        In case of gimbal lock, the third angle is arbitrarily set to 0.
    """
    return unitquat_to_euler(convention, roma.rotmat_to_unitquat(rotmat), as_tuple=as_tuple, degrees=degrees, epsilon=epsilon)