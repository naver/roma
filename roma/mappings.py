# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Various mappings between different rotation representations.
"""

import torch
import roma.internal

class _ProcrustesManualDerivatives(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, force_rotation, regularization, gradient_eps):
        assert (M.dim() == 3 and M.shape[1] == M.shape[2]), "Input should be a BxDxD batch of matrices."
        # Singular values of D are sorted in descending order
        U, D, V = roma.internal.svd(M)
        if force_rotation:
            # We flip the smallest singular value to ensure getting a rotation matrix
            with torch.no_grad():
                flip = (torch.det(U) * torch.det(V) < 0)            
            # in-place modifications of variables not used afterwards.
            DS = D
            DS[flip,-1] *= -1
            del D
            US = U
            US[flip,:,-1] *= -1
            del U
        else:
            flip = None
            DS = D
            US = U
        R = US @ V.transpose(-1, -2)
        # Store data for backprop:
        ctx.save_for_backward(US, DS, V, M, R)
        ctx.gradient_eps = gradient_eps
        ctx.regularization = regularization
        return R, DS

    @staticmethod
    def backward(ctx, grad_R, grad_DS):
        US, DS, V, M, R = ctx.saved_tensors
        gradient_eps = ctx.gradient_eps

        USik_Vjl = torch.einsum('bik,bjl -> bklij', US, V)
        USil_Vjk = USik_Vjl.transpose(1,2)
        DSl = DS[:,None,:,None,None]
        DSk = DS[:,:,None,None,None]
        Omega_klij = (USik_Vjl - USil_Vjk) * roma.internal._pseudo_inverse(DSk + DSl, gradient_eps)
        # Note: this intermediary matrix may require lots of memory for large dimensional cases.
        # Diagonal k==l should always be 0 thanks to the clamping of the pseudo-inverse.
        
        grad_M = torch.einsum('bnm, bnk, bklij, bml -> bij', grad_R, US, Omega_klij, V)
        # Gradient contribution from singular values
        grad_M += (US * grad_DS[:,None,:]) @ V.transpose(-1, -2)
        if ctx.regularization != 0.0:
            # Add a regularization term in the direction of the orthonormalized output.
            grad_M += ctx.regularization * (M - R)
        return grad_M, None, None, None

def procrustes(M, force_rotation=False, regularization=0.0, gradient_eps=1e-5, return_singular_values : bool = False):
    r""" 
    Returns the orthonormal matrix :math:`R` minimizing Frobenius norm :math:`\| M - R \|_F`.

    Args:
        M (...xNxN tensor): batch of square matrices.
        force_rotation (bool): if True, forces the output to be a rotation matrix.
        regularization (float >= 0): weight of a regularization term added to the gradient.
            Using this regularization is equivalent to adding a term :math:`regularization * \| M - R \|_F^2` to the training loss function.
        gradient_eps (float > 0): small value used to enforce numerical stability during backpropagation.
    Returns:
        batch of orthonormal matrices (...xNxN tensor) and optional singular values.
        For advanced users, singular values of the SVD decomposition with sign flipping (... tensor) can optionally be returned by setting the argument :code:`return_singular_values` to :code:`True`.
    """
    M, batch_shape = roma.internal.flatten_batch_dims(M, -3)
    R, DS = _ProcrustesManualDerivatives.apply(M, force_rotation, regularization, gradient_eps)
    R = roma.internal.unflatten_batch_dims(R, batch_shape)
    if not return_singular_values:
        return R
    else:
        DS = roma.internal.unflatten_batch_dims(DS, batch_shape)
        return R, DS

def special_procrustes(M, regularization=0.0, gradient_eps=1e-5, return_singular_values : bool = False):
    r"""
    Returns the rotation matrix :math:`R` minimizing Frobenius norm :math:`\| M - R \|_F`.

    Args:
        M (...xNxN tensor): batch of square matrices.
        regularization (float >= 0): weight of a regularization term added to the gradient.
            Using this regularization is equivalent to adding a term :math:`regularization * \| M - R \|_F^2` to the training loss function.
        gradient_eps (float > 0): small value used to enforce numerical stability during backpropagation.
    Returns:
        batch of rotation matrices (...xNxN tensor).
        For advanced users, singular values of the SVD decomposition with sign flipping (... tensor) can optionally be returned by setting the argument :code:`return_singular_values` to :code:`True`.

    """
    return procrustes(M, True, regularization, gradient_eps, return_singular_values)

def procrustes_naive(M, force_rotation : bool = False, return_singular_values : bool = False):
    r"""
    Implementation of :func:`~roma.mappings.procrustes` relying on default backward pass of autograd and SVD decomposition.
    Could be slightly less stable than :func:`~roma.mappings.procrustes`.
    """
    M, batch_shape = roma.internal.flatten_batch_dims(M, -3)
    assert (M.dim() == 3 and M.shape[1] == M.shape[2]), "Input should be a BxDxD batch of matrices."
    U, D, V = roma.internal.svd(M)
    # D is sorted in descending order
    SVt = V.transpose(-1,-2)
    if force_rotation:
        # We flip the smallest singular value to ensure getting a rotation matrix
        with torch.no_grad():
            flip = (torch.det(U) * torch.det(V) < 0)
        if torch.is_grad_enabled():
            # This is needed to avoid a runtime error "one of the variables needed for gradient computation has been modified by an inplace operation"
            SVt = SVt.clone()
        SVt[flip,-1,:] *= -1
    R = U @ SVt
    R = roma.internal.unflatten_batch_dims(R, batch_shape)
    if not return_singular_values:
        return R
    else:
        DS = D.clone()
        if force_rotation:
            DS[flip, -1] *= -1
        del D
        return R, DS


def special_procrustes_naive(M, return_singular_values : bool = False):
    r"""
    Implementation of :func:`~roma.mappings.special_procrustes` relying on default backward pass of autograd and SVD decomposition.
    Could be slightly less stable than :func:`~roma.mappings.special_procrustes`.
    """
    return procrustes_naive(M, force_rotation=True, return_singular_values=return_singular_values)

def special_gramschmidt(M, epsilon=0):
    r"""
    Returns the 3x3 rotation matrix obtained by Gram-Schmidt orthonormalization of two 3D input vectors (first two columns of input matrix M).

    Args:
        M (...x3xN tensor): batch of 3xN matrices, with N >= 2.
            Only the first two columns of the matrices are used for orthonormalization.
        epsilon (float >= 0): optional clamping value to avoid returning *Not-a-Number* values in case of ill-defined input.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    Warning:
        In case of ill-defined input (colinear input column vectors), the output will not be a rotation matrix.
    """
    M, batch_shape = roma.internal.flatten_batch_dims(M, -3)
    assert(M.dim() == 3)
    x = M[:,:,0]
    y = M[:,:,1]
    x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), epsilon)
    y = y - torch.sum(x*y, dim=-1, keepdim=True) * x
    y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), epsilon)
    z = torch.cross(x,y, dim=-1)    
    R = torch.stack((x, y, z), dim=-1)
    return roma.internal.unflatten_batch_dims(R, batch_shape)

def symmatrix_to_projective_point(A):
    r"""
    Converts a DxD symmetric matrix A into a projective point represented by a unit vector :math:`q` minimizing :math:`q^T A q`.
    Only the lower part of the matrix is considered in practice.

    Args:
        A (...xDxD tensor): batch of symmetric matrices. Only the lower triangular part is considered.
    Returns:
        batch of unit vectors :math:`q` (...xD tensor).
    Reference:
        V. Peretroukhin, M. Giamou, D. M. Rosen, W. N. Greene, N. Roy, and J. Kelly, “A Smooth Representation of Belief over SO(3) for Deep Rotation Learning with Uncertainty,” 2020, doi: 10.15607/RSS.2020.XVI.007.

    Warning:
        - This mapping is unstable when the smallest eigenvalue of A has a multiplicity strictly greater than 1.
        - The eigenvalue decomposition may fail, in particular when using single precision numbers.
        - Current implementation is rather slow due to the implementation of ``torch.symeig``.
          The CuSolver library provides a faster eigenvalue decomposition alternative, but results where found to be unreliable.
    """
    A, batch_shape = roma.internal.flatten_batch_dims(A, end_dim=-3)
    B, D1, D2 = A.shape
    assert (D1,D2) == (4,4), "Input should be a symmetric Bx4x4 matrix."
    eigenvalues, eigenvectors = roma.internal.symeig_lower(A)
    # Eigenvalues are sorted in ascending order
    q = eigenvectors[:,:,0]
    return roma.internal.unflatten_batch_dims(q, batch_shape)

def symmatrixvec_to_unitquat(x):
    r"""
    Converts a 10D vector into a unit quaternion representation.
    Based on :func:`~roma.mappings.symmatrix_to_projective_point`.
    
    Args:
        x (...x10 tensor): batch of 10D vectors.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    Reference:
        V. Peretroukhin, M. Giamou, D. M. Rosen, W. N. Greene, N. Roy, and J. Kelly, “A Smooth Representation of Belief over SO(3) for Deep Rotation Learning with Uncertainty,” 2020, doi: 10.15607/RSS.2020.XVI.007.
    """
    x, batch_shape = roma.internal.flatten_batch_dims(x, end_dim=-2)
    batch_size, D = x.shape
    assert(D) == 10, "Input should be a Bx10 tensor."
    A = torch.empty((batch_size, 4, 4), dtype=x.dtype, device=x.device)
    # Fill only the lower triangular part of the matrix.
    A[:, :,0] = x[:,0:4]
    A[:,1:,1] = x[:,4:7]
    A[:,2:,2] = x[:,7:9]
    A[:, 3,3] = x[:,9]
    return roma.internal.unflatten_batch_dims(symmatrix_to_projective_point(A), batch_shape)    

def rotvec_to_unitquat(rotvec):
    r"""
    Converts rotation vector into unit quaternion representation.

    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    rotvec, batch_shape = roma.internal.flatten_batch_dims(rotvec, end_dim=-2)
    num_rotations, D = rotvec.shape
    assert D == 3, "Input should be a Bx3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L621
    
    norms = torch.norm(rotvec, dim=-1)
    small_angle = (norms <= 1e-3)
    large_angle = ~small_angle

    scale = torch.empty((num_rotations,), device=rotvec.device, dtype=rotvec.dtype)
    scale[small_angle] = (0.5 - norms[small_angle] ** 2 / 48 +
                          norms[small_angle] ** 4 / 3840)
    scale[large_angle] = (torch.sin(norms[large_angle] / 2) /
                          norms[large_angle])

    quat = torch.empty((num_rotations, 4), device=rotvec.device, dtype=rotvec.dtype)
    quat[:, :3] = scale[:, None] * rotvec
    quat[:, 3] = torch.cos(norms / 2)
    return roma.internal.unflatten_batch_dims(quat, batch_shape)

def unitquat_to_rotvec(quat, shortest_arc=True):
    r"""
    Converts unit quaternion into rotation vector representation.

    Based on the representation of a rotation of angle :math:`{\theta}` and unit axis :math:`(x,y,z)` 
    by the unit quaternions :math:`\pm [\sin({\theta} / 2) (x i + y j + z k) + \cos({\theta} / 2)]`.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
        shortest_arc (bool): if True, the function returns the smallest rotation vectors corresponding 
            to the input 3D rotations, i.e. rotation vectors with a norm smaller than :math:`\pi`.
            If False, the function may return rotation vectors of norm larger than :math:`\pi`, depending on the sign of the input quaternions.
    Returns:
        batch of rotation vectors (...x3 tensor).
    Note:
        Behavior is undefined for inputs ``quat=torch.as_tensor([0.0, 0.0, 0.0, -1.0])`` and ``shortest_arc=False``, 
        as any rotation vector of angle :math:`2 \pi` could be a valid representation in such case.
    """
    quat, batch_shape = roma.internal.flatten_batch_dims(quat, end_dim=-2)
    # We perform a copy to support auto-differentiation.
    quat = quat.clone()
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L1006-L1073
    if shortest_arc:
        # Enforce w > 0 to ensure 0 <= angle <= pi.
        # (Otherwise angle can be arbitrary within ]-2pi, 2pi]).
        quat[quat[:, 3] < 0] *= -1
    half_angle = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    angle = 2 * half_angle
    small_angle = (torch.abs(angle) <= 1e-3)
    large_angle = ~small_angle

    num_rotations = len(quat)
    scale = torch.empty(num_rotations, dtype=quat.dtype, device=quat.device)
    scale[small_angle] = (2 + angle[small_angle] ** 2 / 12 +
                          7 * angle[small_angle] ** 4 / 2880)
    scale[large_angle] = (angle[large_angle] /
                          torch.sin(half_angle[large_angle]))

    rotvec = scale[:, None] * quat[:, :3]
    return roma.internal.unflatten_batch_dims(rotvec, batch_shape)

def unitquat_to_rotmat(quat):
    r"""
    Converts unit quaternion into rotation matrix representation.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L912
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = - x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = - x2 - y2 + z2 + w2
    return matrix

def rotmat_to_unitquat(R):
    r"""
    Converts rotation matrix to unit quaternion representation.

    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    matrix, batch_shape = roma.internal.flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert((D1, D2) == (3,3)), "Input should be a Bx3x3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return roma.internal.unflatten_batch_dims(quat, batch_shape)

def rotvec_to_rotmat(rotvec: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    r"""
    Converts rotation vector to rotation matrix representation.
    Conversion uses Rodrigues formula in general, and a first order approximation for small angles.
    
    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
        epsilon (float): small angle threshold.
    Returns:
        batch of rotation matrices (...x3x3 tensor).        
    """
    rotvec, batch_shape = roma.internal.flatten_batch_dims(rotvec, end_dim=-2)
    batch_size, D = rotvec.shape
    assert(D == 3), "Input should be a Bx3 tensor."

    # Rotation angle
    theta = torch.norm(rotvec, dim=-1)
    is_angle_small = theta < epsilon
    
    # Rodrigues formula for angles that are not small.
    # Note: we use clamping to avoid non finite values for small angles
    # (torch.where produces nan gradients in such case).
    axis = rotvec / torch.clamp_min(theta[...,None], epsilon)
    kx, ky, kz = axis[:,0], axis[:,1], axis[:,2]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    xs = kx*sin_theta
    ys = ky*sin_theta
    zs = kz*sin_theta
    xyc = kx*ky*one_minus_cos_theta
    xzc = kx*kz*one_minus_cos_theta
    yzc = ky*kz*one_minus_cos_theta
    xxc = kx**2*one_minus_cos_theta
    yyc = ky**2*one_minus_cos_theta
    zzc = kz**2*one_minus_cos_theta
    R_rodrigues = torch.stack([1 - yyc - zzc, xyc - zs, xzc + ys,
                     xyc + zs, 1 - xxc - zzc, -xs + yzc,
                     xzc - ys, xs + yzc, 1 -xxc - yyc], dim=-1).reshape(-1, 3, 3)

    # For small angles, use a first order approximation
    xs, ys, zs = rotvec[:,0], rotvec[:,1], rotvec[:,2]
    one = torch.ones_like(xs)
    R_first_order = torch.stack([one, -zs, ys,
                                 zs, one, -xs,
                                 -ys, xs, one], dim=-1).reshape(-1, 3, 3)
    # Select the appropriate expression
    R = torch.where(is_angle_small[:,None,None], R_first_order, R_rodrigues)
    return roma.internal.unflatten_batch_dims(R, batch_shape)
    
def rotmat_to_rotvec(R):
    r"""
    Converts rotation matrix to rotation vector representation.

    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of rotation vectors (...x3 tensor).
    """
    q = rotmat_to_unitquat(R)
    return unitquat_to_rotvec(q)

def quat_xyzw_to_wxyz(xyzw):
    r"""
    Convert quaternion from XYZW to WXYZ convention.

    Args:
        xyzw (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, WXYZ convention).
    """
    assert xyzw.shape[-1] == 4
    return torch.cat((xyzw[...,-1,None], xyzw[...,:-1]), dim=-1)

def quat_wxyz_to_xyzw(wxyz):
    r"""
    Convert quaternion from WXYZ to XYZW convention.

    Args:
        wxyz (...x4 tensor, WXYZ convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    assert wxyz.shape[-1] == 4
    return torch.cat((wxyz[...,1:], wxyz[...,0,None]), dim=-1)