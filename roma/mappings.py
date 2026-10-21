# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Various mappings between different rotation representations.
"""

import functools
import math
import torch
import roma.internal


class _ProcrustesManualDerivatives(torch.autograd.Function):
    generate_vmap_rule = True

    # Explicitely cast inputs to float32 for CPU and CUDA devices when using autocast,
    # as svd is not supported with bfloat16 and float16 on CPU and CUDA devices.
    @staticmethod
    @roma.internal.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
    @roma.internal.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(M, force_rotation, gradient_eps):
        assert M.dim() == 3 and M.shape[1] == M.shape[2], "Input should be a BxDxD batch of matrices."
        # Singular values of D are sorted in descending order
        U, D, Vt = torch.linalg.svd(M)
        V = Vt.transpose(-2, -1)
        if force_rotation:
            # We flip the smallest singular value to ensure getting a rotation matrix
            with torch.no_grad():
                flip = torch.det(U) * torch.det(V) < 0
                sign = 1.0 - 2.0 * flip.to(U.dtype)
                # Matrix of ones, except for its last column equal to +-1 depending on the flip,
                # built out-of-place for vmap compatibility.
                flip_matrix = torch.cat((torch.ones_like(D[:, :-1]), sign[:, None]), dim=-1)
            DS = D * flip_matrix
            del D
            US = U * flip_matrix[:, None, :]
            del U
        else:
            DS = D
            US = U
        R = US @ V.transpose(-1, -2)
        # US and V are returned to make them available to setup_context,
        # and are discarded by the public wrapper.
        return R, DS, US, V

    @staticmethod
    def setup_context(ctx, inputs, output):
        M, force_rotation, gradient_eps = inputs
        R, DS, US, V = output
        # Store data for backprop and forward-mode differentiation:
        ctx.save_for_backward(US, DS, V)
        ctx.save_for_forward(US, DS, V)
        ctx.mark_non_differentiable(US, V)
        ctx.gradient_eps = gradient_eps

    @staticmethod
    def backward(ctx, grad_R, grad_DS, grad_US, grad_V):
        # Backward-mode differentiation (vector-Jacobian product).
        US, DS, V = ctx.saved_tensors
        gradient_eps = ctx.gradient_eps
        # Disable autocast to perform computations in full precision.
        with roma.internal.autocast_disabled(US.device.type):
            # Omega_klij = (US_ik V_jl - US_il V_jk) / (DS_k + DS_l), antisymmetric with respect to (k,l),
            # using a damped pseudo-inverse (see roma.internal._pseudo_inverse). Diagonal k==l is 0 by antisymmetry.
            # Note: this intermediary tensor may require lots of memory for large dimensional cases.
            USik_Vjl = torch.einsum("bik,bjl -> bklij", US, V)
            USil_Vjk = USik_Vjl.transpose(1, 2)
            DSl = DS[:, None, :, None, None]
            DSk = DS[:, :, None, None, None]
            Omega_klij = (USik_Vjl - USil_Vjk) * roma.internal._pseudo_inverse(DSk + DSl, gradient_eps)
            grad_M = torch.einsum("bnm, bnk, bklij, bml -> bij", grad_R, US, Omega_klij, V)
            # Gradient contribution of the singular values.
            grad_M = grad_M + (US * grad_DS[:, None, :]) @ V.transpose(-1, -2)
        return grad_M, None, None

    @staticmethod
    def jvp(ctx, dM, *args):
        # Forward-mode differentiation (Jacobian-vector product).
        US, DS, V = ctx.saved_for_forward
        # Disable autocast to perform computations in full precision.
        with roma.internal.autocast_disabled(US.device.type):
            dM = dM.to(US.dtype)
            A = US.transpose(-1, -2) @ dM @ V
            # Z_kl = (A_kl - A_lk) / (DS_k + DS_l), antisymmetric with respect to (k,l),
            # using a damped pseudo-inverse (see roma.internal._pseudo_inverse). Diagonal k==l is 0 by antisymmetry.
            Z = (A - A.transpose(-1, -2)) * roma.internal._pseudo_inverse(
                DS[:, :, None] + DS[:, None, :], ctx.gradient_eps
            )
            dR = US @ Z @ V.transpose(-1, -2)
            # Tangent contribution of the singular values.
            dDS = torch.diagonal(A, dim1=-2, dim2=-1)
        return dR, dDS, None, None


# Closed-form eigenvectors of symmetric 4x4 matrices,
# used for the 3D special Procrustes problem and for the symmetric matrix representation of rotations.
#
# Following Horn's quaternion formulation of absolute orientation (B.K.P. Horn, JOSA A 1987), tr(R^T M) = q^T N(M) q
# for any rotation R of unit quaternion q, where N(M) is a symmetric traceless 4x4 matrix depending linearly on M.
# The rotation maximizing tr(R^T M) is therefore given by the eigenvector of N(M) associated to its largest eigenvalue,
# and the eigenvalues of N(M) are known in closed form from the singular values of M.
# Moreover, M -> N(M) is a bijection between 3x3 matrices and symmetric traceless 4x4 matrices: any symmetric 4x4 matrix A
# can be written A = tr(A)/4 I + N(M), so that its eigenvectors can be computed in the same way (see symmatrix_to_projective_point).
# In both cases, the eigenvector is obtained with elementwise and tiny matrix operations only, without any batched SVD or
# eigendecomposition.
# Computations are performed in float64, as in float32 this closed form is less accurate than a batched SVD or eigendecomposition.

_GAP_TOL = 1e-7  # Relative eigenvalue gap below which an eigenvalue is considered as repeated (non-unique solution).
_SHIFT = 1e-13  # Relative offset of the inverse iteration shift, to avoid an exactly singular system.


@functools.lru_cache(maxsize=None)
def _horn_map(device):
    r"""
    :meta private:
    Linear map (9x16 tensor) from the 9 entries of a 3x3 matrix M to the 16 entries of the symmetric 4x4 matrix N(M)
    of Horn's paper, for quaternions in XYZW convention. Its rows are orthogonal with squared norm 4.
    """
    e = torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)
    (m11, m12, m13), (m21, m22, m23), (m31, m32, m33) = (row.unbind(-1) for row in e.unbind(-2))
    # fmt: off
    horn = torch.stack([m11 - m22 - m33, m12 + m21, m13 + m31, m32 - m23,
                        m12 + m21, m22 - m11 - m33, m23 + m32, m13 - m31,
                        m13 + m31, m23 + m32, m33 - m11 - m22, m21 - m12,
                        m32 - m23, m13 - m31, m21 - m12, m11 + m22 + m33], dim=-1)
    # fmt: on
    return horn.to(device)


def _horn_matrix(M):
    r"""
    :meta private:
    Horn's symmetric matrix N(M) (...x4x4 tensor) of a batch of 3x3 float64 matrices M, such that tr(R^T M) = q^T N(M) q
    for any rotation matrix R of unit quaternion q.
    """
    return (M.flatten(-2) @ _horn_map(M.device)).unflatten(-1, (4, 4))


def _horn_matrix_adjoint(X):
    r"""
    :meta private:
    Adjoint of the linear map M -> N(M) (see _horn_matrix): for a batch of 4x4 float64 matrices X, returns the 3x3 matrices Y
    (...x3x3 tensor) such that <Y, M> = <X, N(M)> for any 3x3 matrix M, where <., .> denotes the Frobenius inner product.
    Since N(M) is symmetric and traceless for any M, only the symmetric traceless part of X contributes to Y
    (in particular, Y = 0 for X = I).

    Two special cases are used in this module:

    - X = q q^T with q a unit quaternion: Y is the rotation matrix R of q.
      Indeed, Horn's identity tr(R^T M) = q^T N(M) q reads <R, M> = <q q^T, N(M)> for any M, which is the defining property of Y.
    - X = N(M0) for some 3x3 matrix M0: Y = 4 M0.
      Indeed, the rows of the Horn map are orthogonal with squared norm 4, so that <N(M0), N(M)> = 4 <M0, M> for any M.
      Y / 4 therefore inverts M -> N(M) on symmetric traceless 4x4 matrices.
    """
    return (X.flatten(-2) @ _horn_map(X.device).T).unflatten(-1, (3, 3))


def _matmul_small(a, b):
    r"""
    :meta private:
    Batched matrix product a @ b for tiny matrices.
    On GPU, a broadcasted multiply-and-sum is several times faster than torch.matmul for large batches of 3x3 or 4x4 matrices,
    whereas torch.matmul is faster on CPU.
    """
    if a.device.type == "cpu":
        return a @ b
    return (a[..., :, :, None] * b[..., None, :, :]).sum(dim=-2)


def _det3(x):
    r"""
    :meta private:
    Determinant of a batch of 3x3 matrices.
    """
    return (x[..., 0, :] * torch.linalg.cross(x[..., 1, :], x[..., 2, :], dim=-1)).sum(dim=-1)


def _symmetric_eigenvector(A, lam, lam_others):
    r"""
    :meta private:
    Unit eigenvector (...xD tensor) of a batch of symmetric matrices A (...xDxD tensor) for the eigenvalue lam (... tensor),
    given estimates of the other eigenvalues lam_others (...x(D-1) tensor).
    If lam has a multiplicity greater than 1 (up to a tolerance _GAP_TOL relative to ||A||),
    some vector of the corresponding eigenspace is returned.
    """
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    # First estimate: a column of the spectral projector onto the eigenspace of lam, i.e. of the product of
    # (A - lam_j I) over the other eigenvalues. Eigenvalues (nearly) equal to lam are skipped.
    norm_A = torch.linalg.matrix_norm(A)
    factors = A[..., None, :, :] - lam_others[..., None, None] * eye
    separated = (lam[..., None] - lam_others).abs() > _GAP_TOL * norm_A[..., None]
    factors = torch.where(separated[..., None, None], factors, eye)
    P = factors[..., 0, :, :]
    for j in range(1, factors.shape[-3]):
        P = _matmul_small(P, factors[..., j, :, :])
    i = P.diagonal(dim1=-2, dim2=-1).argmax(dim=-1)  # Index of the column of largest norm.
    q = torch.gather(P, -1, i[..., None, None].expand(*i.shape, eye.shape[0], 1))

    # Refinement by two steps of inverse iteration, the second one using the Rayleigh quotient of q as shift.
    # The shift is slightly offset to avoid an exactly singular system.
    def inverse_iteration(q, lam):
        shift = lam + _SHIFT * norm_A
        q = torch.linalg.solve_ex(A - shift[..., None, None] * eye, q, check_errors=False)[0]
        return q / q.norm(dim=-2, keepdim=True)

    q = inverse_iteration(q, lam)
    q = inverse_iteration(q, (q * _matmul_small(A, q)).sum(dim=(-2, -1)))
    return q[..., 0]


def _horn_eigenvector(M):
    r"""
    :meta private:
    Unit eigenvector (...x4 tensor) associated to the largest eigenvalue of Horn's matrix N(M), for a batch of 3x3 float64 matrices M,
    i.e. unit quaternion (XYZW convention) of the rotation maximizing tr(R^T M) (closed form, see above).
    If this eigenvalue is repeated (up to a relative tolerance _GAP_TOL), some vector of the eigenspace is returned.
    """
    eye3 = torch.eye(3, dtype=M.dtype, device=M.device)
    # Normalization to ||M||_F = 1 (zero matrices are replaced by the identity).
    norm = M.flatten(-2).norm(dim=-1)[..., None, None]
    M = torch.where(norm == 0, eye3 / math.sqrt(3.0), M / norm)
    det = _det3(M)
    # Singular values s1 >= s2 >= s3 of M: s1^2 is the largest eigenvalue of S = M^T M (of trace 1), given by the
    # trigonometric solution of the characteristic polynomial.
    # References:
    # - O.K. Smith, "Eigenvalues of a symmetric 3 x 3 matrix", Communications of the ACM 4(4), p. 168, 1961, doi: 10.1145/355578.366316
    # - Wikipedia contributors. (2026, March 30).
    #
    # With K = S - I/3 traceless (Tr(S)=1 since M is normalized),
    # the eigenvalues k of K are the roots of the characteristic polynomial det(k I - K) = k^3 - 3 p k - det(K) = 0 where p = tr(K^2) / 6.
    # The trigonometric solution of this cubic equation consist of the three roots 2 sqrt(p) cos(phi + 2 j pi / 3), j = 0, 1, 2,
    # with phi = acos(det(K) / (2 p^(3/2))) / 3 in [0, pi/3], the largest being 2 sqrt(p) cos(phi) >= 0.
    # Therefore s1^2 = 1/3 + 2 sqrt(p) cos(phi) >= 1/3.
    # s2 and s3 then follow from s2^2 + s3^2 = 1 - s1^2 and s2 s3 = |det(M)| / s1.
    # (Small singular values would be inaccurate if computed directly as square roots of eigenvalues of M^T M.)
    S = _matmul_small(M.transpose(-1, -2), M)
    K = S - eye3 / 3
    p = K.flatten(-2).square().sum(dim=-1) / 6
    phi = torch.acos((_det3(K) / (2 * p**1.5).clamp_min(1e-300)).clamp(-1, 1)) / 3
    s1 = (1 / 3 + 2 * p.sqrt() * torch.cos(phi)).sqrt()
    prod = det / s1  # s2 * sign(det) s3
    sumsq = (1 - s1 * s1).clamp_min(2 * prod.abs())
    # u = s2 + sign(det) s3 and v = s2 - sign(det) s3.
    u, v = (sumsq + 2 * prod).sqrt(), (sumsq - 2 * prod).sqrt()
    # Eigenvalues of N(M): for the SVD M = U Sigma V^T with U, V rotations and Sigma = diag(s1, s2, sign(det) s3),
    # N(Sigma) is diagonal with entries (s1 - u, -s1 + v, -s1 - v, s1 + u) in XYZW order, and N(M) = Q^T N(Sigma) Q
    # for the orthogonal map Q: q -> conj(q_U) q q_V (Horn's identity in the frame of U and V). The largest eigenvalue is s1 + u.
    return _symmetric_eigenvector(_horn_matrix(M), s1 + u, torch.stack([s1 - u, -s1 + v, -s1 - v], dim=-1))


def _horn_eigenvector_pseudo_inverse(M, q, Y, eps):
    r"""
    :meta private:
    Linear operator mapping a perturbation of Horn's matrix N(M) to the resulting perturbation of one of its unit eigenvectors q,
    damped where q is ill-conditioned, and applied to Y.

    Notations, for a batch of 3x3 float64 matrices M (...x3x3 tensor):

    - N = N(M): Horn's symmetric 4x4 matrix (see _horn_matrix).
    - q (...x4 tensor): a unit eigenvector of N (in practice the one of largest eigenvalue).
    - lambda = q^T N q: the eigenvalue of N associated to q.
    - C = lambda I - N: symmetric 4x4 matrix whose kernel is spanned by q (when lambda is simple).

    Given Y (...x4xK tensor), returns the solution X (...x4xK tensor) of the singular system C X = Y in the damped least-squares sense,
    i.e. the minimizer of ||C X - Y||^2 + eps^2 ||X||^2, obtained by solving the 4x4 linear system (C^2 + eps^2 I) X = C Y.

    By first-order perturbation theory, dq = C^+ dN q, where C^+ is the pseudo-inverse of C. C has eigenvalue 0 along q
    (dropped by C^+, so that dq is orthogonal to q), and its other eigenvalues c are the gaps between lambda and the other
    eigenvalues of N. For the largest eigenvalue, these gaps are twice the pairwise sums of the (signed) singular values of M,
    and they vanish when q is not unique. The damped solution X replaces each factor 1 / c of C^+ Y by c / (c^2 + eps^2),
    which is bounded by 1 / (2 eps), consistently with roma.internal._pseudo_inverse.

    Typical uses, with N^* the adjoint Horn map (see _horn_matrix_adjoint):

    - forward mode: dq = X with Y = N(dM) q.
    - reverse mode: dL/dM = N^*(X q^T) with Y = dL/dq, since the map Y -> X is symmetric.
    """
    eye4 = torch.eye(4, dtype=M.dtype, device=M.device)
    N = _horn_matrix(M)
    qq = q[..., :, None] * q[..., None, :]
    lam = (qq * N).sum(dim=(-2, -1))
    C = lam[..., None, None] * eye4 - N
    # C Y is orthogonal to q, so that the term ||C||^2 q q^T does not change the solution,
    # but prevents the system from being ill-conditioned in the direction of q when eps is small.
    damped = _matmul_small(C, C) + eps**2 * eye4 + C.square().sum(dim=(-2, -1))[..., None, None] * qq
    return torch.linalg.solve_ex(damped, _matmul_small(C, Y), check_errors=False)[0]


class _HornEigenvectorManualDerivatives(torch.autograd.Function):
    r"""
    :meta private:
    Unit eigenvector associated to the largest eigenvalue of N(M), for a batch of 3x3 float64 matrices M (see _horn_eigenvector),
    with damped derivatives (see _horn_eigenvector_pseudo_inverse).
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(M, gradient_eps):
        assert M.dim() == 3 and M.shape[1:] == (3, 3), "Input should be a Bx3x3 batch of matrices."
        assert M.dtype == torch.float64, "Input should be a float64 tensor."
        return _horn_eigenvector(M)

    @staticmethod
    def setup_context(ctx, inputs, output):
        M, gradient_eps = inputs
        ctx.save_for_backward(M, output)
        ctx.save_for_forward(M, output)
        ctx.gradient_eps = gradient_eps

    @staticmethod
    def backward(ctx, grad_q):
        # Backward-mode differentiation (vector-Jacobian product).
        M, q = ctx.saved_tensors
        with roma.internal.autocast_disabled(M.device.type):
            grad_N = _horn_eigenvector_pseudo_inverse(M, q, grad_q[..., None], ctx.gradient_eps) * q[..., None, :]
            return _horn_matrix_adjoint(grad_N), None

    @staticmethod
    def jvp(ctx, dM, _):
        # Forward-mode differentiation (Jacobian-vector product).
        M, q = ctx.saved_for_forward
        with roma.internal.autocast_disabled(M.device.type):
            dNq = _matmul_small(_horn_matrix(dM), q[..., None])
            return _horn_eigenvector_pseudo_inverse(M, q, dNq, ctx.gradient_eps)[..., 0]


class _ProcrustesRegularization(torch.autograd.Function):
    r"""
    :meta private:
    Identity on the orthonormalized matrices R of a batch of matrices M, adding a term regularization * (M - R) to the gradient
    of M during backpropagation (see the regularization argument of procrustes).
    This term does not depend on the gradient of R, and therefore cannot be expressed as the derivative of any function.
    It only affects backpropagation: forward-mode differentiation (jvp) leaves the tangent of R unchanged.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(M, R, regularization):
        return R.view_as(R)

    @staticmethod
    def setup_context(ctx, inputs, output):
        M, R, regularization = inputs
        ctx.save_for_backward(M, R)
        ctx.regularization = regularization

    @staticmethod
    def backward(ctx, grad_R):
        M, R = ctx.saved_tensors
        return ctx.regularization * (M - R), grad_R, None

    @staticmethod
    def jvp(ctx, dM, dR, _):
        return dR


try:
    from torch.compiler import is_compiling as _is_torch_compiling
except ImportError:  # torch < 2.1
    from torch._dynamo import is_compiling as _is_torch_compiling


# Workaround for a PyTorch bug (Dynamo, observed with torch<=2.13):
# when a functorch transform such as torch.func.jvp is applied inside a torch.compile'd function,
# Dynamo inlines the forward() of the autograd Function and silently discards its custom jvp rule,
# returning potentially incorrect or unstable tangents.
# We therefore run the autograd Function eagerly (outside of the compiled graph)
# whenever a functorch transform is active during compilation.
_procrustes_apply_eager = torch._dynamo.disable(_ProcrustesManualDerivatives.apply)
_horn_eigenvector_apply_eager = torch._dynamo.disable(_HornEigenvectorManualDerivatives.apply)


# Otherwise, Dynamo (observed with torch<=2.13) refuses to trace autograd Functions defining a custom jvp
# whenever gradients are required, which breaks torch.compile(fullgraph=True) with backpropagation.
# We therefore apply variants without custom jvp during compilation.
class _ProcrustesManualDerivativesNoJvp(_ProcrustesManualDerivatives):
    jvp = torch.autograd.Function.jvp


class _HornEigenvectorManualDerivativesNoJvp(_HornEigenvectorManualDerivatives):
    jvp = torch.autograd.Function.jvp


class _ProcrustesRegularizationNoJvp(_ProcrustesRegularization):
    jvp = torch.autograd.Function.jvp


def _procrustes_apply(M, force_rotation, gradient_eps):
    if _is_torch_compiling():
        if torch._C._are_functorch_transforms_active():
            return _procrustes_apply_eager(M, force_rotation, gradient_eps)
        return _ProcrustesManualDerivativesNoJvp.apply(M, force_rotation, gradient_eps)
    return _ProcrustesManualDerivatives.apply(M, force_rotation, gradient_eps)


def _horn_eigenvector_apply(M, gradient_eps):
    if _is_torch_compiling():
        if torch._C._are_functorch_transforms_active():
            return _horn_eigenvector_apply_eager(M, gradient_eps)
        return _HornEigenvectorManualDerivativesNoJvp.apply(M, gradient_eps)
    return _HornEigenvectorManualDerivatives.apply(M, gradient_eps)


def _procrustes_regularization_apply(M, R, regularization):
    # Inlined by Dynamo under functorch transforms, its forward has the same derivative as its custom jvp.
    if _is_torch_compiling():
        return _ProcrustesRegularizationNoJvp.apply(M, R, regularization)
    return _ProcrustesRegularization.apply(M, R, regularization)


def _horn_procrustes(M, gradient_eps):
    r"""
    :meta private:
    Rotation matrices maximizing tr(R^T M) for a batch of 3x3 matrices M (Bx3x3 tensor), computed as R = adjoint Horn map of q q^T
    where q is the eigenvector of _horn_eigenvector. Derivatives are those of q (see _HornEigenvectorManualDerivatives),
    propagated by autograd. Computations are performed in float64 (not affected by autocast), and R has the dtype of M.
    The eigenvalue gaps of N(M) are twice the pairwise sums of the singular values of M involved in the SVD-based path,
    hence a damping 2 * gradient_eps to keep the meaning of gradient_eps identical for every dimension.
    """
    assert M.dim() == 3 and M.shape[1:] == (3, 3), "Input should be a Bx3x3 batch of matrices."
    q = _horn_eigenvector_apply(M.double(), 2 * gradient_eps)
    # Equivalent to unitquat_to_rotmat(q), as a single matmul.
    return _horn_matrix_adjoint(q[..., :, None] * q[..., None, :]).to(M.dtype)


def _supports_float64(device):
    r"""
    :meta private:
    Whether a device supports float64 tensors (required by the closed-form eigenvector computations).
    """
    # Apple MPS is the only PyTorch backend without float64 support.
    return device.type != "mps"


def procrustes(
    M,
    force_rotation=False,
    regularization=0.0,
    gradient_eps=1e-5,
    return_singular_values: bool = False,
    method: str = "auto",
):
    r"""
    Returns the orthonormal matrix :math:`R` minimizing Frobenius norm :math:`\| M - R \|_F`.

    Args:
        M (...xNxN tensor): batch of square matrices.
        force_rotation (bool): if True, forces the output to be a rotation matrix.
        regularization (float >= 0): weight of a regularization term added to the gradient during backpropagation.
            Using this regularization is equivalent to adding a term :math:`regularization * \| M - R \|_F^2` to the training loss function.
            It only affects backpropagation through the output matrix (not through the optional singular values).
            Forward-mode differentiation (jvp) returns the true directional derivative and ignores it.
        gradient_eps (float > 0): damping of the derivative when the solution is ill-conditioned.
            The derivative involves the inverse of pairwise sums :math:`t` of (signed) singular values of :math:`M`, which vanish when the solution is not unique:
            :math:`1 / t` is replaced by the damped pseudo-inverse :math:`t / (t^2 + gradient\_eps^2)`, bounded by :math:`1 / (2\, gradient\_eps)`.
        return_singular_values (bool): if True, also returns the singular values of the SVD decomposition with sign flipping (... tensor), for advanced users.
            Not compatible with ``method="horn"``.
        method (str): algorithm used to solve the problem.

            - ``"svd"``: batched singular value decomposition (any dimension).
            - ``"horn"``: closed-form solution for 3x3 matrices with :code:`force_rotation=True`, based on Horn's quaternion formulation
              of absolute orientation.
              Scales better with the batch size than the SVD: slower for small batches, but significantly faster for large ones (especially on GPU),
              and more accurate for single precision inputs.
              Computations are performed in float64, which is not supported on MPS devices.
              A :code:`ValueError` is raised if the method is not applicable.
            - ``"auto"`` (default): ``"horn"`` whenever applicable, ``"svd"`` otherwise.

    Returns:
        batch of orthonormal matrices (...xNxN tensor) and optional singular values.
    References:
        - P.H. Schönemann, “A generalized solution of the orthogonal Procrustes problem,” Psychometrika, 31(1), 1966, doi: 10.1007/BF02289451.
        - B.K.P. Horn, “Closed-form solution of absolute orientation using unit quaternions,” Journal of the Optical Society of America A, 4(4), 1987, doi: 10.1364/JOSAA.4.000629.

    Note:
        Both methods are optimal up to numerical precision, but may differ when the nearest rotation is nearly ambiguous
        (relative gap between singular values of :math:`M` below :math:`10^{-7}`), as the closed-form solution does not resolve such gaps.
    """
    assert M.is_floating_point(), "Input should be a floating-point tensor."
    M, batch_shape = roma.internal.flatten_batch_dims(M, -3)
    # Under autocast, the input is cast to float32 (as done by the custom_fwd decorators of _ProcrustesManualDerivatives),
    # so that the gradient contributions of the solver and of the regularization are summed in float32.
    M = roma.internal.autocast_to_float32(M)
    if method == "auto":
        horn_applicable = (
            force_rotation and M.shape[-2:] == (3, 3) and not return_singular_values and _supports_float64(M.device)
        )
        method = "horn" if horn_applicable else "svd"
    if method == "horn":
        if not (force_rotation and M.shape[-2:] == (3, 3)):
            raise ValueError("method='horn' is only applicable to 3x3 matrices with force_rotation=True.")
        if return_singular_values:
            raise ValueError("method='horn' does not compute singular values.")
        if not _supports_float64(M.device):
            raise ValueError("method='horn' requires float64 support, which is not available on this device.")
        R = _horn_procrustes(M, gradient_eps)
    elif method == "svd":
        R, DS = _procrustes_apply(M, force_rotation, gradient_eps)[:2]
    else:
        raise ValueError(f"Unknown method '{method}', expected 'auto', 'svd' or 'horn'.")
    if regularization != 0.0:
        R = _procrustes_regularization_apply(M, R, regularization)
    R = roma.internal.unflatten_batch_dims(R, batch_shape)
    if not return_singular_values:
        return R
    else:
        DS = roma.internal.unflatten_batch_dims(DS, batch_shape)
        return R, DS


def special_procrustes(
    M, regularization=0.0, gradient_eps=1e-5, return_singular_values: bool = False, method: str = "auto"
):
    r"""
    Returns the rotation matrix :math:`R` minimizing Frobenius norm :math:`\| M - R \|_F`.

    Args:
        M (...xNxN tensor): batch of square matrices.
        regularization (float >= 0): weight of a regularization term added to the gradient during backpropagation.
            Using this regularization is equivalent to adding a term :math:`regularization * \| M - R \|_F^2` to the training loss function.
            It only affects backpropagation through the output matrix (not through the optional singular values).
            Forward-mode differentiation (jvp) returns the true directional derivative and ignores it.
        gradient_eps (float > 0): damping of the derivative when the solution is ill-conditioned.
            The derivative involves the inverse of pairwise sums :math:`t` of (signed) singular values of :math:`M`, which vanish when the solution is not unique:
            :math:`1 / t` is replaced by the damped pseudo-inverse :math:`t / (t^2 + gradient\_eps^2)`, bounded by :math:`1 / (2\, gradient\_eps)`.
        return_singular_values (bool): if True, also returns the singular values of the SVD decomposition with sign flipping (... tensor), for advanced users.
            Not compatible with ``method="horn"``.
        method (str): algorithm used to solve the problem, see :func:`~roma.mappings.procrustes`.
            ``"svd"`` (any dimension), ``"horn"`` (closed-form solution for 3x3 matrices, scaling better with the batch size)
            or ``"auto"`` (default: ``"horn"`` whenever applicable, ``"svd"`` otherwise).
    Returns:
        batch of rotation matrices (...xNxN tensor) and optional singular values.

    Note:
        Both methods are optimal up to numerical precision, but may differ when the nearest rotation is nearly ambiguous
        (relative gap between singular values of :math:`M` below :math:`10^{-7}`), as the closed-form solution does not resolve such gaps.
    """
    return procrustes(M, True, regularization, gradient_eps, return_singular_values, method)


def procrustes_naive(M, force_rotation: bool = False, return_singular_values: bool = False):
    r"""
    Implementation of :func:`~roma.mappings.procrustes` relying on default backward pass of autograd and SVD decomposition.
    Could be slightly less stable than :func:`~roma.mappings.procrustes`.
    """
    M, batch_shape = roma.internal.flatten_batch_dims(M, -3)
    assert M.dim() == 3 and M.shape[1] == M.shape[2], "Input should be a BxDxD batch of matrices."
    U, D, Vt = torch.linalg.svd(M)
    # D is sorted in descending order
    if force_rotation:
        # We flip the smallest singular value to ensure getting a rotation matrix
        with torch.no_grad():
            flip = torch.det(U) * torch.det(Vt) < 0
        # This is needed to avoid a runtime error "one of the variables needed for gradient computation has been modified by an inplace operation"
        SVt = Vt.clone() if torch.is_grad_enabled() else Vt
        SVt[flip, -1, :] *= -1
        R = U @ SVt
    else:
        R = U @ Vt
    R = roma.internal.unflatten_batch_dims(R, batch_shape)
    if not return_singular_values:
        return R
    else:
        DS = D.clone()
        if force_rotation:
            DS[flip, -1] *= -1
        del D
        DS = roma.internal.unflatten_batch_dims(DS, batch_shape)
        return R, DS


def special_procrustes_naive(M, return_singular_values: bool = False):
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
    assert M.dim() == 3
    x = M[:, :, 0]
    y = M[:, :, 1]
    x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), epsilon)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), epsilon)
    z = torch.cross(x, y, dim=-1)
    R = torch.stack((x, y, z), dim=-1)
    return roma.internal.unflatten_batch_dims(R, batch_shape)


def symmatrix_to_projective_point(A, gradient_eps=1e-5, method: str = "auto"):
    r"""
    Converts a 4x4 symmetric matrix A into a projective point represented by a unit vector :math:`q` minimizing :math:`q^T A q`,
    i.e. a unit eigenvector of A associated to its smallest eigenvalue.

    Args:
        A (...x4x4 tensor): batch of symmetric matrices. Symmetry is assumed but not checked: for a non-symmetric input,
            the result depends on the method (``"eigh"`` only considers its lower triangular part, ``"horn"`` its symmetric part
            :math:`(A + A^T) / 2`). With both methods, the gradient with respect to A is symmetric.
        gradient_eps (float > 0): damping of the derivative when the solution is ill-conditioned (``method="horn"`` only).
            The derivative involves the inverse of the gaps :math:`c` between the smallest eigenvalue of :math:`A` and the other ones,
            which vanish when the solution is not unique:
            :math:`1 / c` is replaced by the damped pseudo-inverse :math:`c / (c^2 + gradient\_eps^2)`, bounded by :math:`1 / (2\, gradient\_eps)`.
        method (str): algorithm used to compute the eigenvector.

            - ``"eigh"``: batched eigendecomposition (``torch.linalg.eigh``), differentiated by autograd.
              ``gradient_eps`` is ignored, and the derivative is unstable when the smallest eigenvalue is (nearly) repeated.
            - ``"horn"``: closed-form solution shared with :func:`~roma.mappings.special_procrustes`, computed in float64
              (for a 3x3 matrix :math:`M`, the quaternion of the nearest rotation is the eigenvector of a symmetric 4x4 matrix
              associated to its largest eigenvalue, following Horn's absolute orientation formulation, and any symmetric 4x4 matrix
              can be written in this form up to a multiple of the identity).
              Scales better with the batch size than ``"eigh"``, which is faster for small batches but may fail for large ones on GPU
              (batched cuSOLVER eigendecomposition), and more accurate for single precision inputs.
              A :code:`ValueError` is raised on devices without float64 support (MPS).
            - ``"auto"`` (default): ``"horn"`` whenever applicable, ``"eigh"`` otherwise.

    Returns:
        batch of unit vectors :math:`q` (...x4 tensor). The sign of :math:`q` is arbitrary.
    References:
        - V. Peretroukhin, M. Giamou, D. M. Rosen, W. N. Greene, N. Roy, and J. Kelly, “A Smooth Representation of Belief over SO(3) for Deep Rotation Learning with Uncertainty,” 2020, doi: 10.15607/RSS.2020.XVI.007.
        - B.K.P. Horn, “Closed-form solution of absolute orientation using unit quaternions,” Journal of the Optical Society of America A, 4(4), 1987, doi: 10.1364/JOSAA.4.000629.

    Warning:
        When the smallest eigenvalue of A is repeated (up to a relative tolerance of :math:`10^{-7}`), some vector of the corresponding
        eigenspace is returned, and the derivative is damped (see ``gradient_eps``).
    """
    A, batch_shape = roma.internal.flatten_batch_dims(A, end_dim=-3)
    B, D1, D2 = A.shape
    assert (D1, D2) == (4, 4), "Input should be a symmetric Bx4x4 matrix."
    # Under autocast, the input is cast to float32
    A = roma.internal.autocast_to_float32(A)
    if method == "auto":
        method = "horn" if _supports_float64(A.device) else "eigh"
    if method == "eigh":
        # Eigenvalues are sorted in ascending order.
        q = torch.linalg.eigh(A, UPLO="L")[1][:, :, 0]
    elif method == "horn":
        if not _supports_float64(A.device):
            raise ValueError("method='horn' requires float64 support, which is not available on this device.")
        # A written as -N(M) + c I for some 3x3 matrix M (see _horn_matrix_adjoint, which only depends on the symmetric part
        # of its input and annihilates the identity component, so that the gradient with respect to A is symmetric).
        # The eigenvector associated to the smallest eigenvalue of A is the one associated to the largest eigenvalue of N(M).
        M = -_horn_matrix_adjoint(A.double()) / 4
        q = _horn_eigenvector_apply(M, gradient_eps).to(A.dtype)
    else:
        raise ValueError(f"Unknown method '{method}', expected 'auto', 'eigh' or 'horn'.")
    return roma.internal.unflatten_batch_dims(q, batch_shape)


def symmatrixvec_to_unitquat(x, gradient_eps=1e-5, method: str = "auto"):
    r"""
    Converts a 10D vector into a unit quaternion representation.
    Based on :func:`~roma.mappings.symmatrix_to_projective_point`.

    Args:
        x (...x10 tensor): batch of 10D vectors.
        gradient_eps (float > 0): damping of the derivative when the solution is ill-conditioned, see :func:`~roma.mappings.symmatrix_to_projective_point`.
        method (str): algorithm used to compute the eigenvector (``"auto"``, ``"eigh"`` or ``"horn"``), see :func:`~roma.mappings.symmatrix_to_projective_point`.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    Reference:
        V. Peretroukhin, M. Giamou, D. M. Rosen, W. N. Greene, N. Roy, and J. Kelly, “A Smooth Representation of Belief over SO(3) for Deep Rotation Learning with Uncertainty,” 2020, doi: 10.15607/RSS.2020.XVI.007.
    """
    x, batch_shape = roma.internal.flatten_batch_dims(x, end_dim=-2)
    batch_size, D = x.shape
    assert (D) == 10, "Input should be a Bx10 tensor."
    x00, x10, x20, x30, x11, x21, x31, x22, x32, x33 = x.unbind(dim=-1)
    # fmt: off
    A = torch.stack((x00, x10, x20, x30,
                    x10, x11, x21, x31,
                    x20, x21, x22, x32,
                    x30, x31, x32, x33), dim=-1).reshape(-1,4,4)
    # fmt: on
    return roma.internal.unflatten_batch_dims(symmatrix_to_projective_point(A, gradient_eps, method), batch_shape)


def _sinc_threshold(x):
    r"""
    Default magnitude below which :func:`sinc` and :func:`inv_sinc` switch to their Taylor expansion.

    The forward pass is accurate on both sides of the switch, so the threshold is chosen to minimize
    the error of the first derivative, where the two branches fail in opposite directions:

    - Taylor branch: the derivative omits a term in :math:`x^5` (:math:`x^5/840` for sinc,
      :math:`31 x^5/2520` for inv_sinc), i.e. a relative truncation error in :math:`x^4`
      that grows with :math:`x`.
    - Closed-form branch: the derivative, :math:`(x \cos(x) - \sin(x)) / x^2` for sinc and
      :math:`(\sin(x) - x \cos(x)) / \sin(x)^2` for inv_sinc,
      subtracts two nearly equal numbers of magnitude :math:`x` for :math:`x` small,
      each with a rounding error of about :math:`\epsilon x`, to obtain a value of
      only about :math:`\mp x^3/3` (Taylor expansion of :math:`x \cos(x) - \sin(x)`).
      In both cases this yields a relative error of
      about :math:`3 \epsilon / x^2` that grows as :math:`x` shrinks.

    The errors cross when :math:`x^6 = 840 \epsilon` for sinc and :math:`x^6 = (2520/31) \epsilon`
    for inv_sinc. Since both functions share this threshold, we use the smaller crossover,
    :math:`(2520/31)^{1/6} \epsilon^{1/6} \approx 2 \epsilon^{1/6}`.
     This gives roughly 5e-3 for float64 and 0.14 for float32.
    """
    return 2.0 * torch.finfo(x.dtype).eps ** (1 / 6)


def sinc(x, threshold=None):
    r"""
    sinc function :math:`\mathrm{sinc}(x) = \sin(x) / x`.

    Args:
        x (... tensor): input values.
        threshold (float > 0 or None): magnitude below which a Taylor expansion is used.
            If None, a default value depending on the floating point type of ``x`` is used,
            chosen to minimize the error of the first derivative.
    Returns:
        batch of sinc values (... tensor).
    """
    if threshold is None:
        threshold = _sinc_threshold(x)
    mask = torch.abs(x) < threshold
    # Only replace the denominator in the unused small-angle branch.
    denominator = torch.where(mask, torch.ones_like(x), x)
    return torch.where(mask, 1 - x**2 / 6 + x**4 / 120, torch.sin(x) / denominator)


def inv_sinc(x, threshold=None):
    r"""
    Inverse of the sinc function :math:`\mathrm{inv\_sinc}(x) = x / \sin(x)`.

    Args:
        x (... tensor): input values.
        threshold (float > 0 or None): magnitude below which a Taylor expansion is used.
            If None, a default value depending on the floating point type of ``x`` is used,
            chosen to minimize the error of the first derivative.
    Returns:
        batch of inv_sinc values (... tensor).
    """
    if threshold is None:
        threshold = _sinc_threshold(x)
    mask = torch.abs(x) < threshold
    # sin(x) can be small outside the Taylor region, e.g. near pi for long arcs.
    denominator = torch.where(mask, torch.ones_like(x), torch.sin(x))
    return torch.where(mask, 1 + x**2 / 6 + 7 * x**4 / 360, x / denominator)


def rotvec_to_unitquat(rotvec):
    r"""
    Converts rotation vector into unit quaternion representation.

    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    rotvec, batch_shape = roma.internal.flatten_batch_dims(rotvec, end_dim=-2)
    _, D = rotvec.shape
    assert D == 3, "Input should be a Bx3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L621

    norms = torch.norm(rotvec, dim=-1)
    scale = sinc(norms / 2) / 2.0
    quat = torch.cat((scale[:, None] * rotvec, torch.cos(norms / 2)[:, None]), dim=-1)
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
        sign = (quat[:, 3] > 0).to(quat.dtype) * 2.0 - 1.0
        quat = quat * sign[:, None]
    half_angle = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    scale = 2.0 * inv_sinc(half_angle)
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
    # fmt: off
    return torch.stack((x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),
                    2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw),
                    2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2), dim=-1).reshape(quat.shape[:-1] + (3, 3))
    # fmt: on


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
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

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
    assert D == 3, "Input should be a Bx3 tensor."

    # Rotation angle
    theta = torch.norm(rotvec, dim=-1)
    is_angle_small = theta < epsilon

    # Rodrigues formula for angles that are not small.
    # Note: we use clamping to avoid non finite values for small angles
    # (torch.where produces nan gradients in such case).
    axis = rotvec / torch.clamp_min(theta[..., None], epsilon)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    xs = kx * sin_theta
    ys = ky * sin_theta
    zs = kz * sin_theta
    xyc = kx * ky * one_minus_cos_theta
    xzc = kx * kz * one_minus_cos_theta
    yzc = ky * kz * one_minus_cos_theta
    xxc = kx**2 * one_minus_cos_theta
    yyc = ky**2 * one_minus_cos_theta
    zzc = kz**2 * one_minus_cos_theta
    # fmt: off
    R_rodrigues = torch.stack([1 - yyc - zzc, xyc - zs, xzc + ys,
                     xyc + zs, 1 - xxc - zzc, -xs + yzc,
                     xzc - ys, xs + yzc, 1 -xxc - yyc], dim=-1).reshape(-1, 3, 3)
    # fmt: on

    # For small angles, use a first order approximation
    xs, ys, zs = rotvec[:, 0], rotvec[:, 1], rotvec[:, 2]
    one = torch.ones_like(xs)
    # fmt: off
    R_first_order = torch.stack([one, -zs, ys,
                                 zs, one, -xs,
                                 -ys, xs, one], dim=-1).reshape(-1, 3, 3)
    # fmt: on
    # Select the appropriate expression
    R = torch.where(is_angle_small[:, None, None], R_first_order, R_rodrigues)
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
    return torch.cat((xyzw[..., -1, None], xyzw[..., :-1]), dim=-1)


def quat_wxyz_to_xyzw(wxyz):
    r"""
    Convert quaternion from WXYZ to XYZW convention.

    Args:
        wxyz (...x4 tensor, WXYZ convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    assert wxyz.shape[-1] == 4
    return torch.cat((wxyz[..., 1:], wxyz[..., 0, None]), dim=-1)
