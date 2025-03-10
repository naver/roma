# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Set of functions for internal module use.
"""

import torch

# SVD decomposition
# Default behavior: vanilla svd
_IS_TORCH_BATCH_SVD_AVAILABLE = False
try:
    # Should raise an AttributeError exception if undefined.
    torch.linalg.svd
    def svd(M):
        r"""
        Singular Value Decomposition wrapper.
        
        Args:
            M (BxMxN tensor): batch of real matrices.
        Returns:
            (U,D,V) decomposition, such as :math:`M = U @ diag(D) @ V^T`.
        """
        U, D, Vt = torch.linalg.svd(M)
        return (U, D, Vt.transpose(-2,-1))
except (NameError, AttributeError):
    # deprecated in torch 2.0
    svd = torch.svd

# With PyTorch < 1.8,
# we observed some significant speed-ups using torch_batch_svd (https://github.com/KinglittleQ/torch-batch-svd) instead of torch.svd on NVidia GPUs.
# In more recent versions, this is no longer required (https://github.com/pytorch/pytorch/pull/48436).
_torch_version_major, _torch_version_minor = [int(s) for s in torch.__version__.split(".")[:2]]
if  _torch_version_major == 0 or (_torch_version_major == 1 and _torch_version_minor < 8):
    try:
        import torch_batch_svd
        _IS_TORCH_BATCH_SVD_AVAILABLE = True
        def svd(M):
            r"""
            Singular Value Decomposition wrapper, using efficient batch decomposition on GPU.

            Args:
                M (BxMxN tensor): batch of real matrices.
            Returns:
                (U,D,V) decomposition, such as :math:`M = U @ diag(D) @ V^T`.
            """
            if M.is_cuda and M.shape[1] < 32 and M.shape[2] < 32:
                return torch_batch_svd.svd(M)
            else:
                return torch.svd(M)
    except ModuleNotFoundError:
        pass
del _torch_version_major, _torch_version_minor

def flatten_batch_dims(tensor, end_dim):
    r"""
    :meta private:
    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
    """
    batch_shape = tensor.shape[:end_dim+1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape

def unflatten_batch_dims(tensor, batch_shape):
    r"""
    :meta private:
    Revert flattening of a tensor.
    """
    # Note: alternative to tensor.unflatten(dim=0, sizes=batch_shape) that was not supported by PyTorch 1.6.0.
    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)

def _pseudo_inverse(x, eps):
    r"""
    :meta private:
    Element-wise pseudo inverse.
    """
    inv = 1.0/x
    inv[torch.abs(x) < eps] = 0.0
    return inv    

# Batched eigenvalue decomposition.
# Recent version of PyTorch deprecated the use of torch.symeig.
try:
    torch.linalg.eigh
    def symeig_lower(A):
        r"""
        Batched eigenvalue decomposition. Only the lower part of the matrix is considered.
        """
        return torch.linalg.eigh(A, UPLO='L')
except (NameError, AttributeError):
    # Older PyTorch version
    def symeig_lower(A):
        r"""
        Batched eigenvalue decomposition. Only the lower part of the matrix is considered.
        """
        return torch.symeig(A, upper=False, eigenvectors=True)
    
# L2 normalization
try:
    torch.linalg.norm
    def norm(x, dim=None, keepdim=False):
        return torch.linalg.norm(x, dim=dim, keepdim=keepdim)
except AttributeError:
    # torch.linalg.norm was introduced in PyTorch 1.7, and torch.norm is deprecated.
    def norm(x, dim=None, keepdim=False):
        return torch.norm(x, dim=dim, keepdim=keepdim)
    
try:
    torch.hypot
    hypot = torch.hypot
except AttributeError:
    # torch.hypot is not available in PyTorch 1.6.
    def hypot(x, y):
        return torch.sqrt(torch.square(x) + torch.square(y))
