# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.
"""
Set of functions for internal module use.
"""

import torch

try:
    import torch_batch_svd
    _fast_gpu_svd = torch_batch_svd.svd
except ModuleNotFoundError:
    if torch.cuda.is_available():
        print("WARNING: torch_batch_svd (https://github.com/KinglittleQ/torch-batch-svd) is not installed and is required for maximum efficiency of special_procrustes. Using torch.svd as a fallback.")
    _fast_gpu_svd = torch.svd

def flatten_batch_dims(tensor, end_dim):
    """
    :meta private:
    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
    """
    batch_shape = tensor.shape[:end_dim+1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape

def unflatten_batch_dims(tensor, batch_shape):
    """
    :meta private:
    Revert flattening of a tensor.
    """
    return tensor.unflatten(dim=0, sizes=batch_shape) if len(batch_shape) > 0 else tensor.squeeze(0)

def _pseudo_inverse(x, eps):
    """
    :meta private:
    Element-wise pseudo inverse.
    """
    inv = 1.0/x
    inv[torch.abs(x) < eps] = 0.0
    return inv    

def svd(M):
    """
    Singular Value Decomposition wrapper, using efficient batch decomposition on GPU when available.

    Args:
        M (BxMxN tensor): batch of real matrices.
    Returns:
        (U,D,V) decomposition, such as :math:`M = U @ diag(D) @ V^T`.
    """
    if M.is_cuda:
        return _fast_gpu_svd(M)
    else:
        return torch.svd(M)