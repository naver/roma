# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Set of functions for internal module use.
"""

import torch
import contextlib


def flatten_batch_dims(tensor, end_dim):
    r"""
    :meta private:
    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
    """
    batch_shape = tensor.shape[: end_dim + 1]
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
    return torch.where(torch.abs(x) < eps, torch.zeros_like(x), 1.0 / x)


def autocast_disabled(device_type):
    r"""
    :meta private:
    Context manager disabling autocast for the considered device type, if supported.
    Substitute for torch.amp.custom_bwd, which does not support autograd Functions defining a separate setup_context method.
    """
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except RuntimeError:
        # Autocast may not be supported for every device type.
        return contextlib.nullcontext()
