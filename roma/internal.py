# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Set of functions for internal module use.
"""

import torch
import contextlib
import functools


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
    Element-wise damped pseudo-inverse x / (x^2 + eps^2), a smooth approximation of 1 / x bounded by 1 / (2 eps).
    """
    return x / (x * x + eps * eps)


# torch.amp.custom_fwd and torch.amp.custom_bwd were added in PyTorch 2.4.
try:
    custom_fwd = torch.amp.custom_fwd
    custom_bwd = torch.amp.custom_bwd
except AttributeError:
    try:
        # PyTorch < 2.4: equivalents exist for CUDA only, and do not support the device_type argument.
        # Should raise an AttributeError exception if undefined.
        torch.cuda.amp.custom_fwd
        torch.cuda.amp.custom_bwd

        def custom_fwd(device_type, cast_inputs=None):
            if device_type == "cuda":
                return torch.cuda.amp.custom_fwd(cast_inputs=cast_inputs)
            elif cast_inputs is None:
                return lambda fwd: fwd
            else:
                # No equivalent for other device types: emulate the behavior of torch.amp.custom_fwd,
                # i.e. when autocast is enabled, run forward with autocast disabled and floating-point input tensors
                # of the considered device type (except float64 ones) cast to cast_inputs.
                def cast(value):
                    if (
                        isinstance(value, torch.Tensor)
                        and value.is_floating_point()
                        and value.device.type == device_type
                        and value.dtype != torch.float64
                    ):
                        return value.to(cast_inputs)
                    return value

                def decorator(fwd):
                    @functools.wraps(fwd)
                    def decorate_fwd(*args, **kwargs):
                        if not autocast_enabled(device_type):
                            return fwd(*args, **kwargs)
                        with autocast_disabled(device_type):
                            return fwd(*map(cast, args), **{key: cast(value) for key, value in kwargs.items()})

                    return decorate_fwd

                return decorator

        def custom_bwd(device_type, **kwargs):
            if device_type == "cuda":
                return torch.cuda.amp.custom_bwd
            else:
                return lambda bwd: bwd
    except AttributeError:
        # Fallback for older versions: no-op decorators.
        def custom_fwd(**kwargs):
            return lambda fwd: fwd

        def custom_bwd(**kwargs):
            return lambda bwd: bwd


def autocast_enabled(device_type):
    r"""
    :meta private:
    Whether autocast is enabled for the considered device type (CPU or CUDA only, consistently with custom_fwd).
    """
    if device_type not in ("cpu", "cuda"):
        return False
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        # PyTorch < 2.4: no device_type argument.
        return torch.is_autocast_cpu_enabled() if device_type == "cpu" else torch.is_autocast_enabled()


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


def autocast_to_float32(x):
    r"""
    :meta private:
    Under autocast, casts a tensor to float32 unless it is float64, as autocast does for linear algebra operations.
    Subsequent computations then return float32 outputs, and during backpropagation the gradient is computed in float32
    before a single rounding to the input dtype.
    """
    if autocast_enabled(x.device.type) and x.dtype != torch.float64:
        return x.float()
    return x
