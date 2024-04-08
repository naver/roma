# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import torch
import itertools
import traceback
import sys
import pdb
import functools

def is_close(A, B, eps1 = 1.0, eps2 = 1e-5):
    return (torch.norm(A - B) / (torch.norm(torch.abs(A) + torch.abs(B)) + eps1)) < eps2

def central_difference(func, x, v, eps):
    xp = x + eps * v
    xm = x - eps * v
    fp = func(xp)
    fm = func(xm)
    df = (fp - fm) / (2 * eps)
    return df

def numerical_jacobian(func, x, eps):
    """
    jacobian: a tensor of shape (func(x).shape) times x.shape.
    """
    output = func(x)
    jacobian = torch.zeros(output.shape + x.shape)
    for input_indices in itertools.product(*[range(s) for s in x.shape]):
        v = torch.zeros_like(x)
        v[input_indices] = 1
        jacobian[(...,) + input_indices] = central_difference(func, x, v, eps)
    return jacobian

def automatic_jacobian(func, x):
    """
    jacobian: a tensor of shape (func(x).shape) times x.shape.
    """
    output = func(x)
    jacobian = torch.zeros(output.shape + x.shape)
    for output_indices in itertools.product(*[range(s) for s in output.shape]):
        x = x.data.clone().requires_grad_()
        y = func(x)
        y[output_indices].backward()
        jacobian[output_indices] = x.grad
    return jacobian

def debug_on(*exceptions):
    """
    Useful decorator to launch pdb in case of error in tests.
    """
    if not exceptions:
        exceptions = (AssertionError, )
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info) 
                pdb.post_mortem(info[2])
        return wrapper
    return decorator