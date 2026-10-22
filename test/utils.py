# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import torch

# (major, minor) version of PyTorch, to skip tests relying on features unsupported by older versions.
torch_version = tuple(int(s) for s in torch.__version__.split(".")[:2])


def is_close(A, B, eps1=1.0, eps2=1e-5):
    return (torch.norm(A - B) / (torch.norm(torch.abs(A) + torch.abs(B)) + eps1)) < eps2


def central_difference(func, x, v, eps):
    xp = x + eps * v
    xm = x - eps * v
    fp = func(xp)
    fm = func(xm)
    df = (fp - fm) / (2 * eps)
    return df
