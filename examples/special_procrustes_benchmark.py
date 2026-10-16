# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Execution speed benchmark of roma.special_procrustes for 3x3 matrices (closed-form solution)
against the SVD-based implementation roma.special_procrustes_naive, for the forward pass and for the forward+backward passes.
"""

import time
import torch
import roma
import numpy as np

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
batch_sizes = (1, 10, 100, 1000, 10000, 100000, 1000000) if device.type == "cuda" else (1, 10, 100, 1000, 10000, 100000)
torch.manual_seed(0)


def profile(func, repeat=7, inner=5):
    """Median duration (ms) of a call to func."""
    for _ in range(2):
        func()  # Warm-up.
    durations = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(inner):
            func()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1e3 / inner)
    return np.median(durations)


def forward_backward(func, M, G):
    def run():
        M.grad = None
        R = func(M)
        torch.sum(R * G).backward()

    return run


for dtype in (torch.float32, torch.float64):
    print(f"\n{dtype} ({device}): median duration in ms")
    print(f"{'batch size':>12} {'SVD fwd':>12} {'closed fwd':>12} {'SVD fwd+bwd':>12} {'closed fwd+bwd':>15}")
    for n in batch_sizes:
        M = torch.randn((n, 3, 3), dtype=dtype, device=device)
        G = torch.randn((n, 3, 3), dtype=dtype, device=device)
        with torch.no_grad():
            t_svd = profile(lambda: roma.special_procrustes_naive(M))
            t_closed = profile(lambda: roma.special_procrustes(M))
        M_svd = M.clone().requires_grad_()
        M_closed = M.clone().requires_grad_()
        t_svd_bwd = profile(forward_backward(roma.special_procrustes_naive, M_svd, G))
        t_closed_bwd = profile(forward_backward(roma.special_procrustes, M_closed, G))
        print(f"{n:>12} {t_svd:>12.3f} {t_closed:>12.3f} {t_svd_bwd:>12.3f} {t_closed_bwd:>15.3f}")
