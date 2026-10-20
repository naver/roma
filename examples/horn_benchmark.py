# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Benchmark of the two methods available for 3x3 rotations:

- roma.special_procrustes: batched SVD (method="svd") vs closed-form Horn solution (method="horn"),
- roma.symmatrix_to_projective_point: batched eigendecomposition (method="eigh") vs closed-form Horn solution (method="horn").

Reports the median execution time of the forward pass, of the forward+backward passes and of forward-mode differentiation (jvp),
as well as the maximum angular deviation of each method from a float64 SVD/eigh reference.
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


def format_result(value, width, spec):
    """Formatted value, or 'failed' if not available (e.g. batched cuSOLVER eigh fails for large batches)."""
    return f"{'failed':>{width}}" if value is None else f"{value:>{width}{spec}}"


def random_matrix(n, dtype):
    return torch.randn((n, 3, 3), dtype=dtype, device=device)


def random_symmatrix(n, dtype):
    A = torch.randn((n, 4, 4), dtype=dtype, device=device)
    return A + A.transpose(-1, -2)


# name: (input generator, function of the method, methods (reference first), angular distance)
benchmarks = {
    "special_procrustes": (
        random_matrix,
        lambda method: lambda M: roma.special_procrustes(M, method=method),
        ("svd", "horn"),
        roma.rotmat_geodesic_distance,
    ),
    "symmatrix_to_projective_point": (
        random_symmatrix,
        lambda method: lambda A: roma.symmatrix_to_projective_point(A, method=method),
        ("eigh", "horn"),
        # Unit vectors are defined up to sign, as unit quaternions.
        roma.unitquat_geodesic_distance,
    ),
}

for name, (generate, function, methods, distance) in benchmarks.items():
    for dtype in (torch.float32, torch.float64):
        print(f"\n{name}, {dtype} ({device}): median duration in ms")
        modes = ("fwd", "fwd+bwd", "jvp")
        print(f"{'batch size':>12}" + "".join(f"{f'{m} {mode}':>14}" for mode in modes for m in methods))
        for n in batch_sizes:
            x = generate(n, dtype)
            dx = torch.randn_like(x)
            durations = dict.fromkeys(((m, mode) for m in methods for mode in modes), None)
            for method in methods:
                func = function(method)
                try:
                    G = torch.randn_like(func(x))
                except torch.linalg.LinAlgError:
                    continue
                x_grad = x.clone().requires_grad_()

                def forward_backward():
                    x_grad.grad = None
                    torch.sum(func(x_grad) * G).backward()

                with torch.no_grad():
                    durations[method, "fwd"] = profile(lambda: func(x))
                durations[method, "fwd+bwd"] = profile(forward_backward)
                durations[method, "jvp"] = profile(lambda: torch.func.jvp(func, (x,), (dx,)))
            print(f"{n:>12}" + "".join(format_result(durations[m, mode], 14, ".3f") for mode in modes for m in methods))

    # Accuracy, with respect to the reference method applied in float64.
    n = 10000
    print(f"\n{name}: max angular deviation (rad) from {methods[0]} in float64 (batch size {n})")
    print(f"{'dtype':>16}" + "".join(f"{m:>12}" for m in methods))
    x = generate(n, torch.float64)
    reference = function(methods[0])(x)
    for dtype in (torch.float32, torch.float64):
        errors = [distance(function(m)(x.to(dtype)).double(), reference).max().item() for m in methods]
        print(f"{str(dtype):>16}" + "".join(f"{e:>12.1e}" for e in errors))
