# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Benchmark of the two methods available for 3x3 rotations:

- roma.special_procrustes: batched SVD (method="svd") vs closed-form Horn solution (method="horn"),
- roma.symmatrix_to_projective_point: batched eigendecomposition (method="eigh") vs closed-form Horn solution (method="horn").

Reports the median execution time of the forward pass, of the forward+backward passes and of forward-mode differentiation (jvp),
as well as the accuracy of the output and of the gradient of each method with respect to a float64 SVD/eigh reference.
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


benchmarks = {
    "special_procrustes": dict(
        generate=random_matrix,
        function=lambda method: lambda M: roma.special_procrustes(M, method=method),
        methods=("svd", "horn"),  # Reference first.
        distance=roma.rotmat_geodesic_distance,
        invariant=lambda R: R,
    ),
    "symmatrix_to_projective_point": dict(
        generate=random_symmatrix,
        function=lambda method: lambda A: roma.symmatrix_to_projective_point(A, method=method),
        methods=("eigh", "horn"),
        # Unit vectors are defined up to sign, as unit quaternions.
        distance=roma.unitquat_geodesic_distance,
        # Sign-invariant quantity on which the loss is defined for gradient accuracy.
        invariant=lambda q: q[..., :, None] * q[..., None, :],
    ),
}

for name, b in benchmarks.items():
    function, methods = b["function"], b["methods"]
    for dtype in (torch.float32, torch.float64):
        print(f"\n{name}, {dtype} ({device}): median duration in ms")
        modes = ("fwd", "fwd+bwd", "jvp")
        print(f"{'batch size':>12}" + "".join(f"{f'{m} {mode}':>14}" for mode in modes for m in methods))
        for n in batch_sizes:
            x = b["generate"](n, dtype)
            dx = torch.randn_like(x)
            durations = dict.fromkeys(((m, mode) for m in methods for mode in modes), None)
            for method in methods:
                func = function(method)
                try:
                    G = torch.randn_like(func(x))
                    x_grad = x.clone().requires_grad_()

                    def forward_backward():
                        x_grad.grad = None
                        torch.sum(func(x_grad) * G).backward()

                    with torch.no_grad():
                        durations[method, "fwd"] = profile(lambda: func(x))
                    durations[method, "fwd+bwd"] = profile(forward_backward)
                    durations[method, "jvp"] = profile(lambda: torch.func.jvp(func, (x,), (dx,)))
                except RuntimeError:
                    # e.g. batched cuSOLVER eigh fails for large batches (the error type depends on the cuSOLVER version).
                    pass
            print(f"{n:>12}" + "".join(format_result(durations[m, mode], 14, ".3f") for mode in modes for m in methods))

    # Accuracy with respect to the reference method applied in float64: max angular deviation of the output,
    # and relative error (median and max over the batch) of the gradient of a random linear loss.
    n = 10000
    x = b["generate"](n, torch.float64)
    G = torch.randn_like(b["invariant"](function(methods[0])(x)))

    def evaluate(method, dtype):
        x_grad = x.to(dtype).clone().requires_grad_()
        output = function(method)(x_grad)
        torch.sum(b["invariant"](output) * G.to(output.dtype)).backward()
        return output.detach().double(), x_grad.grad.double()

    output_ref, grad_ref = evaluate(methods[0], torch.float64)
    columns = ("angle max", "grad med", "grad max")
    print(f"\n{name}: deviation from {methods[0]} in float64 (batch size {n}, angles in rad, relative gradient errors)")
    print(f"{'dtype':>16}" + "".join(f"{f'{m} {c}':>16}" for m in methods for c in columns))
    for dtype in (torch.float32, torch.float64):
        row = f"{str(dtype):>16}"
        for method in methods:
            output, grad = evaluate(method, dtype)
            relative = (grad - grad_ref).flatten(1).norm(dim=-1) / grad_ref.flatten(1).norm(dim=-1)
            for e in (b["distance"](output, output_ref).max(), relative.median(), relative.max()):
                row += f"{e.item():>16.1e}"
        print(row)
