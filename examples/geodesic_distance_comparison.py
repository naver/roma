# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Study of numerical stability of rotmat_geodesic_distance_naive and rotmat_geodesic_distance.
Plot value and gradient of a function consisting given an angle theta in:
- generating a rotation matrix of angle theta
- estimating its geodesic distance with respect to the identity rotation (i.e. the absolute value of theta).
"""

from numpy.lib.utils import safe_eval
import torch
import roma
import numpy as np
import matplotlib.pyplot as plt

# Numerical errors are much more important when considering float32 numbers
dtype = torch.float32

# Enforce 0 and pi values to be considered
for label, input_angles in [("zero", np.concatenate((np.linspace(-0.005, 0.0, 1000, endpoint=True), np.linspace(0.0, 0.005, 1000, endpoint=True)))),
                            ("pi", np.concatenate((np.linspace(np.pi-0.005, np.pi, 1000, endpoint=True), np.linspace(np.pi, np.pi+0.005, 1000, endpoint=True))))]:

    gradients = []
    output_angles = []
    gradients_naive = []
    output_angles_naive = []

    for val in input_angles:

        theta = torch.scalar_tensor(val, dtype=dtype, requires_grad=True)

        zero = torch.zeros_like(theta)
        rotvec = torch.stack((zero, zero, theta), dim=-1)

        R = roma.rotvec_to_rotmat(rotvec)
        I = torch.eye(3, dtype=dtype)

        if theta.grad is not None:
            theta.grad.fill_(0)
        output = roma.rotmat_geodesic_distance(I, R)
        output.backward(retain_graph=True)
        output_angles.append(output.item())
        gradients.append(theta.grad.item())

        if theta.grad is not None:
            theta.grad.fill_(0)
        output = roma.rotmat_geodesic_distance_naive(I, R)
        output.backward()
        output_angles_naive.append(output.item())
        gradients_naive.append(theta.grad.item())

    fig, ax = plt.subplots(1)
    fig.patch.set_alpha(0.) # no background
    ax.patch.set_alpha(0.) # no background
    plt.plot(input_angles, np.asarray(output_angles_naive), label='rotmat_geodesic_distance_naive')
    plt.plot(input_angles, np.asarray(output_angles), label='rotmat_geodesic_distance')
    plt.ylabel("Geodesic distance (radians)")
    plt.xlabel("Input angle (radians)")
    plt.legend()
    plt.savefig(f"rotmat_geodesic_distance_{label}.png")
    plt.savefig(f"rotmat_geodesic_distance_{label}.svg")

    fig, ax = plt.subplots(1)
    fig.patch.set_alpha(0.) # no background
    ax.patch.set_alpha(0.) # no background
    plt.plot(input_angles, gradients_naive, c="C00", label='rotmat_geodesic_distance_naive')
    # Plot NaN
    mask = ~np.isfinite(gradients_naive)
    nan_count = np.count_nonzero(mask)
    if nan_count > 0:
        plt.scatter(np.asarray(input_angles)[mask], 0.5 * np.ones(nan_count), marker='x', s=1.0, c="C02", label='rotmat_geodesic_distance_naive NaN')

    plt.plot(input_angles, gradients, c="C01", label='rotmat_geodesic_distance')
    # Plot NaN
    mask = ~np.isfinite(gradients)
    nan_count = np.count_nonzero(mask)
    if nan_count > 0:
        plt.scatter(np.asarray(input_angles)[mask], -0.5 * np.ones(nan_count), marker='x', s=1.0, c="C03", label='rotmat_geodesic_distance NaN')
    plt.ylabel("Geodesic distance derivative")
    plt.xlabel("Input angle (radians)")
    plt.legend()
    plt.savefig(f"rotmat_geodesic_distance_grads_{label}.png")
    plt.savefig(f"rotmat_geodesic_distance_grads_{label}.svg")

