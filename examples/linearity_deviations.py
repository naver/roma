# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
"""
Plot the deviation of various differentiable 3D rotation mappings with respect to their tangent plane, for different step sizes.
"""

import torch
import numpy as np
import collections
import roma
import matplotlib.pyplot as plt
from tqdm import tqdm

# Generate random data
n =10000

torch.manual_seed(666)
vecs1 = torch.nn.functional.normalize(torch.randn(n, 3, 1), dim=1)
vecs2 = torch.nn.functional.normalize(torch.randn(n, 1, 3), dim=2)

def compute_deviations(func, shape, epsilon, sigma = 1.0):
    """
    Compute an estimation of the 'linearity' of a mapping operator f.
    """
    deviations = []
    for i in range(n):
        M = sigma * torch.randn(shape)
        M.requires_grad = True
        vec1 = vecs1[i]
        vec2 = vecs2[i]
        R = func(M).squeeze(dim=0)
        val = (vec2 @ R  @ vec1)
        val.backward()
        if True:
            alpha = epsilon     
        else:
            alpha = epsilon / torch.mean(M.grad * M.grad)
        Rbis = func(M.data - alpha * M.grad).squeeze(dim=0)
        valbis = (vec2 @ Rbis  @ vec1)

        expected  = val.data - epsilon * torch.sum(M.grad * M.grad)
        deviation = (valbis - expected).item()
            
        deviations.append(deviation)
    deviations = np.array(deviations)
    return deviations


if True:
    # Pretty print for LaTeX output
    W = 3.5    # Figure width in inches, approximately A4-width - 2*1.25in margin
    plt.rcParams.update({
        'figure.figsize': (W, W/(1.61)),     # aspect ratio
        'font.size' : 7,                   # Set font size to 11pt
        'axes.labelsize': 7,               # -> axis labels
        'legend.fontsize': 7,              # -> legends
        'font.family': 'DejaVu Sans',
        'text.usetex': True,
        'text.latex.preamble': (            # LaTeX preamble
            r'\usepackage{times}'
        )
    })

def unnormalized_quaternion_to_rotation_matrix(quat):
    quat = torch.nn.functional.normalize(quat)
    return roma.unitquat_to_rotmat(quat)

def angleaxis_to_small_rotation(x):
    # Dirty hack to test something
    norm = torch.norm(x, dim=-1)
    x = torch.tanh(norm) * np.pi/2 * x / (norm + 1e-6)
    return roma.rotvec_to_rotmat(x)
    
def symmatrix_to_rotation_matrix(x):
    quat = roma.symmatrixvec_to_unitquat(x)
    R = roma.unitquat_to_rotmat(quat)
    return R


epsilons = np.linspace(0, 0.5, 6)
deviations = collections.defaultdict(lambda : [])
for epsilon in tqdm(epsilons):
    deviations['Quaternion'].append(compute_deviations(unnormalized_quaternion_to_rotation_matrix, (1, 4), epsilon))
    deviations['6D'].append(compute_deviations(roma.special_gramschmidt, (1, 3, 2), epsilon))
    deviations['Procrustes'].append(compute_deviations(roma.special_procrustes, (1, 3, 3), epsilon))
    deviations['Rotation vector (small angle)'].append(compute_deviations(angleaxis_to_small_rotation, (1, 3), epsilon))
    deviations['SymMatrix'].append(compute_deviations(symmatrix_to_rotation_matrix, (1, 10), epsilon))
        
keys = list(deviations.keys())
for key in keys:
    deviations[key] = np.asarray(deviations[key])
    
    
plt.figure(0)
plt.clf()

colors = {
        'Rotation vector (small angle)': 'C3',
        'Angle-Axis (large angle)': 'C5',
        'Quaternion': 'C0',
        '6D': 'C1',
        'Procrustes': 'C2',
        'SymMatrix': 'C4'}

for key in keys:
    if True:
        absolute_deviations = np.abs(deviations[key])
    else:
        absolute_deviations = deviations[key]
    if True:
        thresholds = np.percentile(absolute_deviations, (25, 75),  axis=1)
        plt.fill_between(epsilons, thresholds[0], thresholds[1], facecolor=colors[key], alpha=0.2)
        plt.plot(epsilons, np.median(absolute_deviations, axis=-1), c=colors[key], label=key)
        plt.ylabel('Absolute deviation')
    else:
        plt.plot(epsilons, np.linalg.norm(absolute_deviations, axis=-1), c=colors[key], label=key)
        plt.ylabel('RMS deviation')
plt.legend(loc='upper left')

plt.xlim(0, max(epsilons))
plt.xlabel('Step size')
plt.grid('on')
plt.tight_layout()
plt.savefig("linearity_deviations.svg")
plt.savefig("linearity_deviations.png")