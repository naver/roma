# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Execution speed benchmark for different mappings on the 3D rotation space.
Requires a CUDA-enabled GPU.
""" 
 
import torch
import roma
import numpy as np

torch.manual_seed(0)
device = torch.device(0)
torch.backends.cudnn.benchmark = False

# Batch size
n =1000
# Input for Procrustes and Gram-Schmidt.
M = torch.randn((n, 3, 3), device = device)
M_with_grad = M.clone().detach().requires_grad_()
# Input for symmatrix.
N = torch.randn((n, 10), device = device)
N_with_grad = N.clone().detach().requires_grad_()
# Input for rotvec
O = torch.randn((n, 3), device = device)
O_with_grad = O.clone().detach().requires_grad_()

# Number of repetitions considered to evaluate computation time.
repeat = 10
inner_repeat = 10

# Target rotations, to define the loss used to benchmark backward pass.
Rtarget = torch.randn((n, 3, 3), device = device)

def symmatrix(x):
    return roma.unitquat_to_rotmat(roma.symmatrixvec_to_unitquat(x))

def profile(func, label):
    print(f"Profiling {label}:")
    print("=================================================")
    # Dry run
    func()
    
    durations = []
    for i in range(repeat + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # Run the function a couple of times
        for i in range(inner_repeat):
            func()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end))
            
    # Print info
    print("-Mean (ms):", np.mean(durations))
    print("-Median (ms)", np.median(durations))
    print("-Min (ms)", np.min(durations))
    print("-Max (ms)", np.max(durations))

profile(lambda : roma.special_procrustes(M), "special_procrustes")
profile(lambda : roma.special_gramschmidt(M), "special_gramschmidt")
profile(lambda : roma.symmatrixvec_to_unitquat(N), "symmatrix")
profile(lambda : roma.rotvec_to_rotmat(O), "rotvec")


def backward(func, input):
    if input.grad is not None:
        input.grad.data.zero_()
    R = func(input)
    loss = torch.mean((R - Rtarget)**2)
    loss.backward()

def backward_symmatrix():
    if N_with_grad.grad is not None:
        N_with_grad.grad.data.zero_()
    R = roma.unitquat_to_rotmat(roma.symmatrixvec_to_unitquat(N_with_grad))
    loss = torch.mean((R - Rtarget)**2)
    loss.backward()

profile(lambda : backward(roma.special_procrustes, M_with_grad), "special_procrustes (forward/backward)")
profile(lambda : backward(roma.special_gramschmidt, M_with_grad), "special_gramschmidt (forward/backward)")
profile(lambda : backward(lambda x: roma.unitquat_to_rotmat(roma.symmatrixvec_to_unitquat(x)), N_with_grad), "symmatrix (forward/backward)")
profile(lambda : backward(roma.rotvec_to_rotmat, O_with_grad), "rotvec (forward/backward)")