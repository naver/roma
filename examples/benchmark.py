# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.
 
import torch
import roma
import numpy as np

torch.manual_seed(0)
device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = False

n =1000
M = torch.randn((n, 3, 3), device = device)
M_with_grad = M.clone().detach().requires_grad_()

N = torch.randn((n, 10), device = device)
N_with_grad = N.clone().detach().requires_grad_()

repeat = 10
inner_repeat = 10


Mtarget = torch.randn((n, 3, 3), device = device)

def special_procrustes(x):
    return roma.special_procrustes(x)

def special_procrustes_naive(x):
    return roma.special_procrustes_naive(x)


def special_gramschmidt(x):
    return roma.special_gramschmidt(x)



def symmatrix(x):
    return roma.unitquat_to_rotmat(roma.symmatrixvec_to_unitquat(x))


def backward(func):
    if M_with_grad.grad is not None:
        M_with_grad.grad.data.zero_()
    R = func(M_with_grad)
    loss = torch.mean((R - Mtarget)**2)
    loss.backward()

def profile(func, label):
    print("Profiling", label)
    
    # Dry run
    func()
    
    durations = []
    for i in range(repeat + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(inner_repeat):
            func()
        end.record()
    
        # Waits for everything to finish running
        torch.cuda.synchronize()
        
        # Ignore the first run
        durations.append(start.elapsed_time(end))
            
    # Print info
    print("-Mean (ms):", np.mean(durations))
    print("-Median (ms)", np.median(durations))
    print("-Min (ms)", np.min(durations))
    print("-Max (ms)", np.max(durations))


profile(lambda : special_procrustes(M), "special_procrustes")
profile(lambda : special_procrustes_naive(M), "special_procrustes_naive")
profile(lambda : special_gramschmidt(M), "special_gramschmidt")
profile(lambda : symmatrix(N), "symmatrix")

profile(lambda : backward(special_procrustes), "special_procrustes_backward")
profile(lambda : backward(special_gramschmidt), "special_gramschmidt_backward")

# We repeat it just to be sure
profile(lambda : special_procrustes(M), "special_procrustes")
profile(lambda : special_procrustes_naive(M), "special_procrustes_naive")
profile(lambda : special_gramschmidt(M), "special_gramschmidt")

profile(lambda : backward(special_procrustes), "special_procrustes_backward")
profile(lambda : backward(special_gramschmidt), "special_gramschmidt_backward")
