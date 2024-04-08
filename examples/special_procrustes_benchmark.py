# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
"""
Plot computation time of special_procrustes using regular torch.svd decomposition versus batch SVD decomposition when available.
Requires a CUDA-enabled GPU.
"""
 
import torch
import roma
import numpy as np
import roma.internal
import matplotlib.pyplot as plt

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
        durations.append(start.elapsed_time(end))            
    # Print info
    print("-Mean (ms):", np.mean(durations))
    print("-Median (ms)", np.median(durations))
    print("-Min (ms)", np.min(durations))
    print("-Max (ms)", np.max(durations))
    return durations

if not roma.is_torch_batch_svd_available():
    print("Error: The 'torch_batch_svd' module should be installed to run this benchmark.")
    exit()


torch.manual_seed(0)
device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = True

repeat = 20
inner_repeat = 10

batch_sizes = np.asarray(np.logspace(start=0, stop=3, num=10, base=10, endpoint=True), dtype=np.int64)#[1, 2, 3, 5]#, 10, 100, 1000]
batch_durations = []
batch_durations_basic = []
means_basic = []
for n in batch_sizes:
    M = torch.randn((n, 3, 3), device = device)
    M_with_grad = M.clone().detach().requires_grad_()

    N = torch.randn((n, 10), device = device)
    N_with_grad = N.clone().detach().requires_grad_()

    Mtarget = torch.randn((n, 3, 3), device = device)

    durations = profile(lambda : roma.special_procrustes(M), "special_procrustes")
    batch_durations.append(durations)

    # Replace the internal SVD with basic torch.svd
    foo = roma.internal.svd
    roma.internal.svd = torch.svd
    durations = profile(lambda : roma.special_procrustes(M), "special_procrustes with basic svd")
    batch_durations_basic.append(durations)
    roma.internal.svd = foo

fig, ax = plt.subplots(1)
fig.patch.set_alpha(0.) # no background
ax.patch.set_alpha(0.) # no background
for label, b_durs in (("Pytorch SVD", batch_durations_basic), ("Batch SVD", batch_durations)):
    means = np.mean(b_durs, axis=1)
    quantiles = np.quantile(b_durs, [0.05, 0.95], axis=1)
    plt.fill_between(batch_sizes,
                    quantiles[0],
                    quantiles[1],
                    alpha = 0.2)
    plt.plot(batch_sizes, means, label = label)

plt.xlabel("Batch size")
plt.ylabel("Computation time (ms)")
plt.title("Special Procrustes (forward pass)")
plt.loglog()

plt.xticks(batch_sizes)
plt.legend()

plt.savefig("special_procrustes_benchmark.png", transparent=True)
plt.savefig("special_procrustes_benchmark.svg", transparent=True)
