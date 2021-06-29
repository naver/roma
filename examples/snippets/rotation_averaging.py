import torch, roma
n = 5
R_i = roma.random_rotmat(n) # Batch of n 3x3 rotation matrices to average
w_i = torch.rand(n) # Weight of each matrix, between 0 and 1
M = torch.sum(w_i[:,None, None] * R_i, dim=0) # 3x3 matrix
R = roma.special_procrustes(M) # weighted average.