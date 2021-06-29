import torch, roma
batch_shape = (5,)
M = torch.randn(batch_shape + (3,3))
R = roma.special_procrustes(M)
assert roma.is_rotation_matrix(R, epsilon=1e-5)