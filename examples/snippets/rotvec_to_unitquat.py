import torch, roma
batch_shape = (3, 2)
rotvec = torch.randn(batch_shape + (3,))
q = roma.rotvec_to_unitquat(rotvec)