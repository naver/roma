import torch, roma
steps = torch.linspace(0, 1.0, 5)
rotvec0, rotvec1 = torch.randn(3), torch.randn(3)
rotvec_interpolated = roma.rotvec_slerp(rotvec0, rotvec1, steps)