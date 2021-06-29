import torch, roma
print(roma.rotvec_to_rotmat(torch.randn(3)).shape) # -> torch.Size([3, 3])
print(roma.rotvec_to_rotmat(torch.randn(5, 3)).shape) # -> torch.Size([5, 3, 3])
print(roma.rotvec_to_rotmat(torch.randn(2, 5, 3)).shape) # -> torch.Size([2, 5, 3, 3])