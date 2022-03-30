import torch, roma
R_gt = roma.random_rotmat()
t_gt = torch.randn(1, 3)
src = torch.randn(100, 3) # source points / vectors
target = src @ R_gt.T # target vectors
R_predicted = roma.rigid_vectors_registration(src, target)
print(f"R_gt\n{R_gt}")
print(f"R_predicted\n{R_predicted}")

target = src @ R_gt.T + t_gt # target points
R_predicted, t_predicted = roma.rigid_points_registration(src, target)
print(f"R_gt\n{R_gt}")
print(f"R_predicted\n{R_predicted}")
print(f"t_gt\n{t_gt}")
print(f"t_predicted\n{t_predicted}")