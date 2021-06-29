import torch, roma
R1, R2 = roma.random_rotmat(size=5), roma.random_rotmat(size=5)
theta = roma.rotmat_geodesic_distance(R1, R2) # In radian
cos_theta = roma.rotmat_cosine_angle(R1.transpose(-2, -1) @ R2)