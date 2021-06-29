import torch, roma
rotvecs = [roma.random_rotvec() for _ in range(3)] # Random rotation vectors
r = roma.rotvec_composition(rotvecs) # Composition of an arbitrary number of rotations
rinv = roma.rotvec_inverse(r) # Rotation vector corresponding to the inverse rotation