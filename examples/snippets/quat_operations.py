import torch, roma
q = torch.randn(4) # Random unnormalized quaternion
qconv = roma.quat_conjugation(q) # Quaternion conjugation
qinv = roma.quat_inverse(q) # Quaternion inverse
print(roma.quat_product(q, qinv)) # -> [0,0,0,1] identity quaternion