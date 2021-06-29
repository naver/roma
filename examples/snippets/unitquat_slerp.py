import torch, roma
q0, q1 = roma.random_unitquat(size=10), roma.random_unitquat(size=10)
steps = torch.linspace(0, 1.0, 5)
q_interpolated = roma.utils.unitquat_slerp(q0, q1, steps)
idx = 1 # Print interpolations for an arbitrary element of the batch
print('q0:\n', q0[idx])
print('q1:\n', q1[idx])
print('q_interpolated:\n', q_interpolated[:,idx])