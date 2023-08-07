import torch, roma
# Rigid transformation parameterized by a rotation matrix and a translation vector
T0 = roma.Rigid(linear=roma.random_rotmat(), translation=torch.randn(3))

# Rigid transformations parameterized by a unit quaternion and a translation vector
T1 = roma.RigidUnitQuat(linear=roma.random_unitquat(), translation=torch.randn(3))
T2 = roma.RigidUnitQuat(linear=roma.random_unitquat(), translation=torch.randn(3))

# Inverting and composing transformations
T = (T1.inverse() @ T2)

# Normalization to ensure that T is actually a rigid transformation.
T = T.normalize()

# Direct access to the translation part
T.translation += 0.5

# Transformation of points:
points = torch.randn(100,3)
# Adjusting the shape of T for proper broadcasting.
transformed_points = T[None].apply(points)

# Transformation of vectors:
vectors = torch.randn(10,20,3)
# Adjusting the shape of T for proper broadcasting.
transformed_vectors = T[None,None].linear_apply(vectors)

# Casting the transformation into an homogeneous 4x4 matrix.
M = T.to_homogeneous()