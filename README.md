# *RoMa*: A lightweight library to deal with 3D rotations in PyTorch.
[![Unit tests](https://github.com/naver/roma/actions/workflows/main.yml/badge.svg)](https://github.com/naver/roma/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/roma.svg)](https://badge.fury.io/py/roma)
[![Downloads](https://pepy.tech/badge/roma)](https://pepy.tech/project/roma)

*RoMa* (which stands for Rotation Manipulation) provides differentiable mappings between 3D rotation representations, mappings from Euclidean to rotation space, and various utilities related to rotations.

It is implemented in PyTorch and aims to be an easy-to-use and reasonably efficient toolbox for Machine Learning and gradient-based optimization.

## Documentation
Latest documentation is available here: https://naver.github.io/roma/.

Below are some examples of use of *RoMa*:
```python
import torch
import roma

# Arbitrary numbers of batch dimensions are supported, for convenience.
batch_shape = (2, 3)

# Conversion between rotation representations
rotvec = torch.randn(batch_shape + (3,))
q = roma.rotvec_to_unitquat(rotvec)
R = roma.unitquat_to_rotmat(q)
Rbis = roma.rotvec_to_rotmat(rotvec)

# Regression of a rotation from an arbitrary input:
# Special Procrustes orthonormalization of a 3x3 matrix
R1 = roma.special_procrustes(torch.randn(batch_shape + (3, 3)))
# Conversion from a 6D representation
R2 = roma.special_gramschmidt(torch.randn(batch_shape + (3, 2)))
# From the 10 coefficients of a 4x4 symmetric matrix
q = roma.symmatrixvec_to_unitquat(torch.randn(batch_shape + (10,)))

# Metrics on the rotation space
R1, R2 = roma.random_rotmat(size=5), roma.random_rotmat(size=5)
theta = roma.utils.rotmat_geodesic_distance(R1, R2)
cos_theta = roma.utils.rotmat_cosine_angle(R1.transpose(-2, -1) @ R2)

# Operations on quaternions
q_identity = roma.quat_product(roma.quat_conjugation(q), q)

# Spherical interpolation between rotation vectors (shortest path)
rotvec0, rotvec1 = torch.randn(batch_shape + (3,)), torch.randn(batch_shape + (3,))
rotvec_interpolated = roma.rotvec_slerp(rotvec0, rotvec1, steps)
```


## Installation
The easiest way to install *RoMa* is to use pip:
```
pip install roma
```
We also recommend installing [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)
to achieve significant speed-up with the _special_procrustes_ function on a CUDA GPU.

Alternatively one can install the latest version of *RoMa* directly from the source repository:
```
pip install git+https://github.com/naver/roma
```

## License
*RoMa*, Copyright (c) 2021 NAVER Corp., is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license (see [license](https://github.com/naver/roma/blob/master/LICENSE)).

Bits of code were adapted from SciPy. Documentation is generated, distributed and displayed with the support of Sphinx and other materials (see [notice](https://github.com/naver/roma/blob/master/NOTICE)).

## References
For a more in-depth discussion regarding differentiable mappings on the rotation space, please refer to:
- [__Romain Brégier, Deep Regression on Manifolds: a 3D Rotation Case Study.__ in _2021 International Conference on 3D Vision (3DV)_, 2021.](https://arxiv.org/abs/2103.16317)

Please cite this work in your publications:
```
@inproceedings{bregier2021deepregression,
	title={Deep Regression on Manifolds: a {3D} Rotation Case Study},
	author={Br{\'e}gier, Romain},
	journal={2021 International Conference on 3D Vision (3DV)},
	year={2021}
}
```

