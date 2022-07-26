# *RoMa*: A lightweight library to deal with 3D rotations in PyTorch.

*RoMa* (which stands for Rotation Manipulation) provides differentiable mappings between 3D rotation representations, mappings from Euclidean to rotation space, and various utilities related to rotations.

It is implemented in PyTorch and aims to be an easy-to-use and reasonably efficient toolbox for Machine Learning and gradient-based optimization.

## Documentation
Latest documentation is available here: https://naver.github.io/roma/.



We test *RoMa* with PyTorch versions 1.6.0 and latest: [![Unit tests](https://github.com/naver/roma/actions/workflows/main.yml/badge.svg)](https://github.com/naver/roma/actions/workflows/main.yml)

## Installation
The easiest way to install *RoMa* is to use pip:
```
pip install roma
```
We also recommend installing [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)
to achieve significant speed-up with _special_procrustes_ function on a CUDA GPU.

Alternatively one can install the latest version of *RoMa* directly from the source repository:
```
pip install git+https://github.com/naver/roma
```
or include the source repository (https://github.com/naver/roma) as a Git submodule.

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

