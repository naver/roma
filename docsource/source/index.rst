RoMa: A lightweight library to deal with 3D rotations in PyTorch.
=================================================================

*RoMa* (which stands for Rotation Manipulation) provides differentiable mappings between 3D rotation representations, mappings from Euclidean to rotation space, and various utilities related to rotations.
It is implemented in PyTorch and aims to be an easy-to-use and reasonably efficient toolbox for Machine Learning and gradient-based optimization.

Installation
------------
The easiest way to install *RoMa* is to use pip::

    pip install roma

We also recommend installing `torch-batch-svd <https://github.com/KinglittleQ/torch-batch-svd>`_
to achieve significant speed-up with :func:`~roma.mappings.procrustes` on a CUDA GPU (see section :ref:`Why a new library?`).
You can check that this module is properly loaded using the function :func:`~roma.utils.is_torch_batch_svd_available()`.

Alternatively one can install the latest version of *RoMa* directly from the source repository::

    pip install git+https://github.com/naver/roma

..  contents::
        :depth: 3     

Main features
=============

Supported rotation representations
----------------------------------
Rotation vector (rotvec)
    - Encoded using a ...x3 tensor.
    - 3D vector *angle* * *axis* represents a rotation of angle *angle* (expressed in radians) around a unit 3D *axis*.

Unit quaternion (unitquat)
    - Encoded as ...x4 tensor.
    
    ..  note::
        - We use XYZW quaternion convention, *i.e.* components of quaternion :math:`x i + y j + z k + w` 
          are represented by the 4D vector :math:`(x,y,z,w)`.
        - We assume unit quaternions to be of unit length, and do not perform implicit normalization.
Rotation matrix (rotmat)
    - Encoded as a ...xDxD tensor (D=3 for 3D rotations).
    - We use column-vector convention, i.e. :math:`R X` is the transformation of a 1xD vector :math:`X`  by a rotation matrix :math:`R`.

Euler and Tait-Bryan angles are *NOT* currently supported.
    This is because of the many different existing conventions, and because of the limited interest of such parameterization for numerical applications.


Mappings between rotation representations
-----------------------------------------
*RoMa* provides functions to convert between rotation representations.

Example mapping a batch of rotation vectors into corresponding unit quaternions:

.. literalinclude :: ../../examples/snippets/rotvec_to_unitquat.py
    :language: python

Mappings from Euclidean to 3D rotation space
--------------------------------------------
Mapping an arbitrary tensor to a valid rotation can be useful *e.g.* for Machine Learning applications.
While rotation vectors or Euler angles can be used for such purpose, they suffer from various shortcomings, and we therefore provide the following alternative mappings:

:func:`~roma.mappings.special_gramschmidt`
    Mapping from a 3x2 tensor to 3x3 rotation matrix, using special Gram-Schmidt orthonormalization (*6D* representation, popularized by `Zhou et al. <https://arxiv.org/abs/1812.07035>`_).
:func:`~roma.mappings.special_procrustes`
    Mapping from a nxn arbitrary matrix to a nxn rotation matrix, using special orthogonal Procrustes orthonormalization.
:func:`~roma.mappings.symmatrixvec_to_unitquat`
    Mapping from a 10D vector to an antipodal pair of quaternion through eigenvector decomposition of a 4x4 symmetric matrix, proposed by `Peretroukhin et al. <https://arxiv.org/abs/2006.01031>`_.
    
For general purpose applications, we recommend the use of :func:`~roma.mappings.special_procrustes` which projects an arbitrary square matrix onto the closest matrix of the rotation space,
considering Frobenius norm. Please refer to this `paper <https://arxiv.org/abs/2103.16317>`_ for more insights.

Example mapping random 3x3 matrices to valid rotation matrices:

.. literalinclude :: ../../examples/snippets/special_procrustes.py
    :language: python


Support for an arbitrary number of batch dimensions
----------------------------------------------------
For convenience, functions accept an arbitrary number of batch dimensions:

.. literalinclude :: ../../examples/snippets/batch_dims.py
    :language: python

Quaternion operations
---------------------

.. literalinclude :: ../../examples/snippets/quat_operations.py
    :language: python

Rotation composition and inverse
---------------------------------
Example using rotation vector representation:

.. literalinclude :: ../../examples/snippets/composition_inverse.py
    :language: python

Rotation metrics
----------------
*RoMa* implements some usual similarity measures over the 3D rotation space:

.. literalinclude :: ../../examples/snippets/metrics.py
    :language: python


Weighted rotation averaging
---------------------------
:func:`~roma.mappings.special_procrustes` can be used to easily average rotations:

.. literalinclude :: ../../examples/snippets/rotation_averaging.py
    :language: python

To be precise, it consists in the Fréchet mean considering the chordal distance.

.. note::
    The same average could be performed using quaternion representation and *symmatrix* mapping (slower batched implementation on GPU).

Rigid registration
-------------------
:func:`~roma.utils.rigid_points_registration` and :func:`~roma.utils.rigid_vectors_registration` enable to align ordered sets of points/vectors:

.. literalinclude :: ../../examples/snippets/rigid_registration.py
    :language: python

Spherical linear interpolation (SLERP)
--------------------------------------

SLERP between batches of unit quaternions:

.. literalinclude :: ../../examples/snippets/unitquat_slerp.py
    :language: python


SLERP between rotation vectors (shortest path interpolation):

.. literalinclude :: ../../examples/snippets/rotvec_slerp.py
    :language: python

Why a new library?
==================
We could not find a PyTorch library satisfying our needs, so we built our own.
    We wanted a reliable, easy-to-use and efficient toolbox to deal with rotation representations in PyTorch.
    While Kornia provides some utility functions to deal with 3D rotations, it included several major bugs at the time of writting (early 2021) (see e.g. https://github.com/kornia/kornia/issues/723 or https://github.com/kornia/kornia/issues/317).

Care for numerical precision
    *RoMa* is implemented with numerical precision in mind, e.g. with a special handling of small angle rotation vectors
    or through the choice of appropriate algorithms.

    As an example, below is plotted a function that takes as input an angle :math:`\theta`,
    produces a rotation matrix :math:`R_z(\theta)` of angle :math:`\theta` and estimates its geodesic distance with respect to the identity matrix, using 32 bits floating point arithmetic.
    We observe that :func:`~roma.utils.rotmat_geodesic_distance` is much more precise for this task than an other implementation
    often found in academic code: :func:`~roma.utils.rotmat_geodesic_distance_naive`. 
    Backward pass through :func:`~roma.utils.rotmat_geodesic_distance_naive` leads to unstable gradient estimations and produces *Not-a-Number* values for small angles,
    whereas :func:`~roma.utils.rotmat_geodesic_distance_naive` is well-behaved, and returns *Not-a-Number* only for 0.0 angle where gradient is mathematically undefined.

    .. image:: rotmat_geodesic_distance_zero.svg
    
    .. image:: rotmat_geodesic_distance_grads_zero.svg


Computation efficiency
    *RoMa* favors code clarity, but aims to be reasonably efficient. 
    In particular, for Procrustes orthonormalization it can use on NVidia GPUs a batched SVD decomposition
    that provides orders of magnitude speed-ups for large batch sizes compared to vanilla ``torch.svd()``
    (tested with random 3x3 matrices, PyTorch 1.7, a NVidia Tesla T4 GPU and CUDA 11.0).

    .. image:: special_procrustes_benchmark.svg

Syntactic sugar
    *RoMa* aims to be easy-to-use with a simple syntax, and supports of an arbitrary number of batch dimensions to let its users focus on their applications.    

API Documentation
=================

Mappings
----------
.. automodule:: roma.mappings
   :members:

Utils
----------
.. automodule:: roma.utils
   :members:

Advanced
===============

Running unit tests
-------------------
from source repository::

    python -m unittest

Building Sphinx documentation
-----------------------------
From source repository::

    ./build_doc.sh   

License
=======
*RoMa*, Copyright (c) 2021 NAVER Corp., is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license (see `license <https://github.com/naver/roma/blob/master/LICENSE>`_).

Bits of code were adapted from SciPy. Documentation is generated, distributed and displayed with the support of Sphinx and other materials (see `notice <https://github.com/naver/roma/blob/master/NOTICE>`_).


References
==========
For a more in-depth discussion regarding differentiable mappings on the rotation space, please refer to:

**Romain Brégier, Deep Regression on Manifolds: A 3D Rotation Case Study.** in 2021 International Conference on 3D Vision (3DV), 2021. (https://arxiv.org/abs/2103.16317).

Please cite this work in your publications:
::

    @inproceedings{bregier2021deepregression,
        title={Deep Regression on Manifolds: a {3D} Rotation Case Study},
        author={Br{\'e}gier, Romain},
        journal={2021 International Conference on 3D Vision (3DV)},
        year={2021}
    }

Changelog
==========
Version 1.3.1:
    - Removed spurious code in :func:`~roma.mappings.procrustes()`.
    - Replaced warning about missing 'torch_batch_svd' module by a test function: :func:`~roma.utils.is_torch_batch_svd_available()`.
    - Added :func:`roma.utils.unitquat_slerp_fast()`.
    - Improved documentation and tests.
Version 1.3.0:
    - Added :func:`roma.utils.quat_action()`.
    - Change of underlying algorithm for :func:`~roma.utils.random_unitquat()` to avoid potential divisions by 0.
    - Fix of :func:`roma.utils.unitquat_slerp()` which was always performing interpolation along the shortest arc regardless of the value of the ``shortest_path`` argument (renamed ``shortest_arc`` in the new version).
Version 1.2.7:
    - Fix of :func:`~roma.internal.unflatten_batch_dims()` to ensure compatibility with PyTorch 1.6.0.
    - Fix of :func:`~symmatrixvec_to_unitquat()` that was not producing a lower triangular matrix.
Version 1.2.6:
    - Added an optional `regularization` argument to :func:`~roma.mappings.special_procrustes()`.
    - Added an optional `clamping` argument to :func:`~roma.utils.rotmat_geodesic_distance()`.
    - Fix: :func:`~roma.mappings.rotvec_to_rotmat` no longer produces nonfinite gradient for null rotation vectors.
Version 1.2.5:
    - Added an optional `regularization` argument for Procrustes orthonormalization.
    - Added a rigid registration example in the documentation.
Version 1.2.4:
    - Procrustes: automatic fallback to vanilla SVD decomposition for large dimensions.
Version 1.2.3:
    - Improved support for double precision tensors.
Version 1.2.2:
    - Added :func:`~roma.utils.rigid_points_registration()` and :func:`~roma.utils.rigid_vectors_registration()`.
    - Added :func:`~roma.utils.rotmat_slerp()`.
    - Circumvented a deprecation warning with :func:`torch.symeig()` when using recent PyTorch versions.
Version 1.2.1:
    - Open-source release.