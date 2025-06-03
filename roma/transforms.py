# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
r"""
Spatial transformations parameterized by rotation matrices, unit quaternions and more.

Example of use
~~~~~~~~~~~~~~

.. literalinclude :: ../../examples/snippets/transforms.py
    :language: python

.. _apply-transformation:

Applying a transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

When applying a transformation to a set of points of coordinates :code:`v`,
The batch shape of :code:`v` should be broadcastable with the batch shape of the transformation.

For example, one can sample a unique random rigid 3D transformation and use it to transform 100 random 3D points as follows:

.. code-block:: python

    roma.Rigid(roma.random_rotmat(), torch.randn(3))[None].apply(torch.randn(100,3))

To apply a different transformation to each point, one could use instead:

.. code-block:: python

    roma.Rigid(roma.random_rotmat(100), torch.randn(100,3)).apply(torch.randn(100,3))

.. _aliasing_issues:    

Aliasing issues
~~~~~~~~~~~~~~~

.. warning::

    For efficiency reasons, transformation objects do not copy input data. Be careful if you intend to do some in-place data modifications, and use the :code:`clone()` method when required.

"""
import torch
import roma
import warnings
 
class Linear:
    r"""
    A linear transformation parameterized by a matrix :math:`M \in \mathcal{M}_{D,C}(\mathbb{R})`,
    transforming a point :math:`x \in \mathbb{R}^C` into :math:`M x`.

    :var linear: (...xDxC tensor): batch of matrices specifying the transformations considered.
    """
    def __init__(self, linear):
        self.linear = linear

    def linear_compose(self, other):
        r"""
        Compose the linear part of two transformations.

        Args:
            other: an other transformation of same type.
        Returns:
            a tensor representing the composed transformation.
        """
        assert len(self.linear.shape) == len(other.linear.shape), "Expecting the same number of batch dimensions for the two transformations."
        return torch.einsum("...ik, ...kj -> ...ij", self.linear, other.linear)
    
    def linear_inverse(self):
        r"""
        Returns:
            The inverse of the linear transformation, when applicable.
        """
        return torch.inverse(self.linear)
    
    def linear_apply(self, v):
        r"""
        Transforms a tensor of vector coordinates.

        Args:
            v (...xD tensor): tensor of vector coordinates to transform.

        Returns:
            The transformed vector coordinates.

        See note in :func:`~roma.transforms.Linear.apply()` regarding broadcasting.
        """
        assert len(self.linear.shape) == len(v.shape) + 1, "Expecting the same number of batch dimensions for the transformation and the vector."
        return torch.einsum("...ik, ...k -> ...i", self.linear, v)

    def linear_normalize(self):
        return self.linear
    
    def compose(self, other):
        r"""
        Compose a transformation with the current one.

        Args:
            other: an other transformation of same type.

        Returns:
            The resulting transformation.
        """
        return type(self)(self.linear_compose(other))
    
    def inverse(self):
        r"""
        Returns:
            The inverse transformation, when applicable.
        """
        return type(self)(self.linear_inverse())
    
    def apply(self, v):
        r"""
        Transforms a tensor of points coordinates. See :ref:`apply-transformation`.

        Args:
            v (...xD tensor): tensor of point coordinates to transform.

        Returns:
            The transformed point coordinates.
        """
        return self.linear_apply(v)
    
    def normalize(self):
        r"""
        Returns:
            Copy of the transformation, normalized to ensure the class properties
            (for example to ensure that a :class:`Rotation` object is an actual rotation).
        """
        return type(self)(self.linear_normalize())
    
    def __matmul__(self, other):
        r"""
        Overloading of the `@` operator for composition.
        """
        return self.compose(other)
    
    def __getitem__(self, args):
        r"""
        Slicing operator, for convenience.
        """
        return type(self)(self.linear[args])
    
    def __repr__(self):
        return f"{type(self).__name__}(linear={self.linear.__repr__()})"
    
    def clone(self):
        r"""
        Returns:
            A copy of the transformation (useful to avoid aliasing issues).
        """
        return type(self)(self.linear.clone())

class Orthonormal(Linear):
    r"""
    An orthogonal transformation represented by an orthonormal matrix :math:`M \in \mathcal{M}_{D,D}(\mathbb{R})`,
    transforming a point :math:`x \in \mathbb{R}^D` into :math:`M x`.

    :var linear: (...xDxD tensor): batch of matrices :math:`M` specifying the transformations considered.
    """
    def __init__(self, linear):
        assert linear.shape[-1] == linear.shape[-2], "Expecting same dimensions for input and output."
        super().__init__(linear)

    def linear_inverse(self):
        return self.linear.transpose(-1,-2)

    def linear_normalize(self):
        r"""
        Returns:
            Linear transformation normalized to an orthonormal matrix (...xDxD tensor).
        """        
        return roma.mappings.procrustes(self.linear)
    
class Rotation(Orthonormal):
    r"""
    A rotation represented by a rotation matrix :math:`R \in \mathcal{M}_{D,D}(\mathbb{R})`,
    transforming a point :math:`x \in \mathbb{R}^D` into :math:`R x`.

    :var linear: (...xDxD tensor): batch of matrices :math:`R` defining the rotation.
    """
    def __init__(self, linear):
        super().__init__(linear)

    def linear_normalize(self):
        r"""
        Returns:
            Linear transformation normalized to a rotation matrix (...xDxD tensor).
        """
        return roma.mappings.special_procrustes(self.linear)
    
class RotationUnitQuat(Linear):
    r"""
    A 3D rotation represented by a unit quaternion.
    
    :var linear: (...x4 tensor, XYZW convention): batch of unit quaternions defining the rotation.

    Note:
        Quaternions are assumed to be of unit norm, for all internal operations.
        Use :code:`normalize()` if needed.
    """
    def __init__(self, linear):
        self.linear = linear

    def linear_compose(self, other):
        return roma.utils.quat_product(self.linear, other.linear)
    
    def linear_apply(self, v):
        return roma.utils.quat_action(self.linear, v, is_normalized=True)

    def linear_inverse(self):
        return roma.utils.quat_conjugation(self.linear)
                                       
    def linear_normalize(self):
        r"""
        Returns:
            Normalized unit quaternion (...x4 tensor).
        """
        unitquat = self.linear / torch.norm(self.linear, dim=-1, keepdim=True)
        return unitquat

class _BaseAffine:
    r"""
    Abstract base class representing an affinity transformation.
    """
    def __init__(self, linear, translation):
        self.linear = linear
        self.translation = translation

    def compose(self, other):
        linear = self.linear_compose(other)
        translation = self.linear_apply(other.translation) + self.translation
        return type(self)(linear, translation)
    
    def inverse(self):
        inv_linear = self.linear_inverse()
        res = type(self)(inv_linear, self.translation)
        res.translation = -res.linear_apply(self.translation)
        return res
    
    def apply(self, v):
        return self.linear_apply(v) + self.translation
    
    def normalize(self):
        linear = self.linear_normalize()
        return type(self)(linear, self.translation)
    
    def __getitem__(self, args):
        r"""
        Slicing operator, for convenience.
        """
        return type(self)(self.linear[args], self.translation[args])
    
    def squeeze(self, dim):
        r"""
        Return a view of the transformation in which a batch dimension equal to 1 has been squeezed.

        :var dim: positive integer: The dimension to squeeze.
        """
        assert dim >= 0, "Only positive dimensions are supported to avoid ambiguities."
        assert self.linear.shape[dim] == self.translation.shape[dim] == 1, ""
        return type(self)(self.linear.squeeze(dim), self.translation.squeeze(dim))
    
    def __len__(self):
        return len(self.linear)
    
    def __repr__(self):
        return f"{type(self).__name__}(linear={self.linear.__repr__()}, translation={self.translation.__repr__()})"

    def as_tuple(self):
        r"""
        Returns:
            a tuple of tensors containing the linear and translation parts of the transformation respectively.
        """
        return self.linear, self.translation
    
    def clone(self):
        r"""
        Returns:
            A copy of the transformation (useful to avoid aliasing issues).
        """
        return type(self)(self.linear.clone(), self.translation.clone())
    

    def to_homogeneous(self, output=None):
        r"""
        Args:
            output (...x(D+1)x(C+1) tensor or None): optional tensor in which to store the result.

        Returns:
            A ...x(D+1)x(C+1) tensor of homogeneous matrices representing the transformation, normalized with a last row equal to (0,...,0,1).
        """
        batch_shape, D, C = self.linear.shape[:-2], self.linear.shape[-2], self.linear.shape[-1]
        output_shape = batch_shape + (D+1,C+1)
        if output is None:
            output = torch.empty(output_shape, device=self.translation.device, dtype=self.translation.dtype)
        else:
            assert output.shape == output_shape
        output[...,:D,:C] = self.linear
        output[...,:D,C] = self.translation
        output[...,D,:C] = 0.0
        output[...,D,C] = 1.0
        return output

    @classmethod
    def from_homogeneous(class_object, matrix):
        r"""
        Instantiate a new transformation from an input homogeneous (D+1)x(C+1) matrix.
        The input matrix is assumed to be normalized and to satisfy the properties of the transformation. No checks are performed.

        Args:
            matrix (...x(D+1)x(C+1) tensor): tensor of transformations expressed in homogeneous coordinates, normalized with a last row equal to (0,...,0,1).

        Returns:
            The corresponding transformation.
        """
        H1, H2 = matrix.shape[-2:]
        D = H1 - 1
        C = H2 - 1
        linear = matrix[...,:D, :C]
        translation = matrix[...,:D, C]
        return class_object(linear, translation)
                                       
class Affine(_BaseAffine, Linear):
    r"""
    An affine transformation represented by a linear and a translation part.

    :var linear: (...xCxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor or None): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        if translation is None:
            # Set a default null translation.
            translation = torch.zeros(linear.shape[:-2] + (linear.shape[-1],), dtype=linear.dtype, device=linear.device)
        assert translation.shape[-1] == linear.shape[-2], "Incompatible linear and translation dimensions."
        assert len(linear.shape[:-2]) == len(translation.shape[:-1]), "Batch dimensions should be broadcastable."
        _BaseAffine.__init__(self, linear, translation)


class Isometry(Affine, Orthonormal):
    r"""
    An isometric transformation represented by an orthonormal and a translation part.

    :var linear: (...xDxD tensor or None): batch of matrices specifying the linear part.
    :var translation: (...xD tensor or None): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        if linear is None:
            # Set a default identity linear part.
            batch_dims = translation.shape[:-1]
            D = translation.shape[-1]
            if len(batch_dims) == 0:
                linear = torch.eye(D, dtype=translation.dtype, device=translation.device)
            else:
                linear = torch.eye(D, dtype=translation.dtype, device=translation.device)[[None] * len(batch_dims)].expand(batch_dims + (-1,-1))
        else:
            assert linear.shape[-1] == linear.shape[-2], "Expecting same dimensions for input and output."
        Affine.__init__(self, linear, translation)

    @classmethod
    def Identity(cls, *args, **kwargs):
        warnings.warn(
            "Isometry.Identity is deprecated and will be removed in a future release."
            "Use Isometry.identity instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return cls.identity(*args, **kwargs)

    @classmethod
    def identity(cls, dim, batch_shape=tuple(), dtype=torch.float32, device=None):
        r"""
        Return a default identity transformation.

        :var dim: (strictly positive integer): dimension of the space in which the transformation operates (e.g. `dim=3` for 3D transformations).
        :var batch_shape: (tuple): batch dimensions considered.
        """
        translation = torch.zeros(batch_shape + (dim,), dtype=dtype, device=device)
        return cls(linear=None, translation=translation)
    
    @classmethod
    def identity_like(cls, other):
        r"""
        Return an identity transformation with the same batch shape, type and device as the input transformation.

        :var other: (Isometry): input transformation.
        """
        return cls.identity(dim=other.translation.shape[-1], batch_shape=other.translation.shape[:-1], dtype=other.linear.dtype, device=other.linear.device)

class Rigid(Isometry, Rotation):
    r"""
    A rigid transformation represented by an rotation and a translation part.

    :var linear: (...xDxD tensor or None): batch of matrices specifying the linear part.
    :var translation: (...xD tensor or None): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        Isometry.__init__(self, linear, translation)

    def to_rigidunitquat(self):
        r"""
        Returns the corresponding RigidUnitQuat transformation.

        Note:
            Original and resulting transformations share the same translation tensor. Be careful in case of in-place modifications.
        """
        return RigidUnitQuat(roma.rotmat_to_unitquat(self.linear), self.translation)

class RigidUnitQuat(_BaseAffine, RotationUnitQuat):
    r"""
    A rigid transformation represented by a unit quaternion and a translation part.

    :var linear: (...x4 tensor): batch of unit quaternions defining the rotation.
    :var translation: (...x3 tensor): batch of matrices specifying the translation part.    

    Note:
        Quaternions are assumed to be of unit norm, for all internal operations.
        Use the :code:`normalize()` method if needed.
    """
    def __init__(self, linear, translation):
        assert linear.shape[-1] == 4 and translation.shape[-1] == 3, "Expecting respectively a ...x4 quaternion vector and a ...x3 translation vector"
        assert len(linear.shape[:-1]) == len(translation.shape[:-1]), "Batch dimensions should be broadcastable."
        _BaseAffine.__init__(self, linear, translation)

    def to_homogeneous(self, output=None):
        r"""
        Args:
            output (...x4x4 tensor or None): tensor in which to store the result (optional).

        Returns:
            A ...x4x4 tensor of homogeneous matrices representing the transformation, normalized with a last row equal to (0,...,0,1).
        """
        batch_shape = self.translation.shape[:-1]
        output_shape = batch_shape + (4,4)
        if output is None:
            output = torch.zeros(output_shape, device=self.translation.device, dtype=self.translation.dtype)
        else:
            assert output_shape == output_shape
        output[...,:3,:3] = roma.unitquat_to_rotmat(self.linear)
        output[...,:3,3] = self.translation
        # Set the homogeneous line
        # Note: this is redundant with zeros initialization, but .
        output[...,3,:3] = 0.0
        output[...,3,3] = 1.0
        return output

    @staticmethod
    def from_homogeneous(matrix):
        r"""
        Instantiate a new transformation from an input homogeneous (D+1)x(D+1) matrix.

        Args:
            matrix (...x(D+1)x(D+1) tensor): tensor of transformations expressed in homogeneous coordinates, normalized with a last row equal to (0,...,0,1).

        Returns:
            The corresponding transformation.

        Note:
            - The input matrix is not tested to ensure that it satisfies the required properties of the transformation.
            - Components of the resulting transformation may consist in views of the input matrix. Be careful if you intend to modify it in-place.
            
        """
        H1, H2 = matrix.shape[-2:]
        assert H1 == H2
        D = H1 - 1
        linear = roma.rotmat_to_unitquat(matrix[...,:D, :D])
        translation = matrix[...,:D, D]
        return RigidUnitQuat(linear, translation)
    
    def to_rigid(self):
        r"""
        Returns the corresponding Rigid transformation.

        Note:
            Original and resulting transformations share the same translation tensor. Be careful in case of in-place modifications.
        """
        return Rigid(roma.unitquat_to_rotmat(self.linear), self.translation)