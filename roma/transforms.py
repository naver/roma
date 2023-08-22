# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.
"""
Spatial transformations parameterized by rotation matrices, unit quaternions and more.

Warning:
    For efficiency reasons, transformation classes do not copy data. Be careful if you intend to use some in-place data modifications.

Example of use:

.. literalinclude :: ../../examples/snippets/transforms.py
    :language: python
"""
import torch
import roma
 
class Linear:
    """
    A linear transformation parameterized by a matrix :math:`M \in \mathcal{M}_{D,D}(\mathbb{R})`,
    transforming a point :math:`x \in \mathbb{R}^D` into :math:`M x`.

    :var linear: (...xDxD tensor): batch of matrices specifying the transformations considered.
    """
    def __init__(self, linear):
        self.linear = linear

    def linear_compose(self, other):
        """
        Compose the linear part of two transformations.

        Args:
            other: an other transformation of same type.
        Returns:
            a tensor representing the composed transformation.
        """
        return torch.einsum("...ik, ...kj -> ...ij", self.linear, other.linear)
    
    def linear_inverse(self):
        """
        Returns:
            The inverse of the linear transformation.
        """
        return torch.inverse(self.linear)
    
    def linear_apply(self, v):
        """
        Transforms a tensor of vector coordinates.

        Args:
            v (...xD tensor): tensor of vector coordinates to transform.

        Returns:
            The transformed vector coordinates.

        See note in :func:`~roma.transforms.Linear.apply()` regarding broadcasting.
        """
        return torch.einsum("...ik, ...k -> ...i", self.linear, v)

    def linear_normalize(self):
        return self.linear
    
    def compose(self, other):
        """
        Compose a transformation with the current one.

        Args:
            other: an other transformation of same type.

        Returns:
            The resulting transformation.
        """
        return type(self)(self.linear_compose(other))
    
    def inverse(self):
        """
        Returns:
            The inverse transformation.
        """
        return type(self)(self.linear_inverse())
    
    def apply(self, v):
        """
        Transforms a tensor of points coordinates.

        Args:
            v (...xD tensor): tensor of point coordinates to transform.

        Returns:
            The transformed point coordinates.

        Note:
            The batch shape of :code:`v` should be broadcastable with the batch shape of the transformation.
            For example, one can transform 100 points by the same 3x3 linear transformation using:
            :code:`roma.Linear(torch.randn(3,3))[None].apply(torch.randn(100,3))`.
        """
        return self.linear_apply(v)
    
    def normalize(self):
        """
        Returns:
            Copy of the transformation, normalized to ensure the class properties
            (for example to ensure that a :class:`Rotation` object is an actual rotation).
        """
        return type(self)(self.linear_normalize())
    
    def __matmul__(self, other):
        """
        Overloading of the `@` operator for composition.
        """
        return self.compose(other)
    
    def __getitem__(self, args):
        """
        Slicing operator, for convenience.
        """
        return type(self)(self.linear[args])
    
    def __repr__(self):
        return f"{type(self).__name__}(linear={self.linear.__repr__()})"
    
    def clone(self):
        """
        Returns:
            A copy of the transformation (useful to avoid aliasing issues).
        """
        return type(self)(self.linear.clone())

class Orthonormal(Linear):
    """
    An orthogonal transformation represented by an orthonormal matrix :math:`M \in \mathcal{M}_{D,D}(\mathbb{R})`,
    transforming a point :math:`x \in \mathbb{R}^D` into :math:`M x`.

    :var linear: (...xDxD tensor): batch of matrices :math:`M` specifying the transformations considered.
    """
    def __init__(self, linear):
        super().__init__(linear)

    def linear_inverse(self):
        return self.linear.transpose(-1,-2)

    def linear_normalize(self):
        """
        Returns:
            Linear transformation normalized to an orthonormal matrix (...xDxD tensor).
        """        
        return roma.mappings.procrustes(self.linear)
    
class Rotation(Orthonormal):
    """
    A rotation represented by a rotation matrix :math:`R \in \mathcal{M}_{D,D}(\mathbb{R})`,
    transforming a point :math:`x \in \mathbb{R}^D` into :math:`R x`.

    :var linear: (...xDxD tensor): batch of matrices :math:`R` defining the rotation.
    """
    def __init__(self, linear):
        super().__init__(linear)

    def linear_normalize(self):
        """
        Returns:
            Linear transformation normalized to a rotation matrix (...xDxD tensor).
        """
        return roma.mappings.special_procrustes(self.linear)
    
class RotationUnitQuat(Linear):
    """
    A 3D rotation represented by a unit quaternion.
    
    :var linear: (...x4 tensor, XYZW convention): batch of unit quaternions defining the rotation.

    Warning:
        Quaternions are assumed to be of unit norm, for all internal operations.
        Use :func:`roma.transforms.RotationUnitQuat.normalize()` if needed.
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
        """
        Returns:
            Normalized unit quaternion (...x4 tensor).
        """
        unitquat = self.linear / torch.norm(self.linear, dim=-1, keepdim=True)
        return unitquat

class _BaseAffine:
    """
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
        """
        Slicing operator, for convenience.
        """
        return type(self)(self.linear[args], self.translation[args])    
    
    def __len__(self):
        return len(self.linear)
    
    def __repr__(self):
        return f"{type(self).__name__}(linear={self.linear.__repr__()}, translation={self.translation.__repr__()})"

    def as_tuple(self):
        """
        Returns:
            a tuple of tensors containing the linear and translation parts of the transformation respectively.
        """
        return self.linear, self.translation
    
    def clone(self):
        """
        Returns:
            A copy of the transformation (useful to avoid aliasing issues).
        """
        return type(self)(self.linear.clone(), self.translation.clone())
    

    def to_homogeneous(self, output=None):
        """
        Args:
            output (...x(D+1)x(D+1) tensor or None): optional tensor in which to store the result.

        Returns:
            A tensor of homogeneous matrices representing the transformation, normalized with a last row equal to (0,...,0,1) (...x(D+1)x(D+1) tensor).
        """
        batch_shape, D = self.translation.shape[:-1], self.translation.shape[-1]
        H = D + 1
        output_shape = batch_shape + (H,H)
        if output is None:
            output = torch.empty(output_shape, device=self.translation.device, dtype=self.translation.dtype)
        else:
            assert output.shape == output_shape
        output[...,:D,:D] = self.linear
        output[...,:D,D] = self.translation
        output[...,D,:D] = 0.0
        output[...,D,D] = 1.0
        return output

    @classmethod
    def from_homogeneous(class_object, matrix):
        """
        Instantiate a new transformation from an input homogeneous (D+1)x(D+1) matrix.

        Args:
            matrix (...x(D+1)x(D+1) tensor): tensor of transformations expressed in homogeneous coordinates, normalized with a last row equal to (0,...,0,1).

        Returns:
            The corresponding transformation.

        Warning:
            - The input matrix is assumed to be normalized and to satisfy the properties of the transformation. No checks are performed.
            - The resulting transformation may consist in views of the input matrix. Use the :code:`clone()` method if you intend to modify data in-place. 

        """
        H1, H2 = matrix.shape[-2:]
        assert H1 == H2
        D = H1 - 1
        linear = matrix[...,:D, :D]
        translation = matrix[...,:D, D]
        return class_object(linear, translation)
                                       
class Affine(_BaseAffine, Linear):
    """
    An affine transformation represented by a linear and a translation part.

    :var linear: (...xDxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        assert translation.shape[-1] == linear.shape[-1], "Incompatible linear and translation dimensions."
        assert len(linear.shape[:-2]) == len(translation.shape[:-1]), "Batch dimensions should be broadcastable."
        _BaseAffine.__init__(self, linear, translation)


class Isometry(Affine, Orthonormal):
    """
    An isometric transformation represented by an orthonormal and a translation part.

    :var linear: (...xDxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        Affine.__init__(self, linear, translation)


class Rigid(Affine, Rotation):
    """
    A rigid transformation represented by an rotation and a translation part.

    :var linear: (...xDxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        Affine.__init__(self, linear, translation)       

class RigidUnitQuat(_BaseAffine, RotationUnitQuat):
    """
    A rigid transformation represented by a unit quaternion and a translation part.

    :var linear: (...x4 tensor): batch of unit quaternions defining the rotation.
    :var translation: (...x3 tensor): batch of matrices specifying the translation part.    

    Warning:
        Quaternions are assumed to be of unit norm, for all internal operations.
        Use the :code:`normalize()` method if needed.
    """
    def __init__(self, linear, translation):
        assert linear.shape[-1] == 4 and translation.shape[-1] == 3, "Expecting respectively a ...x4 quaternion vector and a ...x3 translation vector"
        assert len(linear.shape[:-1]) == len(translation.shape[:-1]), "Batch dimensions should be broadcastable."
        _BaseAffine.__init__(self, linear, translation)

    def to_homogeneous(self, output=None):
        """
        Args:
            output (...x(D+1)x(D+1) tensor or None): tensor in which to store the result (optional).

        Returns:
            A tensor of homogeneous matrices representing the transformation, normalized with a last row equal to (0,...,0,1) (...x(D+1)x(D+1) tensor).
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
        """
        Instantiate a new transformation from an input homogeneous (D+1)x(D+1) matrix.

        Args:
            matrix (...x(D+1)x(D+1) tensor): tensor of transformations expressed in homogeneous coordinates, normalized with a last row equal to (0,...,0,1).

        Returns:
            The corresponding transformation.

        Warning:
            - The input matrix is not tested to ensure that it satisfies the required properties of the transformation.
            - Components of the resulting transformation may consist in views of the input matrix. Be careful if you intend to modify it in-place.
            
        """
        H1, H2 = matrix.shape[-2:]
        assert H1 == H2
        D = H1 - 1
        linear = roma.rotmat_to_unitquat(matrix[...,:D, :D])
        translation = matrix[...,:D, D]
        return RigidUnitQuat(linear, translation)