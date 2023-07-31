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
            Current transformation normalized to ensure the properties of its class (e.g. ensuring that :class:`Rotation` is an actual rotation).
        """
        return type(self)(self.linear_normalize())
    
    def __matmul__(self, other):
        """
        Overloading of the `@` operator for composition.
        """
        return self.compose(other)
    
    def __getitem__(self, args):
        """
        Convenience slicing function.
        """
        return type(self)(self.linear[args])
    
    def __repr__(self):
        return f"{type(self).__name__}(linear={self.linear.__repr__()})"
    
    def clone(self):
        """
        Return a copy of the transformation, with cloned data.
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
            Linear transformation normalized to ensure that it is an orthonormal matrix (...xDxD tensor).
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
            Linear transformation normalized to ensure that it is a rotation matrix (...xDxD tensor).
        """
        return roma.mappings.special_procrustes(self.linear)
    
class RotationUnitQuat(Linear):
    """
    A 3D rotation represented by a unit quaternion.
    
    :var linear: (...x4 tensor): batch of unit quaternions defining the rotation.

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
        Convenience slicing operation
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
        return type(self)(self.linear.clone(), self.translation.clone())
                                       
class Affine(_BaseAffine, Linear):
    """
    An affine transformation represented by a linear and a translation part.

    :var linear: (...xDxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        _BaseAffine.__init__(self, linear, translation)  

class Isometry(_BaseAffine, Orthonormal):
    """
    An isometric transformation represented by an orthonormal and a translation part.

    :var linear: (...xDxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        _BaseAffine.__init__(self, linear, translation)

class Rigid(_BaseAffine, Rotation):
    """
    A rigid transformation represented by an rotation and a translation part.

    :var linear: (...xDxD tensor): batch of matrices specifying the linear part.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.
    """
    def __init__(self, linear, translation):
        _BaseAffine.__init__(self, linear, translation)

class RigidUnitQuat(_BaseAffine, RotationUnitQuat):
    """
    A rigid transformation represented by a unit quaternion and a translation part.

    :var linear: (...x4 tensor): batch of unit quaternions defining the rotation.
    :var translation: (...xD tensor): batch of matrices specifying the translation part.    

    Warning:
        Quaternions are assumed to be of unit norm, for all internal operations.
        Use :func:`roma.transforms.RotationUnitQuat.normalize()` if needed.
    """
    def __init__(self, linear, translation):
        assert linear.shape[-1] == 4 and translation.shape[-1] == 3, "Expecting respectively a 4D quaternion vector and a 3D translation vector"
        assert len(linear.shape[:-1]) == len(translation.shape[:-1]), "Batch dimensions should at least be broadcastable."
        _BaseAffine.__init__(self, linear, translation)