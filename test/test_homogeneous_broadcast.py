import unittest

import torch

import roma


class TestHomogeneousBroadcast(unittest.TestCase):
    def test_matrix_transform(self):
        for Type in [roma.Affine, roma.Isometry, roma.Rigid]:
            for linear_shape, translation_shape in [((2, 1), (1, 3)), ((1, 3), (2, 1))]:
                with self.subTest(Type=Type, linear_shape=linear_shape):
                    linear = torch.eye(3, dtype=torch.float64).expand(linear_shape + (3, 3)).clone().requires_grad_()
                    translation = (
                        torch.arange(3.0, dtype=torch.float64).expand(translation_shape + (3,)).clone().requires_grad_()
                    )
                    transform = Type(linear, translation)
                    points = torch.arange(18.0, dtype=torch.float64).reshape(2, 3, 3)
                    before = (linear.detach().clone(), translation.detach().clone())

                    matrix = transform.to_homogeneous()
                    self.assertEqual(matrix.shape, (2, 3, 4, 4))
                    extended = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
                    actual = (matrix @ extended[..., None]).squeeze(-1)
                    expected = transform.apply(points)
                    torch.testing.assert_close(actual[..., :3], expected)
                    torch.testing.assert_close(actual[..., 3], torch.ones(2, 3, dtype=torch.float64))
                    actual_grads = torch.autograd.grad(actual.sum(), (linear, translation), retain_graph=True)
                    expected_grads = torch.autograd.grad(expected.sum(), (linear, translation))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad)
                    torch.testing.assert_close(linear, before[0])
                    torch.testing.assert_close(translation, before[1])
                    restored = Type.from_homogeneous(matrix)
                    torch.testing.assert_close(restored.apply(points), expected)

    def test_rectangular_affine(self):
        for output_dim, input_dim in [(2, 3), (3, 2)]:
            with self.subTest(output_dim=output_dim):
                linear = torch.arange(float(3 * output_dim * input_dim)).reshape(1, 3, output_dim, input_dim)
                translation = torch.ones(2, 1, output_dim)
                transform = roma.Affine(linear, translation)
                points = torch.ones(2, 3, input_dim)
                matrix = transform.to_homogeneous()
                self.assertEqual(matrix.shape, (2, 3, output_dim + 1, input_dim + 1))
                extended = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
                torch.testing.assert_close((matrix @ extended[..., None])[..., :output_dim, 0], transform.apply(points))

    def test_unitquat_transform(self):
        for linear_shape, translation_shape in [((2, 1), (1, 3)), ((1, 3), (2, 1))]:
            with self.subTest(linear_shape=linear_shape):
                raw = (
                    torch.tensor([0.1, 0.2, 0.3, 0.9], dtype=torch.float64)
                    .expand(linear_shape + (4,))
                    .clone()
                    .requires_grad_()
                )
                linear = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
                translation = torch.ones(translation_shape + (3,), dtype=torch.float64, requires_grad=True)
                transform = roma.RigidUnitQuat(linear, translation)
                points = torch.arange(18.0, dtype=torch.float64).reshape(2, 3, 3)
                matrix = transform.to_homogeneous()
                self.assertEqual(matrix.shape, (2, 3, 4, 4))
                extended = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
                actual = (matrix @ extended[..., None]).squeeze(-1)[..., :3]
                expected = transform.apply(points)
                torch.testing.assert_close(actual, expected)
                actual_grads = torch.autograd.grad(actual.sum(), (raw, translation), retain_graph=True)
                expected_grads = torch.autograd.grad(expected.sum(), (raw, translation))
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)
                output = torch.empty_like(matrix)
                self.assertIs(transform.to_homogeneous(output), output)
                torch.testing.assert_close(output, matrix)
                restored = roma.RigidUnitQuat.from_homogeneous(matrix)
                torch.testing.assert_close(restored.apply(points), expected)


if __name__ == "__main__":
    unittest.main()
