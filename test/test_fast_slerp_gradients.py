import unittest

import torch

import roma


class TestFastSlerpGradients(unittest.TestCase):
    def assert_close(self, actual, expected):
        tolerance = 6e-6 if actual.dtype == torch.float32 else 2e-12
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_coincident_endpoint_jacobians(self):
        for dtype in (torch.float32, torch.float64):
            for shortest_arc in (False, True):
                with self.subTest(dtype=dtype, shortest_arc=shortest_arc):
                    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
                    steps = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=dtype)
                    projector = torch.eye(4, dtype=dtype) - torch.outer(q, q)

                    def function(a, b, t):
                        return roma.unitquat_slerp_fast(a, b, t, shortest_arc=shortest_arc)

                    jac0, jac1, jac_steps = torch.autograd.functional.jacobian(function, (q, q.clone(), steps))
                    self.assert_close(jac0, (1 - steps)[:, None, None] * projector)
                    self.assert_close(jac1, steps[:, None, None] * projector)
                    self.assert_close(jac_steps, torch.zeros((len(steps), 4, len(steps)), dtype=dtype))

    def test_shortest_arc_sign_flip_jacobian(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q = torch.full((4,), 0.5, dtype=dtype)
                step = torch.tensor(0.3, dtype=dtype)
                projector = torch.eye(4, dtype=dtype) - torch.outer(q, q)

                def function(a, b):
                    return roma.unitquat_slerp_fast(a, b, step, shortest_arc=True)

                jac0, jac1 = torch.autograd.functional.jacobian(function, (q, -q))
                self.assert_close(function(q, -q), q)
                self.assert_close(jac0, (1 - step) * projector)
                self.assert_close(jac1, -step * projector)

    def test_mixed_batch_values_and_finite_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shortest_arc in (False, True):
                with self.subTest(dtype=dtype, shortest_arc=shortest_arc):
                    angles = torch.tensor([[0.0, 1e-4], [0.9, -1.1]], dtype=dtype)
                    q0 = torch.zeros((2, 2, 4), dtype=dtype)
                    q0[..., 3] = 1
                    q1 = torch.stack(
                        (
                            torch.zeros_like(angles),
                            torch.zeros_like(angles),
                            torch.sin(angles / 2),
                            torch.cos(angles / 2),
                        ),
                        dim=-1,
                    )
                    steps = torch.tensor([[0.0, 0.25], [0.75, 1.0]], dtype=dtype)
                    originals = [x.clone() for x in (q0, q1, steps)]
                    q0.requires_grad_()
                    q1.requires_grad_()
                    steps.requires_grad_()
                    actual = roma.unitquat_slerp_fast(q0, q1, steps, shortest_arc=shortest_arc)
                    expected = torch.empty_like(actual)
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                for l in range(2):
                                    step = originals[2][i, j]
                                    rotor = originals[1][k, l]
                                    if k == 0:
                                        value = (1 - step) * originals[0][k, l] + step * rotor
                                        expected[i, j, k, l] = value / torch.linalg.vector_norm(value)
                                    else:
                                        half_angle = step * angles[k, l] / 2
                                        expected[i, j, k, l] = torch.stack(
                                            (
                                                half_angle * 0,
                                                half_angle * 0,
                                                torch.sin(half_angle),
                                                torch.cos(half_angle),
                                            )
                                        )
                    self.assertEqual(actual.shape, (2, 2, 2, 2, 4))
                    self.assert_close(actual, expected)
                    cotangent = torch.arange(actual.numel(), dtype=dtype).reshape(actual.shape) / actual.numel()
                    gradients = torch.autograd.grad((actual * cotangent).sum(), (q0, q1, steps))
                    self.assertTrue(all(torch.isfinite(x).all() for x in gradients))
                    for current, original in zip((q0, q1, steps), originals):
                        self.assertTrue(torch.equal(current.detach(), original))

    def test_coincident_second_derivatives(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q = torch.full((4,), 0.5, dtype=dtype)
                cotangent = torch.tensor([0.2, -0.3, 0.5, 1.1], dtype=dtype)
                step = torch.tensor(0.25, dtype=dtype)
                # Hessian of c.n(x), where n(x)=x/||x|| and ||q||=1.
                dot = torch.dot(cotangent, q)
                hessian = (
                    -torch.outer(cotangent, q)
                    - torch.outer(q, cotangent)
                    - dot * torch.eye(4, dtype=dtype)
                    + 3 * dot * torch.outer(q, q)
                )

                def function(a, b):
                    return torch.dot(roma.unitquat_slerp_fast(a, b, step), cotangent)

                actual = torch.autograd.functional.hessian(function, (q, q.clone()))
                scales = (1 - step, step)
                for i in range(2):
                    for j in range(2):
                        self.assert_close(actual[i][j], scales[i] * scales[j] * hessian)

    def test_coincident_gradcheck_and_gradgradcheck(self):
        q0 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
        q1 = q0.detach().clone().requires_grad_()
        steps = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)

        def function(a, b, t):
            return roma.unitquat_slerp_fast(a, b, t)

        # gradgradcheck draws cotangents; isolate that new-test RNG consumption.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7391)
            self.assertTrue(torch.autograd.gradcheck(function, (q0, q1, steps), eps=1e-6, atol=1e-5, rtol=1e-4))
            self.assertTrue(torch.autograd.gradgradcheck(function, (q0, q1, steps), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_coincident_forward_and_reverse_jvp(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q = torch.full((4,), 0.5, dtype=dtype)
                step = torch.tensor(0.25, dtype=dtype)
                v0 = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=dtype)
                v1 = torch.tensor([-1.0, 0.75, 2.0, -0.5], dtype=dtype)
                projector = torch.eye(4, dtype=dtype) - torch.outer(q, q)
                expected = projector @ ((1 - step) * v0 + step * v1)

                def function(a, b):
                    return roma.unitquat_slerp_fast(a, b, step)

                _, reverse = torch.autograd.functional.jvp(function, (q, q.clone()), (v0, v1))
                _, forward = torch.func.jvp(function, (q, q.clone()), (v0, v1))
                self.assert_close(reverse, expected)
                self.assert_close(forward, expected)


if __name__ == "__main__":
    unittest.main()
