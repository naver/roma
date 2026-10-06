# Copyright (c) 2026 Afloat16.
# 3-Clause BSD License.
"""A private generator owns the complete rotation sampling stream."""

import unittest

import torch

import roma


class TestSamplingGenerator(unittest.TestCase):
    def test_private_generator_replays_independently_of_global_rng(self):
        for sampler in (roma.random_unitquat, roma.random_rotmat, roma.random_rotvec):
            for dtype in (torch.float32, torch.float64):
                for shape in ((), 7, (2, 3)):
                    with self.subTest(sampler=sampler.__name__, dtype=dtype, shape=shape):
                        with torch.random.fork_rng(devices=[]):
                            first = torch.Generator().manual_seed(213)
                            second = torch.Generator().manual_seed(213)
                            torch.manual_seed(41)
                            expected = sampler(shape, dtype=dtype, generator=first)
                            torch.rand(83)
                            actual = sampler(shape, dtype=dtype, generator=second)
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                            self.assertTrue(torch.equal(first.get_state(), second.get_state()))

    def test_private_generator_does_not_advance_global_rng(self):
        for sampler in (roma.random_unitquat, roma.random_rotmat, roma.random_rotvec):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(sampler=sampler.__name__, dtype=dtype):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(29)
                        state = torch.random.get_rng_state().clone()
                        generator = torch.Generator().manual_seed(431)
                        generator_state = generator.get_state().clone()
                        result = sampler((2, 3), dtype=dtype, generator=generator)
                        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
                        self.assertFalse(torch.equal(generator.get_state(), generator_state))
                        self.assertEqual(result.dtype, dtype)
                        self.assertTrue(torch.isfinite(result).all())

    def test_default_generator_replays_and_rotations_remain_proper(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype), torch.random.fork_rng(devices=[]):
                torch.manual_seed(71)
                first = roma.random_unitquat((2, 3), dtype=dtype)
                torch.manual_seed(71)
                second = roma.random_unitquat((2, 3), dtype=dtype)
                torch.testing.assert_close(first, second, atol=0, rtol=0)
                torch.testing.assert_close(first.square().sum(-1), torch.ones((2, 3), dtype=dtype))
                matrix = roma.unitquat_to_rotmat(first)
                identity = torch.eye(3, dtype=dtype).expand(2, 3, 3, 3)
                torch.testing.assert_close(matrix @ matrix.transpose(-1, -2), identity)
                torch.testing.assert_close(torch.linalg.det(matrix), torch.ones((2, 3), dtype=dtype))
