# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
import torch
import unittest
from test.utils import *

# Test of test utils
class TestDerivatives(unittest.TestCase):
    def test_derivatives(self):
        # Test the derivatives
        x = torch.randn((2,3,4), dtype=torch.float64)
        func = torch.tanh
        auto = automatic_jacobian(func, x)
        num = numerical_jacobian(func, x, 1e-10)
        self.assertTrue(is_close(auto, num, 1e-6))

        func = lambda x : torch.sum(torch.tanh(x))
        auto = automatic_jacobian(func, x)
        num = numerical_jacobian(func, x, 1e-10)
        self.assertTrue(is_close(auto, num, 1e-6))
        
        x = torch.randn((2, 3, 3), dtype=torch.float64)
        func = lambda x: torch.sum(x[0] @ x[1], axis=1)
        auto = automatic_jacobian(func, x)
        num = numerical_jacobian(func, x, 1e-10)
        self.assertTrue(is_close(auto, num, 1e-6))

if __name__ == "__main__":
    unittest.main()        