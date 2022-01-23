import torch

import math
import unittest

import matplotlib.pyplot as plt
from vandermonde import solve, solve_transpose


# def nchoosek(n, k):
# return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def nchoosek(n, k):
    return torch.exp(
        torch.lgamma(torch.tensor(n + 1))
        - (
            torch.lgamma(torch.tensor(k + 1))
            + torch.lgamma(torch.tensor(n - k + 1))
        )
    )


"""
Tests for the case as described in Bjorck and Pereyra:
    alpha_i = 1 / (i+3)
    b_i = 1/2^i  i = 0,1,...,n

"The exact solution can be shown to be:
    x_i = (-1)**i (n+1)Choose(i+1) (1 + (i+1)/2)^n
"""


class TestBjorckPereyra(unittest.TestCase):
    def setUp(self):
        # build the terms for the test
        n = 7
        x = torch.zeros(n + 1)
        for i in range(n + 1):
            x[i] = (
                ((-1) ** i) * nchoosek(n + 1, i + 1) * ((1 + (i + 1) / 2) ** n)
            )

        self.x = torch.Tensor(x)
        self.a = torch.Tensor([1 / (i + 3) for i in range(n + 1)])
        self.b = torch.Tensor([1 / (2 ** i) for i in range(n + 1)])
        pass

    def test_vandermonde(self):
        eps = 0.0001
        result = solve(self.a, self.b)
        # print("result:", result)
        self.assertTrue((torch.abs(self.x - result) < eps).all())

    def test_vandermonde_transpose(self):
        eps = 0.0001
        result = solve_transpose(self.a, self.b)
        print("transpose_result:", result)
        print("self.x is:", self.x)
        print("error is:", torch.abs(self.x - result))
        breakpoint()
        # self.assertTrue((torch.abs(self.x - result) < eps).all())
        plt.plot(torch.abs(self.x - result))
        plt.show()


if __name__ == "__main__":
    unittest.main()
