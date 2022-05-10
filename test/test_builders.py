import torch
import torch.distributions as D
from ortho.builders import (
    get_weight_function_from_sample,
    get_moments_from_function,
    get_moments_from_sample,
    get_gammas_from_moments,
    integrate_function,
    sample_from_function,
    get_poly_from_moments,
    get_gammas_from_sample,
    get_poly_from_sample,
)
import math

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_printoptions(precision=10)


def function_for_sampling(x: torch.Tensor):
    return (1 / (math.sqrt(math.pi * 2))) * torch.exp(-(x ** 2) / 2)


def gauss_moment(n: int) -> int:
    """
    Returns the n-th moment of a standard normal Gaussian distribution.
    """
    if n == 0:
        return 1
    if n % 2 == 0:
        return double_fact(n - 1)
    else:
        return 0


def double_fact(n: int) -> int:
    """
    Evaluates the double factorial of the input integer n
    """
    # assert n % 2 == 0, "n is not even!"
    # p = n - 1
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return n * double_fact(n - 2)


@unittest.skip("NOT COMPLETE")
class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.order = 10
        distribution = D.Normal(0.0, 1.0)
        sample_size = 10000000
        self.sample = distribution.sample((sample_size,))

        self.end_point = torch.tensor(10.0)
        fineness = 1000
        self.input_points = torch.linspace(
            -self.end_point, self.end_point, fineness
        )
        # normal moments:
        normal_moments = []
        for n in range(2 * self.order + 2):
            normal_moments.append(gauss_moment(n))
        self.normal_moments = torch.Tensor(normal_moments)

        self.prob_polynomials = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: x ** 2 - 1,
            lambda x: x ** 3 - 3 * x,
            lambda x: x ** 4 - 6 * x ** 2 + 3,
            lambda x: x ** 5 - 10 * x ** 3 + 15 * x,
        ]

    def test_get_moments_from_sample(self):
        calculated_moments = get_moments_from_sample(self.sample, self.order)
        self.assertTrue(
            torch.allclose(calculated_moments, self.normal_moments)
        )

    def test_get_gammas_from_moments(self):
        moments = self.normal_moments
        gammas = get_gammas_from_moments(moments, self.order)
        self.assertTrue(
            torch.allclose(
                gammas, torch.linspace(0.0, self.order - 1, self.order), 1e-02
            )
        )

    def test_get_gammas_from_sample(self):
        gammas = get_gammas_from_sample(self.sample, self.order)[1:]
        comparison_gammas = torch.linspace(1.0, self.order - 1, self.order - 1)
        # breakpoint()
        self.assertTrue(torch.allclose(gammas, comparison_gammas), 3e-2)

    def test_integrate_function(self):
        # breakpoint()
        integral = integrate_function(
            function_for_sampling,
            self.end_point,
            (1 / (torch.sqrt(math.pi * 2))),
        )
        self.assertTrue(torch.allclose(integral, 1))

    def test_sample_from_function(self):
        new_sample = sample_from_function(
            function_for_sampling,
            self.end_point,
            (1 / (torch.sqrt(torch.tensor(math.pi * 2)))),
        )
        calculated_moments = get_moments_from_sample(new_sample, self.order)
        # breakpoint()
        self.assertTrue(
            torch.allclose(calculated_moments, self.normal_moments, rtol=1e-01)
        )

    def test_get_poly_from_moments(self):
        poly = get_poly_from_moments(self.normal_moments, self.order)
        for i in range(6):
            with self.subTest(i=i):
                polyvals = poly(self.input_points, i, None)
                true_polyvals = self.prob_polynomials[i](self.input_points)
                self.assertTrue(torch.allclose(polyvals, true_polyvals))

    def test_get_poly_from_sample(self):
        poly = get_poly_from_sample(self.sample, self.order)
        # breakpoint()
        for i in range(6):
            with self.subTest(i=i):
                polyvals = poly(self.input_points, i, None)
                true_polyvals = self.prob_polynomials[i](self.input_points)
                # breakpoint()
                self.assertTrue(torch.allclose(polyvals, true_polyvals))


if __name__ == "__main__":
    unittest.main()
