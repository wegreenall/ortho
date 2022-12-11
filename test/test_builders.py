import torch
import torch.distributions as D
from ortho.builders import (
    get_weight_function_from_sample,
    get_moments_from_function,
    get_moments_from_sample,
    get_betas_from_moments,
    get_gammas_from_moments,
    integrate_function,
    sample_from_function,
    get_poly_from_moments,
    get_gammas_from_sample,
    get_poly_from_sample,
    get_orthonormal_basis,
    get_orthonormal_basis_from_sample,
)
from ortho.basis_functions import Basis
import math
from torch.quasirandom import SobolEngine

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_printoptions(precision=10)


def function_for_sampling(x: torch.Tensor):
    return (1 / (math.sqrt(math.pi * 2))) * torch.exp(-(x ** 2) / 2)


# def gauss_moment(n: int) -> int:
# """
# returns the n-th moment of a standard normal gaussian distribution.
# """
# if n == 0:
# return 1
# if n % 2 == 0:
# return double_fact(n - 1)
# else:
# return 0


def gauss_moment(order: int) -> list:
    """
    returns the n-th moment of a standard normal gaussian distribution.
    """
    if order == 0:
        return 1
    if order == 1:
        return 0
    else:
        return (order - 1) * gauss_moment(order - 2)


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


class TestBuilders(unittest.TestCase):
    def setUp(self):
        self.order = 16
        distribution = D.Normal(0.0, 1.0)
        sample_size = 100000
        self.sample = distribution.sample((sample_size,))
        sobol = SobolEngine(dimension=1)
        base_sample = sobol.draw(sample_size)
        # self.sample = D.Normal(0.0, 1.0).icdf(base_sample).squeeze()[2:]

        self.end_point = torch.tensor(10.0)
        fineness = 1000
        self.input_points = torch.linspace(
            -self.end_point, self.end_point, fineness
        )

        normal_moments = [
            gauss_moment(order) for order in range(2 * self.order + 2)
        ]
        self.normal_moments = torch.Tensor(normal_moments)

        self.prob_polynomials = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: x ** 2 - 1,
            lambda x: x ** 3 - 3 * x,
            lambda x: x ** 4 - 6 * x ** 2 + 3,
            lambda x: x ** 5 - 10 * x ** 3 + 15 * x,
        ]

        self.weight_function = lambda x: torch.exp(-(x ** 2) / 2)
        self.example_betas = torch.cat(
            (
                torch.Tensor([1.0]),
                torch.linspace(1.0, self.order - 1, self.order - 1),
            )
        )
        self.example_gammas = torch.cat(
            (
                torch.Tensor([1.0]),
                torch.linspace(1.0, self.order - 1, self.order - 1),
            )
        )

    # @unittest.skip("bad example")
    def test_get_moments_from_sample(self):
        moments = get_moments_from_sample(self.sample, self.order)
        # breakpoint()
        self.assertEqual(moments.shape, torch.Size([self.order + 1]))
        self.assertTrue(torch.allclose(moments, self.normal_moments))

    def test_get_betas_from_moments(self):
        moments = self.normal_moments
        betas = get_betas_from_moments(moments, self.order)
        self.assertTrue(
            torch.allclose(
                betas,
                torch.zeros(self.order),
                1e-02,
            )
        )

    def test_get_gammas_from_sample(self):
        gammas = get_gammas_from_sample(self.sample, self.order)  # [1:]
        print(gammas)

        # comparison_gammas = torch.linspace(1.0, self.order - 1, self.order - 1)
        breakpoint()
        self.assertTrue(torch.allclose(gammas, self.example_gammas), 5e-2)

    def test_get_gammas_from_moments(self):
        moments = self.normal_moments + D.Normal(0.0, 0.0001).sample(
            self.normal_moments.shape
        )
        gammas = get_gammas_from_moments(moments, self.order)
        print("Gammas from moments:", gammas)
        breakpoint()
        self.assertTrue(
            torch.allclose(
                gammas,
                torch.cat(
                    (
                        torch.Tensor([1.0]),
                        torch.linspace(1.0, self.order - 1, self.order - 1),
                    )
                ),
                1e-02,
            )
        )

    def test_get_orthonormal_basis_from_sample(self):
        basis = get_orthonormal_basis_from_sample(
            self.sample, self.weight_function, self.order
        )
        self.assertTrue(isinstance(basis, Basis))

    @unittest.skip("Not implemented")
    def test_get_orthonormal_basis(self):
        basis = get_orthonormal_basis(
            self.example_betas,
            self.example_gammas,
            self.order,
            self.weight_function,
        )

    @unittest.skip("Not implemented")
    def test_get_symmetric_orthonormal_basis(self):
        pass

    @unittest.skip("Not implemented")
    def test_get_weight_function_from_sample(self):
        pass

    @unittest.skip("Not implemented")
    def test_get_moments_from_function(self):
        pass

    @unittest.skip("Not implemented")
    def test_get_gammas_betas_from_moments(self):
        pass

    @unittest.skip("Not Relevant")
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
                self.assertTrue(torch.allclose(polyvals, true_polyvals))


if __name__ == "__main__":
    unittest.main()
