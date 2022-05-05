import torch
import torch.distributions as D
from ortho.orthopoly import (
    OrthogonalPolynomial,
    get_gammas_from_moments,
    get_poly_from_moments,
    get_poly_from_sample,
    get_moments_from_sample,
)
import math

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)


class TestPolyFromMoments(unittest.TestCase):
    def setUp(self):
        self.order = 8
        self.catalans = torch.Tensor(
            [
                1,
                0,
                1,
                0,
                2,
                0,
                5,
                0,
                14,
                0,
                42,
                0,
                132,
                0,
                429,
                0,
                1430,
                0,
                4862,
                0,
                16796,
                0,
                58786,
                0,
                208012,
                0,
                742900,
                0,
                2674440,
                0,
                9694845,
                0,
                35357670,
                0,
                129644790,
                0,
                477638700,
                0,
                1767263190,
                0,
                6564120420,
                0,
            ]
        )
        self.chebyshev_polynomials_basic = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: x ** 2 - 1,
            lambda x: x ** 3 - 2 * x,
            lambda x: x ** 4 - 3 * x ** 2 + 1,
            lambda x: x ** 5 - 4 * x ** 3 + 3 * x,
        ]
        self.moments_2 = torch.Tensor(
            [1, 0, 1, 0, 3, 0, 15, 0, 105, 0, 945]
        ) * (math.sqrt(3) ** torch.Tensor(list(range(11))))

    def test_moments_to_gammas(self):
        moments = self.catalans[: 2 * self.order + 2]
        gammas = get_gammas_from_moments(moments, self.order)
        self.assertTrue(torch.allclose(gammas, torch.ones(gammas.shape)))

    def test_moments_to_gammas(self):
        gammas = get_gammas_from_moments(self.moments_2, 6)
        self.assertTrue(
            torch.allclose(
                gammas, torch.Tensor([1.0, 3.0, 6.0, 9.0, 12.0, 15.0])
            )
        )

    def test_poly_from_moments(self):
        moments = self.catalans[: 2 * self.order + 2]
        poly = get_poly_from_moments(moments, self.order)
        x = torch.linspace(-1, 1, 1000)
        params = dict()
        for i in range(6):
            self.assertTrue(
                torch.allclose(
                    poly(x, i, params), self.chebyshev_polynomials_basic[i](x)
                )
            )

    @unittest.skip("")
    def test_moments_from_sample(self):
        pass

    @unittest.skip("Not implemented yet")
    def test_poly_from_sample(self):
        noise_parameter = torch.Tensor([[1.0]])
        sample_size = 100
        input_sample = D.Normal(0.0, 1.0).sample([sample_size])
        output_sample = test_function(input_sample) + D.Normal(
            0.0, noise_parameter.squeeze()
        ).sample([sample_size])


class TestOrthogonalPolynomials(unittest.TestCase):
    def setUp(self):

        # model hyperparameters
        self.order = 5  # order
        self.sample_size = 1000
        self.ub = 10
        self.lb = 0
        self.inputs = torch.linspace(0, 10, 1000)
        self.eps = 1e-8
        self.prob_polynomials = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: x ** 2 - 1,
            lambda x: x ** 3 - 3 * x,
            lambda x: x ** 4 - 6 * x ** 2 + 3,
            lambda x: x ** 5 - 10 * x ** 3 + 15 * x,
        ]
        return

    @unittest.skip("Not Implemented Yet.")
    def test_orthonormality(self):
        """
        To check the integral of the function, remember
        that the function always maps its upper and lower bounds into
        [-1, 1] via the transformation
        z = 2x - (b+a)/(b-a)

        i.e. the integral:
        # int_a^b f_i(x)^2 dx = int_a^b phi((2*x - (b+a))/(b-a))dx

        Defining z = (2*x - (b+a))/(b-a) we get:

        # int_{z|x = a}^{z|x=b}  phi(z)dx/dz  dz
        # = ((b-a)/2) *  int_{-1}^1 phi(z) dz
        # = (b-a)/2 * 1 since phi orthonormal by construction.

        The integral of f_i(x)^2
        can be calculate by taking the mean of f_i(x)^2 and multiplying by
        (b-a) since if you divide the integral of this by (b-a)/2 you should
        get 1, the result is that we can compare mean(f(sample))2 with 1
        to see if the integral is correct, REGARDLESS OF THE VALUE OF -1, 1.

        The basis is therefore orthonormal WHEN we also multiply by root 2.
        Hence the normalising term being 2 / sqrt(π) rather than
        sqrt(2/π) which appears to be implied by the standard formula
        """
        n = 20

        # unif = torch.distributions.Uniform(lb, ub)
        N = 1000000
        sample = torch.quasirandom.SobolEngine(1).draw(N)

        func_means = torch.zeros(n)
        for i in range(0, n):
            break
        self.assertTrue(
            (torch.abs(torch.mean(func_means) - 1) < 0.00001).all()
        )

    def test_correctness(self):
        order = 5
        betas = torch.zeros(order + 1)
        gammas = torch.linspace(0, order, order + 1)
        # gammas[0] = 0
        poly = OrthogonalPolynomial(order, betas, gammas)
        for i in range(order + 1):
            outputs = poly(self.inputs, i, dict())
            self.assertTrue(
                (
                    torch.abs(outputs - self.prob_polynomials[i](self.inputs))
                    < self.eps
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
