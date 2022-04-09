import torch
import torch.distributions as D
from ortho.orthopoly import OrthogonalPolynomial

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt


# class TestL(unittest.TestCase):
# def setUp(self):
# self.order = 3

# # build random betas and gammas

# # self.betas = torch.ones(self.order)
# # self.gammas = torch.ones(self.order)
# self.betas = D.Exponential(1.0).sample([self.order])
# self.gammas = D.Exponential(1.0).sample([self.order])
# self.gammas[0] = 1

# self.poly = OrthogonalPolynomial(self.order, self.betas, self.gammas)

# self.test_coeffics = [
# torch.Tensor([1.0]),  # for μ_0
# torch.Tensor([-self.betas[0], 1.0]),  # for μ_1
# torch.Tensor([-self.gammas[1], -self.betas[0], 1.0]),  # for μ_2
# torch.Tensor(  # for μ_3
# [
# 0,
# -(self.gammas[1] - self.betas[0] * self.betas[1]),
# -(self.betas[0] + self.betas[1]),
# 1.0,
# ]
# ),
# ]

# def test_L(self):
# # tests for the correct construction of the coefficients for the
# # generation of the moments of the linear function w.r.t a given
# # orthogonal polynomial system is orthogonal
# for order in range(self.order):
# with self.subTest(i=order):
# coeffics = coeffics_list(order)
# loc = (0, order)
# value = 1
# L(loc, value, coeffics, self.poly)
# final_coeffics = coeffics.get_vals()
# # print("final_coeffics", final_coeffics)
# # breakpoint()
# eps = 0.000001
# self.assertTrue(
# (final_coeffics - self.test_coeffics[order] < eps).all()
# )


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
            # duh, this is always the integral:
            # func_mean = torch.mean(func_sample) * (self.ub - self.lb)
            # func_means[i] = func_mean

        # print(torch.mean(func_means))
        # breakpoint()
        self.assertTrue(
            (torch.abs(torch.mean(func_means) - 1) < 0.00001).all()
        )
        # print("integral of square basis function: ", func_mean)

    def test_correctness(self):
        order = 5
        betas = torch.zeros(order + 1)
        gammas = torch.linspace(0, order, order + 1)
        # gammas[0] = 0
        poly = OrthogonalPolynomial(order, betas, gammas)
        for i in range(order + 1):
            outputs = poly(self.inputs, i, dict())
            plt.plot(self.inputs, outputs)
            plt.plot(self.inputs, self.prob_polynomials[i](self.inputs))
            plt.show()
            self.assertTrue(
                (
                    torch.abs(outputs - self.prob_polynomials[i](self.inputs))
                    < self.eps
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
