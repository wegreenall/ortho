import torch
import torch.distributions as D
import unittest


class TestLikelihood(unittest.TestCase):
    def setUp(self):

        # model hyperparameters
        self.order = 10  # order
        self.sample_size = 1000
        return

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
            # duh, this is always the integral:
            func_mean = torch.mean(func_sample) * (ub - lb)
            func_means[i] = func_mean

        # print(torch.mean(func_means))
        # breakpoint()
        self.assertTrue(
            (torch.abs(torch.mean(func_means) - 1) < 0.00001).all()
        )
        # print("integral of square basis function: ", func_mean)


if __name__ == "__main__":
    unittest.main()
