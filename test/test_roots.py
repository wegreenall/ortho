import torch
import torch.distributions as D
from ortho.roots import (
    get_polynomial_maximiser,
    get_polynomial_max,
    get_second_deriv_at_root,
    get_second_deriv_at_max,
)
import math

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_printoptions(precision=10)


class TestRootFinding(unittest.TestCase):
    def setUp(self):
        self.order = 5
        self.coeffics = torch.Tensor([0.0, 0.5, 3.5, -4, -20])
        self.roots = torch.Tensor([-0.4617, -0.1361, 0, 0.3978])
        self.peaks = torch.Tensor([-0.3493, -0.0672, 0.2664])

    def test_get_maximiser(self):
        poly_maximiser = get_polynomial_maximiser(self.coeffics, self.order)
        self.assertTrue(
            torch.allclose(poly_maximiser, torch.tensor(0.2664415115))
        )

    def test_get_max(self):
        poly_max = get_polynomial_max(self.coeffics, self.order)
        # breakpoint()
        self.assertTrue(torch.allclose(poly_max, torch.tensor(0.2052349847)))

    def test_get_second_deriv_at_max(self):
        poly_max = get_second_deriv_at_max(self.coeffics, self.order)
        # print(test_second_derivs[i])
        self.assertTrue(
            torch.allclose(poly_max, torch.tensor(-16.432455248097114))
        )

    def test_get_second_deriv_at_root(self):
        test_second_derivs = torch.Tensor(
            [-33.0792536, 5.8208296, 7, -40.5259616]
        )
        for i, root in enumerate(self.roots):
            poly_max = get_second_deriv_at_root(
                self.coeffics, self.order, root
            )
            # print(poly_max)
            self.assertTrue(torch.allclose(poly_max, test_second_derivs[i]))

    # def test_get_second_deriv_at_root(self):
    # for root in self.roots:

    # return

    # @unittest.skip("Not implemented yet")
    # def test_poly_from_sample(self):
    # noise_parameter = torch.Tensor([[1.0]])
    # sample_size = 100
    # input_sample = D.Normal(0.0, 1.0).sample([sample_size])
    # output_sample = test_function(input_sample) + D.Normal(
    # 0.0, noise_parameter.squeeze()
    # ).sample([sample_size])
