import torch
import math
from ortho.measure import MaximalEntropyDensity
import unittest


class TestMaximalEntropyDensity(unittest.TestCase):
    def setUp(self):
        # test parameters
        self.catalan_numbers = torch.Tensor(
            [
                1.0,
                1.0,
                2.0,
                5.0,
                14.0,
                42.0,
                132.0,
                429.0,
                1430.0,
                4862.0,
                16796.0,
                58786.0,
                208012.0,
                742900.0,
                2674440.0,
                9694845.0,
                35357670.0,
                129644790.0,
                477638700.0,
            ]
        )

        end_point = 10
        self.input_length = 500

        self.order = 8
        self.sample_size = 1000

        self.betas = torch.zeros(2 * self.order)
        self.gammas = torch.ones(2 * self.order)

        self.betas.requires_grad = True
        self.gammas.requires_grad = True
        self.med = MaximalEntropyDensity(self.order, self.betas, self.gammas)

        self.input_points = torch.linspace(
            -end_point, end_point, self.input_length
        )
        pass

    def test_get_poly_term(self):
        poly_term = self.med._get_poly_term(self.input_points)
        self.assertEqual(
            poly_term.shape, torch.Size([self.input_length, self.order])
        )  # 1
        pass

    def test_get_poly_term_zero(self):
        poly_term = self.med._get_poly_term(self.input_points)
        self.assertTrue((poly_term != 0).all())
        pass

    def test_get_lambdas(self):
        lambdas = self.med._get_lambdas()
        self.assertEqual(lambdas.shape, torch.Size([self.order]))  # 1
        pass

    def test_moments(self):
        catalan_numbers = torch.zeros(self.order + 1)
        for i in range(self.order):
            catalan_numbers[0] = 1.0
            if i % 2 != 0:
                catalan_numbers[math.floor(i + 1)] = self.catalan_numbers[
                    math.floor(i / 2) + 1
                ]
        moments, _ = self.med._get_moments()
        # breakpoint()
        self.assertTrue((moments == catalan_numbers).all())
        pass

    @unittest.skip("Not yet implemented test for Hankel matrix")
    def test_hankel(self):
        catalan_numbers = torch.zeros(2 * self.order + 1)
        for i in range(2 * self.order + 1):
            catalan_numbers[0] = 1.0
            if i % 2 != 0:
                catalan_numbers[math.floor(i + 1)] = self.catalan_numbers[
                    math.floor(i / 2) + 1
                ]
        _, moment_matrix = self.med._get_moments()
        self.assertTrue((moment_matrix == catalan_numbers).all())
        pass

    def test_call_shape(self):
        output = self.med(self.input_points)
        self.assertEqual(output.shape, torch.Size([self.input_length]))

    def test_call_sign(self):
        output = self.med(self.input_points)
        self.assertTrue((output >= 0).all())

    def test_call_for_nans(self):
        output = self.med(self.input_points)
        self.assertTrue((output == output).all())
