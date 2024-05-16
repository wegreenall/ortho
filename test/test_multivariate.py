import unittest
import torch
from ortho.polynomials import (
    LaguerrePolynomial,
    HermitePolynomial,
    ProbabilistsHermitePolynomial,
    chebyshev_first,
    chebyshev_second,
)


# @unittest.skip("Not Implemented Yet")
class TestChebyshevPolynomials(unittest.TestCase):
    def setUp(self):
        pass

    def test_chebyshev_first(self):
        # tests whether calculation of the chebyshev bases gets right answer
        self.assertTrue(torch.allclose(values, test_values))

    def test_chebyshev_second(self):
        self.assertTrue(torch.allclose(values, test_values))


if __name__ == "__main__":
    unittest.main()
