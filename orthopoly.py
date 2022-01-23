import torch
from typing import Dict
import matplotlib.pyplot as plt
from framework.basis_functions import Basis

"""
contains classes for construction of orthogonal polynomials. will also contain
implementations of the classical orthogonal polynomials, as well as methods
for getting the measure/weight function
"""


class OrthogonalPolynomial:
    def __init__(self, order, betas, gammas):
        """
        A system of orthogonal polynomials has the property that:

        int P_j(x) P_k(x) dx = δ_ik

        The three-term recursion for orthogonal polynomials is written:

            P_n+1(x) = (β_{n+1} - x) P_n(x) - γ_{n+1} P_{n-1} (x)

        with:
            P_0(x) = 1 and P_1(x) = x

        Construction then of an arbitrary orthogonal polynomial sequence should
        involve setting the betas and the gammas
        """
        self.order = order

        if len(betas) < order or len(gammas) < order:
            raise TypeError(
                "Not enough betas or gammas for the order %d" % (order)
            )
        self.betas = betas
        self.gammas = gammas

    def __call__(self, x: torch.Tensor, deg: int, params: dict):
        """ """
        if deg == 0:
            return 1
        elif deg == 1:
            return x
        else:
            return (self.betas[deg] - x) * self(
                x, deg - 1, params
            ) - self.gammas[deg] * self(x, deg - 2, params)
        pass


if __name__ == "__main__":
    order = 10
    betas = torch.zeros(order)
    gammas = torch.linspace(0, order, order)

    poly = OrthogonalPolynomial(order, betas, gammas)

    x = torch.linspace(-3, 3, 100)
    params: Dict = dict()
    func = poly(x, 9, params)
    plt.plot(x, func)
    plt.show()
