import torch
from ortho.orthopoly import OrthogonalPolynomial
import matplotlib.pyplot as plt

"""
This file implements various families of orthogonal polynomials as direct
instances of the Orthogonal Polynomial class.
"""


class ProbabilistsHermitePolynomial(OrthogonalPolynomial):
    def __init__(self, order):
        betas = torch.zeros(order)
        gammas = torch.linspace(0, order - 1, order)
        super().__init__(order, betas, gammas)


class HermitePolynomial(OrthogonalPolynomial):
    def __init__(self, order):
        betas = torch.zeros(order)
        gammas = 2 * torch.linspace(0, order - 1, order)
        super().__init__(order, betas, gammas, leading=2)


class GeneralizedLaguerrePolynomial(OrthogonalPolynomial):
    def __init__(self, order, alpha):
        ns = torch.linspace(0, order - 1, order)
        betas = 2 * ns + alpha + 1
        gammas = (ns + 1) * (ns + alpha + 1)
        gammas[0] = 1
        super().__init__(order, betas, gammas)


class LaguerrePolynomial(GeneralizedLaguerrePolynomial):
    def __init__(self, order):
        super().__init__(order, 0)


if __name__ == "__main__":
    # Hermite polynomials
    order = 12
    x = torch.linspace(-2, 2, 1000)
    hermite = HermitePolynomial(order)
    params = dict()
    plt.plot(x, hermite(x, 4, params))
    plt.show()

    for i in range(order):
        plt.plot(x, hermite(x, i, params) ** 2)
    plt.show()
