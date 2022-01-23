import torch

"""
Implements the inverse Laplace transform algorithm as explained in:
    Computation of the inverse Laplace transform based on a collocation method
    which uses only real values

    Cuomo, D'Amore, Murli, and Rizzard, 2005.

"""


def chebyshev_roots(order):
    """
    Returns the roots of the Chebyshev polynomial of order N
    for the purpose of constructing the Laguerre-polynomials based inverse
    Laplace transform
    """
