import torch
from typing import Dict
import matplotlib.pyplot as plt

"""
contains classes for construction of orthogonal polynomials. will also contain
implementations of the classical orthogonal polynomials, as well as methods
for getting the measure/weight function
"""


class OrthogonalPolynomial:
    def __init__(self, order, betas, gammas, leading=1):
        """
        A system of orthogonal polynomials has the property that:

        int P_j(x) P_k(x) dx = δ_ik

        The three-term recursion for orthogonal polynomials is written:

            P_n+1(x) = (β_{n+1} - x) P_n(x) - γ_{n+1} P_{n-1} (x)

        with:
            P_0(x) = 1 and P_1(x) = x

        Construction then of an arbitrary orthogonal polynomial sequence should
        involve setting the betas and the gammas appropriate for a given
        application
        """
        self.order = order

        if len(betas) < order or len(gammas) < order:
            raise TypeError(
                "Not enough betas or gammas for the order %d" % (order)
            )
        self.set_betas(betas)
        self.set_gammas(gammas)
        self.leading = leading

    def __call__(self, x: torch.Tensor, deg: int, params: dict):
        """ """
        if deg > self.order:
            raise ValueError("The order of this polynomial is not high enough")

        if deg == 0:
            return torch.ones(x.shape)
        elif deg == 1:
            return self.leading * (x - self.betas[deg - 1])
        else:
            return (self.leading * x - self.betas[deg - 1]) * self(
                x, deg - 1, params
            ) - self.gammas[deg - 1] * self(x, deg - 2, params)
        pass

    def get_order(self):
        """
        Returns the order of this polynomial
        """
        return self.order

    def set_betas(self, betas):
        """
        Setter for betas on orthogonal polynomial.

        validates w.r.t the fact that the betas are positive.
        """
        assert (betas >= 0).all(), "Please make sure all betas are positive"
        self.betas = betas

    def set_gammas(self, gammas):
        """
        Setter for gammas on orthogonal polynomial.

        validates w.r.t the fact that γ_0 is positive.
        """
        # assert gammas[0] == 1, "Please make sure gammas[0] = 1"
        self.gammas = gammas

    def get_betas(self):
        return self.betas

    def get_gammas(self):
        return self.gammas


def get_measure_from_poly(
    poly: OrthogonalPolynomial, initial_moment=1
) -> list:
    """
    Accepts a polynomial and produces from it the density of an absolutely
    continuous measure w.r.t which the polynomial is orthogonal.

    The Favard's theorem states the recurrence relation for the orthogonal
    polynomial sequence to be:
            P_{n+1} = (x-β_{n+1})P_n - γ_{n+1}P_{n-1}
        so: P_{n+1} - (x-β_{n+1})P_n - γ_{n+1}P_{n-1} = 0

    This is the same as:
            xP_n(x) = P_{n+1}(x) + β_nP_n(x) + γ_n P_{n-1}(x)

    Applying the linear moment functional to this relation gives the moment
    recursion implemented here.

    The moments of the measure, which uniquely define it, have the following
    property: for each k, <L, P_k(x)> is 0. However, this implies a sequence
    of relations for the moments w.r.t the coefficients of the polynomial:

        <L, 1> = μ_0      =1 for a probability distribution, the
                          Poisson-equivalent intensity for a Poisson
                          measure

        <L, P_1> = μ_1 - β_0μ_0 = 0
                   => μ_1 = β_0 μ_0   i.e. the mean is β_0.

        <L, P_2> = μ_2 - <L, xP_1(x)> + β_2<L, xP_1> + γ_{2}<L, P_1>
                   => μ_2 =

    The above define a sequence of relations that implicitly set the moment:

        <L, P_{k+1}> = μ_{Κ+1} - {other terms} = 0

    This function uses this implicit relation to acquire values for the moments
    given the betas and gammas on the input orthogonal polynomial.

    Once the moments are acquired, it is possible to construct the
    measure via an inverse laplace transform on the moment-generating
    function
        Σ_ι  μ_i x^i / i!

    """
    raise NotImplementedError


class SymmetricOrthogonalPolynomial(OrthogonalPolynomial):
    def __init__(self, order, gammas):
        """
        If Pn is symmetric, i.e. Pn(-x) = (-1)^n Pn(x),
        then there exist coefficients γ_n != 0 for n>=1, s.t.
                P_{n+1} = xP_n(x) - γ_nP_{n-1}

        with initial conditions P_0(x) = 1 and P_1(x) = x
        This is equivalent to a polynomial sequence with β_n = 0 for each n
        """
        betas = torch.zeros(order)
        super().__init__(order, betas, gammas)


def get_poly_from_moments(moments: list) -> SymmetricOrthogonalPolynomial:
    """
    Accepts a list of moment values and produces from it a
    SymmetricOrthogonalPolynomial.

    The standard recurrence for an orthogonal polynomial series has n equations
    and 2n unknowns - this means that it is not feasible to construct, from a
    given sequence of moments, an orthogonal polynomial sequence that is
    orthogonal     w.r.t the given moments. However, if we impose
    symmetricality, we can build a sequence of symmetric orthogonal polynomials
    from a given set of moments.
    """
    order = len(moments)
    gammas = torch.zeros(order)

    # to construct the polynomial from the sequnce of moments, utilise the
    # sequence of equations:

    return SymmetricOrthogonalPolynomial(order, gammas)


if __name__ == "__main__":
    from torch import distributions as D

    order = 4
    # betas = torch.ones(order)
    # gammas = torch.linspace(0, order, order)
    betas = D.Exponential(10.0).sample([order])
    gammas = D.Exponential(10.0).sample([order])
    gammas[0] = 1

    poly = OrthogonalPolynomial(order, betas, gammas)

    x = torch.linspace(-3, 3, 100)
    params: Dict = dict()
    func = poly(x, 3, params)
    plt.plot(x, func)
    plt.show()

    sym_poly = SymmetricOrthogonalPolynomial(order, gammas)
    sym_func = sym_poly(x, 2, params)
    plt.plot(x, sym_func)
    plt.show()

    # checking the moments from the polynomial thing works
    # coeffics = coeffics_list(order)
    # loc = (0, order)
    # value = 1
    # L(loc, value, coeffics, poly)
    # final_coeffics = coeffics.get_vals()
    # print("final_coeffics:", final_coeffics)
