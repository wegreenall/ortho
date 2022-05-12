import torch
import torch.distributions as D
from ortho.measure import MaximalEntropyDensity
from ortho.orthopoly import (
    OrthonormalPolynomial,
    SymmetricOrthogonalPolynomial,
    SymmetricOrthonormalPolynomial,
)
from ortho.basis_functions import OrthonormalBasis
from ortho.utils import sample_from_function, integrate_function, gauss_moment
from typing import Callable

"""
This file contains builder functions for various components
required for building out the Favard kernel.
"""

torch.set_default_tensor_type(torch.DoubleTensor)


def get_orthonormal_basis(order, betas, gammas) -> OrthonormalBasis:
    """
    For given order, betas and gammas, generates
    an OrthonormalBasis.
    """
    poly = OrthonormalPolynomial(order, betas, gammas)
    weight_function = MaximalEntropyDensity(order, betas, gammas)
    return OrthonormalBasis(poly, weight_function, 1, order)


def get_symmetric_orthonormal_basis(order, gammas) -> OrthonormalBasis:
    """
    For given order, betas and gammas, generates
    an OrthonormalBasis.
    """
    poly = SymmetricOrthonormalPolynomial(order, gammas)
    weight_function = MaximalEntropyDensity(
        order, torch.zeros(2 * order), gammas
    )
    return OrthonormalBasis(poly, weight_function, 1, order)


def get_weight_function_from_sample(
    sample: torch.Tensor, order
) -> "MaximalEntropyDensity":
    """
    For a given sample, returns the maximal entropy weight function that
    correponds to the moments that the sample has - i.e. it creates the
    maximal entropy weight function corresponding to the weight function.
    """
    gammas = get_gammas_from_sample(sample, 2 * order)
    return MaximalEntropyDensity(order, torch.zeros(2 * order), gammas)


def get_moments_from_function(
    target: Callable,
    end_point: torch.Tensor,
    func_max: torch.Tensor,
    order: int,
    sample_size=2000 ** 2,
):
    sample = sample_from_function(
        target, end_point, func_max, sample_size
    ).squeeze()
    return get_moments_from_sample(sample, order)


def get_moments_from_sample(sample: torch.Tensor, order: int) -> torch.Tensor:
    """
    Returns a sequence of moments calculated from a sample.
    The odd-ordered moments are set to 0 to handle the fact that we are
    calculating a symmetric orthogonal polynomial (we only have one equation
    for each parameter that we want to solve for.
    """
    powers_of_sample = sample.repeat(2 * order + 2, 1).t() ** torch.linspace(
        0, 2 * order + 1, 2 * order + 2
    )
    estimated_moments = torch.mean(powers_of_sample, dim=0)

    # build moments
    moments = torch.zeros(2 * order + 2)
    moments[0] = 1
    for i in range(1, 2 * order + 2):
        if i % 2 == 0:  # i.e. even
            moments[i] = estimated_moments[i]
    # breakpoint()
    return moments


def get_gammas_from_moments(moments: torch.Tensor, order: int) -> torch.Tensor:
    """
    Accepts a tensor containing the moments from a given
    distribution, and generates the gammas that correspond to them;
    i.e., the gammas from the orthogonal polynomial series that
    is orthogonal w.r.t the linear moment functional with those moments.
    """
    dets = torch.zeros(order + 2)
    dets[0] = dets[1] = 1.0
    gammas = torch.zeros(order)
    for i in range(order):
        hankel_matrix = moments[: 2 * i + 1].unfold(0, i + 1, 1)
        dets[i + 2] = torch.linalg.det(hankel_matrix)
    gammas = dets[:-2] * dets[2:] / (dets[1:-1] ** 2)
    return gammas


def get_poly_from_moments(
    moments: torch.Tensor,
    order: int,
) -> SymmetricOrthonormalPolynomial:
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
    gammas = torch.zeros(order)

    # to construct the polynomial from the sequnce of moments, utilise the
    # sequence of equations:
    gammas = get_gammas_from_moments(moments, order)

    return SymmetricOrthonormalPolynomial(order, gammas)


def get_gammas_from_sample(sample: torch.Tensor, order: int) -> torch.Tensor:
    """
    Composes get_gammas_from_moments and get_moments_from_sample to
    produce the gammas from a sample. This just allows for simple
    calls to individual functions to construct the necessary
    component in any given situation.
    """
    return get_gammas_from_moments(
        get_moments_from_sample(sample, 2 * order), 2 * order
    )


def get_poly_from_sample(
    sample: torch.Tensor, order: int
) -> SymmetricOrthonormalPolynomial:
    """
    Returns a SymmetricOrthogonalPolynomial calculated by:
         - taking the moments from the sample, with odd moments set to 0;
         - constructing from these the gammas that correspond to the
           SymmetricOrthogonalPolynomial recursion
         - generating the SymmetricOrthogonalPolynomial from these gammas.
    Hence we have a composition of the three functions:
          get_moments_from_sample -> get_poly_from_moments
    """
    moments = get_moments_from_sample(sample, order)
    return get_poly_from_moments(moments, order)


if __name__ == "__main__":
    dist = D.Normal(0.0, 1.0)
    sample = dist.sample([4000 ** 2])
    order = 10
    checked_moments = get_moments_from_sample(sample, order)[: order + 2]
    print(
        "Should be like standard normal moments:",
    )

    # normal moments:
    normal_moments = []
    for n in range(1, 2 * order + 2):
        normal_moments.append(gauss_moment(n))
    normal_moments = torch.Tensor(normal_moments)[: order + 1]
    # breakpoint()
    print(checked_moments[2:] / normal_moments[1:])
