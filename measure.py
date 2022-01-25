import torch
from ortho.orthopoly import OrthogonalPolynomial
import torch.distributions as D
from typing import Callable


def get_measure_from_poly(poly: OrthogonalPolynomial) -> Callable:
    # get the moments:
    moments = get_moments_from_poly(poly)
    """
    The sequence of moments from the polynomial imply a unique linear moment
    functional, but multiple corresponding densities. To that extent, we can
    choose the one that maximises the Shannon informational entropy, subject
    to the constraints implied by the moments -
    see, "Inverse two-sided Laplace transform for probability density
    functions", (Tagliani 1998)
    """
    # to build the density, we solve the following problem:
    # min μ_0 ln(1/μ_0  \int exp(-Σλ_j t^j) dt ) + Σλ_j μ_j

    return


class coeffics_list:
    """
    A small class to handle the use of object-by-reference in Python -
    Used in the construction of the tensor of coefficients for the construction
    of the sequence of moments for a given orthogonal polynomial. That is,
    for a given OrthogonalPolynomial class with given sequences of betas
    and gammas, the resulting moments result from the three-term recursion.
    """

    def __init__(self, order):
        """
        Initialises the data to be a sequence of zeroes of length order + 1;
        this is because an order 3 polynomial will need 4 coefficients in the
        matrix used to solve for the moments
        """
        self.order = order
        self.data = torch.zeros(order + 1)  # the length is the order + 1

    def update_val(self, k, value):
        self.data[k] += value  # we _update_ the value

    def get_vals(self):
        return self.data

    def __len__(self):
        return self.order

    def __repr__(self):
        return self.data.__repr__()


def get_moments_from_poly(poly):
    """
    For a given polynomial, computes the moments of the measure w.r.t which the
    polynomial is orthogonal, utilising the three-term recurrence that defines
    the orthogonal polynomial, in conjunction with the Linear functional
    operator L.

    For more details, check the documentation for L

    """
    order = poly.get_order()
    final_coefficients = torch.zeros([order, order])

    for i in range(order):
        coeffics = coeffics_list(i)
        L((0, i), 1, coeffics, poly)
        final_coefficients[i, : (i + 1)] = coeffics.get_vals()

    coeffics_inv = torch.linalg.inv(final_coefficients)
    targets = torch.zeros(order)
    targets[0] = 1  # assuming we're getting a probability distribution
    mus = torch.einsum("ij,j->i", coeffics_inv, targets)

    return mus


def L(
    loc: tuple,
    value: float,
    coeffics: coeffics_list,
    poly: OrthogonalPolynomial,
):
    """
    Recursively calculates the coefficients for the construction of the moments

    This is a representation of the linear functional operator:
        <L, x^m P_n>  === L(m, n)


    Places into coeffics the coefficients of order len(coeffics).

    The aim is to build the matrix of coefficients to solve for the moments:

        [1       0     0  ... 0] |μ_0|   |1|
        [-β_0    1     0       ] |μ_1|   |0|
        [-γ_1  -β_0    1  ... 0] |μ_2| = |0|
        [          ...         ] |...|   |0|
        [?       ?        ... 1] |μ_k|   |0|

    To find the coefficients for coefficient k, start with a list of length k,
    and pass (0,k) to the function L which represents the linear functional
    operator for the OPS.

    L takes (m,n) and a value and calls itself on [(m+1, n-1), 1*value],
    [(m, n-1), -β_{n-1}*value]
    [(m, n-2), -γ_{n-1}*value]
    Mutiply all the values passed by the value passed in.

    The function places in coeffics a list of length k, the coefficients
    for row k in the above described matrix.

    :param loc: a tuple (m, n) representing that this is L(m,n) == <L, x^m P_n>
    :param value: a coefficient value passed in will multiply the coefficient
                  found in this iteration
    :param coeffics: the coeffics_list object that will be updated
    :param poly: the orthogonal polynomial system w.r.t which we are building
                 the linear moment functional - i.e. this polynomial system is
                 orthogonal with respect to the measure whose moments are
                 produced by this linear moment functional
    """
    local_betas = poly.get_betas()
    local_gammas = poly.get_gammas()

    if loc[1] == 0:  # n is 0
        # i.e loc consists of a moment (L(m, 0))
        # add the value to the corresponding coefficient in the array
        coeffics.update_val(loc[0], value)

    # check if it's the first go:
    if not (loc[1] < 0 or (loc[0] == 0 and loc[1] != len(coeffics))):
        # m != 0, n != 0
        left_loc = (loc[0] + 1, loc[1] - 1)
        centre_loc = (loc[0], loc[1] - 1)
        right_loc = (loc[0], loc[1] - 2)
        L(left_loc, value, coeffics, poly)
        L(centre_loc, value * -local_betas[loc[1] - 1], coeffics, poly)
        L(right_loc, value * -local_gammas[loc[1] - 1], coeffics, poly)
        pass


if __name__ == "__main__":
    # build an orthogonal polynomial
    order = 10

    # build random betas and gammas

    # self.betas = torch.ones(self.order)
    # self.gammas = torch.ones(self.order)
    betas = D.Exponential(1.0).sample([order])
    gammas = D.Exponential(1.0).sample([order])
    gammas[0] = 1

    poly = OrthogonalPolynomial(order, betas, gammas)
    mus = get_moments_from_poly(poly)
    print(mus)
