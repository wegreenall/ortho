# basis_functions.py
import math
from ortho.orthopoly import OrthogonalPolynomial

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributions as D

# from framework import special, utils
from ortho.polynomials import chebyshev_first, chebyshev_second
from typing import Callable

from ortho.special import hermite_function

"""
This contains implementations of basis functions useful for constructing
kernels according to the Mercer representation theorem.

The basis functions are expected to have signature (x, i, params),
where:

x : input value
i : the degree of the given function (in most cases, the degree of the
polynomial that corresponds to that function)

params:  a dictionary of parameters that contains the appropriate titles etc.
for that set of polynomials.

Default versions or generator functions
can be created here to allow for ease of use. (i.e. available here and not
necessary to hand-tune every time

Return value must be a torch.Tensor of dimension [n] only;
not [n, 1] as is the inputs vector - this reflects explicitly that the output
would be 1-d (2-d inputs would be of shape [n,2]). The result of this is that
multi-dimensional problems require the construction of the tensor product 
formulation of an orthonormal basis.
"""


class Basis:
    def __init__(
        self,
        basis_function: Callable,
        dimension: int,
        order: int,
        params: dict = None,
    ):
        self.dimension = dimension
        self.order = order
        self.basis_function = basis_function
        self.params = params
        """
        :param basis function: the basis function to use in this
                               Basis class - i.e., smooth_exponential_basis.

                Expected to have signature:
                                basis_function(x: torch.Tensor,
                                               deg: int,
                                               params: dict) -> torch.Tensor


        """
        return

    def get_dimension(self):
        """
        Getter method for the dimension of the model.
        """
        return self.dimension

    def get_order(self):
        """
        Getter method for the dimension of the model.
        """
        return self.order

    def __call__(self, x):
        """
        Returns the whole basis evaluated at an input.

        output shape: [x.shape[0], self.order]

        """
        # breakpoint()
        # check input shape
        if len(x.shape) <= 1:
            x = x.unsqueeze(-1)
        elif x.shape[1] != self.dimension:
            raise ValueError(
                "The dimension of x should be {dim} for this,\
                              because that is the dim of the\
                              basis.".format(
                    dim=self.dimension
                )
            )

        # can I tensorise the hstack? In basis function?
        result = torch.hstack(
            [
                self.basis_function(x, deg, self.params)
                for deg in range(self.order)
            ]
        )

        return result

    def get_params(self):
        return self.params

    def __add__(self, other):
        """
        When adding two bases, return a basis that
        is of the same
        """
        if self.order != other.order:
            raise ValueError(
                "Other basis is not of the same order as this basis. This Basis order: {this_order}; other Rasis order: {other_order}".format(
                    this_order=self.order, other_order=other.order
                )
            )
        if self.dim != other.dim:
            raise ValueError(
                "Other basis is not of the same dim as this basis. This Basis order: {this_dim}; other Rasis order: {other_dim}".format(
                    this_dim=self.dim, other_dim=other.dim
                )
            )
        new_basis = CompositeBasis(self, other)
        return new_basis


class CompositeBasis(Basis):
    def __init__(self, old_basis, new_basis: Basis):
        self.old_basis = old_basis
        self.new_basis = new_basis

    def __call__(self, x):
        return self.old_basis(x) + self.new_basis(x)


class RandomFourierFeatureBasis(Basis):
    def __init__(
        self,
        dim: int,
        order: int,
        spectral_distribution: D.Distribution,
    ):
        """
        Random Fourier Feature basis for constructing the
        """
        # self.w_dist = D.Normal(torch.zeros(dim), torch.ones(dim))
        self.w_dist = spectral_distribution  # samples are the size of "dim" which is 1-d for stationary, 2-d for non-stationary.
        self.b_dist = D.Uniform(0.0, 2 * math.pi)
        self.b_sample_shape = torch.Size([order])
        self.w_sample_shape = torch.Size([order])

        self.b = self.b_dist.sample(self.b_sample_shape)
        self.w = self.w_dist.sample(self.w_sample_shape)  # .squeeze(1)
        self.order = order
        self.params = None

    def __call__(self, x):
        """
        Returns the value of these random features evaluated at the inputs x.

        output shape: [x.shape[0], self.order]
        """
        if len(x.shape) == 1:  # i.e. the single dimension is implicit
            x = x.unsqueeze(1)

        n = x.shape[0]
        b = self.b.repeat(n, 1).t()
        z = (math.sqrt(2.0 / self.order) * torch.cos(self.w @ x.t() + b)).t()
        # breakpoint()
        # print(z.shape)
        return z

    def get_w(self):
        """
        Getter for feature spectral weights.
        """
        return self.w


class OrthonormalBasis(Basis):
    def __init__(
        self,
        basis_function: OrthogonalPolynomial,
        weight_function: Callable,
        dimension: int,
        order: int,
        params: dict = None,
    ):
        assert isinstance(
            basis_function, OrthogonalPolynomial
        ), "the basis function should be of type OrthogonalPolynomial"
        super().__init__(basis_function, dimension, order, params)
        self.weight_function = weight_function

    def __call__(self, x):
        """
        Returns the whole basis evaluated at an input. The difference between an
        orthonormal basis and a standard set of polynomials is that the weight
        function and normalising constant
        can be applied "outside" the tensor of orthogonal polynomials,
        so it is feasible to do this separately (and therefore faster).
        """
        ortho_poly_basis = super().__call__(x)
        result = torch.sqrt(self.weight_function(x))
        return torch.einsum("ij,i -> ij", ortho_poly_basis, result)

    def set_gammas(self, gammas):
        """
        Updates the gammas on the basis function and the
        weight function.
        """
        self.basis_function.set_gammas(gammas)
        self.weight_function.set_gammas(gammas)


def smooth_exponential_basis(x: torch.Tensor, deg: int, params: dict):
    print("THIS SHOULD NOT BE BEING USED ANYWHERE")
    # breakpoint()
    """
    The smooth exponential basis functions as constructed in Fasshauer (2012),
    "second" paramaterisation. It is orthogonal w.r.t the measure ρ(x)
    described there.

    The Hermite function here is that constructed with the Physicist's Hermite
    Polynomial as opposed to the Probabilist's.


    : param x: the input points to evaluate the function at. Should be of
               dimensions [n,d]
    : param deg: degree of polynomial used in construction of func
    : param params: dict of parameters. keys set on case-by-case basis
    """
    # breakpoint()
    if deg > 42:
        print(
            r"Warning: the degree of the hermite polynomial is relatively"
            + "high - this may cause problems with the hermvals function."
            + "Try lowering the order, or use a different basis function."
        )

    epsilon = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    alpha = torch.diag(params["precision_parameter"])  # precision parameter
    # of the measure w.r.t the hermite functions are orthogonal

    sigma = torch.sqrt(params["variance_parameter"])

    # β = (1 + (2ε/α)^2)^(1/4)
    beta = torch.pow((1 + torch.pow((2 * epsilon / alpha), 2)), 0.25)

    log_gamma = 0.5 * (
        torch.log(beta)
        - (deg) * torch.log(torch.tensor(2, dtype=torch.float))
        - torch.lgamma(torch.tensor(deg + 1, dtype=torch.float))
    )

    delta_2 = (torch.pow(alpha, 2) / 2) * (torch.pow(beta, 2) - 1)

    multiplicative_term = torch.exp(
        log_gamma - (delta_2 * torch.pow(x, torch.tensor(2)))
    )

    # calculate the Hermite polynomial term
    # remember, deg = n-1
    hermite_term = hermite_function(alpha * beta * x, deg)

    # for numerical reasons, we can save the log of the absolute to get the
    #  right value,
    # then exponentiate and multiply by a mask of negative values to get the
    # right value.

    abs_hermite_term = torch.log(torch.abs(hermite_term))
    # breakpoint()
    mask = torch.where(
        hermite_term < 0,
        -torch.ones(hermite_term.shape),
        torch.ones(hermite_term.shape),
    )
    phi_d = mask * torch.exp(abs_hermite_term) * multiplicative_term
    phi = sigma * phi_d
    # if deg == 56 or deg == 57:
    #     breakpoint()
    return phi


def smooth_exponential_eigenvalues(deg: int, params: dict):
    print("THIS SHOULD NOT BE BEING USED ANYWHERE")
    # breakpoint()
    """
    Returns the vector of eigenvalues, up to length deg, using the parameters
    provided in params. This comes from Fasshauer2012 - where it is explained
    that the smooth exponential basis is orthogonal w.r.t a given measure
    (with precision alpha, etc.)

    :param deg: the degree up to which the eigenvalues should be computed.
    :param params: a dictionary of parameters whose keys included
    """

    eigenvalues = torch.zeros(deg)

    for i in range(1, deg + 1):
        epsilon = torch.diag(params["ard_parameter"])  # ε  - of dimension d
        alpha = torch.diag(params["precision_parameter"])  # precision
        # of the measure w.r.t the hermite functions are orthogonal

        beta = torch.pow((1 + torch.pow((2 * epsilon / alpha), 2)), 0.25)

        delta_2 = 0.5 * torch.pow(alpha, 2) * (torch.pow(beta, 2) - 1)

        denominator = torch.pow(alpha, 2) + delta_2 + torch.pow(epsilon, 2)

        left_term = alpha / torch.sqrt(denominator)
        right_term = torch.pow((torch.pow(epsilon, 2) / denominator), i - 1)

        lamda_d = left_term * right_term
        eigenvalue = torch.prod(lamda_d, dim=1)
        eigenvalues[i - 1] = eigenvalue
    # breakpoint()
    # breakpoint()
    return eigenvalues


def smooth_exponential_basis_fasshauer(
    x: torch.Tensor, deg: int, params: dict
):
    """
    The smooth exponential basis functions as constructed in Fasshauer (2012),
    "first" paramaterisation:
        a, b, c.

    The Hermite function here is that constructed with the Physicist's Hermite
    Polynomial as opposed to the Probabilist's.

    : param x: the input points to evaluate the function at. Should be of
               dimensions [n,d]
    : param deg: degree of polynomial used in construction of func
    : param params: dict of parameters.
                    Required keys:
                        - precision_parameter
                        - ard_parameter
    """
    # breakpoint()
    if deg > 42:
        print(
            r"Warning: the degree of the hermite polynomial is relatively"
            + "high - this may cause problems with the hermvals function."
            + "Try lowering the order, or use a different basis function."
        )

    a = torch.diag(params["precision_parameter"])  # precision parameter
    b = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    # of the measure w.r.t the hermite functions are orthogonal
    c = torch.sqrt(a ** 2 + 2 * a * b)
    # sigma = torch.sqrt(params["variance_parameter"])
    if (c != c).any():
        print("c in fasshauer is NaN!")
        breakpoint()

    # β = (1 + (2ε/α)^2)^(1/4)
    log_const_term = -0.5 * (
        deg * torch.log(torch.tensor(2))
        + torch.lgamma(torch.tensor(deg) + 1)
        + 0.5 * (torch.log(a) - torch.log(c))
    )

    # calculate the Hermite polynomial term
    hermite_term = hermite_function(
        torch.sqrt(2 * c) * x,
        deg,
    )

    phi_d = (
        torch.exp(log_const_term - ((c - a) * torch.pow(x, torch.tensor(2))))
        * hermite_term
    )
    return phi_d


def smooth_exponential_eigenvalues_fasshauer(deg: int, params: dict):
    """
    Returns the vector of eigenvalues, up to length deg, using the parameters
    provided in params.
    :param deg: the degree up to which the eigenvalues should be computed.
    :param params: a dictionary of parameters whose keys included
    """
    b = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    a = torch.diag(params["precision_parameter"])  # precision
    c = torch.sqrt(a ** 2 + 2 * a * b)
    left_term = torch.sqrt(2 * a / (a + b + c))
    right_term = b / (a + b + c)

    # construct the vector
    exponents = torch.linspace(0, deg - 1, deg)
    eigenvalues = left_term * torch.pow(right_term, exponents)
    return eigenvalues.squeeze()


def get_linear_coefficients_fasshauer(
    intercept: torch.Tensor, slope: torch.Tensor, params: dict
):
    """
    Returns the coefficients that represent a function that has the
    same slope and intercept at 0 for the fasshauer parameterised smooth
    exponential basis
    ( i.e. smooth_exponential_basis_fasshauer().

    :   param intercept: the intercept of the linear function we want to
                         approximate
    :   param slope:     the slope of the linear function we want to
                         approximate
    """
    b = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    a = torch.diag(params["precision_parameter"])  # precision
    c = torch.sqrt(a ** 2 + 2 * a * b)

    basis_intercept = intercept * torch.pow(a / c, 0.25)
    basis_slope = 0.5 * slope * torch.pow(a, 0.25) / torch.pow(c, 0.75)
    return (basis_intercept, basis_slope)


def standard_chebyshev_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a standard Chebyshev basis function, using Chebyshev
    polynomials of the first kind, evaluated at x, that does
    not impose the zero-endpoint condition of Babolian

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should
                     contain keys 'upper_bound' and 'lower_bound':
                     tensors representing the upper and lower bound
                     over which the function is to be a basis respectively.
    """

    # check that x is of the right dimension:
    # assert len(x.shape) == 2,\
    #    "inputs to basis function should have 2 dimensions."

    if "lower_bound" not in params:
        lower_boundary = torch.tensor(-1, dtype=float)
    else:
        lower_boundary = params["lower_bound"]

    if "upper_bound" not in params:
        upper_boundary = torch.tensor(1, dtype=float)
    else:
        upper_boundary = params["upper_bound"]

    if "chebyshev" not in params:
        # default result: chebyshev polynomials/basis of the second kind
        chebyshev = "second"
    else:
        chebyshev = params["chebyshev"]

    assert chebyshev in {
        "first",
        "second",
    }, 'chebyshev should be either "first" or "second"'

    # Transform to [-1,1] for processing
    z = (2 * x - (upper_boundary + lower_boundary)) / (
        upper_boundary - lower_boundary
    )

    if chebyshev == "first":
        chebyshev_term = chebyshev_first(z, deg)
        # exponent of weight function (1-z**2)
        weight_power = torch.tensor(-0.25)

        # define the normalising constant
        if deg == 0:
            normalising_constant = math.sqrt(2 / math.pi)
        else:
            normalising_constant = 2 / math.sqrt(math.pi)

    elif chebyshev == "second":
        chebyshev_term = chebyshev_second(z, deg)

        # exponent of weight function (1-z**2)
        weight_power = torch.tensor(0.25)

        # define the normalising constant
        normalising_constant = 2 / math.sqrt(
            (upper_boundary - lower_boundary) * (math.pi)
        )

    # define weight function
    weight_term = torch.pow(1 - z ** 2, weight_power)
    if (chebyshev_term != chebyshev_term).any():
        raise ValueError(
            "Chebyshev returning NaNs. Ensure"
            + "it is being evaluated within boundaries."
        )

    return weight_term * chebyshev_term * normalising_constant


def standard_haar_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a standard Chebyshev basis, using Chebyshev
    polynomials of the first kind, that does not impose
    the zero-endpoint condition of Babolian

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should
                     contain keys 'upper_bound' and 'lower_bound':
                     tensors representing the upper and lower bound
                     over which the function is to be a basis respectively.
    """
    pass


if __name__ == "__main__":
    # pass
    dim = 1
    order = 5000
    point_count = 1000
    spectral_distribution = D.Normal(torch.zeros(dim), torch.ones(dim))
    rff = RandomFourierFeatureBasis(dim, order, spectral_distribution)
    # x = torch.linspace(-1, 1, 100)
    # x = torch.ones((order, dim))
    x = torch.linspace(-3, 3, point_count)

    data = rff(x)
    # breakpoint()
    plt.plot(x.numpy().flatten(), torch.sum(data, dim=1).numpy().flatten())
    plt.show()
