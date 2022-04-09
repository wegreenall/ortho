import torch
from torch import nn
from torch.nn import Module
import torch.distributions as D

# from ortho.orthopoly import OrthogonalPolynomial
# from ortho.inverse_laplace import build_inverse_laplace_from_moments
# from typing import Callable
import matplotlib.pyplot as plt


class MaximalEntropyDensity:
    """
    For a given sequence of moments {1, μ_1, μ_2, ...),
    such that the Hankel determinant is positive (i.e. they come from an OPS
    as given by Favard's theorem with  γ_n > 0 for n >= 1.

    To get this, we take e^(-m'M^{_1}x).
    where m+1 is the vector of moments 0 to k, and M is the Hankel matrix of
    moments from 0 to 2k; x is the polynomial up to order k.
        i.e. {1, x, ..., x^k}

    The moment_net output should be equal in size to 2 * order.
    """

    def __init__(self, order: int, betas: torch.Tensor, gammas: torch.Tensor):
        # self.moment_net = moment_net
        self.order = order
        assert (
            betas.shape[0] == 2 * order + 1
        ), "Please provide at least 2 * order + 1 values for beta"
        assert (
            gammas.shape[0] == 2 * order + 1
        ), "Please provide at least 2 * order + 1 values for gamma"
        self.betas = betas
        self.gammas = gammas
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # xsigns = torch.sign(x)
        if len(x.shape) > 1:
            x = x.squeeze()

        poly_term = self.get_poly_term(x)
        lambdas = self._prep_lambdas()

        return torch.exp(-torch.einsum("ij, j -> i", poly_term, lambdas))

    def get_poly_term(self, x):
        z = x.repeat(self.order + 1, 1)
        powers_of_x = torch.pow(
            z.t(), torch.linspace(0, self.order, self.order + 1)
        )
        return powers_of_x

    def _prep_lambdas(self):
        moments = self.moment_generator()
        moment_vector = moments[: self.order + 1]
        moment_matrix = moments.unfold(0, self.order + 1, 1)  # moment matrix

        system_matrix = torch.einsum(
            "i, j, ij -> ij",
            1 / torch.linspace(2, self.order + 1, self.order + 1),
            torch.linspace(1, self.order + 1, self.order + 1),
            moment_matrix,
        )  # ratio matrix

        # print("About to get the lambdas!")
        # breakpoint()
        try:
            lambdas = torch.linalg.solve(system_matrix, moment_vector)
        except RuntimeError:
            print("The system_matrix is singular!")
            breakpoint()
        # print(lambdas)
        # breakpoint()
        return lambdas

    def moment_generator(self):
        order = len(self.betas)
        ones = torch.ones(order)
        cat_matrix = torch.zeros(order + 2, order + 2)
        cat_matrix[0, 0] = 1
        # jordan = jordan_matrix(betas, gammas)
        for n in range(1, order):
            # breakpoint()
            cat_matrix[n, 1:-1] = (
                cat_matrix[n - 1, :-2].clone() * ones
                + cat_matrix[n - 1, 1:-1].clone() * self.betas
                + cat_matrix[n - 1, 2:].clone() * self.gammas
            )

        return cat_matrix[1:-1, 1]


def jordan_matrix(s, t):
    order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([ones[i], s[i], t[i]])
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([ones[i], s[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([ones[i]])
    # for i
    # print(matrix)
    # breakpoint()
    return matrix[:, :].t()


def weight_mask(order) -> (torch.Tensor, torch.Tensor):
    # order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    matrix_2 = torch.zeros((order + 1, order))

    # block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([zeros[i], ones[i], ones[i]])
            matrix_2[i, i : i + 3] = torch.tensor(
                [ones[i], zeros[i], zeros[i]]
            )
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([zeros[i], ones[i]])
            matrix_2[i, i : i + 2] = torch.tensor([ones[i], zeros[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([zeros[i]])
    return matrix[:, :].t(), matrix_2[:, :].t()


# class CatNet(nn.Module):
# """
# A Catalan matrix represented as a Neural net, to allow
# us to differentiate w.r.t. its parameters easily.
# """

# def __init__(self, order, initial_s, initial_t):
# # Check the right size of the initial s
# if (initial_s.shape[0] != 2 * order + 1) or (
# initial_t.shape[0] != 2 * order + 1
# ):
# raise ValueError(
# r"Please provide at least 2 * order + 1 parameters for beta and gamma"
# )
# # breakpoint()
# super().__init__()
# self.order = 2 * order + 1

# self.mask, self.ones_matrix = weight_mask(len(initial_s))
# self.jordan = torch.nn.Parameter(
# jordan_matrix(initial_s, initial_t)
# )  # .t()
# # print(jordan)
# self.layers = []
# self.shared_weights = []
# for i in range(1, 2 * order + 1):
# self.shared_weights.append(
# torch.nn.Parameter(self.jordan[: i + 1, :i])
# )
# layer = nn.Linear(i, i + 1, bias=False)
# with torch.no_grad():
# layer.weight = torch.nn.Parameter(self.jordan[: i + 1, :i])
# self.layers.append(layer)

# return

# def forward(self, x):
# """
# As is, with s = 0 and t = 1, the resulting values should be the Catalan
# numbers. The parameters need to be shared between all the layers...

# This needs to be done by sourcing the parameters from the Jordan matrix
# but i'm not sure that this is what is happening here.
# """
# catalans = torch.zeros(self.order)
# # breakpoint()
# for i, layer in enumerate(self.layers):
# # breakpoint()
# with torch.no_grad():
# print("Shared weights are:", self.shared_weights[3])
# layer.weight *= self.mask[: i + 2, : i + 1]
# layer.weight += self.ones_matrix[: i + 2, : i + 1]
# # breakpoint()
# # print(layer.weight.shape)
# x = layer(x)
# # breakpoint()
# catalans[i] = x[
# -1
# ]  # the last in the layer will be the corresponding catalan no.

# return torch.hstack(
# (
# torch.tensor(
# 1,
# ),
# catalans[:-1],
# )
# )


# class coeffics_list:
# """
# A small class to handle the use of object-by-reference in Python -
# Used in the construction of the tensor of coefficients for the construction
# of the sequence of moments for a given orthogonal polynomial. That is,
# for a given OrthogonalPolynomial class with given sequences of betas
# and gammas, the resulting moments result from the three-term recursion.
# """

# def __init__(self, order):
# """
# Initialises the data to be a sequence of zeroes of length order + 1;
# this is because an order 3 polynomial will need 4 coefficients in the
# matrix used to solve for the moments
# """
# self.order = order
# self.data = torch.zeros(order + 1)  # the length is the order + 1

# def update_val(self, k, value):
# self.data[k] += value  # we _update_ the value

# def get_vals(self):
# return self.data

# def __len__(self):
# return self.order

# def __repr__(self):
# return self.data.__repr__()


# def get_moments_from_poly(poly):
# """
# For a given polynomial, computes the moments of the measure w.r.t which the
# polynomial is orthogonal, utilising the three-term recurrence that defines
# the orthogonal polynomial, in conjunction with the Linear functional
# operator L.

# For more details, check the documentation for L

# """
# order = poly.get_order()
# final_coefficients = torch.zeros([order, order])

# for i in range(order):
# coeffics = coeffics_list(i)
# L((0, i), 1, coeffics, poly)
# final_coefficients[i, : (i + 1)] = coeffics.get_vals()

# coeffics_inv = torch.linalg.inv(final_coefficients)
# targets = torch.zeros(order)
# targets[0] = 1  # assuming we're getting a probability distribution
# mus = torch.einsum("ij,j->i", coeffics_inv, targets)

# return mus


# def get_measure_from_poly(poly) -> Callable:
# # get the moments:
# moments = get_moments_from_poly(poly)
# """
# The sequence of moments from the polynomial imply a unique linear moment
# functional, but multiple corresponding densities. To that extent, we can
# choose the one that maximises the Shannon informational entropy, subject
# to the constraints implied by the moments -
# see, "Inverse two-sided Laplace transform for probability density
# functions", (Tagliani 1998)
# """
# # to build the density, we solve the following problem:
# # min μ_0 ln(1/μ_0  \int exp(-Σλ_j t^j) dt ) + Σλ_j μ_j
# sigma = 2
# b = 1
# measure = build_inverse_laplace_from_moments(moments, sigma, b)
# return measure


# def L(
# loc: tuple,
# value: float,
# coeffics: coeffics_list,
# poly: OrthogonalPolynomial,
# ):
# """
# Recursively calculates the coefficients for the construction of the moments

# This is a representation of the linear functional operator:
# <L, x^m P_n>  === L(m, n)


# Places into coeffics the coefficients of order len(coeffics).

# The aim is to build the matrix of coefficients to solve for the moments:

# [1       0     0  ... 0] |μ_0|   |1|
# [-β_0    1     0       ] |μ_1|   |0|
# [-γ_1  -β_0    1  ... 0] |μ_2| = |0|
# [          ...         ] |...|   |0|
# [?       ?        ... 1] |μ_k|   |0|

# To find the coefficients for coefficient k, start with a list of length k,
# and pass (0,k) to the function L which represents the linear functional
# operator for the OPS.

# L takes (m,n) and a value and calls itself on [(m+1, n-1), 1*value],
# [(m, n-1), -β_{n-1}*value]
# [(m, n-2), -γ_{n-1}*value]
# Mutiply all the values passed by the value passed in.

# The function places in coeffics a list of length k, the coefficients
# for row k in the above described matrix.

# :param loc: a tuple (m, n) representing that this is L(m,n) == <L, x^m P_n>
# :param value: a coefficient value passed in will multiply the coefficient
# found in this iteration
# :param coeffics: the coeffics_list object that will be updated
# :param poly: the orthogonal polynomial system w.r.t which we are building
# the linear moment functional - i.e. this polynomial system is
# orthogonal with respect to the measure whose moments are
# produced by this linear moment functional
# """
# local_betas = poly.get_betas()
# local_gammas = poly.get_gammas()

# if loc[1] == 0:  # n is 0
# # i.e loc consists of a moment (L(m, 0))
# # add the value to the corresponding coefficient in the array
# coeffics.update_val(loc[0], value)

# # check if it's the first go:
# if not (loc[1] < 0 or (loc[0] == 0 and loc[1] != len(coeffics))):
# # m != 0, n != 0
# left_loc = (loc[0] + 1, loc[1] - 1)
# centre_loc = (loc[0], loc[1] - 1)
# right_loc = (loc[0], loc[1] - 2)
# L(left_loc, value, coeffics, poly)
# L(centre_loc, value * -local_betas[loc[1] - 1], coeffics, poly)
# L(right_loc, value * -local_gammas[loc[1] - 1], coeffics, poly)
# pass


if __name__ == "__main__":
    # build an orthogonal polynomial
    order = 30

    test_cat_net = False
    test_moment_gen = True
    # build random betas and gammas
    if test_cat_net:
        for i in range(1, 10):
            start_point = 5
            betas = 0 * torch.ones(2 * order + 1)
            # betas = D.Normal(0.0, 0.2).sample([2 * order + 1])
            # gammas = i * torch.ones(2 * order + 1)
            # betas = D.Exponential(1.0).sample([order])
            gammas = D.Exponential(1 / 8.0).sample([2 * order + 1]) + 2
            # plt.hist(gammas.numpy().flatten())
            # plt.show()
            gammas[0] = 1

            my_net = CatNet(order, betas, gammas)
            my_measure = MaximalEntropyDensity(my_net, order)
            fineness = 2000
            x_axis = torch.linspace(-start_point, start_point, fineness)
            measure_values = my_measure(x_axis)
            breakpoint()
            print("About to plot the measure density")
            plt.plot(x_axis.detach().numpy(), measure_values.detach().numpy())
        plt.show()
    if test_moment_gen:
        betas = 0 * torch.ones(order)
        gammas = 1 * torch.ones(order)

        catalans = MomentGenerator()
        result = catalans(betas, gammas)
        print(result)
    # poly = OrthogonalPolynomial(order, betas, gammas)
    # mus = get_moments_from_poly(poly)
    # print("moments from a random polynomial linear moment functional:", mus)

    # # testing the moment acquirer
    # s = 0 * torch.ones(1 * order)
    # t = 1 * torch.ones(1 * order)
    # order = 8
    # my_net = CatNet2(order, betas, gammas)
    # my_net(torch.Tensor([1.0]))
    # catalans = my_net(torch.Tensor([1.0]))
    # # optimiser = torch.optim.SGD(my_net.shared_weights, 0.001)
    # print(catalans)
