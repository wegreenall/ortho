import torch
import torch.distributions as D
from itertools import product
import matplotlib.pyplot as plt


class MultivariateMonomial:
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d

    def __call__(self, x: torch.Tensor):
        """
        In the beginning, we will have this output the entire r_n shaped vector
        of monomials. If this does not work we will return to choosing
        the multi-index.

        return shape:
            (x.shape[0], r_n)
        """
        # create the stretched value of the right shape
        stretched_x = x.unsqueeze(2).expand(-1, -1, r(self.n, self.d))

        # create the multi-index
        multi_index = self._multi_index(self.n, self.d)
        monomial = torch.pow(stretched_x, multi_index).prod(dim=1)
        assert monomial.shape == (x.shape[0], r(self.n, self.d))
        return monomial

    def _multi_index(self, n: int, d: int) -> torch.Tensor:
        """
        Returns the multi-index of the monomial of exactly degree n in d
        dimensions. Released in lexicographical order.

        Return shape:
            (d, r_n)
        """
        result = filter(
            lambda x: sum(x) == n, product(list(range(n + 1)), repeat=d)
        )
        return torch.Tensor(list(result)).t()


class MultivariateOrthogonalPolynomialSequence:
    """
    Returns a sequence of MultivariateOrthogonalPolynomial objects of degree 0 to N
    """

    def __init__(self, N: int, d: int, sample: torch.Tensor):
        self.N = N
        self.d = d
        self.monomials = MultivariateMonomialSequence(N, d)
        self.L_inv = self._get_moment_matrix(sample)

    def __call__(self, x: torch.Tensor):
        return self.monomials(x) @ self.L_inv

    def _get_moment_matrix(self, sample: torch.Tensor):
        """
        Returns the Gram matrix:
            (G_n)_{i,j} = m(i,j)
            G_n \in R^{R_n x R_n}
            m(i,j) = int φ_i(x) φ_j(x) dμ(x)

        so \hat{m}(i,j) = \frac{1}{n} \sum_{k=1}^n φ_i(x_k) φ_j(x_k)

        Return shape:
            (R_n, R_n)
        """
        value = self.monomials(sample)
        moment_matrix_value = value.t() @ value / sample.shape[0]
        L = torch.linalg.cholesky(moment_matrix_value)
        L_inv = torch.inverse(L)
        return L_inv


class MultivariateMonomialSequence:
    """
    Returns a sequence of MultivariateMonomial objects of degree 0 to N
    """

    def __init__(self, N: int, d: int):
        self.N = N
        self.d = d
        print(N)
        self.monomials = [
            MultivariateMonomial(n, d) for n in range(N + 1)
        ]  # an iterator of the monomials
        print(self.monomials)

    def __call__(self, x: torch.Tensor):
        result = torch.zeros(x.shape[0], int(R(self.N, self.d)))
        for n, monomial in enumerate(self.monomials):
            monom_data = monomial(x)
            result[:, R(n - 1, self.d) : R(n, self.d)] = monom_data
        return result


def moment_matrix(
    n: int,
    d: int,
    sample: torch.Tensor,
    multivariate_monomial_sequence: MultivariateMonomialSequence,
):
    """
    Returns the Gram matrix:
        (G_n)_{i,j} = m(i,j)
        G_n \in R^{R_n x R_n}
        m(i,j) = int φ_i(x) φ_j(x) dμ(x)

    so \hat{m}(i,j) = \frac{1}{n} \sum_{k=1}^n φ_i(x_k) φ_j(x_k)

    Return shape:
        (R_n, R_n)
    """
    value = multivariate_monomial_sequence(sample)
    moment_matrix_value = value.t() @ value / sample.shape[0]
    print(moment_matrix_value.shape)
    plt.imshow(moment_matrix_value.numpy(), cmap="viridis")
    plt.show()

    L = torch.linalg.cholesky(moment_matrix_value)
    L_inv = torch.inverse(L)
    p = L_inv @ value.t()
    return p


"""
In all cases, the phrase "the paper" refers to:
    A Stieltjes algorihtm for generating multivariate orthogonal polynomials,
    Liu and Narayan, 2023.

To get A_{n+1},i, B_{n+1},i, the following calculation process is used, given
A_{n},i, B_{n},i, and B_{n-1},i:

    1. Calculate the moment matrix S_{n,i} and T_{n,i,j} for the given sample.
    2. calculate A_{n+1}, i = S_{n, i}
    3. Calculate \tilde{p}_{n+1} = x_i p_n - A_{n+1}, i p_n - B'_{n, i} p_{n-1}
    4. Calculate T_{n, i, j} = \int \tilde{p}_{n+1} \tilde{p}_{n+1}^t dμ(x)
    5. Calculate B_{n+1, i}:
        5a. Compute U_{n+1, i}, Σ_{n+1}, from B's SVD.
        5b. Compute V_{n+1, i}:
            5b1. Compute subblocks \hat{V}
            5b2. Compute V_{n+1, i} = V_{n+1, i} - A_{n+1, i} U_{n+1, i}
"""


def r(n: int, d: int):
    top = torch.Tensor([n + d - 1])
    bottom = torch.Tensor([n])
    return int(
        torch.exp(
            (
                torch.lgamma(top + 1)
                - torch.lgamma(bottom + 1)
                - torch.lgamma(top - bottom + 1)
            )
        )
    )


def R(n: int, d: int):
    top = torch.Tensor([n + d])
    bottom = torch.Tensor([n])
    return int(
        torch.exp(
            (
                torch.lgamma(top + 1)
                - torch.lgamma(bottom + 1)
                - torch.lgamma(top - bottom + 1)
            )
        )
    )


def S(n: int, i: int, d: int, sample: torch.Tensor):
    """
    Returns the moment matrix S_n,i for the given sample, where:
                 S_n,i = \int x_i p_n p_n^t dμ(x)
    """
    pass


def T(n: int, i: int, j: int, d: int, sample: torch.Tensor):
    """
    Returns the moment matrix T_n,i,j for the given sample, where:
                 T_n,i,j = \int x_i x_j p_n p_n^t dμ(x)
    """
    pass


def B(n: int, i: int, d: int, sample: torch.Tensor, B_prev: torch.Tensor):
    """
    Calculates the matrix B_n,i in the recurrence.
    """
    if n == 0 and d > 2:  # "fall back" to the moment matrices
        """
        See section 5.2.5 of the paper.

        ... Therefore, when d > 2 and n = 0, we use (18) to compute the B_{1,i}
            matrices.

        (18) A_{n+1,i} = \tilde{L}_n^{-1} G_{n,i} \tilde{L}_n^{-T}
             B_{n+1,i} = \tilde{L}_n^{-1} \tilde{G}_{n+1,i} \tilde{L}_{n+1}^{-T}
        """
    elif n > 0 and d > 2:
        """
        See section 5.2.6 of the paper.
        """
    if d == 2:
        """
        See section 5.2.4 of the paper.
        """


def A(n: int, i: int, d: int, sample: torch.Tensor, A_prev: torch.Tensor):
    pass


if __name__ == "__main__":
    multivariate_monomial = MultivariateMonomial(5, 2)

    # plot multivariate polynomial
    fineness = 200

    # sample = D.Normal(0, 6).sample((100000, 2))
    sample = D.MultivariateNormal(
        torch.zeros(2), torch.Tensor([[5.0, 0.0], [0.0, 2.0]])
    ).sample((100000,))

    # monomial_sequence = MultivariateMonomialSequence(5, 2)
    # moment_matrix(5, 2, sample, monomial_sequence)

    multivariate_orthogonal_polynomial_sequence = (
        MultivariateOrthogonalPolynomialSequence(5, 2, sample)
    )

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = torch.linspace(-5, 5, fineness)
    y = torch.linspace(-5, 5, fineness)
    X, Y = torch.meshgrid(x, y)
    Z = multivariate_orthogonal_polynomial_sequence(
        torch.stack([X.flatten(), Y.flatten()], dim=1)
    ).reshape(fineness, fineness, -1)
    print(Z.shape)
    ax.plot_surface(
        X.numpy(),
        Y.numpy(),
        Z[:, :, 12].numpy(),
        rstride=1,
        cstride=1,
        cmap="viridis",
    )
    plt.show()
