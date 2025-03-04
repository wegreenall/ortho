import torch
import torch.distributions as D
from itertools import product
import matplotlib.pyplot as plt
from termcolor import colored


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
        stretched_x = x.unsqueeze(2).expand(-1, -1, MultivariateStieltjes.r(self.n, self.d))

        # create the multi-index
        multi_index = self._multi_index(self.n, self.d)
        monomial = torch.pow(stretched_x, multi_index).prod(dim=1)
        assert monomial.shape == (x.shape[0], MultivariateStieltjes.r(self.n, self.d))
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
        self.sample = sample

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
                  
        self.monomials = [
            MultivariateMonomial(n, d) for n in range(N + 1)
        ]  # an iterator of the monomials
        print(self.monomials)

    def __call__(self,n: int, x: torch.Tensor):
        result = torch.zeros(x.shape[0], int(MultivariateStieltjes.R(n, self.d)))
        for n, monomial in enumerate(self.monomials[:n+1]):
            monom_data = monomial(x)
            result[:, MultivariateStieltjes.R(n - 1, self.d) : MultivariateStieltjes.R(n, self.d)] = monom_data
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
    # plt.imshow(moment_matrix_value.numpy(), cmap="viridis")
    # plt.show()

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

class MultivariateStieltjes:
    def __init__(self, n: int, d: int, sample: torch.Tensor):
        # parameters
        self.n = n
        self.d = d
        self.sample = sample
        self.monomial_sequence = MultivariateMonomialSequence(n, d)

        # matrices
        self.As = []
        self.Bs = []

        # calculate the moment matrix
    def calculate(self):
        """
        The main function to calculate the A and B matrices.
        """
        for n in range(self.n):
            print(f"Calculating A and B for n={n}")
            A_n = torch.zeros(self.r(n, self.d), self.r(n, self.d), self.d)
            B_n = torch.zeros(self.r(n+1, self.d), self.r(n, self.d), self.d)
            for i in range(self.d):
                A_n = self.A(n, self.d)
                B_n = self.B(n, self.d)
                print(colored("A_n", "red"), A_n)
                print(colored("B_n", "blue"), B_n)

            self.As.append(A_n)
            self.Bs.append(B_n)

    def ops(self, n: int, inputs: torch.Tensor):
        """
        Returns the orthogonal polynomial sequence at the given input.
        """
        # assert inputs.shape[0] == self.d
        assert len(self.As)>=n, f"The matrix A_n does not exist for n={n}"
        assert inputs.shape[1] == self.d

        if n == -1:
            return torch.zeros(inputs.shape[0], self.r(0, self.d))
        elif n == 0:
            return torch.ones(inputs.shape[0], self.r(0, self.d))
        elif n == 1: # calculating p_1
            A_n = self.As[1] # A_{1, i} \forall i
            B_n = self.Bs[1] # B_{1, i} \forall i
            p_0 = self.ops(0, inputs)
            # einsum index key:
            # R = r_{n}
            # r = r_{n-1}
            # d = d (dimensions)
            # n = n (sample_size)
            term_1 = torch.einsum("nd, Rrd, nr -> nR", inputs,  B_n,  p_0)
            term_2 = torch.einsum("Rrd, rrd, nr -> nR",  B_n,  A_n,  p_0)
            # term_3 = B @ self.ops(-1, inputs)
            result = term_1 + term_2

        elif n > 1:
            # einsum index key:
            # R = r_{n}
            # r = r_{n-1}
            # p = r_{n-2}
            # d = d (dimensions)
            # n = n (sample_size)
            A_n = self.As[n]
            B_n = self.Bs[n]
            B_n_1 = self.Bs[n-1]
            p_n_1 = self.ops(n-1, inputs)
            p_n_2 = self.ops(n-2, inputs)

            print("getting term_1")
            # breakpoint()
            term_1 = torch.einsum("nd, Rrd, nr -> nR", inputs,  B_n,  p_n_1)
            print("getting term_2")
            term_2 = torch.einsum("Rrd, rrd, nr -> nR",  B_n,  A_n,  p_n_1)
            print("getting term_3")
            term_3 = torch.einsum("Rrd, rpd, np -> nR",  B_n,  B_n_1,  p_n_2)
            
            result = term_1 + term_2 + term_3

        # get the inverse eigenvalue matrix:
        L = torch.einsum("Rrd, Prd -> RP", B_n, B_n)
        # breakpoint()
        # breakpoint()
        L_inv = torch.inverse(L)
        assert result.shape == (inputs.shape[0], self.r(n, self.d))
        return result @ L_inv

    @staticmethod
    def r(n: int, d: int) -> int:
        """
        This is the number of monomials of degree n in d dimensions.
        """
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

    @staticmethod
    def r_delta(n: int, d: int) -> int:
        """
        This is the number of monomials of degree n in d dimensions minus the
        number of monomials of degree n-1 in d dimensions.
        """
        return MultivariateStieltjes.r(n, d) - MultivariateStieltjes.r(n - 1, d)

    @staticmethod
    def R(n: int, d: int) -> int:
        """
        This is the dimensionality of the space spanned by multinomials up to
        degree n in d dimensions.
        """
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


    def S(self, n: int, d: int):
        """
        Returns the moment matrix S_n,i for the given sample, where:
                     S_n,i = \int x_i p_n p_n^t dμ(x)
        """
        output = torch.zeros(self.r(n, d), self.r(n, d))
        polynomial_evaluation = self.ops(n, self.sample)
        result = torch.einsum("nd, ni, nj -> ijd", self.sample, polynomial_evaluation, polynomial_evaluation)/self.sample.shape[0]
        return result

    def T(self, n: int, i: int, j: int, d: int):
        """
        Returns the moment matrix T_n,i,j for the given sample, where:
                     T_n,i,j = \int x_i x_j p_n p_n^t dμ(x)
        """
        pass


    def A(self, n: int, d: int):
        """
        Returns the matrix A_{n, i} in the recurrence for a given polynomial.
        
        Necessary for this calculation is P_{n-1}.
        """
        # polynomial_evaluation = self.ops(n-1, self.sample)
        return self.S(n-1, d)

    def B(self, n: int, d: int):
        """
        Calculates the matrix B_n,i in the recurrence.
        """

        if n == -1:  # "fall back" to the constant polynomial
            result = torch.zeros(self.r(n, d), self.r(n+1, d), d)
        elif n == 0:  # "fall back" to the moment matrices
            """
            See section 5.2.5 of the paper.

            ... Therefore, when d > 2 and n = 0, we use (18) to compute the B_{1,i}
                matrices.

            (18) A_{n+1,i} = \tilde{L}_n^{-1} G_{n,i} \tilde{L}_n^{-T}
                 B_{n+1,i} = \tilde{L}_n^{-1} \tilde{G}_{n+1,i} \tilde{L}_{n+1}^{-T}

            """
            """
            MOMENT MATRIX IS NOT DOING WHAT YOU THINK IT IS DOING
            """
            Gn = self.G(n, d)
            Gn_plus_1 = self.G(n+1, d)
            # breakpoint()

            Ln = torch.linalg.cholesky(Gn)
            Ln_inv = torch.inverse(Ln)
            # breakpoint()
            Ln_tilde = Ln_inv[-self.r(n, d):, :]
            assert Ln_tilde.shape == (self.r(n, d), self.R(n, d))
            print("just made Ln_tilde")

            Ln_plus_1 = torch.linalg.cholesky(Gn_plus_1)
            Ln_plus_1_inv = torch.inverse(Ln_plus_1)
            Ln_plus_1_tilde = Ln_plus_1_inv[-self.r(n+1, d):, :]
            assert Ln_plus_1_tilde.shape == (self.r(n+1, d), self.R(n+1, d))
            print("just made Ln_plus_1_tilde")
            
            # Gni = int x_i p_0 p_0^t dμ(x)
            Gni_plus_1 = torch.einsum("nd, ni, nj -> ijd", self.sample, self.monomial_sequence(n+1, self.sample), self.monomial_sequence(n+1, self.sample))/self.sample.shape[0]
            assert Gni_plus_1.shape == (self.R(n+1, d), self.R(n+1, d), self.d)
            print("just made Gni_plus_1")

            Gni_plus_one_tilde = Gni_plus_1[:self.R(n, d), :]
            assert Gni_plus_one_tilde.shape == (self.R(n, d), self.R(n+1, d), self.d)
            print("just made Gni_plus_1_tilde")

            print("About to make B_n_plus_1")
            # key:
            # r = r_n
            # p = r_n+1
            # R = R_n
            # P = R_{n+1}
            # d = dimension
            B_n_plus_1 = torch.einsum("rR, RPd, pP -> rpd", Ln_tilde, Gni_plus_one_tilde, Ln_plus_1_tilde)

            print("just made B_n_plus_1")
            # breakpoint()
            # B_n_plus_1 = Ln_inv @ Gni_plus_one_tilde @ Ln_plus_1_tilde.t()

            result = B_n_plus_1

        elif n > 0 and d > 2:
            """
            See section 5.2.6 of the paper.
            """
            # Gn = moment_matrix(n, d, self.sample, self.monomial_sequence)
            # Gn_plus_1 = moment_matrix(n+1, d, self.sample, self.monomial_sequence)

            # Ln = torch.linalg.cholesky(Gn)
            # Ln_inv = torch.inverse(Ln)
            # Ln_tilde = Ln_inv[-self.r(n, d):, :]
            # assert Ln_tilde.shape == (self.r(n, d), self.R(n, d))

            # Ln_plus_1 = torch.linalg.cholesky(Gn_plus_1)
            # Ln_plus_1_inv = torch.inverse(Ln_plus_1)
            # Ln_plus_1_tilde = Ln_plus_1_inv[-self.r(n+1, d):, :]
            # assert Ln_plus_1_tilde.shape == (self.r(n+1, d), self.R(n+1, d))

            
            # # Gni = int x_i p_0 p_0^t dμ(x)
            # Gni_plus_1 = torch.einsum("nd, ni, nj -> ij", self.sample, self.ops(n+1, self.sample), self.ops(n+1, self.sample))/self.sample.shape[0]
            # assert Gni_plus_1.shape == (self.R(n+1, d), self.R(n+1, d))

            # G_n_plus_one_tilde = Gni_plus_1[:self.R(n, d), :]
            # B_n_plus_1 = Ln_inv @ G_n_plus_one_tilde @ Ln_plus_1_tilde.t()
            # result = B_n_plus_1
        if d == 2:
            """
            See section 5.2.4 of the paper.
            """
            pass

        # result = torch.eye(self.r(n, d)).repeat(d, 1, 1).permute(1, 2, 0)
        # result = torch.ones(self.r(n, d), self.r(n-1, d), d)
        # breakpoint()
        """
        B_{n+1}, i = \in R^{r_{n-1} x r_{n}}
        B_n = \in R^{dr_{n-1} x r_{n-1}}
        """
        # breakpoint()
        assert result.shape == torch.Size((self.r(n, d), self.r(n+1, d), d)), f"Shape of B_n is {result.shape}, and it should be {(self.r(n, d), self.r(n+1, d), d)}"
        return result

    def U(self, n:int , i: int, d: int):
        """
        The U matrix is the first part of the SVD of the B matrix.
        """
        pass

    def G(self, n:int , d: int):
        """
        The G matrix is the moment matrix used to construct the B matrix.
        """
        Gn = torch.einsum("ni, nj -> ij", self.monomial_sequence(n, self.sample), self.monomial_sequence(n, self.sample))/self.sample.shape[0]
        # breakpoint()
        return Gn
    
    def V_hat(self, n:int , i: int, d: int):
        """
        The V_hat matrix is the first r_n columns of the SVD of the B matrix at n+1.
        """
        pass

    def V_tilde(self, n:int , i: int, d: int):
        pass

    def Sigma(self, n:int , d: int):
        pass


if __name__ == "__main__":
    test_multivariate_monomial = False
    test_stieltjes = True
    if test_multivariate_monomial:
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

    if test_stieltjes:
        msm = MultivariateStieltjes(5, 2, torch.randn(10000, 2))
        msm.calculate()
