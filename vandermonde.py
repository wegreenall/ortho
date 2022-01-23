import torch

"""
Here we implement the Bjorck and Pereya algorithm, based in pytorch for 
differentiability.
"""


def solve(alpha, b):
    """
    Returns the solution c to the system:
        Vc = ψ

    for getting the coefficients that interpolate from a given polynomial.

    V(α_0, α_1, ..., α_n) = [1,     1,   ...,     1]
                            [α_0, α_1,   ...,   α_n]
                            [          ...         ]
                            [α^n_0, α_1, ..., α^n_n]

    From the paper (Bjorck and Pereyra, 1970):
        Algorithm for the primal system:
        Step (i). put d^(0) = b,
            for k = 0, 1, ..., n-1,
            compute:
                d_j^{K+1} = d_i^{k} - α_k d_i-1^{k}  for j = n, n-1, ..., k+1
                          = d_i^{k}                  for j = k, k-1, ...,   0
        Step (ii). put x^{n} = d^{n},
            for k= n-1, ..., 1, 0 compute
            x_j^{(k+1/2)} = x_j^{K}                    j = 0, 1, ..., k
            x_j^{(k+1/2)} = x_j^{K}/(a_i - a_{i-k-1})  j = k+1, ..., n-1, n
            x_j(k) = x_j^{k+1/2}                       j = 0,1,...,k-1,n
            x_j(k) = x_j^{k+1/2} - x_j+1               j = k,..., n-1
    """
    assert len(alpha) == len(b)

    n = len(b)
    x = b.clone()

    for k in range(1, n):
        x[k:n] = x[k:n] - x[k - 1 : n - 1]  # for k to
        x[k:n] /= alpha[k:n] - alpha[0 : n - k]

    for k in range(n - 1, 0, -1):
        x[k - 1 : n - 1] -= alpha[k - 1] * x[k:n]
    return x


def solve_transpose(alpha, b):
    """
    Returns the solution c to the system:
        V'a = f

    for getting the coefficients that interpolate from a given polynomial.

    V(α_0, α_1, ..., α_n) = [1,     1,   ...,     1]
                            [α_0, α_1,   ...,   α_n]
                            [          ...         ]
                            [α^n_0, α_1, ..., α^n_n]

    From the paper (Bjorck and Pereyra, 1970):
        Algorithm for the primal system:
        Step (i). put d^(0) = b,
            for k = 0, 1, ..., n-1,
            compute:
                d_j^{K+1} = d_i^{k} - α_k d_i-1^{k}  for j = n, n-1, ..., k+1
                          = d_i^{k}                  for j = k, k-1, ...,   0
        Step (ii). put x^{n} = d^{n},
            for k= n-1, ..., 1, 0 compute
            x_j^{(k+1/2)} = x_j^{K}                    j = 0, 1, ..., k
            x_j^{(k+1/2)} = x_j^{K}/(a_i - a_{i-k-1})  j = k+1, ..., n-1, n
            x_j(k) = x_j^{k+1/2}                       j = 0,1,...,k-1,n
            x_j(k) = x_j^{k+1/2} - x_j+1               j = k,..., n-1
    """
    n = len(b)
    x = b.clone()

    for k in range(n):
        x[k + 1 : n] -= alpha[k] * x[k : n - 1]

    for k in range(n - 1, 0, -1):
        x[k:n] /= alpha[k:n] - alpha[: n - k]
        x[k - 1 : n - 1] = x[k - 1 : n - 1] - x[k:n]
    return x


if __name__ == "__main__":
    n = 10
    a = torch.Tensor([(1 / (i + 3)) for i in range(n)])
    b = torch.Tensor([(1 / (2 ** i)) for i in range(n)])

    result = solve(a, b)
    result_2 = solve_transpose(a, b)
    # print(result)
    # print(result_2)
