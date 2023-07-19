# iterative_solvers.py


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = sp.dok_matrix((n, n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr()
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1
    return A.tocsr() if as_sparse else A.toarray()


def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    n = A.shape[0]
    i = 1

    # Initialize D
    D_inv = 1/np.diag(A)

    # compute first iteration
    x_0 = np.zeros(n)
    x_1 = x_0 + D_inv * (b - A @ x_0)
    abs_err = np.array(np.linalg.norm(A@x_1 - b, ord=np.inf))

    # loop until tol or maxiter is reached
    while np.linalg.norm(x_1 - x_0, ord=np.inf) >= tol and i <= maxiter:
        x_0 = x_1
        x_1 = x_0 + D_inv * (b - A @ x_0)
        abs_err = np.append(abs_err, np.linalg.norm(A@x_1 - b, ord=np.inf))
        i += 1

    if plot:  # plot the convergence
        plt.title("Convergence of Jacobi Method")
        plt.ylabel("Abs Error of Approx")
        plt.xlabel("Iteration")
        plt.semilogy(np.arange(0, i), abs_err)
        plt.show()

    return x_1  # return approx solution to Ax = b


def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int)he maximum: T number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    n = A.shape[0]
    iter = 0

    # Initialize D
    A_diag = 1 / np.diag(A)

    # compute first iteration
    x_0 = np.zeros(n)

    abs_err = np.array([])
    while iter < maxiter:  # loop until maxiter is reached
        x_1 = x_0.copy()
        for i in range(n):
            x_1[i] = x_0[i] + A_diag[i] * (b[i] - A[i, :] @ x_1)

        abs_err = np.append(abs_err, np.linalg.norm(A @ x_1 - b, ord=np.inf))

        # check tol
        if np.linalg.norm(x_1 - x_0, ord=np.inf) < tol:
            if plot:  # plot the error
                plt.title("Convergence of Gauss-Seidel Method")
                plt.ylabel("Abs Error of Approx")
                plt.xlabel("Iteration")
                plt.semilogy(np.arange(0, iter+1), abs_err)
                plt.show()
            return x_1
        x_0 = x_1
        iter += 1

    return x_1


def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    n = A.shape[0]
    iter = 0

    # compute first iteration
    x_0 = np.zeros(n)

    while iter < maxiter:  # break at maxiter
        x_1 = x_0.copy()
        for i in range(n):

            # use the code they provide to work with sparse matrix
            rowstart = A.indptr[i]
            rowend = A.indptr[i + 1]
            Aix = A.data[rowstart:rowend] @ x_1[A.indices[rowstart:rowend]]
            x_1[i] = x_0[i] + (b[i] - Aix) / A[i, i]

        # check tol
        if np.linalg.norm(x_1 - x_0, ord=np.inf) < tol:
            return x_1
        x_0 = x_1
        iter += 1

    return x_1


def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    n = A.shape[0]
    iter = 0
    A_diag = A.diagonal()

    # compute first iteration
    x_0 = np.zeros(n)

    for iter in range(maxiter):  # Loop through till maxiter
        x_1 = x_0.copy()

        for i in range(n):
            # Use the code given to add omega
            rowstart = A.indptr[i]
            rowend = A.indptr[i + 1]
            Aix = A.data[rowstart:rowend] @ x_1[A.indices[rowstart:rowend]]
            x_1[i] = x_0[i] + omega * (b[i] - Aix) / A_diag[i]

        # check tol
        if np.linalg.norm(x_1 - x_0, ord=np.inf) < tol:
            return x_1, True, iter + 1
        x_0 = x_1

    return x_1, False, iter + 1  # return x, boolean, iterations


def lin_systems_prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """

    # create B using diagonals
    diagonals = [[1 for i in range(n-1)]]
    diagonals.append([-4 for j in range(n)])
    diagonals.append([1 for k in range(n-1)])
    offsets = [-1, 0, 1]
    B = sp.diags(diagonals, offsets, shape=(n, n))

    # add them to a sparse matrix
    A = sp.block_diag((B for i in range(n)))
    A.setdiag([1 for m in range(n**2-n)], -n)
    A.setdiag([1 for m in range(n**2-n)], n)
    return A.tocsr()


def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    # construct the matrix
    A = lin_systems_prob5(n)
    tile = np.array([-100])
    tile = np.append(tile, [0 for i in range(n - 2)])
    tile = np.append(tile, -100)
    b = np.tile(tile, n)

    # use problem 5
    u, conv, iters = sor(A, b, omega, tol=tol, maxiter=maxiter)

    if plot:  # visualize the steady state
        u_reshape = u.reshape((n, n))
        plt.title("Hot Plate Visualization")
        plt.pcolormesh(u_reshape, cmap="coolwarm")
        plt.show()

    return u, conv, iters  # return u, convergence, iterations


def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """

    # initialize variables
    omega_list = np.arange(1, 2, .05)
    n = 20
    iters = []

    for omega in omega_list:  # loop through each omega
        iters.append(hot_plate(n, omega, tol=1e-2, maxiter=1000)[2])

    # plot the iterations against the omega values
    plt.plot(omega_list, iters)
    plt.show()

    return omega_list[np.argmin(iters)]  # return the optimal omega value
