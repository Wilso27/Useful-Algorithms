# newtons_method.py


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import sympy as sy


def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    # initialize counter and convergence boolean
    c = 0
    converges = False

    if np.isscalar(x0): # if n = 1
        # loop through maxiter times and break if tol is met
        for k in range(maxiter):
            c += 1 #count iterations

            # use (9.3)
            x1 = x0 - alpha*f(x0)/Df(x0)
            if abs(x0 - x1) < tol: # check tolerance
                converges = True
                break
            x0 = x1 # iterate
    else: # if n > 1
        for k in range(maxiter):
            c += 1 #count iterations

            # use (9.5)
            x1 = x0 - alpha * la.solve(Df(x0),f(x0))
            if la.norm(x0 - x1,2) < tol: # check tolerance
                converges = True
                break
            x0 = x1 # iterate

    return x1, converges, c


def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    f = lambda r: (P1*((1 + r)**N1 - 1)) - (P2*(1 - (1 + r)**(-N2)))
    df = lambda r: P1*N1*(1 + r)**(N1 - 1) - P2*N2*(1 + r)**(-N2 - 1)
    x0 = 0.1
    r = newton(f,x0,df,)[0]
    return r


def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    # initialize alphas and array to store number of iterations
    alpha = np.linspace(0,1,1000)[1:1000]
    codomain = np.array(())

    # loop through each alpha and store the number of iterations
    for i in alpha:
        codomain = np.append(codomain, newton(f,x0,Df,tol,maxiter,alpha=i)[2])

    # plot alpha against the number of iterations
    plt.plot(alpha,codomain)
    plt.show()

    # return the alpha value
    return alpha[np.where(codomain == np.min(codomain))][0] # optimal alpha


def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    # create functions
    f = lambda x: np.array([5*x[0]*x[1] - x[0] * (1 + x[1]) , -x[0]*x[1] + (1 - x[1]) * (1 + x[1])])
    df = lambda x: np.array([[5*x[1] - (1 + x[1]),5*x[0] - x[0]],[-x[1], 1 - x[0] - 2 * x[1] - 1]])

    # create the square
    X = np.linspace(-.25,0,100,endpoint=False)
    Y = np.linspace(.01,.25,100)

    for x in X: # loop through x and y values in the square
        for y in Y:
            # approximate the zeros
            x0 = newton(f,np.array([x, y]),df)[0]
            x0_1 = newton(f, np.array([x, y]), df,alpha=.55)[0]
            # check if they converged
            if (np.allclose(x0,np.array([0,1])) or np.allclose(x0,np.array([0,-1]))) and np.allclose(x0_1,np.array([3.75,.25])):
                return x,y


def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    #create the grid
    A = domain
    r_min, r_max, i_min, i_max = A[0], A[1], A[2], A[3]
    x_real = np.linspace(r_min, r_max, res)  # Real parts.
    x_imag = np.linspace(i_min, i_max, res)  # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j * X_imag  # Combine real and imaginary parts.
    X_K = X_0
    #loop through the iterations
    for i in range(iters):
        X_K = X_K - (f(X_K) / Df(X_K))
    Y = np.zeros_like(X_K)
    # loop through the resolution grid
    for i in range(res):
        for j in range(res):
            a = X_K[i][j]
            L = []
            for k in range(len(zeros)):
                L.append(abs(a - zeros[k]))
            Y[i][j] = np.argmin(L)
    #plot the result
    plt.pcolormesh(X_real, X_imag, Y.astype(float))
    plt.show()
