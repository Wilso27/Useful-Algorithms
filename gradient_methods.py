# gradient_methods.py


import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import scipy.linalg as la


def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether the algorithm converged.
        (int): The number of iterations computed.
    """
    # initialize variables
    i = 0
    xk = x0

    # loop until maxiter or tol is reached
    while i <= maxiter:
        i += 1 # counter

        # Minimize the function along the line
        alpha = opt.minimize_scalar(fun=lambda a: f(xk - a * Df(xk).T)).x

        # Use the algorithm
        xk = xk - alpha * Df(xk).T

        # Check tolerance
        if la.norm(Df(xk), ord=np.inf) < tol:
            return xk, True, i

    return xk, False, i


def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether the algorithm converged.
        (int): The number of iterations computed.
    """
    # initialize variables
    n = Q.shape[0]
    rk = Q@x0 - b
    dk = -rk
    xk = x0
    k = 0

    # loop until tol reached or max iters surpassed
    while np.linalg.norm(rk, ord=np.inf) >= tol:
        if k >= n:
            return xk, False, k
        ak = rk@rk / (dk @ Q @ dk)  # alphas calculated
        xk = xk + ak * dk  # do the descent algorithm
        r1 = rk + ak * Q @ dk
        beta = r1@r1 / (rk@rk)
        dk = -r1 + beta * dk
        k += 1
        rk = r1

    return xk, True, k


def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether the algorithm converged.
        (int): The number of iterations computed.
    """
    # initialize variables by given equations
    xk = x0
    rk = -df(xk)
    dk = rk
    ak = opt.minimize_scalar(fun=lambda a: f(xk + a * dk)).x
    xk = xk + ak * dk
    k = 1

    # loop through until maxiter or tol
    while la.norm(rk) >= tol:
        if k >= maxiter:
            return xk, False, k
        # use algorithm
        r1 = -df(xk)
        bk = r1@r1 / (rk@rk)
        dk = r1 + bk * dk

        # optimize scalar to get alpha
        ak = opt.minimize_scalar(fun=lambda a: f(xk + a * dk)).x
        xk = xk + ak * dk
        k += 1
        rk = r1

    return xk, True, k



def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    # get A matrix from data
    A = np.loadtxt(filename)

    # get b from first column of data
    b = A[:, 0].copy()

    # set first column to 1
    A[:, 0] = 1

    # create least squares problem
    Q = A.T@A
    b1 = A.T@b

    return conjugate_gradient(Q, b1, x0)[0]


class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        # define the likelihood function
        def L_func(b): return np.sum(np.log(1+np.exp(-(b[0] + b[1]*x))) + (1-y)*(b[0] + b[1]*x))

        # Store as attributes
        self.b0, self.b1 = opt.fmin_cg(L_func, guess, disp=False)

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        sig = 1 / (1+np.exp(-(self.b0 + self.b1*x)))
        return sig


def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    # set up logreg
    data = np.load(filename).T
    reg = LogisticRegression1D()
    reg.fit(data[0], data[1], guess)

    domain = np.linspace(30,100,200)

    # predict damage on that day
    x = 31
    y = reg.predict(x)

    # plot the data
    plt.plot(domain, reg.predict(domain))
    plt.scatter(data[0], data[1], label="Previous Damage")
    plt.scatter(x,y, label="P(Damage) at Launch")
    plt.title("Probability of O-Ring Damage")
    plt.ylabel("O-Ring Damage")
    plt.xlabel("Temperature")

    plt.tight_layout()
    plt.show()

    return y  # return damage for that day
