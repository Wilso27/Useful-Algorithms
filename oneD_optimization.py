# oneD_optimization.py


import numpy as np
import scipy as sp


def golden_section(f, a, b, tol=1e-5, maxiter=100):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converges = False
    x0 = (a + b) / 2 # set the initial minimizer
    phi = (1 + np.sqrt(5)) / 2
    for i in range(maxiter): # iterate only maxiter times at most
        c = (b - a) / phi
        a_ = b - c
        b_ = a + c
        if f(a_) <= f(b_): # get new boundaries for the search interval
            b = b_
        else:
            a = a_
        x1 = (a + b) / 2 # set the minimizer approximation
        if abs(x0 - x1) < tol:
            converges = True
            break # stop iterating if the approximation stops changing enough
        x0 = x1
    return x1, converges, i + 1


def newton1d(df, d2f, x0, tol=1e-5, maxiter=100):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converges = False
    for i in range(maxiter): # stop after maxiter is reached
        # use (12.1)
        x1 = x0 - (df(x0) / d2f(x0))
        if abs(x0 - x1) < tol:  # check tolerance
            converges = True
            break # break once tolerance is met
        x0 = x1  # iterate
    return x1, converges, i + 1


def secant1d(df, x0, x1, tol=1e-5, maxiter=100):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converges = False
    for i in range(maxiter): # stop after maxiter is reached
        # use (12.3)
        dfx0 = df(x0)
        dfx1 = df(x1)
        num = x0 * dfx1 - x1 * dfx0
        denom = dfx1 - dfx0
        x2 = num / denom
        if abs(x1 - x2) < tol:  # check tolerance
            converges = True
            break # break once tolerance is met
        x0 = x1
        x1 = x2  # iterate
    return x2, converges, i + 1


def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    Dfp = np.dot(Df(x),p) # compute these values only once
    fx = f(x)
    # iterate until the condition is met
    while (f(x + alpha * p)) > fx + c * alpha * Dfp:
        alpha = rho * alpha
    return alpha
