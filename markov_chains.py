# markov_chains.py


import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """

    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        n = A.shape[1]
        if A.shape[0] != A.shape[1]:
            raise ValueError('A is not square')
        if np.allclose(A.sum(axis=0), np.ones(A.shape[1])) is False:
            raise ValueError('Matrix must be Column Stochastic')
        
        map = {}
        if states is None:
            for i in range(n):
                map[i] = i
            states = [j for j in range(n)]
        else:
            for i in range(n):
                map[states[i]] = i 
        

        self.trans_mat = A
        self.states = states
        self.map = map
        
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        map_ind = self.map[state]
        col = self.trans_mat[:,map_ind]
        draw = np.random.multinomial(1,col)
        return self.states[np.argmax(draw)]

    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        L = [start]
        current = start
        for i in range(N - 1):
            current = self.transition(current)
            L.append(current)
        return L

    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        L = [start]
        current = start
        while current != stop:
            current = self.transition(current)
            L.append(current)
        return L

    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        n = self.trans_mat.shape[1]
        x_1 = np.random.random(n)
        x_k = self.trans_mat@x_1
        k = 1
        while la.norm(x_k - x_1) >= tol:
            if k > maxiter:
                raise ValueError('Max iterations exceeded')
            x_1 = x_k
            x_k = self.trans_mat@x_1
            k += 1
        return x_k


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        with open(filename, 'r') as file:
            lines = file.read()
        states_list = ['$tart'] + list(set(lines.split())) + ['$top']
        states = {states_list[i]:i for i in range(len(states_list))}
        n = len(states)
        A = np.zeros((n,n))

        for line in lines.split('\n'):
            words = ['$tart'] + line.split() + ['$top']
            r = len(words)
            for i in range(r - 1):
                A[states[words[i+1]]][states[words[i]]] += 1
        A[n-1][n-1] = 1
        
        self.trans_mat = A / np.sum(A, axis=0)[np.newaxis, :]  

        self.states = states_list
        self.map = states
        
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        path = self.path('$tart','$top')
        path.pop(0)
        path.pop()
        return ' '.join(path)
