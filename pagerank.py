# solutions.py


import numpy as np
import networkx as nx
from itertools import combinations as comb


class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """

    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        # initialize variables
        n = A.shape[0]
        sink = np.where(np.sum(A, axis=0) == 0)
        if np.any(sink):  # check for sinks
            A.T[sink[0]] = np.ones(n)

        # check labels
        if labels is None:
            labels = np.arange(0, n, 1)
        if len(labels) != n:
            raise ValueError("Wrong number of labels")

        # save attributes
        self.A = A / np.sum(A, axis=0)
        self.labels = labels
        self.n = n

    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # create elements of (13.6)
        I = np.eye(self.n)
        Ax = I - epsilon * self.A
        b = ((1 - epsilon) / self.n) * np.ones(self.n)

        p = np.linalg.solve(Ax, b)  # solve the system

        return dict(zip(self.labels, p))

    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # use (13.7)
        E = np.ones((self.n, self.n))
        e = epsilon
        B = e * self.A + ((1 - e) / self.n) * E

        # get eigenvecs and vals
        vals, vecs = np.linalg.eig(B)

        # find the max and the corresponding vec
        index = np.argmax(vals.real)
        p = vecs[:, index].real
        p = p / np.sum(p)

        return dict(zip(self.labels, p))

    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # initialize variables
        p_ = np.zeros(self.n)
        p = np.ones(self.n) / self.n
        e = epsilon
        i = 0

        # loop until maxiter or tolerance level reached
        while i < 100 and np.linalg.norm(p - p_, ord=1) >= tol:
            p_ = p
            p = e * self.A @ p_ + (1 - e) * np.ones(self.n) / self.n
            i += 1

        return dict(zip(self.labels, p))


def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # get keys and vals from dictionary
    keys = np.array((list(d.keys())))
    vals = np.array((list(d.values())))

    # sort vals and apply to labels
    labels_sorted = keys[np.argsort(vals)][::-1]

    return list(labels_sorted)


def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    # read in the data
    with open(filename) as file:
        data = file.read()
    id = np.array(())

    # Get sorted list of unique id's
    for line in data.splitlines():
        info = line.split('/')
        id = np.append(id, info)
    id = np.unique(id)
    id = list(np.sort(id))

    # construct the adjacency matrix
    n = len(id)
    A = np.zeros((n, n))
    for line in data.splitlines():
        info = line.split('/')
        val = info[0]
        key = id.index(val)
        for i in info[1:]:
            ind = id.index(i)
            A[ind, key] = 1

    # use previous functions to get list of ranked webpages
    obj = DiGraph(A,labels = id)
    d = obj.itersolve(epsilon=epsilon)

    return get_ranks(d)


def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    # read in the data
    with open(filename, 'r') as file:
        data = file.read()
    id = np.array(())

    # Get sorted list of unique id's
    for line in data.splitlines()[1:]:
        info = line.split(',')
        id = np.append(id, info)
    id = np.unique(id)
    id = list(np.sort(id))

    # create the adjacency matrix
    n = len(id)
    A = np.zeros((n, n))
    for line in data.splitlines()[1:]:
        info = line.split(',')
        row = id.index(info[0])
        col = id.index(info[1])
        A[row, col] += 1

    # use previous functions to get list of ranked teams
    obj = DiGraph(A, labels=id)
    d = obj.itersolve(epsilon=epsilon)

    return get_ranks(d)


def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    # read in the data
    with open(filename, 'r', encoding="utf-8") as file:
        data = file.read()
    graph = nx.DiGraph()

    # loop through each pair of actors on each line and add weights
    for line in data.splitlines():
        names = line.split('/')[1:]
        for c in list(comb(names, 2)):
            if graph.has_edge(c[1], c[0]):
                graph[c[1]][c[0]]["weight"] += 1
            else:
                graph.add_edge(c[1], c[0], weight = 1)

    # rank the actors and return the list
    d = nx.pagerank(graph, alpha = epsilon)
    return get_ranks(d)
