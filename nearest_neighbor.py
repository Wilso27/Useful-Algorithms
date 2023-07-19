# nearest_neighbor.py


import numpy as np
from scipy import linalg as la
from sklearn.cluster import k_means
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import scipy


def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #array broadcasting
    return X[np.argmin(la.norm((X-z),axis=1))],sum(X[np.argmin(la.norm((X-z),axis=1))])


class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        '''Initialize value, left, right, and pivot'''
        if type(x) != np.ndarray:
            raise TypeError('Must be of type ndarray')
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None


class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        node = KDTNode(data)
        # if tree is empty
        if self.root is None:
            self.root = node
            self.k = len(node.value)
            self.root.pivot = 0
        # check if input is correct dimension
        elif self.k != len(node.value):
            raise ValueError('Data being inserted must be of k-th dimension')
        # Bottom-up to insert
        else:
            current = self.root
            node.pivot = 0
            # Loop until it is a leaf
            while True:
                # Check for duplicates
                if current.value is node.value:
                    raise ValueError('Node is already in tree')
                #if node value is less than current value
                if node.value[current.pivot] < current.value[current.pivot]:
                    #base case
                    if current.left is None:
                        current.left = node
                        node.pivot += 1
                        node.pivot = node.pivot % self.k
                        break
                    else:
                        current = current.left
                #if node value is greater than or equal to current value
                elif node.value[current.pivot] >= current.value[current.pivot]:
                    #base case
                    if current.right is None:
                        current.right = node
                        node.pivot += 1
                        node.pivot = node.pivot % self.k
                        break
                    else:
                        current = current.right
                node.pivot += 1
                node.pivot = node.pivot % self.k

    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest,d):
            '''searches the KDTree
            
            Parameters:
                current,nearest: k-n=dimensional array
                d: float
            '''
            #base case if there was no right or left in previous iteration
            if current is None:
                return nearest,d
            x = current.value
            i = current.pivot
            #check if the new vector is closer than the previous
            if la.norm(x-z) < d:
                nearest = current
                d = la.norm(x-z)
            #check if the pivot is greater or equal to than or less than
            if z[i] < x[i]:
                nearest,d = KDSearch(current.left,nearest,d)
                if z[i] + d >= x[i]:
                    nearest,d = KDSearch(current.right,nearest,d)
            else:
                nearest,d = KDSearch(current.right,nearest,d)
                if z[i] - d <= x[i]:
                    nearest,d = KDSearch(current.left,nearest,d)
            return nearest,d
        #call recursively
        node,d = KDSearch(self.root,self.root,la.norm(self.root.value-z))
        return node.value,d
        
    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        '''initializes the class'''
        self.n_neighbors = n_neighbors
        self.tree = None
        self.labels = None
    
    def fit(self, X, y):
        '''
        Parameters:
            X (ndarray): m x k
            y (ndarray): 1 dimensional
        '''
        self.tree = KDTree(X)
        self.labels = y
    
    def predict(self, z):
        '''
        Parameters:
            z (ndarray): 1 by k
        '''
        #get indices
        c,indices = self.tree.query(z, k=self.n_neighbors)
        closest = [self.labels[i] for i in indices]
        #check which occurs the most
        return scipy.stats.mode(closest)[0]


def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    #extract the data
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float) # Training data
    y_train = data["y_train"] # Training labels
    X_test = data["X_test"].astype(np.float) # Test data
    y_test = data["y_test"]               #test labels

    #classifier
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train,y_train)

    #check accuracy
    counter = 0
    for i in range(len(y_test)):
        if classifier.predict(X_test[i]) == y_test[i]:
            counter += 1
    return counter / len(y_test)

    plt.imshow(X_test[0].reshape((28,28)), cmap="gray")
    plt.show()
