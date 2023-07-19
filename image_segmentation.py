# image_segmentation.py


import numpy as np
import scipy.sparse as sc
import scipy.linalg as la
from imageio import imread
from matplotlib import pyplot as plt
import scipy.sparse.linalg 


def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    DList = A.sum(axis=0)
    D = sc.diags(DList).toarray()
    L = D - A
    return L


def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    Ieigs = la.eigvals(L)

    #extract real components
    eigs = np.real(Ieigs)
    
    #count how many nonzero eigenvals there are
    connected = np.count_nonzero(eigs < tol)

    #return number of connected nodes and algebraic connectivity
    return connected,sorted(eigs)[1]


def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        I = imread(filename)
        self.image = I/255
        if len(self.image.shape) == 3:
            self.gray = False
            self.bright = np.ravel(self.image.mean(axis=2))
        else:
            self.gray = True
            self.bright = np.ravel(self.image.copy())
        
    def show_original(self):
        """Display the original image."""
        if self.gray is True:
            plt.imshow(self.image, cmap='gray')
        else:
            plt.imshow(self.image)
        plt.axis('off')
        plt.show()

    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        
        #check if gray or color to get right dimension
        if self.gray is False:
            m,n,z = self.image.shape
        else:
            m,n = self.image.shape

        A = sc.lil_matrix((m*n,m*n))
        D = np.zeros(shape=(m*n))

        #get weights from equation in specs
        for i in range(m*n):
            
            neighbors,distance = get_neighbors(i,r,m,n)

            weights = np.exp(
                -(abs(self.bright[i] - self.bright[neighbors]))/(sigma_B2) 
                -(distance)/(sigma_X2))

            A[i, neighbors] = weights
            D[i] = np.sum(weights)
        return A.tocsc(),D

    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        #get m x n
        if self.gray is False:
            m,n,z = self.image.shape
        else:
            m,n = self.image.shape

        L = sc.csgraph.laplacian(A)
        #create d inverse sqrt
        D_12 = sc.diags(1/np.sqrt(D)).tocsc()
        AAA = D_12@L@D_12
        #get eigvecs
        eigvals,eigvecs = scipy.sparse.linalg.eigsh(D_12@L@D_12,which='SM',k=2)

        #create mask
        mask = eigvecs[:,1].reshape(m,n) > 0

        return mask

    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        #get mask
        A,D = self.adjacency(r,sigma_B,sigma_X)
        mask = self.cut(A,D)

        if self.gray is True: #Plot gray
            plt.subplot(131)
            plt.imshow(self.image, cmap='gray')
            plt.axis('off')
            plt.title('Original')

            plt.subplot(132)
            plt.imshow(mask * self.image, cmap='gray')
            plt.axis('off')
            plt.title('Positive')

            plt.subplot(133)
            plt.imshow(~mask * self.image, cmap='gray')
            plt.axis('off')
            plt.title('Negative')
        else: #Plot color image
            plt.subplot(131)
            plt.imshow(self.image)
            plt.axis('off')
            plt.title('Original')
            #apply to all rgb
            im2 = np.zeros(self.image.shape)
            im2[:,:,0] = mask * self.image[:,:,0]
            im2[:,:,1] = mask * self.image[:,:,1]
            im2[:,:,2] = mask * self.image[:,:,2]
            plt.subplot(132)
            plt.imshow(im2)
            plt.axis('off')
            plt.title('Positive')

            im3 = np.zeros(self.image.shape)
            im3[:,:,0] = ~mask * self.image[:,:,0]
            im3[:,:,1] = ~mask * self.image[:,:,1]
            im3[:,:,2] = ~mask * self.image[:,:,2]
            plt.subplot(133)
            plt.imshow(im3)
            plt.axis('off')
            plt.title('Negative')
        
        plt.suptitle('Masks')
        plt.show()
