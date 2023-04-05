from typing import Callable
import cvxopt
import numpy as np
from numpy import ndarray


class SVM:
    def __init__(self, kernel_function: Callable, C, min_lagrange_multiplier):
        self.support_vectors_labels = None
        self.support_vectors = None
        self.alphas = None
        self.b = None
        self.kernel_function = kernel_function
        self.C = C
        self.min_lagrange_multiplier = min_lagrange_multiplier

    def train(self, x: ndarray, y: ndarray):
        n, labels = x.shape

        """
        Calculate kernel matrix
        kernel = M(n, n)
        kernel[i][j] = K(Xi, Xj)
        """
        kernel = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                kernel[i][j] = self.kernel_function(x[i], x[j])

        """
        Calculate P parameter to cvxopt solver
        P = M(n, n), P[i][j] = H[i][j] = Yi*Yj*K[i][j]
        Yi*Yj = numpy.outer(y, y)
        """
        P = cvxopt.matrix(np.outer(y, y) * kernel, tc="d")

        """
        Calculate q parameter to cvxopt solver
        q = M(n, 1)
        qi = -1
        """
        q = cvxopt.matrix(-np.ones(n))

        """
        Calculate G parameter to cvxopt solver
        G = M(2n, n)
        Top "diagonal" is -1
        Bottom "diagonal is +1"
        -1  0  0
         0 -1  0
         0  0 -1
         1  0  0
         0  1  0
         0  0  1
        """
        bottom = np.eye(n)
        top = - bottom
        G = cvxopt.matrix(np.vstack((top, bottom)))

        """
        Calculate h parameter to cvxopt solver
        h = M(2n, 1)
        hi = 0 for i < n
        hi = C else
        """
        h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        """
        Calculate A parameter to cvxopt solver
        A = M(1, n)
        A = y^T
        """
        A = cvxopt.matrix(y, (1, n), tc="d")

        """
        Calculate b parameter to cvxopt solver
        b = M(1, 1)
        b = [0]
        """
        b = cvxopt.matrix(0.0)

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        best_alpha = np.ravel(solution['x'])

        """
        Filter x vectors and extract support vectors
        support vectors have alpha > 0 (min_alpha)
        """

        best_indexes = best_alpha > self.min_lagrange_multiplier

        self.alphas = best_alpha[best_indexes]
        self.support_vectors = x[best_indexes]
        self.support_vectors_labels = y[best_indexes]
        """
        Calculate b
        b = 1/Ns * sum(ys - sum(am*ym*xm * xs)) 
        """
        bias = 0.0
        for s in range(len(self.support_vectors)):
            bias += self.support_vectors_labels[s]
            for m in range(len(self.support_vectors)):
                bias -= self.alphas[m] * self.support_vectors_labels[m] * kernel[m][s]
        self.b = bias / len(self.support_vectors)

    def predict(self, x: ndarray):
        labels = []
        for sample in x:
            temp = np.sum(
                [self.alphas[i] * self.support_vectors_labels[i] * self.kernel_function(self.support_vectors[i], sample)
                 for i in range(len(self.alphas))])
            labels.append(np.sign(temp + self.b))
        return np.array(labels)
