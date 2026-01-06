import numpy as np
from scipy.linalg import solve_lyapunov


def spectral_abscissa(W):
    return np.max(np.linalg.eig(W)[0].real)


class GramianStore():
    def __init__(self, x, modes, evals):
        self.gramian = x
        self.modes = modes
        self.evals = evals

    def top(self, n):
        return self.modes[:,0:n]

    def get(self,n):
        return self.modes[:,n]

class Gramians():
    def __init__(self, a, b=None, c=None):
        self.a = a
        self.n, _ = np.shape(a)
        self.b = np.eye(self.n) if b == None else b
        self.c = np.eye(self.n) if c == None else b
        self.O = self.construct_obsv(a, self.c)
        self.C = self.construct_ctrl(a, self.b)
 
    @staticmethod
    def ctrl(a, b=None):
        n, _ = np.shape(a)
        b = np.eye(n) if b is None else b
        return solve_lyapunov(a, -b.dot(b.T))

    @staticmethod
    def obsv(a, c=None):
        n, _ = np.shape(a)
        c = np.eye(n) if c is None else c
        return solve_lyapunov(a.T, -c.T.dot(c))
 
    def construct_ctrl(self, a, b):
        x = self.ctrl(a, b)
        modes, evals, _ = np.linalg.svd(x, full_matrices=False)
        return GramianStore(x, modes, evals)

    def construct_obsv(self, a, c):
        x = self.obsv(a, c)
        modes, evals, _ = np.linalg.svd(x, full_matrices=False)
        return GramianStore(x, modes, evals)

