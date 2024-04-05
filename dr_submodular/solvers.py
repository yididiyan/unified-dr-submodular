import math

import numpy as np
import cvxopt
from dr_submodular.utils import reduce_dim, lifting, lower_lifting

"""
Pieces taken from 

Zhang Q, Deng Z, Chen Z, Zhou K, Hu H, Yang Y. Online Learning for Non-monotone DR-Submodular Maximization: From Full Information to Bandit Feedback. InInternational Conference on Artificial Intelligence and Statistics 2023 Apr 11 (pp. 3515-3537). PMLR.
"""

class Online_non_convex:
    def __init__(self, n, utility_function):
        self.dim = n
        self.x = np.zeros((n, 1))
        self.utility_function = utility_function

    def update(self, vector1, vector2, M):
        dim = self.dim * M
        all_1 = np.ones((dim, 1))
        vector1 = lifting(vector1, M)
        vector2 = lower_lifting(vector2, M)
        self.x = lower_lifting(self.x, M)
        self.x += vector1 * (all_1 - vector2)
        if M == 1:
            A1 = self.utility_function.A
        else:
            A1 = self.utility_function.A / M
            for i in range(M - 1):
                A1 = np.concatenate((A1, self.utility_function.A / M), axis=1)
        G = cvxopt.matrix(np.concatenate((A1, np.eye(dim), -np.eye(dim)), axis=0))
        g = cvxopt.matrix(np.concatenate((np.ones((A1.shape[0], 1)), np.ones((dim, 1)), np.zeros((dim, 1))), axis=0))
        P = cvxopt.matrix(2 * np.eye(dim))
        q = cvxopt.matrix(-2 * self.x)
        sol = cvxopt.solvers.qp(P, q, G, g)
        self.x = np.array(sol['x'])
        self.x = self.x.reshape(dim, 1)
        self.x = np.maximum(self.x, 0)
        self.x = reduce_dim(self.x, self.dim, M)
        # print(self.x>=0)


# In[4]:


# In[5]:


class Online_LP:
    def __init__(self, n, utility_function):
        self.dim = n
        self.x = np.zeros((n, 1))
        self.utility_function = utility_function

    def update(self, vector, T):
        self.x += vector / math.sqrt(T)
        G = cvxopt.matrix(np.concatenate((self.utility_function.A, np.eye(self.dim), -np.eye(self.dim)), axis=0))
        g = cvxopt.matrix(
            np.concatenate((np.ones((self.utility_function.A.shape[0], 1)), np.ones((self.dim, 1)), np.zeros((self.dim, 1))),
                           axis=0))
        P = cvxopt.matrix(2 * np.eye(self.dim))
        q = cvxopt.matrix(-2 * self.x)
        sol = cvxopt.solvers.qp(P, q, G, g)
        self.x = np.array(sol['x'])
        self.x = self.x.reshape(self.dim, 1)
        self.x = np.maximum(self.x, 0)


class Online_LP1:
    def __init__(self, n, delta, alpha, utility_function):
        self.dim = n
        self.delta = delta
        self.alpha = alpha
        self.utility_function = utility_function
        self.r = utility_function.r
        self.x = self.delta * np.ones((n, 1))

    def update(self, vector, T):
        all_1 = np.ones((self.dim, 1))
        self.x += vector / (math.sqrt(T))
        G = cvxopt.matrix(np.concatenate((self.utility_function.A, np.eye(self.dim), -np.eye(self.dim)), axis=0))
        g = cvxopt.matrix(
            np.concatenate((np.ones((self.utility_function.A.shape[0], 1)), np.ones((self.dim, 1)), np.zeros((self.dim, 1))),
                           axis=0))
        P = cvxopt.matrix((2 * (1 - self.alpha) ** 2) * np.eye(self.dim))
        q = cvxopt.matrix(-2 * (1 - self.alpha) * (self.x - self.delta * all_1))
        sol = cvxopt.solvers.qp(P, q, G, g)
        self.x = np.array(sol['x'])
        self.x = self.x.reshape(self.dim, 1)
        self.x = np.maximum(self.x, 0)
        # print(self.x)
        self.x = (1 - self.alpha) * self.x + self.delta * all_1
        # print(self.x)
