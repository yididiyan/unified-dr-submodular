from abc import abstractmethod

import numpy as np
from dr_submodular.quadratic_programming.constraints import QuadraticConstraint
from dr_submodular.function import Function, CompositeFunction




class SingleQuadraticFunction(Function):
    # we set the utility function as 1/2 x^{T}Hx+h^{T}x+c where
    # the element of H sampled from U[-1,0]
    # h=-1/n*H^{T}\mathbf{1}, c=-1/2*\sum{H}
    # domain as P={Ax\le b and x\le\mathbf{1}}
    # we set b=\mathbf{1} and the element of A sampled from U[0,1]

    # Note about constraints
    # for the problem to make sense, all objective functions must have the same set of constraints; so we want to set those for the T objective functions

    # change -- move A, r, noise inside init
    # A=None  #-- this is linear constraint coeff
    # r=None
    # noise=0.1
    def __init__(self, n, m, constraint=None, value_noise=None, grad_noise=0.1, h_scale=10, construction_type=1):
        """

        :param n: dimension
        :param m: constraint dimension
        :param constraint:
        :param value_noise: standard iid noise parameter for the function values
        :param h_scale: scalar for the coefficients
        """
        self.dim = n
        self.dim_constraint = m
        self.h_scale = h_scale

        if construction_type == 1:
            self._construct_1()
        else:
            raise NotImplementedError('Use construction type 1; ')

        # added
        self.grad_noise = grad_noise

        self.constraint = constraint
        # mod -- make a _exact and _noisy version

        self.value_noise = value_noise




    def _construct_1(self):
        """
        Construction 1
        Original Zhang's construction of the quadratic function
        """
        self.H = None
        self.h = None
        self.c = None
        all_1 = np.ones((self.dim, 1))
        self.H = np.random.rand(self.dim, self.dim)
        self.H = 0.5 * (self.H + self.H.T)
        self.H = -self.H * self.h_scale
        self.h = -0.1 * self.H.dot(all_1)
        self.c = -0.5 * np.sum(self.H)



    # def gradient(self,x):
    #     return self.H.dot(x)+self.h+self.grad_noise*np.random.randn(self.dim,1)
    def value(self, x):
        val = float(0.5 * x.T.dot(self.H.dot(x)) + self.h.T.dot(x) + self.c)
        if self.value_noise:
            val += self.value_noise * np.random.rand()
        assert val > 0., 'function value can\'t be negative '
        return val

    # added (at least for sanity checks, maybe for offline)
    def gradient(self, x):
        return self.gradient_noisy(x)

    def gradient_exact(self, x):
        return self.H.dot(x) + self.h

    def gradient_noisy(self, x):
        noise = np.random.randn(self.dim, 1)
        normalized_noise = (noise / np.linalg.norm(noise, 2))
        return self.gradient_exact(x) + self.grad_noise * normalized_noise


# made -- wrapper for sum (or individual) quadratic functions
# functions defined for returning values, exact gradients, noisy
#   gradients for individual f_t's or \sum{i=1}^t f_i
class CompositeQuadraticFunction(CompositeFunction):
    """
    Composition of above quadratic function, one for each timestep T
    """

    def __init__(self, n, m, T, constraint=None, fdict=None, grad_noise=None, h_scale=None, construction_type=2):
        """

        :param n: number of dimensions
        :param m: dimensionality constrain
        :param T: horizon
        :param constraint:
        :param fdict: can specify the dictionary of functions -- used in stochastic case
        """
        self.dim = n
        self.dim_constraint = m
        self.T = T

        # generate constraints
        if constraint:
            self.constraints = constraint
        else:
            self.constraints = QuadraticConstraint(n, m)
            self.constraints.generate_constraints()

        # generate a dictionary of T quadratic functions
        self.fdict = fdict or {t: SingleQuadraticFunction(n, m, grad_noise=grad_noise, h_scale=h_scale, construction_type=construction_type) for t in range(1, T + 1)}

    def get_constraints_problem(self):
        return self.constraints.A, self.constraints.b

    def get_constraints_full(self):
        return self.constraints.Afull, self.constraints.bfull

    def value_individual(self, x, t):
        # This returns objective value for just a single quadratic function
        # x is a solution (n dimensional vector in the hypercube)
        # t is an integer in 1, .., T
        return self.fdict[t].value(x)

    def value_sum(self, x, t):
        # This returns sum of objective value f_1, ... f_t
        # x is a solution (n dimensional vector in the hypercube)
        # t is an integer in 1, .., T
        vals = [self.fdict[i].value(x) for i in range(1, t + 1)]
        return np.sum(vals)

    def gradient_exact_individual(self, x, t):
        # This returns the exact gradient for just a single quadratic function
        # x is a solution (n dimensional vector in the hypercube)
        # t is an integer in 1, .., T
        return self.fdict[t].gradient_exact(x)

    def gradient_exact_sum(self, x, t):
        # This returns the sum of exact gradients
        # x is a solution (n dimensional vector in the hypercube)
        # t is an integer in 1, .., T

        grad_sum = np.zeros((self.dim, 1))
        for i in range(1, t + 1):
            grad_sum += self.fdict[i].gradient_exact(x)
        # vals = [self.fdict[i].gradient_exact(x) for i in range(1,t+1)]
        return grad_sum  # np.sum(vals)

    def gradient_noisy_individual(self, x, t):
        # This returns the exact gradient for just a single quadratic function
        # x is a solution (n dimensional vector in the hypercube)
        # t is an integer in 1, .., T

        return self.fdict[t].gradient_noisy(x)

    def gradient_noisy_sum(self, x, t):
        # This returns the sum of noisy gradients
        # x is a solution (n dimensional vector in the hypercube)
        # t is an integer in 1, .., T

        grad_sum = np.zeros((self.dim, 1))
        for i in range(1, t + 1):
            grad_sum += self.fdict[i].gradient_noisy(x)
        # vals = [self.fdict[i].gradient_exact(x) for i in range(1,t+1)]
        return grad_sum  # np.sum(vals)





if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    n = 25
    m = 15

    constraint = QuadraticConstraint(n, m) 
    constraint.generate_constraints()

    T = 200  # from the experiment section
    K = int(T ** (2 / 3))
    L = int(T ** (7 / 9))
    Q = int(T ** (2 / 9))



    # initialize K utility function that
    u_funcs = [SingleQuadraticFunction(n, m) for k in range(T)]

