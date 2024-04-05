import math
import numpy as np
import random
from scipy.optimize import linprog

from dr_submodular.utils import sample_on_the_surface_of_ball
from dr_submodular.solvers import Online_LP, Online_LP1, Online_non_convex
from dr_submodular.function import CompositeFunction

# Mitra2021submod -- Algorithm 2
# they provided code, though
class Offline_Meas_Grd_FW:
    # n: dimension of solutions
    # x: current point (Mitra pseudo-code 'y^(t)')
    # t: current time
    # epsilon: user specified quality control parameter
    # func: the func object (can query exact gradient, value, constraint coeff)

    def __init__(self, n, epsilon, func: CompositeFunction, t_func, constraint):
        """
        T - func time steps
        t - algorithm local time steps 
        """


        self.dim = n
        self.x = np.zeros((n, 1))
        self.epsilon = epsilon
        self.t = 0  # the local steps  't' in their pseudo-code

        self.func = func
        self.t_func = t_func  # the func time -- so F(x) = sum_{i=1}^T f_i(x)
        self.constraint = constraint

    def solve(self):
        # run the algorithm until iteration clock t passes 1
        while self.t < 1.0:
            self.step()


        # return solution and value
        return self.x, self.func.value_sum(self.x, self.t_func)

    def step(self):
        # take one step (iteration of while loop)

        # initializing variables
        t_func = self.t_func  # the number of funcs to use
        t = self.t  # the local clock ('t' in the pseudocode); between 0 and 1
        x = self.x
        n = self.dim
        epsilon = self.epsilon
        func = self.func

        # at end may jump past t=1 reduce epsilon (within loop; leave self.epsilon alone)
        if t + epsilon > 1:
            epsilon = 1 - t

        # first evaluate gradient at the current point
        # will access gradient for F(x) = sum_{i=1}^t f_i(x)
        grad = func.gradient_exact_sum(x, t_func)

        #####  USING CVXOPT.SOLVERS.LP -- feasibility issues

        # #generate coefficients for linear optimization problem
        # # element-wise multiplication
        # # take -coef in order to maximize (solvers.lp by default minimizes)
        # coef = -np.multiply(np.ones((n,1)) - x , grad)
        # coef = cvxopt.matrix(coef)

        # #prepare constraint matrices for LP # https://cvxopt.org/examples/tutorial/lp.html
        # A,b = func.get_constraints_full()
        # A = cvxopt.matrix(A)
        # b = cvxopt.matrix(b)

        # # sol=solvers.lp(c,A,b)
        # #find solution to linear program
        # #cvxopt's LP is for minimization; we want maximization
        # cvxopt.solvers.options['feastol']=10**-15 # with default, was getting constraint violations
        # cvxopt.solvers.options['show_progress'] = False

        # sol = cvxopt.solvers.lp(coef,A,b,)
        # s = sol['x']
        # #print(sol['x'])
        # if np.min(s)<0.0:
        #     print('s is not a non-negative direction')

        #########   SCIPY.LINPROG -- compare (feasibility)

        # See Mitra2021submod+concave's code 'quadraticprogramming.py'

        # we want to maximize; linprog by default minimizes
        coef = -np.multiply(np.ones((n, 1)) - x, grad)

        # linprog allows box constraints separated
        A, b = self.constraint.get_constraints_problem()

        # separately set up bounds for individual decision variables
        bounds_arg = []
        for i in range(n):
            bounds_arg.append((0, 1))
        res = linprog(coef, A_ub=A, b_ub=b, bounds=bounds_arg)
        s_ = res.x
        s_ = np.vstack(s_)  # convert to have dims (n,1)

        if np.min(s_) < 0.0:
            print('s is not a non-negative direction')

        # # Debug -- cvxopt.solvers.lp versus scipy.linprog
        # print(np.concatenate((np.array(s),s_),axis=1))

        # now update x
        x = x + epsilon * np.multiply(np.ones((n, 1)) - x, s_)

        if np.min(x) < 0 or np.max(x) > 1:
            print('x is outside of the hyper-cube')
            print(x)

        # set value
        self.x = x

        # increment clock
        self.t += epsilon  # should set self.t=1.0 at last iteration

    # def update(self,vector,T):
    #     self.x+=vector/math.sqrt(T)
    #     G=cvxopt.matrix(np.concatenate((utility_function.A,np.eye(self.dim),-np.eye(self.dim)),axis=0))
    #     g=cvxopt.matrix(np.concatenate((np.ones((utility_function.A.shape[0],1)),np.ones((self.dim,1)),np.zeros((self.dim,1))),axis=0))
    #     P=cvxopt.matrix(2*np.eye(self.dim))
    #     q=cvxopt.matrix(-2*self.x)
    #     sol=cvxopt.solvers.qp(P,q,G,g)
    #     self.x=np.array(sol['x'])
    #     self.x=self.x.reshape(self.dim,1)
    #     self.x=np.maximum(self.x,0)






