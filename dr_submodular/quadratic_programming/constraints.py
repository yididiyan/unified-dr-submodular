
import numpy  as np

##### Non-Convex/Non-Concave Quadratic Programming


class QuadraticConstraint:
    """
    Generate constraint for quadratic constraints problem
    """
    # domain as P={Ax\le b and x\le\mathbf{1}}
    # we set b=\mathbf{1} and the element of A sampled from U[0,1]

    def __init__(self, n, m):
        self.dim = n
        self.dim_constraint = m  # additional constraints to the hyper-cube

        # problem constraints -- will not include hypercube constraints
        # scipylinprog does not need box (hypercube) constraints included
        self.A = None
        self.b = None

        # including hypercube constraints -- which cvxopt needs
        self.Afull = None
        self.bfull = None

        self.r = None  # unsure how used -- kept from orig code

    # generate a set of constraints (will be common for different objective functions)
    def generate_constraints(self):
        m = self.dim_constraint  # the number of linear constraints
        n = self.dim  # solutions in [0,1]^n

        A = np.random.rand(m, n)

        self.r = min([1 / np.linalg.norm(A[i]) for i in range(m)])

        # Question -- should we do some adjusting for A if (1,1,1..) feasible?
        # for L1 norm, if it is 1 or bigger, whole hypercube satisfies that constraint
        print('maybe adjust any row of A with L1 norm >= 1')

        b = np.ones((m, 1))

        # Now add in hyper-cube constraints
        # 0 \leq x \leq 1 -->  -np.eye x \leq 0  np.eye x \leq 1
        Afull = np.concatenate((A, np.eye(n), -np.eye(n)), axis=0)
        bfull = np.concatenate((b, np.ones((n, 1)), np.zeros((n, 1))), axis=0)

        self.A = A
        self.b = b
        self.Afull = Afull
        self.bfull = bfull

    def get_constraints_problem(self):
        return self.A, self.b

    def get_constraints_full(self):
        return self.Afull, self.bfull

    # moved (out of the utility function class) - generate once (common feasible region)



class MonotonousQuadraticConstraint:

    # Constraints for monotonous Monotonous functions
    # Black Box Submodular Maximization: Discrete and Continuous Settings
    # https://arxiv.org/pdf/1901.09515.pdf -- Section 5
    def __init__(self, n, m):
        self.dim = n
        self.dim_constraint = m  # additional constraints to the hyper-cube

        # problem constraints -- will not include hypercube constraints
        # scipylinprog does not need box (hypercube) constraints included
        self.A = None
        self.b = None


    # generate a set of constraints (will be common for different objective functions)
    def generate_constraints(self):
        bound = 20 # predefined bound
        m = self.dim_constraint  # the number of linear constraints
        n = self.dim  # solutions in [0,1]^n

        n_groups = np.array_split(np.array(range(n)), m) # split the dimensions into m groups
        # construct three constraints

        assert len(n_groups) == m

        A = np.zeros((m, n))


        for m_, n_group in enumerate(n_groups):
            A[m_, n_group] = 1

        b = np.ones((m, 1)) * bound

        self.A = A
        self.b = b

    def get_constraints_problem(self):
        return self.A, self.b




if __name__ == '__main__':
    constraint = MonotonousQuadraticConstraint(100, 3)
    constraint.generate_constraints()
    print(constraint.A)
    print(constraint.b)