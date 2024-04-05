import copy
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

from dr_submodular.algorithms import Offline_Meas_Grd_FW
from dr_submodular.generalized_fw import GeneralizedMetaFrankWolfe, SemiBanditFrankWolfe
from dr_submodular.quadratic_programming.functions import SingleQuadraticFunction, CompositeQuadraticFunction
from dr_submodular.quadratic_programming.constraints import QuadraticConstraint



plt.rcParams['font.size'] = 12

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


class QuadraticExperiment:

    def __init__(self,
                 n, m, T,
                 alg='meta',
                 beta=3/2,
                 stochastic=False,
                 grad_noise=0.1,
                 gmfw_beta=1 / 2,
                 gmfw_L=None,
                 include_trajectory=False,
                 h_scale=10,
                 construction_type=2,
                 t_init=1,  ## init step for calculating offline solutions
                 epsilon=1 / 50
                 ):
        """
        :param n: Number of dims for the actions (actions in subset of [0,1]^n)
        :param m: Number of (linear)constraints
        :param T: Number of time steps

        # Generalized Meta FW options
        :param gmfw_beta: beta argument for GMFW
        :param gmfa_L: user-supplied L -- set to 1

        """

        self.n = n
        self.m = m
        self.T = T

        self.alg = alg
        self.beta = beta
        self.gmfw_beta = gmfw_beta
        self.gmfw_L = gmfw_L
        self.stochastic = stochastic
        self.include_trajectory = include_trajectory


        self.grad_noise = grad_noise

        self.constraint = QuadraticConstraint(self.n, self.m)
        self.constraint.generate_constraints()

        print('\tUsing values in mitra2021submod+concave for similar experiments')
        self.epsilon = epsilon

        # initialize the function
        if self.stochastic:
            raise NotImplementedError('Not included in implementation')
        else:
            self.quad_func = CompositeQuadraticFunction(self.n,
                                                        self.m,
                                                        self.T,
                                                        constraint=self.constraint,
                                                        grad_noise=self.grad_noise,
                                                        h_scale=h_scale,
                                                        construction_type=construction_type)

        self.solutions = None
        self.values = None
        self.t_init = t_init



    def generalized_mfw_method(self):

        n_rounds = 1  # 100
        final_values = []
        trajectories = []
        for _ in range(n_rounds):
            ## int rounds to nearest lower integer

            L = self.gmfw_L or int(self.T ** ((1 - 2 * self.gmfw_beta) / 3 ))
            K = int(self.T ** ((1 + self.gmfw_beta) / 3 ))

            Q = ceil(self.T/L)

            u_funcs = [self.quad_func.fdict[f] for f in sorted(self.quad_func.fdict.keys())]  # list of function

            # Generalized Meta Frank Wolfe
            trajectory, final_value = GeneralizedMetaFrankWolfe(
                n=self.n,
                T=self.T,
                L=L,
                Q=Q,
                K=K,
                T_utility=u_funcs,
                utility_function=self.constraint,
                include_trajectory=self.include_trajectory
            )

            ## calculate the running average
            final_value = np.cumsum(final_value) / np.arange(1, len(final_value) + 1)

            final_values.append(final_value)
            trajectories.append(trajectory)

        return trajectories, np.mean(final_values, axis=0)

    def generalized_sbfw_method(self):

        n_rounds = 1  # 100
        final_values = []
        for _ in range(n_rounds):
            ## int rounds to nearest lower integer

            K = int(self.T ** (1/4))
            L = int(self.T ** (1/2))
            Q = ceil(self.T / L)

            u_funcs = [self.quad_func.fdict[f] for f in sorted(self.quad_func.fdict.keys())]  # list of function

            # Semibandit Frank Wolfe
            final_value = SemiBanditFrankWolfe(
                n=self.n,
                T=self.T,
                L=L,
                Q=Q,
                K=K,
                T_utility=u_funcs,
                utility_function=self.constraint
            )

            ## calculate the running average
            final_value = np.cumsum(final_value) / np.arange(1, len(final_value) + 1)

            final_values.append(final_value)

        return np.mean(final_values, axis=0)

    def offline_method(self):
        if self.solutions and self.values:
            print('Using precomuted offline values')
            return self.solutions, self.values
        print('Running offline solver ')

        solutions, values = [], []

        for t_func in range(1, self.T + 1):
            if t_func < self.t_init:
                solutions.append(None)
                values.append(0.)
                continue
            offline_solver = Offline_Meas_Grd_FW(
                self.n,
                self.epsilon,
                self.quad_func,
                t_func=t_func,
                constraint=self.constraint
            )
            sol, val = offline_solver.solve()

            solutions.append(sol)  # normalizing the solution
            values.append(val / t_func)

        self.solutions = solutions
        self.values = values

        return solutions, values

    # linear constrained non-negative quadratic optimization
    def run(self):
        trajectories = []
        online_values = []

        if self.alg == 'gmfw': # generalized meta frank wolfe
            trajectories, online_values = self.generalized_mfw_method()

        elif self.alg == 'sbfw':  # generalized meta frank wolfe
            online_values = self.generalized_sbfw_method()

        _, offline_values = self.offline_method()

        print('\n\nThe Measured Greedy Frank Wolfe result:')
        print(f'Normalized by T: {offline_values}')

        print(f'Online values - {online_values}')

        return trajectories, offline_values, online_values



if __name__ == '__main__':
    import argparse
    import os
    import pickle
    import time

    parser = argparse.ArgumentParser('argument ')
    parser.add_argument('--output-dir', action='store', type=str, default='output' )
    parser.add_argument('--seed', action='store', type=int, default=1888)
    parser.add_argument('--T', action='store', type=int, default=100)
    parser.add_argument('--alg', action='store', type=str, default=None)
    parser.add_argument('--beta', action='store', type=float, default=3/2)
    parser.add_argument('--gmfw-beta', action='store', type=float, default=0.)
    parser.add_argument('--gmfw-L', action='store', type=int, default=None)
    parser.add_argument('--trajectory', action='store', type=bool, default=False)
    parser.add_argument('--grad-noise', action='store', type=float, default=0.1)
    parser.add_argument('--h-scale', action='store', type=float, default=10.0)
    parser.add_argument('--c-type', action='store', type=int, default=1)
    parser.add_argument('--t-init', action='store', type=int, default=None)


    args = parser.parse_args()

    seed = args.seed

    np.random.seed(seed)




    stochastic = False
    T = args.T #200

    N = [ 25, 40, 50]
    M = [ 15, 20, 50]


    y_lims = [(-1, 16), (-2, 35), (-5, 40), (-5, 40)]

    alg_options = [
        ('gmfw', 0, 'GMFW(0)'),
        ('gmfw', 1 / 4, 'GMFW(1/4)'),
        ('gmfw', 1 / 2, 'GMFW(1/2)'),
        ('sbfw', None, 'SBFW'),
    ]

    ## filter algs 
    if args.alg is not None:
        alg_options = [ a for a in alg_options if a[0] == args.alg]


    for y_lim, n,m in zip(y_lims, N, M):
        print(n, m, 'N M')
        q = QuadraticExperiment(
            n,
            m,
            T,
            alg=args.alg,
            beta=args.beta,
            gmfw_beta=args.gmfw_beta,
            gmfw_L=args.gmfw_L,
            grad_noise=args.grad_noise,
            h_scale=args.h_scale,
            include_trajectory=args.trajectory,
            construction_type=args.c_type,
            t_init=1
        )

        q.offline_method()

        plt.figure()
        for alg_option in alg_options:
            # update algorithm settings
            q.alg = alg_option[0]
            q.beta = q.gmfw_beta = alg_option[1]


            # prepare output directory
            output_dir = f'{args.output_dir}_{args.seed}/{args.T}/{alg_option[0]}'

            if alg_option[0] == 'gmfw':
                output_dir = f'{output_dir}/beta_{alg_option[1]}'
            elif alg_option[0] == 'meta':
                output_dir = f'{output_dir}/beta_{alg_option[1]}'

            os.makedirs(output_dir, exist_ok=True)
            # /prepare

            print('Running online solver..')
            start_time = time.time()
            _, offline_values, online_values = q.run()
            total_time = time.time() - start_time
            print(f'Total time {total_time:.2f}')


            plt.plot(np.array(offline_values) - np.array(online_values), label=alg_option[2])


            plt.ylabel('1/e-regret/t')
            plt.xlabel('Iteration index(t)')

            xticks = list(range(0, T+1, 25))
            yticks = list(range(0, y_lim[1] + 3, 5))

            plt.xticks(xticks, xticks)
            plt.title(f'n={n}, m={m}, {total_time:.2f}')
            plt.legend()

            filename = f'{output_dir}/n_{n}__m_{m}_{("stochastic" if stochastic else "adversarial")}_{args.grad_noise}'
            print(f'Saving results under ..{filename}')
            plt.savefig(f'{filename}.png', bbox_inches="tight", dpi=300)
            

            with open(f'{filename}.pkl', 'wb') as f_:
                pickle.dump({
                    'online_values': online_values,
                    'offline_values': offline_values,
                    'n': n,
                    'm': m,
                    'T': T,
                    'alg': alg_option[0],
                    'stochastic': stochastic,
                    'gmfw_beta': alg_option[1],
                    'beta': alg_option[1],
                    'L': args.gmfw_L,
                    'grad_noise': args.grad_noise,
                    'h_scale': args.h_scale,
                    'total_time': total_time,
                    'c_type': args.c_type
                }, f_)

        plt.show()