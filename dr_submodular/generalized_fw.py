
import numpy as np
import random

from dr_submodular.solvers import Online_LP


"""
Reward functions 
"""
def reward_non_monotone(v, g, x):
    """
    Setting (B) in the paper
    :param v: oracle's update direction
    :param g: gradient
    :param x: current position
    :return:
    """
    return (g * (1-x))

def reward(v, g):
    """
    Setting (A), (C) and (D)
    :param v: oracle's update direction
    :param g: gradient
    :return: reward the oracle gets
    """
    raise NotImplementedError("Unimplemented reward function: " + update_function.__name__)




"""
functions for updating X 
"""

def update_x_monotone(x, v, step_size):
    """
    Setting(A) in the paper -- Monotone, zero enclosed constraint set (downward closed)

    :param x: point
    :param v: oracle's update direction
    :return: the updated x
    """
    return x + step_size * v

def update_x_non_monotone(x, v, step_size):
    """

    Setting (B) in the paper -- Non-monotone, zero enclosed constraint set  (downward closed)
    :param x: point
    :param v: oracle's update direction
    :return: the updated x
    """
    return x + step_size * v * (1 - x)




"""
General note 

* We need to use Online_LP instead on Online LP1

"""

def GeneralizedMetaFrankWolfe(
        n, T, L, Q, K,
        T_utility,
        utility_function,
        update_function=update_x_non_monotone,
        reward_function=reward_non_monotone,
        include_trajectory=False):
    """
    Implementation of the Algorithm 1 - Generalized-Meta-Frank-Wolfe in the paper
    :param n: number of nodes
    :param T: total number of steps -- T= LQ
    :param L: size of each block
    :param Q: number of blocks
    :param K: number of oracles
    :param T_utility: T dimensional functions to evaluate
    :param reward_function: oracle reward function
    :return:
    """
    assert K >= L, "Number of oracles must be integer multiples of block size L"

    utility = T_utility
    LP_oracle = [Online_LP(n, utility_function) for i in range(K)]
    all_1 = np.ones((n, 1))
    final_value = [0] * T

    ## Line -- 2 -- zeros as initial point
    lower_vector = np.zeros((n, 1))
    trajectory = []

    # for Q blocks
    for q in range(Q):
        iter_point = [0] * K
        point = lower_vector
        for k in range(K):
            # receive the output of the oracle
            update_direction = LP_oracle[k].x
            if update_function.__name__ == 'update_x_non_monotone':
                step = 1/K
                point = update_x_non_monotone(point, update_direction, step)
            elif update_function.__name__ == 'update_x_monotone':
                step = 1 / K
                point = update_x_monotone(point, update_direction, step)
            else:
                raise NotImplementedError("Unimplemented update function: " + update_function.__name__)

            iter_point[k] = point

        # Line 8 -  generate random permutation
        random_num = random.sample(list(range(L)), L)

        grads = [[] for _ in range(K)] # list to collect the gradients

        for l in range(L):
            t = q * L + l

            if t >= T:
                break

            x_t = iter_point[-1]

            # Get the reward for the current time steps
            f_t = utility[t].value(x_t)
            final_value[t] = f_t

            # find the corresponding l_prime - the position of l in the permuted list
            l_prime = random_num.index(l)

            # observe gradients
            # get set of all ks to observe their corresponding gradient
            k_prime = [k for k in range(K) if (k - l_prime) % L == 0 ]

            for k_index in k_prime:
                x_q_k = iter_point[k_index]

                # evaluate the gradient
                grads[k_index].append(utility[t].gradient(x_q_k))

        # Reward the oracles
        for k in range(K):
            update_direction = LP_oracle[k].x

            # average the gradients
            if len(grads[k]):
                assert len(grads[k]) == 1, 'one grad sample per oracle, less if time has elapsed'

                grad_mean = np.mean(grads[k], 0)

                if reward_function.__name__ == 'reward':
                    # oracle reward
                    LP_oracle[k].update(
                        reward(update_direction, grad_mean), Q
                    )
                elif reward_function.__name__ == 'reward_non_monotone':
                    # oracle reward
                    LP_oracle[k].update(
                        reward_non_monotone(update_direction, grad_mean, iter_point[k]), Q
                    )
                else:
                    return NotImplementedError("Unimplemented reward function: " + reward_function.__name__)

        if include_trajectory:
            trajectory.extend([point])
    return trajectory, final_value




def SemiBanditFrankWolfe(
        n, T, L, Q, K,
        T_utility,
        utility_function,
        update_function=update_x_non_monotone,
        reward_function=reward_non_monotone):
    """
    Implementation of the Algorithm 1 - Generalized-Meta-Frank-Wolfe in the paper
    :param n: number of nodes
    :param T: total number of steps -- T= LQ
    :param L: size of each block
    :param Q: number of blocks
    :param K: number of oracles
    :param T_utility: T dimensional functions to evaluate
    :param reward_function: oracle reward function
    :return:
    """
    assert K <= L, "Number of oracles must be leq to block size L"

    utility = T_utility
    LP_oracle = [Online_LP(n, utility_function) for i in range(K)]
    all_1 = np.ones((n, 1))
    final_value = [0] * T

    ## Line -- 2 -- zeros as initial point
    lower_vector = np.zeros((n, 1))

    # for Q blocks
    for q in range(Q):
        iter_point = [0] * K
        point = lower_vector
        for k in range(K):
            # receive the output of the oracle
            update_direction = LP_oracle[k].x
            if update_function.__name__ == 'update_x_non_monotone':
                step = 1/K
                point = update_x_non_monotone(point, update_direction, step)
            elif update_function.__name__ == 'update_x_monotone':
                step = 1 / K
                point = update_x_monotone(point, update_direction, step)
            else:
                raise NotImplementedError("Unimplemented update function: " + update_function.__name__)

            iter_point[k] = point

        # Line 8 -  generate random permutation
        random_num = random.sample(list(range(L)), L)

        grads = [[] for _ in range(K)]  # list to collect the gradients

        for l in range(L):
            t = q * L + l
            if t >= T:
                break

            if l in random_num[:K]:
                # exploration
                k_prime = random_num.index(l)
                assert k_prime < K, 'index out of bound '

                y_t = iter_point[k_prime]

                # Get the reward for the current time steps
                f_t = utility[t].value(y_t)
                final_value[t] = f_t


                # observe the gradient
                x_q_k = iter_point[k_prime]

                # evaluate the gradient
                grads[k_prime].append(utility[t].gradient(x_q_k))

            else:
                # exploitation -- just play the last oracle's proposal(action)
                y_t = iter_point[-1]

                # Get the reward for the current time steps
                f_t = utility[t].value(y_t)
                final_value[t] = f_t




        # Reward the oracles
        for k in range(K):
            update_direction = LP_oracle[k].x

            if grads[k]:
                # average the gradients
                grad_mean = np.mean(grads[k], 0)

                if reward_function.__name__ == 'reward':
                    # oracle reward
                    LP_oracle[k].update(
                        reward(update_direction, grad_mean), Q
                    )
                elif reward_function.__name__ == 'reward_non_monotone':
                    # oracle reward
                    LP_oracle[k].update(
                        reward_non_monotone(update_direction, grad_mean, iter_point[k]), Q
                    )
                else:
                    return NotImplementedError("Unimplemented reward function: " + reward_function.__name__)


    return final_value


