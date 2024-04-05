import math
import numpy as np

"""
Pieces taken from 

Zhang Q, Deng Z, Chen Z, Zhou K, Hu H, Yang Y. Online Learning for Non-monotone DR-Submodular Maximization: From Full Information to Bandit Feedback. InInternational Conference on Artificial Intelligence and Statistics 2023 Apr 11 (pp. 3515-3537). PMLR.
"""

def lifting(vector,M):
    if M==1:
        return vector
    else:
        a=vector/M
        for i in range(M-1):
            a=np.concatenate((a,vector/M),axis=0)
        return a
def lower_lifting(vector,M):
    length=vector.shape[0]
    new_vector=np.zeros((length*M,1))
    for i in range(length):
        new_value=int(math.floor(vector[i][0]*M))
        if new_value!=0:
            for j in range(new_value):
                new_vector[j*length+i][0]=1
    return new_vector
def reduce_dim(vector,n,M):
    new_vector=np.zeros((n,1))
    for i in range(n):
        new_vector[i][0]=min(np.sum(vector[i::n])/M,1)
    return new_vector



def sample_on_the_surface_of_ball(d):
    x=np.random.randn(d,1)
    x=x/(np.sum(x**2)**0.5)
    return x.reshape((d,1))