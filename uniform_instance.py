import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel
def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, O_low, O_high, T_low, T_high, f,  c, flg):
    n_max=n_m
    M_num_f = n_m * f
    time0 = np.random.uniform(low=T_low, high=T_high, size=(1, n_j, n_max, int(M_num_f)))
    time1 = np.random.uniform(low=-1, high=-1, size=(1, n_j, n_max, n_m-int(M_num_f)))
    time2 = np.concatenate((time0, time1), -1)
    n_o = np.trunc(np.random.uniform(low=O_low, high=O_high+1, size=n_j))
    for j in range(1):
        for i in range(n_j):
            time2[j][i] = permute_rows(time2[j][i])

    for i in range(n_j):
        if int(n_o[i]) <= n_max:
            for n in range(int(n_o[i]),n_max):
                time2[0][i][n]= np.full(shape=(1,n_m), fill_value=-1, dtype=int)

    dur=time2[0]
    pij=[]
    mean = []
    for i in range(n_j):
        ni_p=0
        dur_mean = []
        for j in range(n_max):
            durmch = dur[i][j][np.where(dur[i][j] >= 0)]
            if len(durmch) != 0:
                ni_p = ni_p + durmch.mean()
                dur_mean.append(durmch.mean().tolist())
            else:
                ni_p = ni_p + 0
                dur_mean.append(0)
        pij.append(ni_p)
        mean.append(dur_mean)
                                                                                            
    input_mean = np.array(mean)
    pij = np.array(pij)

    ri = np.random.uniform(low=0, high=20, size=n_j)
    di=ri+c*pij

    n_o = n_o.reshape(n_j, 1)
    ri = ri.reshape(n_j, 1)
    di = di.reshape(n_j, 1)

    ord = np.concatenate((n_o, ri, di), -1)
        
    return dur, ord, n_max