#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys
from scipy.sparse import csr_matrix


# If you are on LA server, then it's runable.
# If not, make sure you install libpmf python interface
sys.path.append('/usr0/home/wchang2/pkg/libpmf-1.41/python/')
import libpmf


def evaluate(K, K_ind, U, V):
    nnz = K.nonzero()[0].shape[0]
    tmp = (K - U.dot(V.T))*K_ind 
    mae = abs(tmp).sum() / float(nnz)
    rmse = np.sqrt((tmp**2).sum() / float(nnz))
    return mae, rmse

# K is rating matrix, d is rank
def libmf_solver(K, d=10, lambda_u=2**(-5), max_iter=10):
    n = K.shape[0]
    rowList, colList = K.nonzero()
    K_ind = np.zeros([n,n], dtype=np.float)
    K_ind[rowList, colList] = 1.0

    param_str = '-k %d -l %f -n 8 -t %d' % (d, lambda_u, max_iter)
    model = libpmf.train(K, param_str, zero_as_missing=True)
    mae, rmse = evaluate(K, K_ind, model['H'], model['W'])
    print('param %s, mae %6f rmse %6f' % (param_str, mae, rmse))
    
    return model['H'], model['W']
