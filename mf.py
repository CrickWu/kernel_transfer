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

''' return the approximated matrix, mae, rmse'''
def get_sym_approx(K, d=10, lambda_u=2**(-5), max_iter=10):
    n = K.shape[0]
    rowList, colList = K.nonzero()
    K_ind = np.zeros([n,n], dtype=np.float)
    K_ind[rowList, colList] = 1.0

    H, W = libmf_solver(K, d, lambda_u, max_iter)
    ret_K = H.dot(W.T)
    ret_K = (ret_K + ret_K.T) / 2.0
    mae, rmse = evaluate(K, K_ind, H, W)
    return ret_K, mae, rmse

def evaluate_K(K, K_approx, K_ind):
    nnz = K_ind.nonzero()[0].shape[0]
    tmp = (K - K_approx)*K_ind
    mae = abs(tmp).sum() / float(nnz)
    rmse = np.sqrt((tmp**2).sum() / float(nnz))
    return mae, rmse

''' return the approximated matrix, mae, rmse'''
def get_sym_K_approx(K, offset, complete_flag=True, d=10, lambda_u=2**(-5), max_iter=10):
    param_str = '-k %d -l %f -n 8 -t %d -N 1' % (d, lambda_u, max_iter) # -N 1 is the non-negative option
    if len(offset) == 2: # semi mf
        K_ind = np.ones(K.shape, dtype=np.int)
        model = libpmf.train(K, param_str, zero_as_missing=False)
    else:
        assert len(offset) == 6 # transfer learning mf
        K_ind = np.ones(K.shape, dtype=np.int)
        if complete_flag:
            K_ind[offset[2]:offset[4],0:offset[1]] = 0
            K_ind[0:offset[1],offset[2]:offset[4]] = 0
        else:
            K_ind[offset[2]:offset[5],0:offset[1]] = 0
            K_ind[0:offset[1],offset[2]:offset[5]] = 0
        K_ind[xrange(offset[2],offset[5]),xrange(0,offset[1])] = 1
        K_ind[xrange(0,offset[1]),xrange(offset[2],offset[5])] = 1
        # get the real index for calling train_coo
        row_idx, col_idx = np.where(K_ind)
        obs_val = K[row_idx, col_idx]
        m, n = K.shape
        model = libpmf.train_coo(row_idx=row_idx, col_idx=col_idx, obs_val=obs_val, m=m, n=n)

    mae, rmse = evaluate(K, K_ind, model['H'], model['W'])
    ret_K = model['H'].dot(model['W'].T)
    ret_K = (ret_K + ret_K.T) / 2.0
    mae, rmse = evaluate_K(K, ret_K, K_ind)
    # sys.stderr.write('param %s, mae %6f rmse %6f\n' % (param_str, mae, rmse))
    return ret_K, mae, rmse

# K is rating matrix, d is rank
def libmf_solver(K, d=10, lambda_u=2**(-5), max_iter=10):
    n = K.shape[0]
    rowList, colList = K.nonzero()
    K_ind = np.zeros([n,n], dtype=np.float)
    K_ind[rowList, colList] = 1.0

    param_str = '-k %d -l %f -n 8 -t %d -N 1' % (d, lambda_u, max_iter)
    model = libpmf.train(K, param_str, zero_as_missing=True)
    mae, rmse = evaluate(K, K_ind, model['H'], model['W'])
    sys.stderr.write('param %s, mae %6f rmse %6f\n' % (param_str, mae, rmse))

    return model['H'], model['W']
