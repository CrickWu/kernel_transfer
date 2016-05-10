#!/usr/bin/env python
# encoding: utf-8
# This file tries to do semi-supervised learning on target domain test data
# using all data [source_train, source_test, source_para, target_train, target_test, target_para].
# We do not complete `K`.
# The solution for prediction `f` is exact.
# The tuning parameters `w_2` for coefficient for regularization term. (we use default gamme, which is the sqrt of dimension for each domain)

import sys
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from dataclass import DataClass
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import scipy.sparse as sp 
import cvxopt
from cvxopt import matrix
np.random.seed(123)


def evalulate(y_true, y_prob):
    y_true = (y_true + 1) / 2.0
    auc = roc_auc_score(y_true, y_prob)
    ap = label_ranking_average_precision_score([y_true], [y_prob])
    rl = label_ranking_loss([y_true], [y_prob])
    return auc, ap, rl

def solve_and_eval(y, I, K, offset, w_2):
    # closed form
    n = y.shape[0]
    D = np.diag( np.sum(K, axis=1) )
    lap = D - K

    P = lap * w_2 + np.diag( I )
    q = -I * y
    G = -np.diag(np.ones(n))
    h = np.zeros(n)
    #f = np.linalg.lstsq(P,-q)[0]
    # using cvxopt quadratic programming:
    #    min_x  1/2 xTPx + qTx
    #    s.t.   Gx <= h
    #           Ax = b
    # reference: https://github.com/cvxopt/cvxopt
    #            http://cvxopt.org/examples/
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    f = np.array(sol['x'])[:,0]

    # for calculating ap
    start_offset = offset[3]
    end_offset = offset[4]

    #loss1 = ((f - y)**2 * I).sum()
    #loss2 = f.T.dot(lap).dot(f) * w_2
    #loss = loss1+loss2

    #ap = average_precision_score(y[start_offset:end_offset], f[start_offset:end_offset])
    y_true = y[start_offset:end_offset]
    y_prob = f[start_offset:end_offset]
    auc, ap, rl = evalulate(y_true, y_prob)
    return auc, ap, rl


def nCk(n,k):
    f = math.factorial
    return f(n) / f(k) / f(n-k)

# alpha step random walk
def get_K_rw(K_rw, offset, alpha=1):
    W_s = K_rw[:offset[2], :offset[2]]
    #W_s = normalize(W_s, norm='l1', axis=1) 
    W_t = K_rw[offset[2]:, offset[2]:]
    #W_t = normalize(W_t, norm='l1', axis=0)
    n_s = W_s.shape[0]
    n_t = W_t.shape[0]
    Y_st = K_rw[:offset[2], offset[2]:] 
    K_st = np.zeros((n_s,n_t))
    #K_st = Y_st
    for i in xrange(0, alpha+1):
        tmp1 = np.linalg.matrix_power(W_s,i)
        tmp2 = np.linalg.matrix_power(W_t,alpha-i)
        K_st += nCk(alpha,i)*tmp1.dot(Y_st.dot(tmp2)) 
    #K_st = K_st / np.max(K_st)
    K_st[offset[1]:, (offset[4]-offset[2]):] /= 2.0
    
    K_rw[:offset[2], offset[2]:] = K_st
    K_rw[offset[2]:, :offset[2]] = K_st.T
    return K_rw

# grid search hyperparameter on valid set
# aList: beta for exp kernel
# wList: weight for Manifold regularization term
# pList: sparsity for kernel (p-nearest neighbor)
# kernel_type: 'rbf' or 'cosine'
# source_data_type: 'full' or 'normal'
# kernel_normal: whether to normalized the W or not
# zero_diag_flag: whether zero out the diagonal or not
def grid(kernel_type='cosine', zero_diag_flag=False, kernel_normal=False, aList=None, wList=None, pList=None):
    dc = DataClass(valid_flag=True, kernel_normal=kernel_normal)
    dc.kernel_type = kernel_type
    dc.zero_diag_flag = zero_diag_flag
    y, I, K, offset = dc.get_TL_Kernel()
    
    best_a = -1
    best_w = -1
    best_p = -1
    best_auc = -1
    for alpha in aList:
        K_rw = K.copy()
        K_rw = get_K_rw(K_rw, offset, alpha=alpha)
 
        for log2_w in wList:
            for log2_p in pList:
                if log2_p == -1:
                    K_sp = K_rw
                else:
                    K_sp = DataClass.sym_sparsify_K(K_rw, 2**log2_p)
                #print DataClass.sparse_K_stat(K_sp, offset)
                auc, ap, rl = solve_and_eval(y, I, K_sp, offset, 2**log2_w)
                print('alpha %3d log2_w %3d log2_p %3d auc %8f ap %6f rl %6f' %(alpha, log2_w, log2_p, auc, ap, rl))
                if best_auc < auc:
                    best_a = alpha
                    best_w = log2_w
                    best_p = log2_p
                    best_auc = auc
    
    print('best parameters: alpha %3d log2_w %3d log2_p %3d auc %6f' \
            % (best_a, best_w, best_p, best_auc))


def run_testset(kernel_type='cosine', zero_diag_flag=False, kernel_normal=False, alpha=None, log2_w=None, log2_p=None):
    dc = DataClass(valid_flag=False, kernel_normal=kernel_normal)
    dc.kernel_type = kernel_type
    dc.zero_diag_flag = zero_diag_flag
    y, I, K, offset = dc.get_TL_Kernel()
    
    K_rw = K.copy()
    K_rw = get_K_rw(K_rw, offset, alpha=alpha)
    if log2_p == -1:
        K_sp = K_rw
    else:
        K_sp = DataClass.sym_sparsify_K(K_rw, 2**log2_p)
    auc, ap, rl = solve_and_eval(y, I, K_sp, offset, 2**log2_w)
    print('test set: auc %8f ap %6f rl %6f' %(auc, ap, rl))

if __name__ == '__main__':
    aList = np.arange(1,6,1)
    wList = np.arange(-12, -10, 2) 
    pList = np.arange(-1, 11, 2)
    #grid(kernel_type='cosine', zero_diag_flag=False, kernel_normal=False, aList=aList, wList=wList, pList=pList)
    run_testset(kernel_type='cosine', zero_diag_flag=False, kernel_normal=False, alpha=1, log2_w=-6, log2_p=3)

