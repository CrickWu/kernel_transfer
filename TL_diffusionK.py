#!/usr/bin/env python
# encoding: utf-8
# This file tries to do semi-supervised learning on target domain test data
# using all data [source_train, source_test, source_para, target_train, target_test, target_para].
# We do not complete `K`.
# The solution for prediction `f` is exact.
# The tuning parameters `w_2` for coefficient for regularization term. (we use default gamme, which is the sqrt of dimension for each domain)

import sys
import numpy as np
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
    D = np.diag( K.sum(1) )
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


# grid search hyperparameter on valid set
# bList: beta for exp kernel
# wList: weight for Manifold regularization term
# pList: sparsity for kernel (p-nearest neighbor)
# kernel_type: 'rbf' or 'cosine'
# source_data_type: 'full' or 'normal'
# kernel_normal: whether to normalized the W or not
# zero_diag_flag: whether zero out the diagonal or not
def grid_diffK(kernel_type='cosine', zero_diag_flag=False, kernel_normal=True, bList=None, wList=None, pList=None):
    dc = DataClass(valid_flag=True, kernel_normal=kernel_normal)
    dc.kernel_type = kernel_type
    dc.zero_diag_flag = zero_diag_flag
    y, I, K, offset = dc.get_TL_Kernel()

    # run eigen decomposition on K
    W_s = K[:offset[2], :offset[2]]
    W_t = K[offset[2]:, offset[2]:]
    max_k = 128
    v_s, Q_s = sp.linalg.eigsh(W_s, k=max_k)
    v_t, Q_t = sp.linalg.eigsh(W_t, k=max_k)
    Y_st = K[:offset[2], offset[2]:]
    
    best_b = -1
    best_w = -1
    best_p = -1
    best_auc = -1
    for log2_b in bList:
        beta = 2**log2_b
        Lambda_s = np.diag(np.exp(beta*v_s))
        Lambda_t = np.diag(np.exp(beta*v_t))
        K_ss = Q_s.dot(Lambda_s.dot(Q_s.T)) 
        K_tt = Q_t.dot(Lambda_t.dot(Q_t.T))
        K_st = K_ss.dot(Y_st.dot(K_tt))
        if not kernel_normal:
            K_st = normalize(K_st)
        K[:offset[2], offset[2]:] = K_st
        K[offset[2]:, :offset[2]] = K_st.T 
        for log2_w in wList:
            auc, ap, rl = solve_and_eval(y, I, K, offset, 2**log2_w)
            print('log2_b %3d log2_w %3d log2_p  -1 auc %8f ap %6f rl %6f' %(log2_b, log2_w, auc, ap, rl))
            if best_auc < auc:
                best_b = log2_b
                best_w = log2_w
                best_p = -1

            for log2_p in pList:
                K_sp = DataClass.sym_sparsify_K(K, 2**log2_p)
                auc, ap, rl = solve_and_eval(y, I, K_sp, offset, 2**log2_w)
                print('log2_b %3d log2_w %3d log2_p %3d auc %8f ap %6f rl %6f' %(log2_b, log2_w, log2_p, auc, ap, rl))
                if best_auc < auc:
                    best_b = log2_b
                    best_w = log2_w
                    best_p = log2_p

    print('best parameters: log2_b %3d log2_w %3d log2_p %3d auc %6f' \
            % (best_b, best_w, best_p, best_auc))


def run_testset(kernel_type='cosine', 
        zero_diag_flag=False, kernel_normal=True, 
        log2_b=2**(-14), log2_w=2**(-8), log2_p=2**8):
    dc = DataClass(valid_flag=False, kernel_normal=kernel_normal)
    dc.kernel_type = kernel_type
    dc.zero_diag_flag = zero_diag_flag
    y, I, K, offset = dc.get_TL_Kernel()

    # run eigen decomposition on K
    W_s = K[:offset[2], :offset[2]]
    W_t = K[offset[2]:, offset[2]:]
    max_k = 128
    v_s, Q_s = sp.linalg.eigsh(W_s, k=max_k)
    v_t, Q_t = sp.linalg.eigsh(W_t, k=max_k)
    Y_st = K[:offset[2], offset[2]:]
   
    beta = 2**log2_b
    Lambda_s = np.diag(np.exp(beta*v_s))
    Lambda_t = np.diag(np.exp(beta*v_t))
    K_ss = Q_s.dot(Lambda_s.dot(Q_s.T)) 
    K_tt = Q_t.dot(Lambda_t.dot(Q_t.T))
    K_st = K_ss.dot(Y_st.dot(K_tt))
    if not kernel_normal:
        K_st = normalize(K_st)
    K[:offset[2], offset[2]:] = K_st
    K[offset[2]:, :offset[2]] = K_st.T 
    if log2_p != -1:
        K = DataClass.sym_sparsify_K(K, 2**log2_p)
    auc, ap, rl = solve_and_eval(y, I, offset, K, 2**log2_w)
    print('test set: auc %6f ap %6f rl %6f' % (auc, ap, rl))


if __name__ == '__main__':
    source_data_type = 'full'
    kernel_type = 'cosine'
    zero_diag_flag = True

    bList = np.arange(-14, -4, 2)
    wList = np.arange(-14, 0, 2) 
    pList = np.arange(0, 12, 2)
    #grid_diffK(kernel_type='cosine', zero_diag_flag=False, kernel_normal=True, bList=bList, wList=wList, pList=pList)
    run_testset()
