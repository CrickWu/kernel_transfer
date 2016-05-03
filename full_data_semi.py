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
    sol = cvxopt.solvers.qp(matrix(P), matrix(q))#, matrix(G), matrix(h))
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
# gList: gamma for rbf kernel
# wList: weight for Manifold regularization term
# pList: sparsity for kernel (p-nearest neighbor)
# kernel_type: rbf or cosine
# complete_flag: 1 or 0 (default 0), whether complete K_st and K_ts block of kernel or not 
def grid_search(gList, wList, pList, kernel_type, complete_flag=0):
    dc = DataClass()
    dc.kernel_type = kernel_type
    y, I, K, offset = dc.get_TL_Kernel()

    n = len(y)
    f = np.zeros(n)
    best_auc = 0.0
    best_gs = 0.0
    best_gt = 0.0
    best_w = 0.0
    best_p = -1
    
    if kernel_type == 'cosine' and len(gList) != 1:
        raise Warning('For cosine kernel, no need to tune gamma!')
 
    for g_s in gList:
        for g_t in gList:
            dc.source_gamma = 2.0**g_s
            dc.target_gamma = 2.0**g_t
            y, I, K, offset = dc.get_TL_Kernel()
            if complete_flag == 1:
                K = DataClass.complete_TL_Kernel(K, offset)


            for w in wList:
                w_2 = 2.0**w
                auc, ap, rl = solve_and_eval(y, I, K, offset, w_2)
                print('log_2gs %3d log_2gt %3d log_2w %3d log_2p  -1 auc %6f ap %6f rl %6f' % (g_s, g_t, w, auc, ap, rl))
                if auc > best_auc:
                    best_auc = auc
                    best_gs = g_s
                    best_gt = g_t
                    best_w = w

                for p in pList:
                    _p = 2**p
                    K_sp = DataClass.sym_sparsify_K(K, _p)
                    auc, ap, rl = solve_and_eval(y, I, K_sp, offset, w_2)
                    print('log_2gs %3d log_2gt %3d log_2w %3d log_2p %3d auc %6f ap %6f rl %6f' \
                            % (g_s, g_t, w, p, auc, ap, rl))
                    if auc > best_auc:
                        best_auc = auc
                        best_gs = g_s
                        best_gt = g_t
                        best_w = w
                        best_p = p

    print('best parameters: log_2gs %3d log_2gt %3d log_2w %3d log_2p %3d auc %6f' \
            % (best_gs, best_gt, best_w, best_p, best_auc))



def run_testset(kernel_type='cosine', log_2gs=-12, log_2gt=-12, log_2w=-2, log_2p=-1, complete_flag=1):
    dc = DataClass(valid_flag=True)
    dc.kernel_type = kernel_type
    dc.source_gamma = 2**log_2gs
    dc.target_gamma = 2**log_2gt
    y, I, K, offset = dc.get_TL_Kernel()
    
    w_2 = 2**log_2w
    p = 2**log_2p
    
    if complete_flag == 1:
        K = DataClass.complete_TL_Kernel(K, offset)

    # log_2p == -1 means that using full kernel w/o sparsify
    if log_2p != -1:
        K = DataClass.sym_sparsify_K(K, p)
    
   
    auc, ap, rl = solve_and_eval(y, I, K, offset, w_2)
    print('tst test: auc %6f ap %6f rl %6f' %(auc, ap, rl))


if __name__ == '__main__':
    kernel_type = 'rbf'
    #kernel_type = 'cosine'
    complete_flag = 0
    if kernel_type == 'rbf':
        gList = np.arange(-12, 1, 2)
    elif kernel_type == 'cosine':
        gList = np.arange(-12, -10, 2)
    else:
        raise ValueError('unknown kernel type')
    
    wList = np.arange(-12, 10, 2)
    pList = np.arange(4, 10, 1)
    grid_search(gList, wList, pList, kernel_type, complete_flag)
    #run_testset(kernel_type='cosine', log_2w=-2, log_2p=-1, complete_flag=1)
