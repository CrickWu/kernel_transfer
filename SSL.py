#!/usr/bin/env python
# encoding: utf-8
# This file tries to do semi-supervised learning on target domain test data
# using target domain [train, test] data only.
# The solution for prediction `f` is exact.
# We tune parameters over `gamma` for the kernel, `w_2` for coefficient for regularization term
# and number of nearest neighbor to make the kernel sparse.

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from dataclass import DataClass
from solve import solve_and_eval

import cvxopt
from cvxopt import matrix
np.random.seed(123)

# grid search hyperparameter on valid set
# gList: gamma for rbf kernel.
# wList: weight for Manifold regularization term
# pList: sparsity for kernel (p-nearest neighbor)
# kernel_type: rbf or cosine
# zero_diag_flag: whether zero out the diagonal or not
def grid_search(gList, wList, pList, kernel_type, zero_diag_flag=True):
    dc = DataClass()
    dc.kernel_type = kernel_type
    dc.zero_diag_flag = zero_diag_flag
    y, I, K, offset = dc.get_SSL_Kernel()

    best_auc = 0.0
    best_g = 0.0
    best_w = 0.0
    best_p = 0.0

    if kernel_type == 'cosine' and len(gList) != 1:
        raise Warning('For cosine kernel, no need to tune gamma!')


    for g in gList:
        dc.target_gamma = 2**g
        y, I, K, offset = dc.get_SSL_Kernel()

        for w in wList:
            # weighting coefficient for the second term
            w_2 = 2.0**w
            auc, ap, rl = solve_and_eval(y, I, K, offset, w_2)
            print('log_2g %3d log_2w %3d log_2p  -1 auc %6f ap %6f rl %6f' % (g, w, auc, ap, rl))
            if auc > best_auc:
                best_auc = auc
                best_g = g
                best_w = w
                best_p = -1


            for p in pList:
                _p = 2**p
                K_sp = DataClass.sym_sparsify_K(K, _p)
                auc, ap, rl = solve_and_eval(y, I, K_sp, offset, w_2)
                print('log_2g %3d log_2w %3d log_2p %3d auc %6f ap %6f rl %6f' \
                        % (g, w, p, auc, ap, rl))
                if auc > best_auc:
                    best_auc = auc
                    best_g = g
                    best_w = w
                    best_p = p

    print('best parameters: log_2g %3d log_2w %3d log_2p %3d auc %6f' \
            % (best_g, best_w, best_p, best_auc))


def run_testset(kernel_type='cosine', log_2g=-12, log_2w=-12, log_2p=6, zero_diag_flag=True):
    dc = DataClass(valid_flag=False)
    dc.kernel_type = kernel_type
    dc.target_gamma = 2**log_2g
    dc.zero_diag_flag = zero_diag_flag
    y, I, K, offset = dc.get_SSL_Kernel()

    w_2 = 2**log_2w
    p = 2**log_2p

    # log_2p == -1 means that using full kernel w/o sparsify
    if log_2p != -1:
        K = DataClass.sym_sparsify_K(K, p)

    auc, ap, rl = solve_and_eval(y, I, K, offset, w_2)
    print('tst test: auc %6f ap %6f rl %6f' %(auc, ap, rl))


if __name__ == '__main__':
    kernel_type = 'cosine'
    #kernel_type = 'rbf'
    if kernel_type == 'rbf':
        gList = np.arange(-12, 1, 2)
    elif kernel_type == 'cosine':
        gList = np.arange(-12, -10, 2)
    else:
        raise ValueError('unknown kernel type')

    wList = np.arange(-12, 10, 2)
    pList = np.arange(4, 10, 1)
    # grid_search(gList, wList, pList, kernel_type, zero_diag_flag=True)
    run_testset(kernel_type='cosine', log_2g=-12, log_2w=-8, log_2p=6, zero_diag_flag=True) # zero_diag
    # run_testset(kernel_type='cosine', log_2g=-12, log_2w=-12, log_2p=6, zero_diag_flag=False) # non_zero_diag
