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

np.random.seed(123)


def evalulate(y_true, y_prob):
    y_true = (y_true + 1) / 2.0
    auc = roc_auc_score(y_true, y_prob)
    ap = label_ranking_average_precision_score([y_true], [y_prob])
    rl = label_ranking_loss([y_true], [y_prob])
    return auc, ap, rl


if __name__ == '__main__':

    dc = DataClass()
    y, I, K, offset = dc.get_SSL_Kernel()
    start_offset = offset[0] # for calculating `ap`

    n = len(y)
    # w_2 = 100.0 / n

    f = np.zeros(n)
    best_auc = 0.0
    best_g = 0.0
    best_i = 0.0
    best_j = 0.0


    for g in xrange(-12, 0, 2):
        dc.target_gamma = 2**g
        y, I, K, offset = dc.get_SSL_Kernel()

        for i in xrange(-12, 10, 2):
            # weighting coefficient for the second term
            w_2 = 2.0**i
            # closed form
            D = np.diag( K.sum(1) )
            lap = D - K

            A = lap * w_2 + np.diag( I )
            b = I * y
            f = np.linalg.lstsq(A,b)[0]

            start_offset = I.sum()

            loss1 = ((f - y)**2 * I).sum() # prediction loss
            loss2 = f.T.dot(lap).dot(f) * w_2 # regularization loss
            loss = loss1+loss2

            y_true = y[start_offset:]
            y_prob = f[start_offset:]
            auc, ap, rl = evalulate(y_true, y_prob)
            print('log_2g %3d log_2w %3d sparsity 000 auc %6f ap %6f rl %6f' % (g, i, auc, ap, rl))
            if auc > best_auc:
                best_auc = auc
                best_g = g
                best_i = i
                best_j = 0


            for j in [10, 20, 40, 80, 160, 320, 500]:
                K_sp = DataClass.sym_sparsify_K(K, j)

                # sparse
                D = np.diag( K_sp.sum(1) )
                lap = D - K_sp

                A = lap * w_2 + np.diag( I )
                b = I * y
                f = np.linalg.lstsq(A,b)[0]

                start_offset = I.sum()

                loss1 = ((f - y)**2 * I).sum()
                loss2 = f.T.dot(lap).dot(f) * w_2
                loss = loss1+loss2
                
                y_true = y[start_offset:]
                y_prob = f[start_offset:]
                auc, ap, rl = evalulate(y_true, y_prob)
                print('log_2g %3d log_2w %3d sparsity %3d auc %6f ap %6f rl %6f' \
                        % (g, i, j, auc, ap, rl))
                if auc > best_auc:
                    best_auc = auc
                    best_g = g
                    best_i = i
                    best_j = j


    print('best parameters: log_2g %3d log_2w %3d sparsity %3d auc %6f' \
            % (best_g, best_i, best_j, best_auc))


