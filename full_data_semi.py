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

np.random.seed(123)


def evalulate(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    ap = label_ranking_average_precision_score([y_true], [y_prob])
    rl = label_ranking_loss([y_true], [y_prob])
    return auc, ap, rl


if __name__ == '__main__':
    dc = DataClass()
    y, I, K, offset = dc.get_TL_Kernel()

    n = len(y)
    # weighting coefficient for the second term
    # w_2 = 100000.0 / n

    f = np.zeros(n)
    best_auc = 0.0
    best_gs = 0.0
    best_gt = 0.0
    best_i = 0.0
    best_auc_complete = 0.0
    best_gs_complete = 0.0
    best_gt_complete = 0.0
    best_i_complete = 0.0


   
    for g_s in xrange(-12, 0, 2):
        for g_t in xrange(-12, 0, 2):
            dc.source_gamma = 2.0**g_s
            dc.target_gamma = 2.0**g_t
            y, I, K, offset = dc.get_TL_Kernel()
            K_comp = DataClass.complete_TL_Kernel(K, offset)
            
            for i in xrange(-8, 12, 2):
                w_2 = 2.0**i
                # closed form
                D = np.diag( K.sum(1) )
                lap = D - K

                A = lap * w_2 + np.diag( I )
                b = I * y
                # solve equation Af = b for exact solution of `f`
                f = np.linalg.lstsq(A,b)[0]

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
                print('TL:          log_2gs %3d log_2gt %3d log_2w %3d auc %6f ap %6f rl %6f' % (g_s, g_t, i, auc, ap, rl))
                if auc > best_auc:
                    best_auc = auc
                    best_gs = g_s
                    best_gt = g_t
                    best_i = i


                # for complete version of kernel
                D = np.diag( K_comp.sum(1) )
                lap = D - K_comp
                A = lap * w_2 + np.diag( I )
                b = I * y
                f = np.linalg.lstsq(A,b)[0]

                # for calculating ap
                start_offset = offset[3]
                end_offset = offset[4]

                #loss1 = ((f - y)**2 * I).sum()
                #loss2 = f.T.dot(lap).dot(f) * w_2
                #loss = loss1+loss2
                y_true = y[start_offset:end_offset]
                y_prob = f[start_offset:end_offset]
                auc, ap, rl = evalulate(y_true, y_prob)
                print('TL_Complete: log_2gs %3d log_2gt %3d log_2w %3d auc %6f ap %6f rl %6f' % (g_s, g_t, i, auc, ap, rl))
                if ap > best_ap_complete:
                    best_ap_complete = ap
                    best_gs_complete = g_s
                    best_gt_complete = g_t
                    best_i_complete = i


    print('best parameters: log_2gs log_2gt %3d log_2w %3d auc %6f' \
            % (best_gs, best_gt, best_i, best_auc))
    print('best parameters: log_2gs log_2gt %3d log_2w %3d auc_complete %6f' \
            % (best_gs_complete, best_gt_complete, best_i, best_auc_complete))

