#!/usr/bin/env python
# encoding: utf-8
# This file tries to do semi-supervised learning on target domain test data
# using all data [source_train, source_test, source_para, target_train, target_test, target_para].
# We do not complete `K`.
# The solution for prediction `f` is exact.
# The tuning parameters `w_2` for coefficient for regularization term. (we use default gamme, which is the sqrt of dimension for each domain)

import numpy as np
from joblib import Parallel, delayed
from dataloader import default_y_I_K, get_target_y_I_K
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score

np.random.seed(123)
if __name__ == '__main__':
    y, I, K, offset = default_y_I_K()

    n = len(y)
    # weighting coefficient for the second term
    # w_2 = 100000.0 / n

    f = np.zeros(n)
    # y, I, K, offset = get_target_y_I_K(gamma=gamma)
    for i in xrange(-10, 20):
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

        loss1 = ((f - y)**2 * I).sum()
        loss2 = f.T.dot(lap).dot(f) * w_2
        loss = loss1+loss2

        print 'w_2: {}'.format(w_2), 'loss:', loss1, loss2, loss
        print '\t\tap', average_precision_score( y[start_offset:end_offset], f[start_offset:end_offset], average='macro' )

