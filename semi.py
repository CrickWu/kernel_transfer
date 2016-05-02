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
from dataclass import DataClass

np.random.seed(123)
if __name__ == '__main__':

    dc = DataClass()
    y, I, K, offset = dc.get_target_y_I_K()
    start_offset = offset[0] # for calculating `ap`

    n = len(y)
    # w_2 = 100.0 / n

    f = np.zeros(n)

    for g in xrange(-10, -4):
        gamma = 2.0**g
        dc.target_gamma = gamma
        y, I, K, offset = dc.get_target_y_I_K()

        for i in xrange(-10, 10):
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

            print 'gamma: {} w_2: {}'.format(gamma, w_2), 'loss:', loss1, loss2, loss
            print '\t\tap', average_precision_score( y[start_offset:], f[start_offset:], average='macro' )


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

                print 'gamma: {} w_2: {}'.format(gamma, w_2), 'loss:', loss1, loss2, loss
                print '\tsparse {}\tap'.format(j), average_precision_score( y[start_offset:], f[start_offset:], average='macro' )



