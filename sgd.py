#!/usr/bin/env python
# encoding: utf-8
# This file uses GD to update `U` and `f` simultaneously.
# Note that the current alternating process does not
# provide a good result due to either the bad approxmimation
# of U or the slow convergance speed. Refer to full_data_semi.py
# for a fixed `K` (kernel) and precise `f` solution.

import numpy as np
from joblib import Parallel, delayed
from dataloader import default_y_I_K
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score

np.random.seed(123)
if __name__ == '__main__':
    # # The following is synthesized data for testing
    # # y: (ns+nt): true_values
    # # f: (ns+nt): prediction
    # # K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
    # # U: (ns+nt) * d (UU^T is the low rank approximation)
    # # I: (ns+nt) observation indicator

    # y = np.array([1,1,0,1,0])
    # f = np.array([0.1,1,0,0.5,0.2])
    # I = np.array([0,1,0,1,1])
    # U = np.matrix([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]], dtype=np.float)

    # V = np.matrix([[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6]], dtype=np.float)
    # # K = np.matrix([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
    # K = V.dot(V.T)

    # # d: rank
    # # n: # of instances
    # d = U.shape[1]
    # n = U.shape[0]


    def grad_U(i):
        grad = np.zeros(d)
        for j in xrange(n):
            grad += U[j,:] * (f[i] - f[j])**2 # may improve using matrix factorization

        for j in xrange(n):
            # if not i ~ j
            if K[i, j] == 0:
                continue
            coeff = -2 * (K[i, j] - U[i,:].dot(U[j,:].T))
            grad += U[j,:] * coeff
        return grad * w_3

    def grad_f():
        K_approx = U.dot(U.T)
        # D = np.diag( np.asarray(K_approx.sum(1)).flatten() )
        D = np.diag( K_approx.sum(1) )
        lap =  D - K_approx
        grad = I * (f - y) * 2
        # grad += np.asarray(lap.dot(f)).flatten() * 2
        grad += lap.dot(f) * 2 * w_2
        return grad


    y, I, K, offset = default_y_I_K()

    d = 10
    n = len(y)
    # weighting coefficient for the second and third terms
    w_2 = 10.0 / n
    w_3 = 1.0 / n

    # lr: learning rate
    lr_U = 1.0 / n * 1e0
    lr_f = 1.0 / n * 1e0

    U = np.random.randn(n, d)
    # U = np.zeros([n, d])
    f = np.zeros(n)

    row_idx, col_idx = np.nonzero(K)
    K_ind = np.zeros([n,n], dtype=np.float)
    K_ind[row_idx, col_idx] = 1.0

    hist_grad_f = np.zeros(n)
    # doing optimization
    max_iter = 100
    try:
        for iteration in xrange(max_iter):
            loss1 = ((f - y)**2 * I).sum()

            K_approx = U.dot(U.T)
            # D = np.diag( np.asarray(K_approx.sum(1)).flatten() )
            D = np.diag( K_approx.sum(1) )
            lap = D - K_approx

            loss2 = f.T.dot(lap).dot(f) * w_2
            loss3 = ((U.dot(U.T) - K)**2 * K_ind).sum() * w_3
            loss = loss1+loss2+loss3

            print 'at iteration', iteration, loss1, loss2, loss3, loss
            # target test offset range
            start_offset = offset[3]
            end_offset = offset[4]
            print 'ap', average_precision_score( y[start_offset:end_offset], f[start_offset:end_offset], average='macro' )
            def update_U(x):
                return grad_U(x) * lr_U

            ret_grad_U = Parallel(n_jobs=16, verbose=5)(delayed(update_U)(i) for i in xrange(n))
            for i in xrange(n):
                # U[i,:] -= ret_grad_U[i] * lr_U
                U[i,:] -= ret_grad_U[i]

            # project back, which ensures U*U^T 's entires are large than 0
            U[U<0] = 0

            # f -=  grad_f() * lr_f
            # adagrad
            cur_grad_f = grad_f()
            hist_grad_f += cur_grad_f ** 2
            cur_grad_f = cur_grad_f / (1e-6 + np.sqrt(hist_grad_f))
            f -= cur_grad_f * lr_f
    except KeyboardInterrupt as e:
        print 'Training interrupt'

    np.savetxt('U.txt', U, fmt='%f')
    np.savetxt('UUT.txt', U.dot(U.T), fmt='%f')
    np.savetxt('K.txt', K, fmt='%f')

    np.savetxt('f.pred', np.transpose([f, y, I]), fmt='%f')

