#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from joblib import Parallel, delayed
from dataloader import default_y_I_K
from sklearn.datasets import load_svmlight_file

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
            coeff = 2 * K[i, j] * U[i,:].dot(U[j,:].T)
            grad += U[j,:] * coeff
        return grad * w_2

    def grad_f():
        K_approx = U.dot(U.T)
        # D = np.diag( np.asarray(K_approx.sum(1)).flatten() )
        D = np.diag( K_approx.sum(1) )
        lap =  D - K_approx
        grad = I * (f - y) * 2
        # grad += np.asarray(lap.dot(f)).flatten() * 2
        grad += lap.dot(f) * 2 * w_3
        return grad


    y, I, K = default_y_I_K()

    d = 10
    n = len(y)
    # weighting coefficient for the second and third terms
    w_2 = 1.0 / n
    w_3 = 1.0 / n

    # lr: learning rate
    lr_U = 1.0 / n * 1e4
    lr_f = 1.0 / n * 1e4

    U = np.random.randn(n, d)
    # U = np.zeros([n, d])
    f = np.zeros(n)

    row_idx, col_idx = np.nonzero(K)
    K_ind = np.zeros([n,n], dtype=np.float)
    K_ind[row_idx, col_idx] = 1.0

    # doing optimization
    max_iter = 100
    for iteration in xrange(max_iter):
        loss = ((f - y)**2 * I).sum()

        K_approx = U.dot(U.T)
        # D = np.diag( np.asarray(K_approx.sum(1)).flatten() )
        D = np.diag( K_approx.sum(1) )
        lap = D - K_approx

        loss += f.T.dot(lap).dot(f) / 2.0 * w_2
        loss += ((U.dot(U.T) - K)**2 * K_ind).sum() * w_3

        print 'at iteration', iteration, loss
        def update_U(x):
            return grad_U(x) * lr_U

        ret_grad_U = Parallel(n_jobs=16, verbose=5)(delayed(update_U)(i) for i in xrange(n))
        for i in xrange(n):
            # U[i,:] -= ret_grad_U[i] * lr_U
            U[i,:] -= ret_grad_U[i]

        f -=  grad_f() * lr_f

    with open('f.pred') as fout:
        for f_ele, y_ele in zip(f, y):
            fout.write('%g\t%g\n' % (f_ele, y_ele))

