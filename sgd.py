#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
    # y: (ns+nt): true_values
    # f: (ns+nt): prediction
    # K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
    # U: (ns+nt) * d (UU^T is the low rank approximation)
    # I: (ns+nt) observation indicator

    y = np.array([1,1,0,1,0])
    f = np.array([0.1,1,0,0.5,0.2])
    I = np.array([0,1,0,1,1])
    U = np.matrix([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]], dtype=np.float)

    V = np.matrix([[4,5,6],[4,5,6],[4,5,6],[4,5,6],[4,5,6]], dtype=np.float)
    # K = np.matrix([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
    K = V.dot(V.T)

    # d: rank
    # n: # of instances
    d = U.shape[1]
    n = U.shape[0]

    # lr: learning rate
    lr_U = 1.0 / n * 1e-5
    lr_f = 1.0 / n * 1e-5

    def grad_U(i):
        grad = np.zeros([1,d])
        for j in xrange(n):
            grad += U[j,:] * (f[i] - f[j])**2 # may improve using matrix factorization

        for j in xrange(n):
            # if not i ~ j
            if K[i, j] == 0:
                continue
            coeff = 2 * K[i, j] * U[i,:].dot(U[j,:].T)[0,0]
            grad += U[j,:] * coeff
        # grad +=
        return grad

    def grad_f():
         grad = I * (f - y) * 2
         grad += np.asarray(U.dot(U.T).dot(f)).flatten() * 2
         return grad

    # doing optimization
    max_iter = 1
    for iteration in xrange(max_iter):
        for i in xrange(n):
            U[i] -= grad_U(i) * lr_U
        f -=  grad_f() * lr_f

