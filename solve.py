#!/usr/bin/env python
# encoding: utf-8
# This file solves the optimization problem using sqaured loss.

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

import cvxopt
from cvxopt import matrix


# assuming y_true \in {-1, 1}
def evalulate(y_true, y_prob):
    # the following deal with {0,1} and {-1,1} ambiguities
    # but may slow down the process (?? not sure)
    # if -1 in y_true:
        # y_true = (y_true + 1) / 2.0
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
    if len(offset) == 2: # SSL setting
        start_offset = offset[0]
        end_offset = offset[1]
    else: # TL setting
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

