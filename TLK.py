#!/usr/bin/env python
# encoding: utf-8
# This file tries to do semi-supervised learning on target domain test data
# using all data [source_train, source_test, source_para, target_train, target_test, target_para].
# We do not complete `K`.
# The solution for prediction `f` is exact.
# The tuning parameters `w_2` for coefficient for regularization term. (we use default gamme, which is the sqrt of dimension for each domain)

import sys, os
import scipy
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import accuracy_score
from dataclass import DataClass
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import cvxopt
from cvxopt import matrix

np.random.seed(123)

def eval_binary(y_true, y_prob):
    unique_class = np.unique(y_true)
    assert(len(unique_class) == 2)
    if np.min(y_true) == -1:
        y_true = (y_true + 1) / 2.0

    n_pos = y_true[y_true == 1].shape[0]
    y_pred = np.zeros(y_true.shape[0])
    y_pred[np.argsort(y_prob)[::-1][:n_pos]] = 1
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    #ap = label_ranking_average_precision_score([y_true], [y_prob])
    #rl = label_ranking_loss([y_true], [y_prob])
    #return auc, ap, rl
    return auc, acc

# for least squared loss
def cvxopt_solver(y, I, K, offset, w_2):
    # closed form
    n = y.shape[0]
    D = np.diag( np.sum(K, axis=1) )
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
    start_offset = offset[3]
    end_offset = offset[4]

    y_true = y[start_offset:end_offset]
    y_prob = f[start_offset:end_offset]
    return y_true, y_prob


# sgd solver for L2-loss
# I use graident descent at this point
def sgd_l2(y, I, K, offset, gamma=2**(-10), nr_epoch=100, stepsize=2**(-1)):
    n = y.shape[0]
    batch_size = n
    nr_batch = n // batch_size
    D = np.diag( np.sum(K, axis=1) )
    lap = D - K
    f = np.ones(n)
    grad = np.zeros(n)
    hist_grad = 0.0

    for eidx in range(nr_epoch):
        idxList = np.arange(nr_batch)
        np.random.shuffle(idxList)
        for idx in idxList:
            start, end = idx*batch_size, (idx+1)*batch_size if idx < nr_batch else n
            
            # calculate active set and objective value
            obj = np.dot(f, np.dot(lap, f)) * gamma
            J = np.zeros(n, dtype='int64')
            for i in range(n):
                dv = 1.0 - f[i]*y[i]
                if I[i] > 0 and dv > 0:
                    J[i] = 1
                    obj += (dv**2)
            # calculate gradient and update f by adagrad
            grad = gamma*np.dot(lap, f) + J*f - J*y
            gnorm = np.sqrt(np.sum(grad**2))
            hist_grad += grad**2
            grad = grad / (1e-6 + np.sqrt(hist_grad))
            f = f - stepsize*grad
            sys.stderr.write('epoch %d obj %f gnorm %f\n' % (eidx, obj, gnorm))
        if gnorm < 1e-3:
            break

    # for calculating ap
    start_offset = offset[3]
    end_offset = offset[4]

    y_true = y[start_offset:end_offset]
    y_prob = f[start_offset:end_offset]
    return y_true, y_prob


def eigen_decompose(K, offset, max_k=None):
    W_s = K[:offset[2], :offset[2]]
    W_t = K[offset[2]:, offset[2]:]
    if max_k == None:
        v_s, Q_s = scipy.linalg.eigh(W_s)
        v_t, Q_t = scipy.linalg.eigh(W_t)
    else:
        v_s, Q_s = sp.linalg.eigsh(W_s, k=max_k)
        v_t, Q_t = sp.linalg.eigsh(W_t, k=max_k)

    return v_s, Q_s, v_t, Q_t


def get_K_exp_by_eigen(K, offset, v_s, Q_s, v_t, Q_t, beta):
    K_exp = K.copy()
    Y_st = K_exp[:offset[2], offset[2]:]
    Lambda_s = np.diag(np.exp(beta*v_s))
    Lambda_t = np.diag(np.exp(beta*v_t))
    K_ss = Q_s.dot(Lambda_s.dot(Q_s.T)) #+ np.identity(Q_s.shape[0]) - Q_s.dot(Q_s.T)
    K_tt = Q_t.dot(Lambda_t.dot(Q_t.T)) #+ np.identity(Q_t.shape[0]) - Q_t.dot(Q_t.T)
    K_st = K_ss.dot(Y_st.dot(K_tt))
    K_st = normalize(K_st, norm='l2', axis=1)
    K_exp[:offset[2], offset[2]:] = K_st
    K_exp[offset[2]:, :offset[2]] = K_st.T
    return K_exp


def run_one(srcPath=None, tgtPath=None, prlPath=None,
            source_n_features=None, target_n_features=None):

    dc = DataClass(srcPath=srcPath, tgtPath=tgtPath, prlPath=prlPath,
                   valid_flag=False, zero_diag_flag=True, source_data_type='full',
                   source_n_features=source_n_features, target_n_features=target_n_features,
                   kernel_type='cosine', kernel_normal=False)
    y, I, K, offset = dc.get_TL_Kernel()

    # run eigen decomposition on K
    v_s, Q_s, v_t, Q_t = eigen_decompose(K, offset, max_k=128)
    beta = 2**(-10)
    K_exp = get_K_exp_by_eigen(K, offset, v_s, Q_s, v_t, Q_t, beta)
    K_exp[K_exp<0] = 0
    #y_true, y_prob = cvxopt_solver(y, I, K_exp, offset, wreg)
    y_true, y_prob = sgd_l2(y, I, K_exp, offset, gamma=2**(-10), nr_epoch=1000, stepsize=2**(-1))
    auc, acc = eval_binary(y_true, y_prob)
    return auc, acc


def run_cls():
    #dataPath = '/usr0/home/wchang2/research/NIPS2016/data/cls/'
    dataPath = '/tmp2/b99902019/AAAI16/data/cls/'
    tgt = ['de', 'fr', 'jp']
    tgtSize = [2, 4, 8, 16, 32]
    #tgtSize= [2]
    domain = ['books', 'dvd', 'music']
    dimDict = {'en':60244, 'de':185922, 'fr':59906, 'jp':75809}
    nr_seed = 3
    for d_s in domain:
        for t in tgt:
            for d_t in domain:
                if d_s == d_t:
                    continue
                acc_result= []
                auc_result = []
                for p in tgtSize:
                    acc_eval = []
                    auc_eval = []
                    for seed in xrange(0, nr_seed):
                        dirPath = dataPath + 'cls_seed_%d/en_%s_%s_%s/' % (seed, d_s, t, d_t)
                        assert(os.path.isdir(dirPath))
                        srcPath = dirPath + 'src.1024'
                        tgtPath = dirPath + 'tgt.%d' % (p)
                        prlPath = dirPath
                        sys.stderr.write('%s\n' % (tgtPath))
                        auc, acc = run_one(srcPath=srcPath, tgtPath=tgtPath, prlPath=prlPath,
                                           source_n_features=dimDict['en'], target_n_features=dimDict[t])
                        #auc, acc = 0.0, 0.0
                        sys.stderr.write('%f %f\n' % (acc, auc))
                        acc_eval.append(acc)
                        auc_eval.append(auc)
                    acc_result.append(acc_eval)
                    auc_result.append(auc_eval)

                acc_result = np.array(acc_result)
                auc_result = np.array(auc_result)
                acc_mean = np.mean(acc_result, axis=1)
                auc_mean = np.mean(auc_result, axis=1)
                acc_std = np.std(acc_result, axis=1)
                auc_std = np.std(auc_result, axis=1)
                print('en_%s_%s_%s: ' % (d_s, t, d_t))
                for i,p in enumerate(tgtSize):
                    print('\ttgtSize-%d: %f/%f %f/%f' % (p, acc_mean[i], acc_std[i], auc_mean[i], auc_std[i]))
                return




if __name__ == '__main__':
    run_cls()
