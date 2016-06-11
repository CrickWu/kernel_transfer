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
    return y_true, y_prob, f

# return obj, grad of logistic loss
def logistic(f, y, I):
    inter_exp = np.exp(- f * y)
    # calculate active set and objective value
    J = I
    grad = -y * inter_exp / (1 + inter_exp) * J / np.log(2)
    obj = np.sum(np.log2(1 + inter_exp) * J)
    return obj, grad

# return obj, grad of hinge loss
def hinge(f, y, I):
    dv = 1 - f * y
    # calculate active set and objective value
    J = np.asarray((I > 0) * (dv > 0), dtype=np.int)
    grad = -y * J
    obj = np.sum(np.maximum(dv * J, 0.0))
    return obj, grad

# return obj, grad of l2 loss
def l2(f, y, I):
    dv = 1 - f * y
    # calculate active set and objective value
    J = np.asarray((I > 0) * (dv > 0), dtype=np.int)
    grad = 2 * (f - y) * J # should multiply by 2
    obj = np.sum(dv ** 2 * J)
    return obj, grad

def gen_weight_lap(K, offset):
    n = K.shape[0]
    W_ss = np.zeros((n,n))
    W_st = np.zeros((n,n))
    W_ts = np.zeros((n,n))
    W_tt = np.zeros((n,n))
    W_ss[:offset[2], :offset[2]] = K[:offset[2], :offset[2]]
    W_st[:offset[2], offset[2]:] = K[:offset[2], offset[2]:]
    W_ts[offset[2]:, :offset[2]] = K[offset[2]:, :offset[2]]
    W_tt[offset[2]:, offset[2]:] = K[offset[2]:, offset[2]:]
    D_ss = np.diag( np.sum(W_ss, axis=1) )
    D_st = np.diag( np.sum(W_st, axis=1) )
    D_ts = np.diag( np.sum(W_ts, axis=1) )
    D_tt = np.diag( np.sum(W_tt, axis=1) )

    alpha = 0.5; beta = 2; gamma = 1;
    lap = np.zeros((n,n))
    lap = alpha*(D_ss-W_ss) + beta*(D_st-W_st) + beta*(D_ts-W_ts) + gamma*(D_tt-W_tt)

    #v, Q = scipy.linalg.eigh(lap)
    #if np.all(v > -1e-10):
    #    raise Warning('This lapcian is not PSD...')
    return lap

# I use graident descent at this point
def sgd_solver(y, I, K, offset, gamma=2**(-10), nr_epoch=100, stepsize=2**(-1), loss='l2'):
    n = y.shape[0]
    batch_size = n
    nr_batch = (n + batch_size - 1) // batch_size
    D = np.diag( np.sum(K, axis=1) )
    lap = D - K
    lap = gen_weight_lap(K, offset)
    f = np.random.rand(n) * 2 - 1
    # f = np.ones(n)
    hist_grad = 0.0

    for eidx in range(nr_epoch):
        idxList = np.arange(nr_batch)
        np.random.shuffle(idxList)
        for idx in idxList:
            start, end = idx*batch_size, (idx+1)*batch_size if idx < nr_batch else n
            tmp_I = np.zeros(n)
            tmp_I [start : end] = 1
            tmp_I = tmp_I * I

            if loss == 'l2':
                lobj, lgrad = l2(f, y, tmp_I)
            elif loss == 'hinge':
                lobj, lgrad = hinge(f, y, tmp_I)
            elif loss == 'logistic':
                lobj, lgrad = logistic(f, y, tmp_I)
            else:
                raise Exception('unknown loss function')
                sys.exit(-1)

            reg = np.dot(f, np.dot(lap, f)) * gamma
            obj = reg + lobj
            grad = 2*gamma*np.dot(lap, f) + lgrad

            gnorm = np.sqrt(np.sum(grad**2))
            hist_grad += grad**2
            grad = grad / (1e-6 + np.sqrt(hist_grad))
            f = f - stepsize*grad
            sys.stderr.write('epoch %d reg %f loss %f obj %f gnorm %f\n' % (eidx, reg, lobj, obj, gnorm))
        if gnorm < 1e-3:
            break

    # for calculating ap
    start_offset = offset[3]
    end_offset = offset[4]

    y_true = y[start_offset:end_offset]
    y_prob = f[start_offset:end_offset]
    return y_true, y_prob, f


def normalize_K(K):
    tmp = np.sum(K, axis=1)
    inv_sqrt_row_sum = np.diag( 1.0 / np.sqrt(tmp))
    tmp = np.dot(K, inv_sqrt_row_sum)
    tmp = np.dot(inv_sqrt_row_sum, tmp)
    assert(np.isnan(tmp).any() == False)
    return tmp

def eigen_decompose(K, offset, max_k=None):
    W_s = K[:offset[2], :offset[2]]
    W_s = normalize_K(W_s)
    W_t = K[offset[2]:, offset[2]:]
    W_t = normalize_K(W_t)
    if max_k == None:
        v_s, Q_s = scipy.linalg.eigh(W_s)
        v_t, Q_t = scipy.linalg.eigh(W_t)
    else:
        v_s, Q_s = sp.linalg.eigsh(W_s, k=max_k)
        v_t, Q_t = sp.linalg.eigsh(W_t, k=max_k)

    return v_s, Q_s, v_t, Q_t


def get_K_exp_by_eigen(K, offset, v_s, Q_s, v_t, Q_t):
    K_exp = K.copy()
    Y_st = K_exp[:offset[2], offset[2]:]
    alpha_s = 2**(-4); alpha_t = 2**(-4)
    Lambda_s = np.diag(np.exp(alpha_s*v_s))
    Lambda_t = np.diag(np.exp(alpha_t*v_t))
    K_ss = Q_s.dot(Lambda_s.dot(Q_s.T)) #+ np.identity(Q_s.shape[0]) - Q_s.dot(Q_s.T)
    K_tt = Q_t.dot(Lambda_t.dot(Q_t.T)) #+ np.identity(Q_t.shape[0]) - Q_t.dot(Q_t.T)
    K_st = K_ss.dot(Y_st.dot(K_tt))
    K_st[K_st < 0] = 0
    K_st = normalize(K_st, norm='l2', axis=1)
    K_exp[:offset[2], offset[2]:] = K_st
    K_exp[offset[2]:, :offset[2]] = K_st.T
    return K_exp


def debug(K, offset, y, I, f):
    D = np.diag( np.sum(K, axis=1) )
    L = D - K
    reg = f.dot(L.dot(f))
    L_ss = L[:offset[2], :offset[2]]
    L_st = L[:offset[2], offset[2]:]
    L_tt = L[offset[2]:, offset[2]:]
    f_s = f[:offset[2]]
    f_t = f[offset[2]:]
    reg_ss = f_s.dot(L_ss.dot(f_s))
    reg_st = f_s.dot(L_st.dot(f_t))
    reg_tt = f_t.dot(L_tt.dot(f_t))
    print('calculate from L, reg %f: reg_ss %f reg_st %f reg_tt %f' % (reg, reg_ss, reg_st, reg_tt))

    reg = 0.0
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            reg += K[i,j]*((f[i] - f[j])**2)
    reg /= 2.0

    reg_ss = 0.0
    for i in range(offset[2]):
        for j in range(offset[2]):
            reg_ss += K[i,j]*((f[i] - f[j])**2)
    reg_ss /= 2.0

    reg_st = 0.0
    for i in range(offset[2]):
        for j in range(offset[2], offset[-1]):
            reg_st += K[i,j]*((f[i] - f[j])**2)
    reg_st /= 2.0

    reg_tt = 0.0
    for i in range(offset[2], offset[-1]):
        for j in range(offset[2], offset[-1]):
            reg_tt += K[i,j]*((f[i] - f[j])**2)
    reg_tt /= 2.0
    print('calculate from W, reg %f: reg_ss %f reg_st %f reg_tt %f' % (reg, reg_ss, reg_st, reg_tt))
    #return reg, reg_ss, reg_st, reg_tt


def run_one(srcPath=None, tgtPath=None, prlPath=None,
            source_n_features=None, target_n_features=None):

    dc = DataClass(srcPath=srcPath, tgtPath=tgtPath, prlPath=prlPath,
                   valid_flag=False, zero_diag_flag=False, source_data_type='full',
                   source_n_features=source_n_features, target_n_features=target_n_features,
                   kernel_type='cosine', kernel_normal=False)
    y, I, K, offset = dc.get_TL_Kernel()
    # make sure diagonal is 1 so that
    # after normalizing lapliacian,
    # nan does not happen
    np.fill_diagonal(K, 1)

    # run eigen decomposition on K
    v_s, Q_s, v_t, Q_t = eigen_decompose(K, offset, max_k=128)
    K_exp = get_K_exp_by_eigen(K, offset, v_s, Q_s, v_t, Q_t)
    #y_true, y_prob, f = cvxopt_solver(y, I, K_exp, offset, 2**(-10))
    #y_true, y_prob, f = sgd_solver(y, I, K_exp, offset, nr_epoch=1000, stepsize=2**(-1), loss='l2')
    y_true, y_prob, f = sgd_solver(y, I, K_exp, offset, gamma=2**(-10), nr_epoch=1000, stepsize=2**(-1))
    auc, acc = eval_binary(y_true, y_prob)
    return auc, acc


def run_cls():
    dataPath = '/usr0/home/wchang2/research/NIPS2016/data/cls/'
    #dataPath = '/tmp2/b99902019/AAAI16/data/cls/'
    tgt = ['de', 'fr', 'jp']
    tgtSize = [2, 4, 8, 16, 32]
    domain = ['books', 'dvd', 'music']
    dimDict = {'en':60244, 'de':185922, 'fr':59906, 'jp':75809}
    nr_seed = 3
    ## test code
    #dataPath = '/Users/crickwu/Work/Research Transfer Learning/code/'
    tgt = ['de', 'fr', 'jp']
    tgtSize= [2, 8]
    nr_seed = 10
    ## test code
    for d_s in domain:
        for t in tgt:
            for d_t in domain:
                if d_s == d_t:
                    continue
                ## test code
                if not (d_s == 'books' and d_t == 'music' and t == 'jp'):
                # if not (d_s == 'books' and d_t == 'dvd' and t == 'de'):
                # if not (d_s == 'books' and d_t == 'music' and t == 'jp')
                # or not (d_s == 'books' and d_t == 'dvd' and t == 'de'):
                    continue
                ## test code
                acc_result= []
                auc_result = []
                for p in tgtSize:
                    acc_eval = []
                    auc_eval = []
                    for seed in xrange(0, nr_seed):
                        dirPath = dataPath + 'cls_seed_%d/en_%s_%s_%s/' % (seed, d_s, t, d_t)
                        assert(os.path.isdir(dirPath))
                        # srcPath = dirPath + 'src.1024'
                        srcPath = dirPath + 'src'
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
