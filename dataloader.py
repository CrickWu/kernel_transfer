#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import rbf_kernel
from configure_path import *

# Load the data
# return values:
# y: (ns+nt): true_values
# I: (ns+nt) observation indicator
# K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
def load_data(source_train, source_test, source_para,
              target_train, target_test, target_para):
    source_n_features = 100000
    target_n_features = 200000

    # source_domain, target_domain dimension should be fixed

    source_train_X, source_train_y = load_svmlight_file(source_train, n_features=source_n_features)
    source_test_X, _ = load_svmlight_file(source_test, n_features=source_n_features)
    source_para_X, _ = load_svmlight_file(source_para, n_features=source_n_features, multilabel=True)

    target_train_X, target_train_y = load_svmlight_file(target_train, n_features=target_n_features)
    target_test_X, _ = load_svmlight_file(target_test, n_features=target_n_features)
    target_para_X, _ = load_svmlight_file(target_para, n_features=target_n_features, multilabel=True)

    source_gamma = 1.0 / np.sqrt(source_train_X.shape[1])
    target_gamma = 1.0 / np.sqrt(target_train_X.shape[1])

    source_data = sp.vstack([source_train_X, source_test_X, source_para_X])
    source_ker = rbf_kernel(source_data, gamma=source_gamma)
    # source_ker = rbf_kernel(source_data)

    target_data = sp.vstack([target_train_X, target_test_X, target_para_X])
    target_ker = rbf_kernel(target_data, gamma=target_gamma)
    # target_ker = rbf_kernel(target_data)

    len_X = [source_train_X.shape[0] , source_test_X.shape[0] , source_para_X.shape[0]
            , target_train_X.shape[0] , target_test_X.shape[0] , target_para_X.shape[0]]
    offset = np.cumsum(len_X)
    print offset
    print len_X

    # K initialize
    n = offset[5]
    K = np.zeros([n, n])
    # source/target, kernel
    K[0:offset[2],0:offset[2]] = source_ker
    K[offset[2]:offset[5],offset[2]:offset[5]] = target_ker
    # parallel data
    K[offset[1]:offset[2],offset[4]:offset[5]] = np.ones([len_X[2], len_X[2]], dtype=np.float)
    K[offset[4]:offset[5],offset[1]:offset[2]] = np.ones([len_X[2], len_X[2]], dtype=np.float)

    # observation Indicator
    I = np.zeros(n, dtype=np.float)
    I[0:offset[0]] = np.ones(len_X[0], dtype=np.float)
    I[offset[2]:offset[3]] = np.ones(len_X[3], dtype=np.float)

    # true values
    y = np.zeros(n, dtype=np.float)
    y[0:offset[0]] = source_train_y
    y[offset[2]:offset[3]] = target_train_y

    return y, I, K

def default_y_I_K():
    source_train = srcPath + '.trn.libsvm'
    source_test = srcPath + '.tst.libsvm'
    source_para = prlPath + 'src.toy.libsvm'

    target_train = tgtPath + '.trn.libsvm'
    target_test = tgtPath + '.tst.libsvm'
    target_para = prlPath + 'tgt.toy.libsvm'

    y, I, K = load_data(source_train, source_test, source_para,
                  target_train, target_test, target_para)
    return y, I, K
if __name__ == '__main__':


    source_train = srcPath + '.trn.libsvm'
    source_test = srcPath + '.tst.libsvm'
    source_para = prlPath + 'src.toy.libsvm'

    target_train = tgtPath + '.trn.libsvm'
    target_test = tgtPath + '.tst.libsvm'
    target_para = prlPath + 'tgt.toy.libsvm'

    y, I, K = load_data(source_train, source_test, source_para,
                  target_train, target_test, target_para)
    print K[1,2], K[1000, 999], K[1891,K.shape[1]-1]
    print sum(I)
    print sum(y)
