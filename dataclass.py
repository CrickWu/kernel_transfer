#!/usr/bin/env python
# encoding: utf-8
# This is the class wrapper around dataloader.
import numpy as np
import scipy.sparse as sp
import os
import configure_path
import sklearn.datasets as sd
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

# wrapper for load_svmlight_file in dealing with empty files / no-existing files
def load_svmlight_file(f, n_features=None, dtype=np.float64, multilabel=False, zero_based='auto', query_id=False):
    if not os.path.exists(f) or (os.stat(f).st_size == 0):
        assert n_features != None
        if multilabel:
            y = [()]
        else:
            y = []
        return sp.csr_matrix((0, n_features)), y
    else:
        return sd.load_svmlight_file(f, n_features=n_features, dtype=dtype, multilabel=multilabel, zero_based=zero_based, query_id=query_id)

class DataClass:
    # attributes:

    # path: srcPath, tgtPath, prlPath
    #   default values are inherited from configure_path
    # feature dimension: source_n_features, target_n_features
    # kernel: source_gamma, target_gamma
    # kernel_type: 'rbf', 'cosine'
    # valid_flag: use .val.libsvm for grid search, or use .tst.libsvm for reporting final results
    # source_data_type:
    #   'full': use src.full.trn.libsvm as the training data in the source domain, leaving test and para data to be empty
    #   'normal': use both train, test, para data in the source domain
    #   'parallel': use only parallel in the source domain
    # zero_diag_flag: (True) zero-out the diagonal, (False) keep the diagonal to be 1s
    def __init__(self, srcPath=None, tgtPath=None, prlPath=None, valid_flag=True, zero_diag_flag=False, source_data_type='full',
                source_n_features=100000, target_n_features=200000, kernel_type='cosine',
                source_gamma=None, target_gamma=None):
        if srcPath == None:
            self.srcPath = configure_path.srcPath
        if tgtPath == None:
            self.tgtPath = configure_path.tgtPath
        if prlPath == None:
            self.prlPath = configure_path.prlPath
        self.source_n_features = source_n_features
        self.target_n_features = target_n_features
        self.kernel_type = kernel_type
        self.valid_flag = valid_flag
        self.zero_diag_flag = zero_diag_flag
        self.source_data_type = source_data_type
        if source_gamma == None:
            self.source_gamma = 1.0 / np.sqrt(source_n_features)
        if target_gamma == None:
            self.target_gamma = 1.0 / np.sqrt(target_n_features)
    def kernel(self, data, **parameters):
        if self.kernel_type == 'cosine':
            return cosine_similarity(data)
        elif self.kernel_type == 'rbf':
            return rbf_kernel(data, gamma=parameters['gamma'])

    # keep the `nn` nearest neighbor for each row
    # nn is # nearest neighbor to keep
    @staticmethod
    def sparsify_K(K, nn):
        ret_K = np.zeros(K.shape)
        for i in xrange(K.shape[0]):
            index = np.argsort(K[i, :])[-nn:]
            ret_K[i, index] = K[i, index]
        return ret_K

    # keep the `nn` nearest neighbor for each row, and make the kernel symmetrical by averaging through its transpose
    # nn is # nearest neighbor to keep
    @staticmethod
    def sym_sparsify_K(K, nn):
        K_sp = DataClass.sparsify_K(K, nn)
        K_sp = (K_sp+K_sp.T) / 2 # in case of non-positive semi-definite
        return K_sp

    # Load the data
    # return values:
    # y: (ns+nt): true_values
    # I: (ns+nt) observation indicator
    # K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
    # offset: [source_train , ... , target_para] offset number
    def get_TL_Kernel(self):
        if self.source_data_type == 'normal':
            source_train = self.srcPath + '.trn.libsvm'
            source_test = self.srcPath + '.val.libsvm' # val is for tuning hyperparameters, in final should report on tst
        elif self.source_data_type == 'full':
            # if srcPath ends with numbers, trail it
            fields = self.srcPath.split('.')
            if fields[-1].isdigit():
                srcPath = '.'.join(fields[:-1])
            else:
                srcPath = self.srcPath
            source_train = srcPath + '.full.trn.libsvm'
            source_test = srcPath + '.full.val.libsvm' # val is for tuning hyperparameters, in final should report on tst
        elif self.source_data_type == 'parallel':
            source_train = '/non/existent/file'
            source_test = '/non/existent/file'
        else:
            raise ValueError('Unknown source domain data option.')
        source_para = self.prlPath + 'prlSrc.libsvm'

        target_train = self.tgtPath + '.trn.libsvm'
        target_test = self.tgtPath + '.val.libsvm'
        target_para = self.prlPath + 'prlTgt.libsvm'
        if self.valid_flag == False:
            target_test = self.tgtPath + '.tst.libsvm'
        return self._get_TL_Kernel(source_train, source_test, source_para,
                                   target_train, target_test, target_para)

    def _get_TL_Kernel(self, source_train, source_test, source_para,
                             target_train, target_test, target_para):
        # source_domain, target_domain dimension should be fixed

        source_train_X, source_train_y = load_svmlight_file(source_train, n_features=self.source_n_features)
        source_test_X, source_test_y = load_svmlight_file(source_test, n_features=self.source_n_features)
        source_para_X, _ = load_svmlight_file(source_para, n_features=self.source_n_features, multilabel=True)

        target_train_X, target_train_y = load_svmlight_file(target_train, n_features=self.target_n_features)
        target_test_X, target_test_y = load_svmlight_file(target_test, n_features=self.target_n_features)
        target_para_X, _ = load_svmlight_file(target_para, n_features=self.target_n_features, multilabel=True)
        ##### default gamma value is taken to be sqrt of the data dimension
        ##### May need to tune and change the calculation of
        # if source_gamma == None:
        #     source_gamma = 1.0 / np.sqrt(source_train_X.shape[1])
        # if target_gamma == None:
        #     target_gamma = 1.0 / np.sqrt(target_train_X.shape[1])

        source_data = sp.vstack([source_train_X, source_test_X, source_para_X])
        # source_ker = rbf_kernel(source_data, gamma=self.source_gamma)
        source_ker = self.kernel(source_data, gamma=self.source_gamma)


        target_data = sp.vstack([target_train_X, target_test_X, target_para_X])
        # target_ker = rbf_kernel(target_data, gamma=self.target_gamma)
        target_ker = self.kernel(target_data, gamma=self.target_gamma)

        len_X = [source_train_X.shape[0] , source_test_X.shape[0] , source_para_X.shape[0]
                , target_train_X.shape[0] , target_test_X.shape[0] , target_para_X.shape[0]]
        offset = np.cumsum(len_X)

        # print 'offset\t', offset

        # K initialize
        n = offset[5]
        K = np.zeros([n, n])
        # source/target, kernel
        K[0:offset[2],0:offset[2]] = source_ker
        K[offset[2]:offset[5],offset[2]:offset[5]] = target_ker
        # parallel data
        K[offset[1]:offset[2],offset[4]:offset[5]] = np.ones([len_X[2], len_X[2]], dtype=np.float)
        K[offset[4]:offset[5],offset[1]:offset[2]] = np.ones([len_X[2], len_X[2]], dtype=np.float)
        if self.zero_diag_flag:
            np.fill_diagonal(K, 0.0)

        # observation Indicator
        I = np.zeros(n, dtype=np.float)
        I[0:offset[0]] = np.ones(len_X[0], dtype=np.float)
        I[offset[2]:offset[3]] = np.ones(len_X[3], dtype=np.float)

        # true values
        y = np.zeros(n, dtype=np.float)
        y[0:offset[0]] = source_train_y
        y[offset[0]:offset[1]] = source_test_y
        y[offset[2]:offset[3]] = target_train_y
        y[offset[3]:offset[4]] = target_test_y
        y = y.astype(np.int)

        return y, I, K, offset

    # Load the target [train,test] data (no parallel)
    # return values:
    # y: (ns+nt): true_values
    # I: (ns+nt) observation indicator
    # K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
    # offset: [train, test] offset
    def get_SSL_Kernel(self):
        target_train = self.tgtPath + '.trn.libsvm'
        target_test = self.tgtPath + '.val.libsvm'
        if self.valid_flag == False:
            target_test = self.tgtPath + '.tst.libsvm'

        target_train_X, target_train_y = load_svmlight_file(target_train, n_features=self.target_n_features)
        target_test_X, target_test_y = load_svmlight_file(target_test, n_features=self.target_n_features)

        # print(type(target_train_X), type(target_train_y), type(target_test_X))
        len_X = [target_train_X.shape[0] , target_test_X.shape[0]]
        offset = np.cumsum(len_X)

        n = offset[1]

        target_data = sp.vstack([target_train_X, target_test_X])
        # target_ker = rbf_kernel(target_data, gamma=self.target_gamma)
        target_ker = self.kernel(target_data, gamma=self.target_gamma)

        y = np.zeros(n, dtype=np.float)
        y[0:offset[0]] = target_train_y
        y[offset[0]:offset[1]] = target_test_y
        y = y.astype(np.int)

        I = np.zeros(n, dtype=np.float)
        I[0:offset[0]] = np.ones(len_X[0], dtype=np.float)

        K = target_ker
        if self.zero_diag_flag:
            np.fill_diagonal(K, 0.0)
        return y, I, K, offset

    @staticmethod
    def complete_TL_Kernel(K_ori, offset):
        # source_train, source_test, source_para
        #            0,           1,           2
        # target_train, target_test, target_para
        #            3,           4,           5
        K = K_ori.copy()
        # complete target parallel
        K[offset[4]:offset[5], 0:offset[1]] = K[offset[1]: offset[2], 0:offset[1]]
        K[offset[2]:offset[4], offset[1]:offset[2]] = K[offset[2]: offset[4], offset[4]:offset[5]]
        K[offset[4]:offset[5], offset[1]:offset[2]] = (K[offset[1]: offset[2], offset[1]:offset[2]] +  K[offset[4]: offset[5], offset[4]:offset[5]]) / 2
        # zero_based should impose 1 to parallel diagonal elements
        xcoos = xrange(offset[4], offset[5])
        ycoos = xrange(offset[1], offset[2])
        K[xcoos, ycoos] = 1.0

        K[0:offset[2], offset[2]:offset[5]] = K[offset[2]:offset[5], 0:offset[2]].transpose()
        return K


    def get_TL_Kernel_completeOffDiag(self):
        y, I, K, offset = self.load_data()
        K = DataClass.complete_TL_Kernel(K, offset)
        return y, I, K, offset

# for testing
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
