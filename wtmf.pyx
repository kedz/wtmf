cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse
import scipy.io
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

ctypedef np.double_t DTYPE_t

class WTMFVectorizer(object):
    def __init__(self, input='file', k=100, w_m=0.01,
                 lam=20, max_iter=20, tokenizer=None, 
                 tf_threshold=3, verbose=False):
        self.input = input
        self.k = k
        self.w_m = w_m
        self.lam = lam
        self.max_iter = max_iter
        if tokenizer is None:
            self.tokenize = _default_tokenizer_
        else:
            self.tokenize = tokenizer
        self.tf_threshold = tf_threshold
        self.verbose = verbose

        self.dict_vectorizer_ = None
        self.tfidf_trans_ = None
        self.wtmf_trans_ = None

    def fit(self, X):
        if self.input == u'file':
            X = self.load_file_(X, self.tf_threshold, True)
        elif self.input == u'content':
            X = self.load_content_(X, self.tf_threshold, True)
        
        X = self.init_tfidf_transform_(X)
        self.wtmf_trans_ = \
            WTMFTransformer(
                k=self.k, w_m=self.w_m, lam=self.lam, 
                max_iter=self.max_iter, verbose=self.verbose).fit(X)
        return self

    def transform(self, X):
        if self.input == u'file':
            X = self.load_file_(X, self.tf_threshold, False)
        elif self.input == u'content':
            X = self.load_content_(X, self.tf_threshold, False)
        X = self.dict_vectorizer_.transform(X)
        X = self.tfidf_trans_.transform(X)
        X = self.wtmf_trans_.transform(X)
        return X

    def load_file_(self, f, tf_threshold, fit=False):
        filter_terms = fit and tf_threshold > 0
        X = []
        if filter_terms:
            global_tf = dict()
        for line in f:
            counts = {}
            for token in self.tokenize(line.strip()):
                counts[token] = counts.get(token, 0) + 1
                if filter_terms:
                    global_tf[token] = global_tf.get(token, 0) + 1
            X.append(counts)
        if filter_terms:
            for x in X:
                for feature, value in x.items():
                    if value < tf_threshold:
                        del x[feature]
        return X

    def load_content_(self, content, tf_threshold, fit=False):
        filter_terms = fit and tf_threshold > 0
        X = []
        if filter_terms:
            global_tf = dict()
        for line in content:
            counts = {}
            for token in self.tokenize(line.strip()):
                counts[token] = counts.get(token, 0) + 1
                if filter_terms:
                    global_tf[token] = global_tf.get(token, 0) + 1
            X.append(counts)
        if filter_terms:
            for x in X:
                for feature, value in x.items():
                    if value < tf_threshold:
                        del x[feature]
        return X

    def init_tfidf_transform_(self, X):
        self.dict_vectorizer_ = DictVectorizer(dtype=np.uint)
        self.tfidf_trans_ = TfidfTransformer(norm=None)

        X = self.dict_vectorizer_.fit_transform(X)
        X = self.tfidf_trans_.fit_transform(X)

        return X
        
def _default_tokenizer_(string):
    return string.split(' ')


cdef class WTMFTransformer(object):
    cdef int k, max_iter;
    cdef double w_m;
    cdef double lam;
    cdef np.ndarray P_;
    cdef object verbose

    def __init__(self, int k=100, double w_m=0.01,
                 double lam=0.1, int max_iter=20, object verbose=False):
        self.k = k
        self.w_m = w_m
        self.lam = lam
        self.max_iter = max_iter
        self.verbose = verbose


    def __reduce__(self):
        return (rebuild, (self.k, self.w_m, self.lam, self.max_iter, self.verbose, self.P_))

    def fit(self, X):
        self.P_ = self.als_grad_descent(
            X, self.k, self.w_m, self.lam, self.max_iter, 
            verbose=self.verbose)
        return self        

    def transform(self, X):
        return self.project_Q(X, self.P_, self.k, self.w_m, self.lam)


    @cython.boundscheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] project_Q(self, object X, np.ndarray[DTYPE_t, ndim=2] P,
                          int k, double w_m, double lam):
        cdef int i, j, start, stop
        cdef int n_docs, n_words
        cdef double w_w = 1.0 - w_m
        cdef np.ndarray[DTYPE_t, ndim=2] Q, pv, pptw, a
        cdef np.ndarray[DTYPE_t, ndim=1] b
        cdef np.ndarray[DTYPE_t, ndim=2] lam_Ik = lam * np.eye(k)


        n_docs, n_words = X.shape
        Q = np.zeros((n_docs, k), dtype=np.double)

        # Initialize column and row sparse views
        Xcsr = scipy.sparse.csr_matrix(X)
        i4d = []
        for i in xrange(len(Xcsr.indptr) - 1):
            start = Xcsr.indptr[i]
            stop = Xcsr.indptr[i + 1]
            i4d.append((Xcsr.indices[start:stop], Xcsr.data[start:stop]))

        pptw = np.dot(P, P.T) * w_m            
        for j in xrange(n_docs):
            pv = P[:,i4d[j][0]]
            a = pptw + np.dot(pv, pv.T) * w_w + lam_Ik
            b = np.dot(pv, i4d[j][1])
            Q[j,:] = np.linalg.solve(a, b)
        return Q


    @cython.boundscheck(False)
    cdef np.ndarray als_grad_descent(self, object X, int k, double w_m, 
                          double lam, int max_iter, bint verbose=False):

        cdef int i, j, n_iter, start, stop
        cdef int n_docs, n_words
        cdef double w_w = 1.0 - w_m
        cdef np.ndarray[DTYPE_t, ndim=2] P, Q, pv, qv, pptw, qqtw, a, 
        cdef np.ndarray[DTYPE_t, ndim=1] b
        cdef np.ndarray[DTYPE_t, ndim=2] lam_Ik = lam * np.eye(k)


        #n_words, n_docs = X.shape
        n_docs, n_words = X.shape
        
        
        if verbose is True:
            import sys
            t_start = time.time()
            print "n_docs = {}, n_words = {}".format(n_docs, n_words)

        # Initialize column and row sparse views
        Xcsr = scipy.sparse.csr_matrix(X)
        i4d = []
        for i in xrange(len(Xcsr.indptr) - 1):
            start = Xcsr.indptr[i]
            stop = Xcsr.indptr[i + 1]
            i4d.append((Xcsr.indices[start:stop], Xcsr.data[start:stop]))

        Xcsc = scipy.sparse.csc_matrix(X)
        i4w = []
        for i in xrange(len(Xcsc.indptr) - 1):
            start = Xcsc.indptr[i]
            stop = Xcsc.indptr[i + 1]
            i4w.append((Xcsc.indices[start:stop], Xcsc.data[start:stop]))

        P = np.random.randn(k, n_words)
        Q = np.zeros((k, n_docs))

        for n_iter in xrange(max_iter):
            if verbose is True:
                sys.stdout.write(
                    "Iter {}, solving Q".format(n_iter + 1))
                sys.stdout.flush()

            pptw = np.dot(P, P.T) * w_m            
            for j in xrange(n_docs):
                pv = P[:,i4d[j][0]]
                a = pptw + np.dot(pv, pv.T) * w_w + lam_Ik
                b = np.dot(pv, i4d[j][1])
                Q[:,j] = np.linalg.solve(a, b)

            if verbose is True:
                sys.stdout.write(
                    " P\n")
                sys.stdout.flush()

            qqtw = np.dot(Q, Q.T) * w_m
            for i in xrange(n_words):
                qv = Q[:,i4w[i][0]]
                a = qqtw + np.dot(qv, qv.T) * w_w + lam_Ik
                b = np.dot(qv, i4w[i][1])
                P[:,i] = np.linalg.solve(a, b)

        if verbose is True:
            t_end = time.time()
            print "Used {} seconds".format(t_end - t_start)

        return P

#standalone function
def rebuild(k, w_m, lam, max_iter, verbose, P):
    trans = WTMFTransformer(k=k, w_m=w_m, lam=lam, max_iter=max_iter, verbose=verbose)
    trans.P_ = P
    return trans
