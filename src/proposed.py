from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import random
import math
import os
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import time
from scipy.sparse import dok_matrix, csc_matrix, csr_matrix, hstack, vstack, dia_matrix
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from scipy.linalg import qr, clarkson_woodruff_transform
from scipy.sparse import diags, load_npz
from scipy.sparse.linalg import eigsh #Find k eigenvalues and eigenvectors of the real symmetric square matrix or complex Hermitian matrix A
from scipy.sparse.linalg import eigs  #Find k eigenvalues and eigenvectors of the square matrix A
from sklearn.utils.extmath import randomized_svd
from scipy.special import softmax
from sklearn.utils import check_random_state, as_float_array

# https://scikit-learn.org/stable/modules/clustering.html
from sklearn.cluster import KMeans, SpectralClustering

from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering

from sklearn.cluster._spectral import discretize

from scipy.sparse.linalg import inv

from scipy.linalg import qr, svd
from scipy.sparse.linalg import svds
from scipy.stats import chi

from functools import wraps

import warnings
warnings.filterwarnings("ignore")

np.get_include()

#os.environ['OPENBLAS_NUM_THREADS'] = '5'

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def load_attr_matrix(data, node):
    if node == 'U':
        filepath = '../datasets/'+data+'/Attribute_U.npz'
    else:
        filepath = '../datasets/'+data+'/Attribute_V.npz'
    print('loading '+filepath)
    X = load_npz(filepath)
    print("Data size:", X.shape)

    return X


def load_matrix(data, node):
    filepath = '../datasets/'+data+'/Edge.txt'
    print('loading '+filepath)

    if data in ['MovieLens', "Google", "Amazon"]:
        IJV = np.fromfile(filepath,sep="\t").reshape(-1,3)
        data = IJV[:,2]
    else:
        IJV = np.fromfile(filepath,sep="\t").reshape(-1,2)
        data = [1]*len(IJV[:,0].astype(np.int))

    if node == 'U':
        row = IJV[:,0].astype(np.int)
        col = IJV[:,1].astype(np.int)
    else:
        row = IJV[:,1].astype(np.int)
        col = IJV[:,0].astype(np.int)

    X = csr_matrix( (data,(row,col)) )
    print("Data size:", X.shape)

    return X

def output(args, labels):
    if not os.path.exists('../cluster/'+args.data+'/'):
        os.makedirs('../cluster/'+args.data+'/')

    with open('../cluster/'+args.data+'/'+args.algo+'.txt', 'w') as f:
        for i in range(len(labels)):
            f.write(str(i)+'\t'+str(labels[i])+'\n')

def read_stat(args):
    filepath = '../datasets/'+args.data+'/stat.txt'
    with open(filepath, 'r') as f:
        line = f.readline()
        nu, nv, m, du, dv, lu, lv = line.split('\t')

        if args.node=='U':
            k = int(lu)
        else:
            k = int(lv)

    return k

@timeit
def orthNMF(X, F, G, T=10):
    print("orthogonal non-negative matrix factorization")
    # https://ranger.uta.edu/~chqding/papers/orthNMF.pdf
    for t in range(T):
        XTF = X.T.dot(F)
        GFTF = G.dot( F.T.dot(F) )
        GFTF[GFTF.nonzero()] = 1.0 / GFTF[GFTF.nonzero()]
        XTF = XTF.multiply(GFTF)
        XTF[XTF<0]=0
        G = G.multiply(XTF)

        XG = X.dot(G)
        FFTXG = F.dot( F.T.dot(XG) )
        FFTXG[FFTXG.nonzero()] = 1.0/FFTXG[FFTXG.nonzero()]
        XG = XG.multiply(FFTXG)
        XG[XG<0]=0
        XG.data = np.sqrt(XG.data)
        F = F.multiply(XG)

    return F

# https://neonnnnn.github.io/pyrfm/_modules/pyrfm/random_feature/orthogonal_random_feature.html#OrthogonalRandomFeature
@timeit
def approxGK(X):
    print("approximate Gaussian kernel")
    d = X.shape[1]
    W = np.random.standard_normal(size=(d, d))
    W, _ = qr(W, mode='economic')
    # S = np.diag(chi.rvs(df=d, size=d, random_state=1024))*1.0/d
    # W = S.dot(W)
    X = X.dot(W.T)
    X_cos = np.cos(X)
    X_sin = np.sin(X)

    X = csr_matrix(np.hstack([X_sin,X_cos])*math.sqrt(math.exp(1)/1.0*d))

    xsum = X.sum(axis=0)
    deg = 1.0/np.sqrt(X.dot(xsum.T))
    X = csr_matrix(X.multiply(deg))

    return X

@timeit
def getL(W):
    c = np.array(np.sqrt(W.sum(axis=0)))#.flatten().tolist()
    c[c==0]=1
    c = 1.0/c
    c = c.flatten().tolist()
    c = diags(c)

    r = np.array(np.sqrt(W.sum(axis=1)))#.flatten().tolist()
    r[r==0]=1
    r_wo_inv = diags(r.flatten().tolist())
    r = 1.0/r
    r = diags(r.flatten().tolist())

    return csr_matrix(r.dot(W.dot(c))), r_wo_inv

@timeit
def trunc_propagate(B, X, alpha, T=5):
    print("feature propagation")
    Xt = X.copy()
    for i in range(T):
        Xt = X + alpha*B.dot(B.T.dot(Xt))

    X = preprocessing.normalize(Xt, norm='l2', axis=1)

    return X

@timeit
def SVD_init(X, k):
    print("SVD initialization")
    U, s, V = randomized_svd(X, n_components=k)
    U = csr_matrix(U)

    V = csr_matrix(V).T.dot(diags(np.array(s)))

    return U, V

def NCI_gen(vectors, T=100):
    vectors = as_float_array(vectors)
    n_samples = vectors.shape[0]
    n_feats = vectors.shape[1]

    labels = vectors.argmax(axis=1)
    print(type(labels), labels.shape)
    vectors_discrete = csc_matrix(
            (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
            shape=(n_samples, n_feats))

    vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
    vectors_sum[vectors_sum==0]=1
    vectors_discrete = vectors_discrete*1.0/vectors_sum
    #vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)

    for _ in range(T):
        Q = vectors.T.dot(vectors_discrete)

        vectors_discrete = vectors.dot(Q)
        vectors_discrete = as_float_array(vectors_discrete)

        labels = vectors_discrete.argmax(axis=1)
        vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_feats))

        
        #vectors_discrete = preprocessing.normalize(vectors_discrete, norm='l2', axis=0)

        vectors_sum = np.sqrt(vectors_discrete.sum(axis=0))
        vectors_sum[vectors_sum==0]=1
        vectors_discrete = vectors_discrete*1.0/vectors_sum

    return labels

@timeit
def TPC(B, X, alpha, dim, k, gamma, tf, tg):
    B, r = getL(B)

    if dim>0 and dim<X.shape[1]:
        print("dimensionality reduction")
        X, s, _ = randomized_svd(X, n_components=dim)
        X = csr_matrix(X).dot(diags(np.array(s)))

    X = trunc_propagate(B, X, alpha, gamma)

    X = approxGK(X)

    U, V = SVD_init(X, k)

    U = orthNMF(X, U, V, tf)

    U = U.todense()
    labels = NCI_gen(U, tg)
    print(labels)

    return labels


def run(args):
    B = load_matrix(args.data, args.node)
    X = load_attr_matrix(args.data, args.node)
    
    start = time.time()
    labels = TPC(B, X, args.alpha, args.d, args.k, args.gamma, args.tf, args.tg)
    elapsedTime = time.time()-start
    print("Elapsed time (secs) for %s clustering: %f"%(args.algo, elapsedTime))

    output(args, labels)

def main():
    parser = ArgumentParser("Our",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', default='default',
                        help='data name.')

    parser.add_argument('--algo', default='TPC',
                        help='method name.')

    parser.add_argument('--node', default='U',
                        help='data name')

    parser.add_argument('--k', default=0, type=int, help='#cluster')
    parser.add_argument('--d', default=-1, type=int, help='attribute dimension')
    parser.add_argument('--gamma', default=10, type=int, help='#iterations for MSA')
    parser.add_argument('--tf', default=5, type=int, help='#iterations for orthogonal NMF')
    parser.add_argument('--tg', default=20, type=int, help='#iterations for NCI generation')
    parser.add_argument('--alpha', default=0.9, type=float, help='decay factor')

    args = parser.parse_args()
    if args.data in ["Google", "Amazon"]:
        args.node = 'V'

    args.k = read_stat(args)
    print(args)

    print("data=%s, #clusters=%d"%(args.data, args.k))
    
    run(args)

if __name__ == "__main__":
    sys.exit(main())
