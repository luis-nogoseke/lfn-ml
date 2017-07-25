import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn import datasets
import timeit

plt.ion()

data = pd.read_csv('../clean2.data')
target = data.ix[:, -1]
d = data.ix[:, 2:168]

# PCA



def pca(matrix, plot=False, dataset_name=''):
    """ Calculate matrix PCA  """
	
    # Standardize the data
    matrix = StandardScaler().fit_transform(matrix.values)

    # Calculate the covariance matrix
    cov_m = np.cov(matrix.T)
    # Calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eigh(cov_m)
    # Sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    # Explained Variance
    total = sum(evals)
    var_exp = [(i / total)*100 for i in sorted(evals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    ind = np.arange(len(var_exp))
    if plot:
        plt.bar(ind, var_exp, align='center')
        plt.plot(ind, cum_var_exp, '-o', color='red')
        x = ['PC %s' %i for i in range(1, len(var_exp)+1)]
        plt.xticks(ind, x)
        plt.title('Explained Variance by Principal Components ({})'.format(dataset_name))
        plt.ylabel('Explained Variance')
        plt.xlim(-1, len(var_exp))
        plt.grid()

    # select the first n eigenvectors
    evecs = evecs[:, :3]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, X.T).T, evals, evecs


# SVD

def svd(data, target, S=2, plot=False):
    # Standardize the data
    data = StandardScaler().fit_transform(data.values)
    #calculate SVD
    U, s, V = np.linalg.svd(data)
    Sig = np.mat(np.eye(S)*s[:S])
    #tak out columns you don't need
    newdata = U[:,:S]

    # this line is used to retrieve the dataset
    # new = U[:,:2]*Sig*V[:2,:]

    samples, features = np.shape(data)
    features += 1
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = ['blue', 'red', 'black']
        for i in range(samples):
            ax.scatter(newdata[i, 0], newdata[i, 1], color=colors[int(target[i])])
        plt.xlabel('SVD1')
        plt.ylabel('SVD2')
        plt.show()


time = 0.
for i in range(500):
    start = timeit.timeit()
    pca(X)
    end = timeit.timeit()
    time += end - start
time / 500.


time = 0.
for i in range(500):
    start = timeit.timeit()
    svd(X, 1)
    end = timeit.timeit()    
    time += end - start
time / 500.

# Kernel PCA
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
time = 0.
for i in range(500):
    start = timeit.timeit()
    kpca.fit_transform(X)
    end = timeit.timeit()
    time += end - start
time / 500.




pipe_rf = Pipeline([('pca', PCA(n_components=0.9)), ('clf',DecisionTreeClassifier())])
scores = model_selection.cross_val_score(pipe_rf, d, target cv=10, scoring='accuracy')
scores.mean()



pipe_rf = Pipeline([('pca', KernelPCA(kernel="rbf")), ('clf',DecisionTreeClassifier())])
scores = model_selection.cross_val_score(pipe_rf, d, target, cv=10, scoring='accuracy')
scores.mean()


pipe_rf = Pipeline([('pca', IncrementalPCA()), ('clf',DecisionTreeClassifier())])
scores = model_selection.cross_val_score(pipe_rf, d, target, cv=10, scoring='accuracy')
scores.mean()

pipe_rf = Pipeline([('pca', SparsePCA()), ('clf',DecisionTreeClassifier())])
scores = model_selection.cross_val_score(pipe_rf, d, target, cv=10, scoring='accuracy')
scores.mean()
