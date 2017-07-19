import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA

plt.ion()

iris = datasets.load_iris()
d = iris['data']
target = iris['target']

# PCA

# Standardize the data
X = StandardScaler().fit_transform(d)

# Calculate the covariance matrix
R = np.cov(X.T)
# Calculate eigenvectors & eigenvalues of the covariance matrix
# the performance gain is substantial
evals, evecs = np.linalg.eigh(R)
# Sort eigenvalue in decreasing order
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]

# Explained Variance

total = sum(evals)
var_exp = [(i / total)*100 for i in sorted(evals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

ind = np.arange(len(var_exp))

plt.bar(ind, var_exp, align='center')
plt.plot(ind, cum_var_exp, '-o', color='red')

x = ['PC %s' %i for i in range(1,len(var_exp)+1)]
plt.xticks(ind, x) 
plt.title('Explained Variance by Principal Components (Iris Dataset)')
plt.ylabel('Explained Variance')
plt.xlim(-1, 4)
plt.grid()


# select the first n eigenvectors 
evecs = evecs[:, :3]
# carry out the transformation on the data using eigenvectors
# and return the re-scaled data, eigenvalues, and eigenvectors
np.dot(evecs.T, X.T).T, evals, evecs


# SVD
pca.explained_variance_ratio_
cumulated_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(cumulated_var)


def svd(data, target, S=2):

    #calculate SVD
    U, s, V = np.linalg.svd(data)
    Sig = np.mat(np.eye(S)*s[:S])
    #tak out columns you don't need
    newdata = U[:,:S]

    # this line is used to retrieve the dataset
    # new = U[:,:2]*Sig*V[:2,:]

    samples, features = np.shape(data)
    features += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['blue', 'red', 'black']
    for i in range(samples):
        ax.scatter(newdata[i,0],newdata[i,1], color= colors[int(target[i])])
    plt.xlabel('SVD1')
    plt.ylabel('SVD2')
    plt.show()


# Kernel PCA
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)

# next
