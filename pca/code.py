import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.ion()

data = pd.read_csv('../wine.data', names=['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'phenols', 'Flavanoids', 'Nonflavanoid', 'Proanthocyanins', 'Color', 'Hue', 'OD280_OD315', 'Proline'])

d = data.ix[:, data.columns != 'Alcohol']

# PCA

# Standardize the data
X = StandardScaler().fit_transform(d.values)

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
plt.plot(ind, cum_var_exp, '-o')

x=['PC %s' %i for i in range(1,len(var_exp)+1)]
plt.xticks(ind, x) 
plt.title('Explained Variance by Principal Components')
plt.ylabel('Explained Variance')
plt.xlim(-1, 12)


# select the first n eigenvectors 
evecs = evecs[:, :3]
# carry out the transformation on the data using eigenvectors
# and return the re-scaled data, eigenvalues, and eigenvectors
np.dot(evecs.T, X.T).T, evals, evecs


# SVD
pca.explained_variance_ratio_
cumulated_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(cumulated_var)

