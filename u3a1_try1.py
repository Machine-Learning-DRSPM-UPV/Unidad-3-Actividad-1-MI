import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_regions import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


#df_rune = pd.read_csv('D.csv', header=None)   # 
#df_rune.info()
ceros = np.zeros((1544,1))
unos = np.ones((1435, 1))
dos = 2*np.ones((1612, 1))
df_rune0 = np.load('dumped/X0.dat')
df_rune0 = np.concatenate((ceros, df_rune0), 1)
df_rune1 = np.load('dumped/X1.dat')
df_rune1 = np.concatenate((unos, df_rune1), 1)
df_rune2 = np.load('dumped/X2.dat')
df_rune2 = np.concatenate((dos, df_rune2), 1)

df_rune = np.vstack((df_rune0, df_rune1, df_rune2))
print(df_rune)
#df_rune = pd.DataFrame(df_rune)#,index=df_rune[:,0])
print(df_rune)

# separate training and test data (70, 30)
#X, y = df_rune.iloc[:, 1:].values, df_rune.iloc[:, 0].values
X, y = df_rune[:, 1:], df_rune[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#print('\nEigenvalues {}'.format(eigen_vals))


# with cumsum we can calculate the cumulative sum of expained variances
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs

# sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#print(eigen_pairs)

feat = 200

# we created a 625 x 100-dimensional projection matrix W from top two eigenvectors.
w = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(feat)])
#w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
#print('Matrix W:\n', w.shape)

X_train_pca = X_train_std.dot(w)
#print(X_train_pca)

# let's visualize the transformed rune training set.
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']


# using PCA from sckit learn
pca = PCA(n_components=feat)
lr = LogisticRegression()
lr1 = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr1.fit(X_train, y_train)
lr.fit(X_train_pca, y_train)
print(X_train_pca.shape[1])
scr = lr.score(X_test_pca, y_test)
scr1 = lr1.score(X_test, y_test)
print("PCA score is: ", scr)
print("Non-PCA score is", scr1)



pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
#print(pca.explained_variance_ratio_)
