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
print(df_rune2.shape)
df_rune2 = np.concatenate((dos, df_rune2), 1)

df_rune = np.vstack((df_rune0, df_rune1, df_rune2))
print(df_rune)
#df_rune = pd.DataFrame(df_rune)#,index=df_rune[:,0])
print(df_rune)

# separate training and test data (70, 30)
#X, y = df_rune.iloc[:, 1:].values, df_rune.iloc[:, 0].values
X, y = df_rune[:, 1:], df_rune[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


#X_train = np.load('dumped/X_train.dat')
#X_test = np.load('dumped/X_test.dat')
#y_train = np.load('dumped/y_train.dat')
#X_test = np.load('dumped/X_test.dat')

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

plt.bar(range(1, 626), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 626), cum_var_exp, where='mid', label='cumilative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs

# sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#print(eigen_pairs)

# we created a 625 x 100-dimensional projection matrix W from top two eigenvectors.
w = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(100)])
#w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
#print('Matrix W:\n', w.shape)

X_train_pca = X_train_std.dot(w)
#print(X_train_pca)

# let's visualize the transformed rune training set.
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
    
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()


# using PCA from sckit learn
pca = PCA(n_components=100)
lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
print(X_train_pca.shape[1])
plot_decision_regions(X_train_pca, y_train, classifier=lr)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()



# test the transformed test dataset
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()


pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)
