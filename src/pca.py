import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA

colors = ['navy', 'turquoise', 'darkorange']


# create data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# build model
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# plot
plt.figure()
plt.title('explained variance ratio (first two components): \n {}'.format(str(pca.explained_variance_ratio_)))
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.savefig('../out/plot.pdf')

# https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
print('explained variance ratio (first two components): {}'.format(str(pca.explained_variance_ratio_)))
print('Component relation with features:')
print(pd.DataFrame(pca.components_, columns=iris.feature_names, index=['PC-1', 'PC-2']))

plt.show()


# TODO
# implement preprocessing / standardscaling of data