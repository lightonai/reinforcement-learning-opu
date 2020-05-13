# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs


ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122)

Z, labels = make_blobs(n_samples=100, n_features=3, centers=3)
z = np.transpose(Z, (1, 0))

ax1.scatter(z[0], z[1], z[2])

R = np.random.normal(0, 1 / np.sqrt(3), size=(3,2))
zz = np.transpose(Z.dot(R), (1, 0))

ax2.scatter(zz[0], zz[1])

plt.show()
