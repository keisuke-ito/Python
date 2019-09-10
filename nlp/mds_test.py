import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

#距離行列の読み込み
datum = np.loadtxt("mds_clu.csv",delimiter=",",usecols=range(1,174))

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)

print(datum)


pos = mds.fit_transform(datum)

labels = np.genfromtxt("mds_clu.csv",delimiter=",",usecols=0,dtype=str)

print(labels)


#図の詳細設定
plt.figure()
 
angle = 120.
theta = (angle/180.) * np.pi
 
rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])
revMatrix = np.array([[-1, 0], [0, 1]])
 
fixed_pos = rotMatrix.dot(revMatrix.dot(pos.T)).T
plt.scatter(fixed_pos[:, 0], fixed_pos[:, 1], marker = 'o')
 
for label, x, y in zip(labels, fixed_pos[:, 0], fixed_pos[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (60, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
       # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
    )
 
plt.show()
