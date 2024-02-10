# Numpy
# 多次元配列（multidimensional array）を扱うライブラリだと
# 例えば、
# 様々な機械学習手法を統一的なインターフェースで利用できる scikit-learn や、
# ニューラルネットワークの記述・学習を行うためのフレームワークである Chainer は、
# NumPy に慣れておくことでとても使いやすくなります。
# ですと。


import numpy as np

X = np.array([
    [2, 3],
    [2, 5],
    [3, 4],
    [5, 9],
])

ones = np.ones((X.shape[0], 1))
X = np.concatenate((ones, X), axis=1)

t = np.array([1, 5, 6, 8])

xx = np.dot(X.T, X)
print("np.dot = ", xx)

xx_inv = np.linalg.inv(xx)
print("np.linalg.inv(xx) = ", xx_inv)

xt = np.dot(X.T, t)
print("np.dot(X.T, t) = ", xt)

w = np.dot(xx_inv, xt)
print("np.dot(xx_inv, xt) = ", w)

