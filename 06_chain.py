import chainer

#chainer.print_runtime_info()

from sklearn.datasets import load_iris

x, t = load_iris(return_X_y=True)

print('x:', x.shape)
print('t:', t.shape)

x = x.astype('float32')
t = t.astype('int32')

from sklearn.model_selection import train_test_split

x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

import chainer.links as L
import chainer.functions as F
