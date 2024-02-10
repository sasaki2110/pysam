#このコード、90%以上 Amazon Code Wispererが書いたんすけど・・・・
#iris dataset を 重回帰分析
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#pltをバックグラウンドで動かすおまじない
import matplotlib
matplotlib.use("Agg")

iris = datasets.load_iris()
X = iris.data
y = iris.target
#データの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#モデルの学習
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
#モデルの評価
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)
#モデルを利用
new_flower = [[5.8, 3.1, 5.2, 1.2]]
#new_flower = [[5.8, 3.1]]
prediction = model.predict(new_flower)
print(prediction)
plt.scatter(X_test[:, 0], y_test, color='red')
plt.scatter(X_test[:, 0], y_pred, color='blue')
plt.savefig("sin.png")

