#このコード、90%以上 Amazon Code Wispererが書いたんすけど・・・・
#カリフォルニア州の住宅価格 を ニューラルネットで 重回帰分析
#データは ./california_housing_train.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("./california_housing_train.csv")
X = df.drop("住宅価格の中央値", axis=1)
y = df["住宅価格の中央値"]
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
new_house = [[115.38,32.83,13,1275,271,865,262,1.9375]]
prediction = model.predict(new_house)
print(prediction)
#モデルの保存
import pickle
pickle.dump(model, open("model.pkl", "wb"))
#モデルの読み込み
model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(new_house)




