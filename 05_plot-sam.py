#matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#pltをバックグラウンドで動かすおまじない
import matplotlib
matplotlib.use("Agg")

df = pd.read_csv('./california_housing_train.csv')
print(df.head(5))

#散布図
#plt.scatter(df['収入の中央値'], df['住宅価格の中央値'])

#ヒストグラム
#plt.hist(df['住宅価格の中央値'], bins=50)

#箱ひげ図はパス

#折れ線グラフ
#x = np.linspace(0, 10, 100)
#y = x + np.random.randn(100)

#plt.plot(y)


#plt.savefig("sin.png")

#統計図（データ分布）
import seaborn as sns

line_plot = sns.distplot(df['人口'])
figure = line_plot.get_figure()
figure.savefig("sin.png")