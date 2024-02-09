#matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./california_housing_train.csv')
print(df.head(5))

#plt.scatter(df['収入の中央値'], df['住宅価格の中央値'])
#plt.show()

x = [1,2,3,4]
y = [2,3,4,5]

# グラフを描画
plt.plot(x, y)
plt.show()