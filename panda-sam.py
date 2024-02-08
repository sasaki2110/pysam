import pandas as pd

#df = pd.read_csv("./california_housing_train.csv", encoding='cp932')
# LANGを指定して、ファイルをutf-8にして、後はutf-8で処理するのがFAやね。
df = pd.read_csv("./california_housing_train.csv", encoding='utf-8')

#print(type(df)) # 読み込んだデータの型表示
#print(df.head(10)) # 先頭10行表示
#print(df['住宅価格の中央値'].head(3)) # 票jするカラムを指定
#print(df.shape) # まあいわゆる行列の構造（行数と列数）

#以降は、データフレーム内部の統計量計算
#print("df.mean() いわゆる平均")
#print("-------------------------------")
#print(df.mean())
#print("-------------------------------")
#print("df.var() いわゆる分散")
#print("-------------------------------")
#print(df.var())
#print("-------------------------------")

# ↓は、データの特徴をおおまかに調べるのに便利なメソッドらしい。
# こっちは、まあまあ意味は解る（古いデータだから、イメージは解んけど）
print(df.describe())

# ↓は、データの相関関係を算出してくれるメソッドらしい
# こっちの相関関係は、まったく意味が解らんなぁ・・・
#print(df.corr())


