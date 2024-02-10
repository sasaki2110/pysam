# Pandas
# データ操作ライブラリ
# CSVのR/Wや、並べ替えとか欠損値の除去 / 補間とかができる

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
#print(df.describe())

# ↓は、データの相関関係を算出してくれるメソッドらしい
# こっちの相関関係は、まったく意味が解らんなぁ・・・
#print(df.corr())


#print(df)
#df_de = df.sort_values(by='合計部屋数', ascending=False)

#print(df_de.head(3))
#print("\n\n\n")
#print(df_de.iloc[2, 2])

#mask = df['住宅価格の中央値'] < 70000
#print(mask)
#print(df[mask].head())

#df['対象'] = None

#print(df.head())

#mask1 = df['住宅価格の中央値'] < 60000
#mask2 = (df['住宅価格の中央値'] >= 60000) & (df['住宅価格の中央値'] < 70000)
#mask3 = (df['住宅価格の中央値'] >= 70000) & (df['住宅価格の中央値'] < 80000)
#mask4 = df['住宅価格の中央値'] >= 80000


#df.loc[mask1, '対象'] = 0
#df.loc[mask2, '対象'] = 1
#df.loc[mask3, '対象'] = 2
#df.loc[mask4, '対象'] = 3

#print(df.head())

#df.iloc[0,0] = None
#print(df.head())
#df_dropna = df.dropna()
#print(df_dropna.head())

#mean = df.mean()
#df_fillna = df.fillna(mean)
#print(df_fillna.head())

print("type(df) = ", type(df))
print("type(df.values) = ", type(df.values))