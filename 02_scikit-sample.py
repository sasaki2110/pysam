# scikit-learn
# scikit-learn は Python のオープンソース機械学習ライブラリ


from sklearn import datasets

# アイリスデータをsklearnから取得
iris = datasets.load_iris()

# 特徴とターゲットに分解。（これが出来る形ってなんだっけ？？？）
# <class 'sklearn.utils._bunch.Bunch'> らしいけど、よくわからんな。
print("type(iris) = ", type(iris))

x = iris.data    # 特徴量（アイリスの花の、「がく片」、「花弁」、「長さ」、「幅」）
y = iris.target  # ターゲット変数（アイリスの種類） 0:Setosa 1:Versicolour 2:Virginica

print("x.shape = ", x.shape) # 150, 4
print("y.shape = ", y.shape) # 150

# 学習データと評価データに分解（20%をテスト評価に、残り80%を学習に
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# 学習　モデルをトレーニングデータにフィットさせる
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# モデルの評価
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))

# モデルを利用
# 新しいアイリスの花のデータ（例：がく片の長さ5.0cm, 幅3.5cm, 花弁の長さ1.5cm, 幅0.2cm）
new_flower = [[5.8, 3.1, 5.2, 1.2]]

# モデルを使って種類を予測
prediction = model.predict(new_flower)

# 種類のリスト
iris_types = ['Setosa', 'Versicolour', 'Virginica']

# 結果の表示
print('Predicted class: ', iris_types[prediction[0]])