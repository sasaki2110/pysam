# chainer
# ディープラーニングフレームワーク
# もう開発が終了していて、PyTorchに移ったみたい。（＝PyTorchも見ないと）
# 要は
#    scikit-learn：マシンラーニング（機械学習）
#    chainer：ディープラーニング（深層学習）
# ってことやね。

import chainer

# Chainer のバージョンや実行環境を確認
#chainer.print_runtime_info()

# Iris データセットの読み込み
from sklearn.datasets import load_iris

# たしか x:特徴値　y:結果のアヤメ種（0～2）だったよね。
x, t = load_iris(return_X_y=True)

print('x:', x.shape)
print('t:', t.shape)

print(x)
print("\n\n\n")
print(t)

# それぞれデータ型を、Chainerに合わせて変換
x = x.astype('float32')
t = t.astype('int32')

# データセットを、訓練用と評価用に分割。
from sklearn.model_selection import train_test_split
x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

# ↓　これなんで２回も分割してるんだっけ？　あ、訓練値と目標値に分解したのか？
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

