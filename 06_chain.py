#---------------------------------------------------------------------------
# chainer
# ディープラーニングフレームワーク
# もう開発が終了していて、PyTorchに移ったみたい。（＝てことは、PyTorchも見ないと）
# 要は
#    scikit-learn：マシンラーニング（機械学習）
#    chainer：ディープラーニング（深層学習）
# ってことやね。
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# いつものパターンで下記を実行する。
# Step 1 : データセットの準備
# Step 2 : ネットワークを決める   (機械学習だと、ここがモデルを決めるだった。)
# Step 3 : 目的関数を決める
# Step 4 : 最適化手法を選択する
# Step 5 : ネットワークを訓練する
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# ここで出来たものを、一度熟読するか。
#---------------------------------------------------------------------------


import chainer
# Chainer のバージョンや実行環境を確認
#chainer.print_runtime_info()

#---------------------------------------------------------------------------
# Step 1 : データセットの準備
#---------------------------------------------------------------------------

# iris データセットをインポート
from sklearn.datasets import load_iris

# ここでは sklearn に定義されている datasets の中から、load_iris を読み込んだのかな？
# なら、ほかにはどんな datasets があるんだろう？
# まあまあ、いろんなデータあんな。
# https://note.nkmk.me/python-sklearn-datasets-load-fetch/

# Iris データセットの読み込み
# x : 特徴値
# t : アヤメの種類
x, t = load_iris(return_X_y=True)

# それぞれデータ型を変換
x = x.astype('float32')
t = t.astype('int32')
# データセットを分割する
# チュートリアルの記述
from sklearn.model_selection import train_test_split
x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
# さらに分割する　このさらに分割の意味が、まったく解らない
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

#---------------------------------------------------------------------------
# Step 2 : ネットワークを決める 
#---------------------------------------------------------------------------

import chainer.links as L      # パラメータを持つ関数（層）：リンク
import chainer.functions as F  # パラメータを持たない関数：ファンクション

# Chainer ではネットワークを作る方法がいくつか用意されています。
# ここでは簡単にネットワークが作ることができる Sequential クラスを利用し、
# 全結合層が 3 つ、活性化関数に ReLU 関数を利用したネットワーク net を作ります。

# Chainerでは全結合層は Linear クラスとして定義されています。 
# これは以下のように、インスタンス化の際に入力の次元数と出力の次元数を引数に取ります。

# 入力次元数が 3、出力次元数が 2 の全結合層
l = L.Linear(3, 2)

# また、ReLU 関数は relu() として定義されています。
# それでは、Sequential クラスと Linear クラス、relu() を使ってネットワークを作ります。
from chainer import Sequential

# net としてインスタンス化
n_input  = 4      # irisの入力変数は4。なので最初の全結合層の入力次元数は4。
n_hidden = 10     # 中間層の次元数は10
n_output = 3      # 最後の全結合層の出力次元数は3

# やべえ、チュートリアルと同じものをwisperが提案してきやがった・・・・
net = Sequential(
    L.Linear(n_input, n_hidden),    F.relu,
    L.Linear(n_hidden, n_hidden),   F.relu,
    L.Linear(n_hidden, n_output)
)

#---------------------------------------------------------------------------
# Step 3 : 目的関数を決める
#---------------------------------------------------------------------------

# 訓練の際に利用する目的関数を決定します。 
# 今回は、分類タスクによく利用される目的関数の中から、交差エントロピーを採用することにします。
# ここでは、Chainerに定義されている softmax_cross_entropy() （注釈3）を利用します。

# 注釈3
# Chainer では、交差エントロピーは、softmax_cross_entropy() という関数を用いて計算されます。
# これは、予測値と正解値の間の誤差を計算するための関数です。
# この関数は、予測値と正解値の間の誤差を計算し、その誤差の平均値
# を目的関数として使用することができます。


#---------------------------------------------------------------------------
# Step 4 : 最適化手法を選択する
#---------------------------------------------------------------------------

# 今回は、最適化手法として、確率的勾配降下法 (SGD) を利用します

# Chainer では chainer.optimizers に各最適化手法を実行するためのクラスが用意されており、
# 確率的勾配降下法は SGD という名前で定義されています。

# 学習率 lr を 0.01としてインスタンス化し、optimizer という名前を付けます。
optimizer = chainer.optimizers.SGD(lr=0.01)
optimizer.setup(net)

#---------------------------------------------------------------------------
# Step 5 : ネットワークを訓練する
#---------------------------------------------------------------------------

# Step 1 〜 Step 4 で準備したものを使ってネットワークの訓練を行います。 
# ネットワークの訓練を行う前に、訓練の際のエポック数とバッチサイズを決めます。
# ここではエポック数 n_epoch と バッチサイズ n_batchsize を以下のようにします。
n_epoch = 30
n_batchsize = 16

# 実際に訓練を実行します。 訓練は以下の処理を繰り返します。
#
#    1.訓練用のバッチを準備
#    2.予測値を計算し、目的関数を適用 (順伝播)
#    3.勾配を計算 (逆伝播)
#    4.パラメータを更新
# これに加えて、訓練がうまくいっているか判断するために、訓練データを利用した分類精度と検証データを利用した目的関数の値と分類精度を計算します。

import numpy as np

iteration = 0

# ログの保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}


for epoch in range(n_epoch):

    # データセット並べ替えた順番を取得
    order = np.random.permutation(range(len(x_train)))

    # 各バッチ毎の目的関数の出力と分類精度の保存用
    loss_list = []
    accuracy_list = []

    for i in range(0, len(order), n_batchsize):
        # バッチを準備
        index = order[i:i+n_batchsize]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]

        # 予測値を出力
        y_train_batch = net(x_train_batch)

        # 目的関数を適用し、分類精度を計算
        loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)

        # 勾配のリセットと勾配の計算
        net.cleargrads()
        loss_train_batch.backward()

        # パラメータの更新
        optimizer.update()

        # カウントアップ
        iteration += 1

    # 訓練データに対する目的関数の出力と分類精度を集計
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # 1エポック終えたら、検証データで評価
    # 検証データで予測値を出力
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = net(x_val)

    # 目的関数を適用し、分類精度を計算
    loss_val = F.softmax_cross_entropy(y_val, t_val)
    accuracy_val = F.accuracy(y_val, t_val)

    # 結果の表示
    print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
        epoch, iteration, loss_train, loss_val.array))

    # ログを保存
    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)

# ここでは簡単のために、1エポックの内で使用してないサンプルサイズがバッチサイズ以下になったら
# 次のエポックに移るようにしています。
# 
# 上のコードで示したように、Chainer では勾配を計算する前に net.cleargrads() を実行し、
# 一つ前の勾配を削除した後、loss_train_batch.backward() で勾配を計算します。
#
# また、検証データに対して予測値の計算を行う際には、chainer.using_config('train', False) と chainer.using_config('enable_backprop', False) の
# 2 つのスコープを指定しています（注釈4）。

# 訓練が終了したので、目的関数の出力値（交差エントロピー誤差）と分類精度がエポック毎にどのように変化したかを可視化します。
import matplotlib.pyplot as plt

#pltをバックグラウンドで動かすおまじない
import matplotlib
matplotlib.use("Agg")

# 目的関数の出力 (loss)
plt.plot(results_train['loss'], label='train')  # label で凡例の設定
plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
plt.legend()  # 凡例の表示
plt.savefig("sin.png")

# 一旦グラフをクリア
matplotlib.pyplot.clf() 

# 分類精度 (accuracy)
plt.plot(results_train['accuracy'], label='train')  # label で凡例の設定
plt.plot(results_valid['accuracy'], label='valid')  # label で凡例の設定
plt.legend()  # 凡例の表示
plt.savefig("sin2.png")

#---------------------------------------------------------------------------
# テストデータを用いた評価
#---------------------------------------------------------------------------

# 訓練済みのネットワークを使ってテストデータに対する評価を行います。
# まずは、テストデータで予測を行います。
# 検証用データのときと同様に chainer.using_config('train', False) と chainer.using_config('enable_backprop', False) を使います。

# テストデータで予測値を計算
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)

# 予測ができたら分類精度を計算を確認します。
accuracy_test = F.accuracy(y_test, t_test)
print(accuracy_test.array)

#---------------------------------------------------------------------------
# ネットワークの保存
#---------------------------------------------------------------------------

# 最後に、訓練したネットワークを保存します。保存する場合は以下のようにします。

chainer.serializers.save_npz('my_iris.net', net)


#---------------------------------------------------------------------------
# 訓練済みネットワークを用いた推論
#---------------------------------------------------------------------------

# 訓練済みのネットワークを用いてテストデータに対して推論を行います。
# 
# まずは保存したネットワークを読み込みます。 
# ネットワークを読み込むには、最初に訓練済みのネットワークと同様のクラスのインスタンスを作ります。
loaded_net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)
# このネットワークのインスタンスに訓練済みのネットワークのパラメータを読み込ませます。
chainer.serializers.load_npz('my_iris.net', loaded_net)

# 訓練済みのネットワークの準備ができたら、実際に推論を行います。
# 推論を行う際は検証用データのときと同様に 
# chainer.using_config('train', False)、chainer.using_config('enable_backprop', False) を使います。

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = loaded_net(x_test)

# テストデータに対して推論ができたので、テストデータの 0 番目のサンプルの予測結果を確認してみます。
# 分類では以下のようにして、予測されたラベルを出力できます。
np.argmax(y_test[0,:].array)

