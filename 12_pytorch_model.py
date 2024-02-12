#--------------------------------------------------------------------------
# PyTorch チュートリアル日本語訳を通じて、ディープラーニングを理解する。
# https://yutaroogawa.github.io/pytorch_tutorials_jp/
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# モデル構築
#
# ニューラルネットワークはモジュール（レイヤー）と呼ばれる
# データ操作の塊で構築されている。
# 
# torch.nn で用意されているクラス・関数ががニューラルネットワーク構築に
# 必要な要素を網羅している。
#
# PyTorchのすべてのモジュールは mm.Module を継承している。
# そしてニューラルネットワークは、モジュール自体が他のモジュールから構成されている。
# この入れ子構造で、複雑なアーキテクチャを容易に構築・管理できる。
#
# ちなみに、nn.Moduleを使わずに、スクラッチでニューラルネットワークを構築もできる。（みたい）
# 
#--------------------------------------------------------------------------

# FashionMNIST datasetの画像データをクラス分類するネットワークモデルを構築
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 利用可能なデバイスをチェック（cudaのはず）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# クラスの定義
# nn.Moduleを継承した独自のネットワークモデルを定義
# その後、__init__ で初期化。
# nn.Moduleを継承したモジュールは、入力データの順伝搬関数である forward() を持つ。

# NuralNetworkというクラス名で、mm.Module を継承
class NuralNetwork(nn.Module):
    # コンストラクタ（第一引数は一般的に自分自身でselfとする）
    def __init__(self):
        # 継承元のコンストラクタを呼び出す
        # んだけど、この第一引数に自分のクラスを渡すってなんだ？
        # って、この記述ってPython2系なの？？？
        #super(NuralNetwork, self).__init__()
        # Python3系では、下記らいしけど、これでもいいよね？
        super().__init__()

        # flatten 平坦化（一次元化）
        # nn.Flatten()は、入力のテンソルの次元数を1にする。
        # self.flatten は自前のメソッドだよね。
        # 継承元（nn.Module）にもあるのかな？ ⇒ないな。完全に自前のメソッド
        self.flatten = nn.Flatten()

        # liner_relu_stackは自分のメソッドだよね。
        # 継承元（nn.Module）にもあるのかな？ ⇒ないな。完全に自前のメソッド

        # なら、ここで出てくるキーワードは３つ
        # nn.Sequential() 
        #     nn.Sequentialは前回の記事でも説明しましたが「モデルとして組み込む関数を1セットとして順番に実行しますよ」というものです。
        #     これは処理の話だから理解できるよね。
        #
        # nn.Linear()
        #     nn.Linear は、皆さんが既に勉強された重回帰分析で説明すると線形結合という意味です。
        # 
        # nn.ReLU()
        #     用語「ReLU（Rectified Linear Unit）／ランプ関数」について説明。
        #     「0」を基点として、0以下なら「0」、
        #     0より上なら「入力値と同じ値」を返す、ニューラルネットワークの活性化関数を指す。
        #         うん、難しくなってきた。。。。。。

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # 入力データをモデルに投入すると、forward()が呼び出される。
    # よってmodel.forward() と記載して入力データを処理しないように注意
    def forward(self, x):
        # 与えられたデータを一次元化し
        x = self.flatten(x)
        # 一連の処理を実行
        logits = self.linear_relu_stack(x)
        # 何を返すんだ？
        # logits とはソフトマックス活性化関数に通す前の値（テンソル）
        return logits
    
# modelを device へ移動し、ネットワークの構造を確認
model = NuralNetwork().to(device)
print(model)

# 28x28の乱数を入力して、出力の確認
X = torch.rand(1, 28, 28, device=device)
# モデルに入力パラメータを食わせると、logitsが帰ってくるんだね
logits = model(X) 
# logits を softmax に与えると、確率を表すベクトルになるのね
pred_probab = nn.Softmax(dim=1)(logits)
# これなんだ？
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


#--------------------------------------------------------------------------
# これ以降は、モデルレイヤーの詳細を確認している。一連の流れではないはず。
# ここでモデルレイヤーを確認

# まずは、3x 28x28 の乱数を入力して、出力の確認
input_image = torch.rand(3,28,28)
print(input_image.size())

# 次にFlattenで一次元化
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 次にLinearで線形変換（重み付け、バイアス設定）
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# 次にReLUで活性化関数　これが一番解らない
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Sequentialでモデルの構造を組み立てる。
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# モデルパラメーター
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# これで一通り流れたけど、結局なんも解っていないなぁ。