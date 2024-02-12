# ニューラルネットワーク入門

# 上記の図はシンプルなフィードフォワードネットワークです。
# 
# 入力を受けると、それがいくつか層を介して次々と伝わっていき、最終的に出力を得ることができます。
# 
# ニューラルネットワークの典型的な学習手順は次のようになります。
# 
#     ・学習可能なパラメータ(重み)を持つニューラルネットワークを定義
#     ・入力のデータセットに対してループ処理
#     ・ネットワークを介して入力を演算
#     ・損失（正解からどのくらい出力が乖離しているか）を計算
#     ・勾配をネットワークのパラメータに逆伝播
#     ・ネットワークの重みを更新。通常は、単純な更新ルールを使用: 重み = 重み - 学習率 * 勾配


# ネットワークの定義

import torch
import torch.nn as nn
import torch.nn.functional as F

# nn.Conv2d は、PyTorchで2次元畳み込み層を実装するためのクラスです。
# 畳み込み層は、畳み込み演算とプーリング演算を組み合わせることで、
# 画像データから特徴量を抽出するために用いられます。


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 一旦、こんなところで勘弁しておこう