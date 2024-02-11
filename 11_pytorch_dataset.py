#--------------------------------------------------------------------------
# PyTorch チュートリアル日本語訳を通じて、ディープラーニングを理解する。
# https://yutaroogawa.github.io/pytorch_tutorials_jp/
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# データセットとデーターローダー
# サンプルデータの記述や処理は複雑で、メンテも大変。
# データセットに関連する事は、モデルの訓練コードから切り離すのが理想。
#--------------------------------------------------------------------------

# DataLoader と Dataset の２種類ある事は解ったが、それぞれの意味や違いは、まだ解らない
# 事前データもあるし、自分で入れる事も可能と。
# PyTorch domain librariesでは、多くのデータセットを提供していると。
# 例えばこんなとこ
# https://pytorch.org/text/stable/index.html
# https://pytorch.org/audio/stable/index.html
# https://pytorch.org/vision/stable/index.html
# 

#--------------------------------------------------------------------------
# Datasetの読み込み
# TorchVisionからFashion-MNISTをロードする例
# https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST
#--------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
#pltをバックグラウンドで動かすおまじない
import matplotlib
matplotlib.use("Agg")


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# ダウンロードと、ダウンロード済みのファイルからロード
# この時点では、まだDataset <class torchvision.datasets.mnist.FashionMNIST>
print("training_data type:", type(training_data))
print("test_data type:", type(test_data))

#--------------------------------------------------------------------------
# Datasetの反復処理と可視化
#--------------------------------------------------------------------------

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# figsizeはできる画像のサイズやな
figure = plt.figure(figsize=(8, 8))

# ここで、3x3と指定しとるんか
cols, rows = 3, 3
# cols * rows 回処理すると。
for i in range(1, cols * rows + 1):
    # torch.randint().item() は、0から指定した数までの乱数を取得する。
    # まあ、訓練データの件数を１要素のテンソルとして作って、item()で取り出すだけかな。
    sample_idx = torch.randint(len(training_data), size=(1,)).item()

    # で、訓練データは２次元配列で、
    # 　カラム１＝実際のimgビットマップ 28x28 のグレースケール イメージ tensorのshapeは[1,28,28]
    # 　カラム２＝ラベル（クラス番号） int
    # を持つんだ。
    img, label = training_data[sample_idx]
    #print("img type = ", type(img)) 
    #print("img shape = ", img.shape) 
    #figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
#plt.show()
figure.savefig("sin.png")

# ここまでの謎はすべて解けたかな。

#--------------------------------------------------------------------------
# カスタムデータセットの作成
# 自分でカスタムしたDatasetクラスを作る際には、 
#   __init__ : コンストラクタ。最初に一回起動
#   __len__ : len()関数に対応する関数。
#   __getitem__ : インデックスを指定して、値を取得する関数。
# の3つの関数は必ず実装する必要があります。
#--------------------------------------------------------------------------
import os
import pandas as pd
from torchvision.io import read_image

# ここで、カスタムデータセットを作成する。
class CustomImageDataset(Dataset):
    # __init__はコンストラクタ。インスタンス化される際に一度だけ実行される。
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # __len__は、len()関数に対応する関数。
    def __len__(self):
        return len(self.img_labels)

    # __getitem__は、インデックスを指定して、値を取得する関数。
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
    
#--------------------------------------------------------------------------
# DataLoader の使用方法
#
# Datasetでも、１つのサンプルの、データとラベルを取り出せる。
#
# でも、モデルの訓練時はミニバッチ（minibatch）の単位でデータを扱いたい。
# また、また各epochでデータはシャッフルされて欲しいです
# （訓練データへの過学習を防ぐ目的です）。
#
# 加えて、Pythonの multiprocessingを使用し、複数データの取り出しを高速化したいところです。
# DataLoaderは上記に示した複雑な処理を簡単に実行できるようにしてくれるAPIとなります。
#
#--------------------------------------------------------------------------
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
#plt.show()
plt.savefig("sin.png")
print(f"Label: {label}")