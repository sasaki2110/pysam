#--------------------------------------------------------------------------
# PyTorch チュートリアル日本語訳を通じて、ディープラーニングを理解する。
# https://yutaroogawa.github.io/pytorch_tutorials_jp/
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Transform
# 
# TorchVisionの全データセットには、特徴量（データ）を変換処理するためのtransformと、
# ラベルを変換処理するためのtarget_transformという2つのパラメータがあります。
# さらに、変換ロジックを記載した callable を受け取ります。
# 
# torchvision.transformsモジュールは、一般的に頻繁に使用される変換を提供しています。
#--------------------------------------------------------------------------
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# さっぱり解らんけど、また必要になったら戻ってくるか。