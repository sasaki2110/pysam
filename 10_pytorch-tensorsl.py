#--------------------------------------------------------------------------
# PyTorch チュートリアル日本語訳を通じて、ディープラーニングを理解する。
# https://yutaroogawa.github.io/pytorch_tutorials_jp/
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# まずはテンソルから
# Numpyのndarrayに似たもの。
# 配列や行列に似ている。まあいわゆるベクトルか。
#--------------------------------------------------------------------------
import torch         # これはPyTorchのライブラリだよね
import numpy as np   # これはNumpyのライブラリだよね

#--------------------------------------------------------------------------
# テンソルの初期化（様々な方法で初期化できる）
#--------------------------------------------------------------------------

# データから直接テンソルに変換
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Numpy配列からテンソルに変換
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 他のテンソルから作成
x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティを維持）ただしデータはすべて「1」
#print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き。データは乱数
#print(f"Random Tensor: \n {x_rand} \n")

x_tensor = torch.as_tensor(x_data) # x_dataのdatatypeを上書き。データはそのままこないかな？
#print(f"Tensor from data: \n {x_tensor} \n")

############################################
# torch の apiリファレンスは、ここにある
# https://pytorch.org/docs/stable/torch.html
############################################

# CSVからの読み込みが上手くできないな。
import pandas as pd
df = pd.read_csv("./california_housing_train.csv", encoding='utf-8')
x_file = torch.from_numpy(df.values)
#print(f"Tensor from file: \n {x_file} \n")

# 乱数や定数でのテンソルの作成
shape = (3, 2,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Ones Tensor: \n {ones_tensor} \n")
#print(f"Zeros Tensor: \n {zeros_tensor}")

# テンソルは、形状（shape）、データ型（dtype）、保存されているデバイスを保持している。
#print(f"Shape of tensor: {ones_tensor.shape}")
#print(f"Datatype of tensor: {ones_tensor.dtype}")
#print(f"Device tensor is stored on: {ones_tensor.device}")

#--------------------------------------------------------------------------
# テンソルの操作（100種類以上の演算があると・・・・）
# これも、さっきのAPIリファレンスを参照　
# https://pytorch.org/docs/stable/torch.html
#--------------------------------------------------------------------------

data = [[ 1,  2,  3,  4],
        [ 5,  6,  7,  8], 
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]]
tensor = torch.tensor(data, dtype=torch.float64,)

print(tensor)
print("Tensor dtype", tensor.dtype)

# 各操作はgpu上で可能だが、デフォルトはcpuなので、明示的に.to でgpuへ移動
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

#print("First row : ", tensor[0])
#print("\n")
#print("First column : ", tensor[:, 0])   # この表現が解らんなぁ。 ：スライスですと。
#print("\n")
#print("Last column : ", tensor[..., -1])  # この表現も　　　　　  ...Ellipsis（3点ドット...）を使うと途中の次元を省略して指定できる。
#print("\n")
#print("Last row : ", tensor[-1, :])
#print("\n")

# なんか : も ... も同じ動きしてるような。。。。まあとりあえず：で覚えておくか。
#tensor[:,1] = 0
#tensor[...,1] = 0
#print(tensor)

t0 = torch.cat([tensor, tensor, tensor], dim=0) # 0次元目がrowなのね
t1 = torch.cat([tensor, tensor, tensor], dim=1) # 1次元目がcolumnなのね

#print(t0)
#print("\n")
#print(t1)

# まず、tensor.Tは行と列を入れ替えたもの
# で行列の掛け算は、いわゆる行列乗算だから、よくわからんけど、そんなやつ
x1 = tensor @ tensor.T
#print("\n")
#print(x1)

# こいつは単純に一致する行列を掛け算するタイプか。
z1 = tensor * tensor
#print("\n")
#print(z1)

# １要素のテンソル。なんのこっちゃ？全部足すと？
# .item()を使用することでPythonの数値型変数に変換できます。なんて言ってる？
#
# やっとでこの日本語の意味が分かった。
# tensor.sum()では、1x1のテンソルが返される。
# それを.item()でPythonの数値型変数に変換してくれる。
agg = tensor.sum()
agg_item = agg.item()
#print("\n")
#print(agg, type(agg))
#print(agg_item, type(agg_item))

# インプレース操作
# 演算結果をオペランドに格納する演算をインプレースと呼ぶ。
# メソッドの最後に接尾辞として _ が付く。
#print(tensor)
#print("\n")
tensor.add_(5)
#print(tensor)

#--------------------------------------------------------------------------
# Numpyとの変換
# CPU上のテンソルとNumpy arraysは同じメモリを共有することができ、
# 相互変換が容易です。
#--------------------------------------------------------------------------

# テンソルをNumpyに変換
t = torch.ones(5)
print(f"t: {t}")   
n = t.numpy()
print(f"n: {n}")
# この際にテンソルが変化すると、Numpyも変化する。
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
# 逆にNumpyが変化すると？テンソルも変化したな。完全に共有されている
n = n + 1
print(f"t: {t}")
print(f"n: {n}")

# Numpyをテンソルに変換
n = np.ones(5)
t = torch.from_numpy(n)
# この際にNumpyが変化すると、テンソルも変化する。
n += 1
print(f"t: {t}")
print(f"n: {n}")
