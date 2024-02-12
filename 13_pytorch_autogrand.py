#--------------------------------------------------------------------------
# PyTorch チュートリアル日本語訳を通じて、ディープラーニングを理解する。
# https://yutaroogawa.github.io/pytorch_tutorials_jp/
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# 自動微分
#
# 
#--------------------------------------------------------------------------

import torch

# x は 入力データ（テンソル）
x = torch.ones(5) 

# y は 出力値 z と ce? することでlossが出てくる
y = torch.zeros(3)

# w は重みベクトル（＜－本当か？）　x　に　乗算する
# requires_grad=Trueで自動微分を許可しているらしいけど・・・
w = torch.randn(5, 3, requires_grad=True)

# b はバイアス（＜－本当か？）　x * w に　加算する
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w)+b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("loss = ", loss)


loss.backward()
print(w.grad)
print(b.grad)

# 計算モデルの図と、コードの関連はまあ理解。でもその真意はまったく解らない。
# 結局 autograd されてどうなった？？？？