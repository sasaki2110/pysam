# 自動微分

import torch

# Autograd: 自動微分
# requires_grad=True で自動微分を可能にする。
#
# ザクっというと、x をインプットとした一連の計算結果をもとに、
# その計算の傾きを勾配ベクトルで求めるってこったな。
# コーディング上のポイントは、tensor生成時に requires_grad=True を指定する。
# 計算後にout.backward()でgradを計算し、x.gradで勾配ベクトルを取得する。




#x = torch.ones(2, 2, requires_grad=True)
data = [[1, 2], [3, 4]]
x = torch.tensor(data, dtype=torch.float32, requires_grad=True)

print("x = ", x)  # 最初は requires_grad=True 

y = x + 2
print("y = ", y)  # grad_fn=<AddBackward0> に変わった。

z = y * y * 3
out = z.mean()    # ちょま、mean() って何？平均値だって。こんなんを覚えれんなぁ・・・・

print("z = ", z)
print("out = ", out)

# ここまでの流れで、requires_grad=Trueを指定しておけば、add や mul、mean が積まれてくることは解った。


# 勾配（Gradients）

# まず大前提として、もともとの入力（いわゆる x）が二次元配列だから、
# 単純に微分で傾き（スカラー値）を求める事ができない。
# そこで、勾配ベクトル（多変数関数の変化率が最大となる方向）で傾きを求める。

# 最初に、out.backward() を実行すると、
# out に対応する微分値がセットされる。
# この微分値は、その後の計算で使われる。

# out.backward() は、計算グラフに基づいて、出力に対する各入力の勾配を計算する関数です。
# out に対する x の勾配が計算され、x.grad に格納されています

out.backward()

#
print("out = ", out)
print("x.grad = ", x.grad)
print("\n\n\n")
# print("y.grad = ", y.grad)  # grad_fn=None になってる。
# print("z.grad = ", z.grad)  # grad_fn=None になってる。

# そしてgradはあくまでインプット（いわゆる x ）に対してつくのね。

# ここから話題が変わって、ヤコビアン積の例
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)