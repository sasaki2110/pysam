# CuPy 
# NumPy と高い互換性を持つ数値計算ライブラリで
# NVIDIA GPU で実行することで簡単に高速化できる
# 構築がややこしそうだったから、cupy コンテナを使う事にした。
#   docker pull cupy/cupy:latest



import numpy as np
import cupy as cp
import time

# 行列AおよびBの生成
#matrix_A = np.random.randn(10000, 20000)
#matrix_B = np.random.randn(20000, 30000)
n = 6 # 家だと、7でkilledになった。　この限界の意味って？
matrix_A = np.random.randn(1000 * n, 2000 * n)
matrix_B = np.random.randn(2000 * n, 3000 * n)

# CPUでの数値計算
start = time.time()
matrix_AB = np.dot(matrix_A, matrix_B)
end = time.time()
print("CPUでの数値計算:", end-start)

# 2つの行列値をGPUメモリへ移動
start = time.time()
matrix_A_gpu = cp.asarray(matrix_A)
matrix_B_gpu = cp.asarray(matrix_B)
end = time.time()
print("メインメモリ->GPUメモリ:", end-start)

# GPUでの数値計算
start = time.time()
matrix_AB_gpu = cp.dot(matrix_A_gpu, matrix_B_gpu)
end = time.time()
print("GPUでの数値計算:", end-start)