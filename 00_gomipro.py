import torch
import torch.nn as nn
import torch.nn.functional as F

linear = nn.Linear(5, 10)
x = torch.ones(3, 5, requires_grad=True)
print("z = y の場合")
print(x)
print(x.shape)
y = x + 2
z = y
print("\n")
print(z)
print(z.shape)
out = z.sum()
out.backward()
print(x.grad)


linear = nn.Linear(5, 10)
x = torch.ones(3, 5, requires_grad=True)
print("z = liner(y) の場合")
print(x)
print(x.shape)
y = x + 2
z = linear(y)
print("\n")
print(z)
print(z.shape)
out = z.sum()
out.backward()
print(x.grad)

