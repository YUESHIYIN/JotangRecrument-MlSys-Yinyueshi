import torch

x = torch.arange(12)

print(x.shape)
print(x.numel())

X = x.reshape(3,-1)
print(X)

y = torch.zeros(2,3,4)
print(y)

z = torch.ones([2,3,4])
print(z)

m = torch.randn(3,4)
print(m)
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y 
print(x+y)
print(x-y) 
print(x*y)
print(x/y)
print(x**y)
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)