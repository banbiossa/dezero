import numpy as np

import dezero.functions as F
from dezero import Variable

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.reshape(x, (6,))
print(y)


x = Variable(x)
y = F.reshape(x, (6,))

y.backward(retain_grad=True)
print(x.grad)

x = np.random.rand(1, 2, 3)
y = x.reshape((2, 3))
y = x.reshape([2, 3])
y = x.reshape(2, 3)


x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.transpose(x)
print(y)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(x.grad)


y = x.transpose()
print(y)
y = x.T
print(y)

A, B, C, D = 1, 2, 3, 4
x = np.random.rand(A, B, C, D)
y = x.transpose(1, 0, 3, 2)
print(y.shape)
