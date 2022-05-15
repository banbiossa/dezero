import numpy as np

from dezero import Variable

x0 = np.array([1, 2, 3])
x1 = np.array([10])
y = x0 + x1
print(y)

x0 = Variable(x0)
x1 = Variable(x1)
y = x0 + x1
print(y)

y.backward()
print(x1.grad)
