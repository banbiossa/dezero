import numpy as np

import dezero.functions as F
from dezero import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.get_item(x, 1)
print(y)

y.backward()
print(x.grad)


indices = np.array([0, 0, 1])
y = F.get_item(x, indices)
print(y)

print(x[1])

print(x[:, 2])

from dezero.models import MLP

model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
print(y)

from dezero import Variable, as_variable


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


print(y)
p = softmax1d(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)

# loss = F.softmax_cross_entropy_simple(y, t)
loss = F.softmax_cross_entropy(y, t)
print(loss)
