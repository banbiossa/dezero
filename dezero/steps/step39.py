import numpy as np

import dezero.functions as F
from dezero import Variable

# x = Variable(np.array([1, 2, 3, 4, 5, 6]))
# y = F.sum(x)
# y.backward()
# print(y)
# print(x.grad)
#

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.sum(x, axis=0)
print(y)
print(x.shape, "->", y.shape)
