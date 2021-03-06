import numpy as np

from dezero import Variable


def f(x):
    return x**4 - 2 * x**2


def gx2(x):
    return 12 * x**2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward()

    # diff = x.grad / gx2(x.data)
    diff = x.grad / gx2(x)
    # x.data = x.data - diff
    x = x - diff

    # x.data -= x.grad / gx2(x.data)
