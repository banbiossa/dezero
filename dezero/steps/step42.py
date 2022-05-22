import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable

np.random.seed(0)

x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

# plt.scatter(x, y)
# plt.show()

x, y = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(W, b, loss)


x_array = np.linspace(min(x.data), max(x.data))
y_pred = predict(x_array).data
plt.scatter(x.data, y.data)
plt.plot(x_array, y_pred, color="red")
plt.show()
