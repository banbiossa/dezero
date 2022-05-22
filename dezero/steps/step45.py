import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable
from dezero.models import Model

# model = Layer()
# model.l1 = L.Linear(5)
# model.l2 = L.Linear(3)
#
#
# def predict(model, x):
#     x = model.l1(x)
#     x = F.sigmoid(x)
#     x = model.l2(x)
#     return x
#

# for p in model.params():
#     print(p)
#
# model.cleargrads()
#


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = self.l2(x)
        return x


# x = Variable(np.random.randn(5, 10), name="x")
# model = TwoLayerNet(100, 10)
# model.plot(x)

np.random.seed(0)
x = np.random.randn(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10_000
hidden_size = 10

model = TwoLayerNet(hidden_size, 1)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
