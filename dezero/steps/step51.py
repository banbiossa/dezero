import dezero
from dezero.datasets import MNIST

train_set = MNIST(train=True, transform=None)
test_set = MNIST(train=False, transform=None)

print(len(train_set), len(test_set))


x, t = train_set[0]
print(type(x), x.shape)
print(t)

import matplotlib.pyplot as plt

plt.imshow(x.reshape(28, 28), cmap="gray")
plt.axis("off")
plt.show()
import numpy as np


def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_set = MNIST(train=True, transform=f)
test_set = MNIST(train=False, transform=f)

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = MNIST(train=True)
test_set = MNIST(train=False)

from dezero.dataloaders import DataLoader

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

from dezero import functions as F
from dezero.models import MLP
from dezero.optimizers import SGD, Adam

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = Adam().setup(model)


for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f"{epoch=}")
    train_loss = sum_loss / len(train_set)
    train_accuracy = sum_acc / len(train_set)
    print(f"{train_loss=}, {train_accuracy=}")

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    test_loss = sum_loss / len(test_set)
    test_accuracy = sum_acc / len(test_set)

    print(f"{test_loss=}, {test_accuracy=}")
