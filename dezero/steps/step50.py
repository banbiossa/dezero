import matplotlib.pyplot as plt

import dezero


class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()
        self.cnt += 1
        return self.cnt


my_iter = MyIterator(5)
for i in my_iter:
    print(i)


from dezero.dataloaders import DataLoader
from dezero.datasets import Spiral

batch_size = 10
max_epoch = 1
train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_dataloader = DataLoader(train_set, batch_size)
test_dataloader = DataLoader(test_set, batch_size, shuffle=True)

for epoch in range(max_epoch):
    for x, t in train_dataloader:
        print(x.shape, t.shape)
        break
    for x, t in test_dataloader:
        print(x.shape, t.shape)
        break

import numpy as np

import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
t = np.array([1, 2, 0])
acc = F.accuracy(y, t)
print(acc)

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)


class History:
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.test_acc = []
        self.test_loss = []

    def add(self, train_loss, train_acc, test_loss, test_acc):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)

    def plot(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(self.train_loss, color="blue", label="train_loss")
        ax[0].plot(self.test_loss, color="orange", label="test_loss")
        ax[0].legend()

        ax[1].plot(self.train_acc, color="blue", label="train_acc")
        ax[1].plot(self.test_acc, color="orange", label="test_acc")
        ax[1].legend()

        plt.show()


history = History()
history.add(1.0, 0.3, 1.1, 0.2)
history.add(0.5, 0.6, 0.6, 0.5)
history.plot()

# plot history
import matplotlib.pyplot as plt

history = History()


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
    train_acc = sum_acc / len(train_set)
    print(f"{train_loss=:.4f}, {train_acc=:.4f}")

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss = sum_loss / len(test_set)
    test_acc = sum_acc / len(test_set)
    print(f"{test_loss=:.4f}, {test_acc :.4f}")
    print()
    history.add(train_loss, train_acc, test_loss, test_acc)

history.plot()
