import numpy as np

from dezero import Parameter

x = np.array([1, 2, 3])
np.save("test.npy", x)

x = np.load("test.npy")
print(x)

x2 = np.array([4, 5, 6])
np.savez("test.npz", x=x, x2=x2)

arrays = np.load("test.npz")

print(list(arrays))

from dezero.layers import Layer

layer = Layer()
l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))

params_dict = {}
layer._flatten_params(params_dict)
print(params_dict)

import os

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.dataloaders import DataLoader
from dezero.datasets import MNIST
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

if os.path.exists("my_mlp.npz"):
    print("load weights")
    model.load_weights("my_mlp.npz")

print(f"start train {max_epoch=}")
for epoch in range(max_epoch):
    sum_loss = 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    train_loss = sum_loss / len(train_set)
    print(f"{epoch=}, {train_loss=:.4f}")

model.save_weights("my_mlp.npz")
