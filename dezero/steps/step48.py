import matplotlib.pyplot as plt
import seaborn as sns

from dezero.datasets import get_spiral

x, t = get_spiral(train=True)


print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])


# plt.scatter(x[:, 0], x[:, 1], marker=t)

# sns.scatterplot(x[:, 0], x[:, 1], hue=t)
# plt.show()

import math

import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

# plot history
# loss_history = [1, 0.3, 0.15]
# plt.plot(loss_history)
# plt.show()

# plot decision boundary
xx, yy = np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))
new_x = np.column_stack((xx.ravel(), yy.ravel()))
new_y = model.forward(new_x)
new_t = np.argmax(new_y.data, axis=1)
zz = new_t.reshape(xx.shape)
plt.contourf(xx, yy, zz, cmap="Paired")
sns.scatterplot(x[:, 0], x[:, 1], hue=t)
plt.show()

loss_history = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f"epoch {epoch+1}, loss {avg_loss}")
    loss_history.append(avg_loss)

plt.plot(loss_history)
plt.show()

# plot decision boundary
xx, yy = np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))
new_x = np.column_stack((xx.ravel(), yy.ravel()))
new_y = model.forward(new_x)
new_t = np.argmax(new_y.data, axis=1)
zz = new_t.reshape(xx.shape)
plt.contourf(xx, yy, zz, cmap="Paired")
sns.scatterplot(x[:, 0], x[:, 1], hue=t)
plt.show()
