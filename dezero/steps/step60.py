import dezero.datasets
from dezero.dataloaders import SeqDataLoader

train_set = dezero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=3)
x, t = next(dataloader)

print(x)
print("-" * 20)
print(t)

import numpy as np

import dezero
import dezero.functions as F
import dezero.layers as L
from dezero.dataloaders import SeqDataLoader
from dezero.datasets import SinCurve
from dezero.models import Model
from dezero.optimizers import Adam

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


model = BetterRNN(hidden_size, 1)
optimizer = Adam().setup(model)


for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss = float(loss.data) / count
    print(f"{epoch=}, {avg_loss=}")
