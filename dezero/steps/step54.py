import numpy as np

dropout_ratio = 0.6
x = np.ones(10)

# direct
# train
mask = np.random.rand(10) > dropout_ratio
y = x * mask

# test
scale = 1 - dropout_ratio
y = x * scale

# train (inverted)
scale = 1 - dropout_ratio
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask / scale

# test
y = x

import dezero.functions as F
from dezero.core import test_mode

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
