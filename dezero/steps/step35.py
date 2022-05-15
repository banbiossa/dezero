import numpy as np

import dezero.functions as F
from dezero import Variable
from dezero.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)


# iters = 0
for iters in range(8):
    print(iters)
    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = f"gx{iters+1}"
    plot_dot_graph(gx, verbose=False, to_file=f"tanh_{gx.name}.png")
