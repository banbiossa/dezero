import numpy as np

from dezero.steps.step01 import Variable
from dezero.steps.step02 import Function, Square


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


A = Square()
B = Exp()
C = Exp()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)
