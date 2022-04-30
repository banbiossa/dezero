from abc import ABC, abstractmethod

import numpy as np

from dezero.steps.step01 import Variable


class Function(ABC):
    def __call__(self, var: Variable) -> Variable:
        x = var.data
        y = self.forward(x)
        output = Variable(y)
        return output

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2


x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)
