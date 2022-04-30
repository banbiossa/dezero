import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)


class Function:
    def __call__(self, var: Variable) -> Variable:
        x = var.data
        y = x**2
        output = Variable(y)
        return output


x = Variable(np.array(10))
f = Function()
y = f(x)
print(type(y))
print(y.data)
