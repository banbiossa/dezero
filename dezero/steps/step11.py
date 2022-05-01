from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} not supported, only np.ndarray")
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: callable) -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function(ABC):
    def __call__(self, *inputs: list[Variable]) -> list[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        # save state
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: list[Variable]) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def backward(self, *gys: list[Variable]) -> list[Variable]:
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1) -> tuple:
        return x0 + x1

    def backward(self, gys: list[Variable]) -> list[Variable]:
        pass


def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))

y = add(x0, x1)
print(y.data)
