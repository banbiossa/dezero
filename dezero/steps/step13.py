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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
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
    def backward(self, *gys: list[Variable]) -> tuple:
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1) -> tuple:
        return x0 + x1

    def backward(self, gy) -> tuple:
        return gy, gy


class Square(Function):
    def forward(self, x) -> tuple:
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def add(x0, x1):
    return Add()(x0, x1)


def square(x):
    return Square()(x)


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

x = Variable(np.array(3.0))
y = add(x, x)
print(f"{y.data=}")

y.backward()
print(f"{x.grad}")
