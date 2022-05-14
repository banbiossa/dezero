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
        self.generation = 0

    def set_creator(self, func: callable) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx
                if x.creator is not None:
                    add_func(x.creator)


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

        self.generation = max([x.generation for x in inputs])
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


generations = [2, 0, 1, 4, 2]
funcs = []

for g in generations:
    f = Add()
    f.generation = g
    funcs.append(f)

print([f.generation for f in funcs])


funcs.sort(key=lambda x: x.generation)
print([f.generation for f in funcs])

f = funcs.pop()
print(f.generation)


x = Variable(np.array(2.0))
a = square(x)

y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)

from memory_profiler import profile

# todo: memory profiler


@profile
def my_func():
    for i in range(10):
        x = Variable(np.random.randn(10_000))
        y = square(square(square(x)))

    del x
    del y
    print("done")


if __name__ == "__main__":
    my_func()
