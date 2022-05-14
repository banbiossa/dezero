from __future__ import annotations

import contextlib
import weakref
from abc import ABC, abstractmethod

import numpy as np


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} not supported, only np.ndarray")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Variable({p})"

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __radd__(self, other):
        return add(self, other)

    def set_creator(self, func: callable) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False) -> None:
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
            gys = [output().grad for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function(ABC):
    def __call__(self, *inputs: list[Variable]) -> list[Variable]:
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            # save state
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

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


class Mul(Function):
    def forward(self, x0, x1) -> tuple:
        y = x0 * x1
        return y

    def backward(self, gy) -> tuple:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Square(Function):
    def forward(self, x) -> tuple:
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def square(x):
    return Square()(x)


def no_grad():
    return using_config("enable_backprop", False)


with using_config("enable_backprop", False):
    x = Variable(np.array(2.0))
    y = square(x)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)


x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)

x = Variable(np.array(2.0))
y = x + 3.0
print(y)

y = 3.0 * x + 1.0
print(y)

x = Variable(np.array([1.0]))
y = np.array([2.0]) + x
print(y)
