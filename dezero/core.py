from __future__ import annotations
from array import array

import contextlib
import weakref
from abc import ABC, abstractmethod

import numpy as np

import dezero
import dezero.cuda

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray,)


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


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        if data is not None and not isinstance(data, array_types):
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

    def __getitem__(self, indices):
        return dezero.functions.get_item(self, indices)

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

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, power):
        return pow(self, power)

    def sum(self, axis=None, keepdims=False):
        import dezero.functions

        return dezero.functions.sum(self, axis, keepdims)

    def transpose(self):
        import dezero.functions

        return dezero.functions.transpose(self)

    @property
    def T(self):
        import dezero.functions

        return dezero.functions.transpose(self)

    def reshape(self, *shape):
        import dezero.functions

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def cleargrad(self):
        self.grad = None

    def set_creator(self, func: callable) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    def backward(self, retain_grad=False, create_graph=False) -> None:
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


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
    def forward(self, x: np.ndarray) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def backward(self, gy: Variable) -> tuple:
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1) -> tuple:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy) -> tuple:
        import dezero.functions

        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1) -> tuple:
        y = x0 * x1
        return y

    def backward(self, gy) -> tuple:
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            import dezero.functions

            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)

        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Square(Function):
    def forward(self, x) -> tuple:
        return x**2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape

        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy

        if self.x0_shape != self.x1_shape:
            import dezero.functions

            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:
            import dezero.functions

            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)

        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def no_grad():
    return using_config("enable_backprop", False)


with using_config("enable_backprop", False):
    x = Variable(np.array(2.0))
    y = square(x)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)


class Parameter(Variable):
    pass
