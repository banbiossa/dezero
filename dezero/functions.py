from typing import Tuple

import numpy as np

from dezero import utils
from dezero.core import Config, Function, Variable, as_array, as_variable


class Sin(Function):
    def forward(self, x: np.ndarray):
        return np.sin(x)

    def backward(self, gy: Variable):
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray) -> tuple:
        return np.cos(x)

    def backward(self, gy: Variable):
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> tuple:
        y = np.exp(x)
        return y

    def backward(self, gy: Variable) -> tuple:
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x: np.ndarray) -> tuple:
        y = np.log(x)
        return y

    def backward(self, gy: Variable) -> tuple:
        (x,) = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray) -> tuple:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> tuple:
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        return y

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(y, t):
    return MeanSquaredError()(y, t)


def linear_simple(x, W, b=None):
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


class Linear(Function):
    def forward(self, x, W, b=None):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> tuple:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + np.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # this is better
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x: np.ndarray) -> tuple:
        y = x[self.slices]
        return y

    def backward(self, gy: Variable) -> tuple:
        (x,) = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


# Variable.__getitem__ = get_item


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy) -> tuple:
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx) -> tuple:
        return get_item(ggx, self.slices)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x: np.ndarray) -> tuple:
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy: Variable) -> tuple:
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x: np.ndarray, t) -> tuple:
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy: Variable) -> tuple:
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(y, t):
    return SoftmaxCrossEntropy()(y, t)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x: np.ndarray) -> tuple:
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy: Variable) -> tuple:
        (x,) = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()

    return Variable(as_array(acc))


class ReLU(Function):
    def forward(self, x: np.ndarray) -> tuple:
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy: Variable) -> tuple:
        (x,) = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


from dezero.core import add, div, mul, neg, pow, rsub, sub
from dezero.functions_conv import (
    average_pooling,
    col2im,
    conv2d,
    conv2d_simple,
    deconv2d,
    im2col,
    pooling,
    pooling_simple,
)
