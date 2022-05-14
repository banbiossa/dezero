import numpy as np

from dezero import Variable


def sphere(x, y):
    z = x**2 + y**2
    return z


def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def goldstein2(x, y):
    z = (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


def goldstein(x, y):
    z = (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


def do(func):
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = func(x, y)
    z.backward()

    return x.grad, y.grad


if __name__ == "__main__":
    for name, func in [
        ("sphere", sphere),
        ("matyas", matyas),
        ("goldstein", goldstein),
    ]:
        print(name)
        print(do(func))
        print()

    for x, y in [(0, 0), (0, 1), (1, 0)]:
        print(x, y)
        print(goldstein(x, y))
        print(goldstein2(x, y))
        print()
