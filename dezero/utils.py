import os
import subprocess
import tempfile
from pathlib import Path

from dezero import Variable


def plot_dot_graph(output: Variable, verbose: bool = True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    with tempfile.TemporaryDirectory() as tmp_dir:
        graph_path = Path(tmp_dir) / "tmp_graph.dot"
        graph_path.write_text(dot_graph)

        # dot command
        extension = Path(to_file).suffix[1:]
        cmd = f"dot {graph_path} -T {extension} -o {to_file}"

        subprocess.run(cmd, shell=True)


def _dot_var(v: Variable, verbose: bool = False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = "" if v.name is None else v.name
    if verbose and v.name is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    return txt


def get_dot_graph(output: Variable, verbose: bool = True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    return "digraph g{" + txt + "}"


import numpy as np

x = Variable(np.random.randn(2, 3))
x.name = "x"
print(_dot_var(x))
print(_dot_var(x, verbose=True))

y = x + 1.0
txt = _dot_func(y.creator)
print(txt)
