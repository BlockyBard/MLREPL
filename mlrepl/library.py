"""Hand-crafted bootstrap graphs.

Used two ways: (1) as ground-truth programs for synthetic recovery benchmarks (render a target
from the graph, search from scratch, compare), and (2) as the seed corpus for library learning
(abstractions mined from these and from solved programs).
All shapes fit in [-1, 1]^3 with +Y up.
"""
from __future__ import annotations

import math

from .graph import Graph

WOOD = {"base_color": [0.55, 0.36, 0.2], "roughness": 0.7}
METAL = {"base_color": [0.7, 0.72, 0.75], "roughness": 0.3, "metallic": 1.0}
CERAMIC = {"base_color": [0.9, 0.88, 0.82], "roughness": 0.25}
STONE = {"base_color": [0.6, 0.58, 0.55], "roughness": 0.9}


def table() -> Graph:
    g = Graph()
    top = g.placed("box", pos=(0, 0.5, 0), b=[0.8, 0.06, 0.5], attrs={"material": WOOD})
    leg = g.add("cylinder", params={"r": [0.06], "h": [0.45]}, attrs={"material": WOOD})
    legs = g.add("instance", [leg], params={"t": [[x, 0, z] for x in (-0.7, 0.7) for z in (-0.4, 0.4)],
                                            "r": [[0, 0, 0]] * 4, "s": [[1.0]] * 4, "logit": [3.0] * 4})
    g.output = g.add("union", [top, legs])
    return g


def mug() -> Graph:
    g = Graph()
    outer = g.placed("cylinder", pos=(-0.15, 0, 0), r=[0.5], h=[0.6], attrs={"material": CERAMIC})
    inner = g.placed("cylinder", pos=(-0.15, 0.15, 0), r=[0.42], h=[0.6])
    body = g.add("subtract", [outer, inner])
    handle = g.placed("torus", pos=(0.38, 0, 0), rot=(math.pi / 2, 0, 0), R=[0.3], r=[0.07],
                      attrs={"material": CERAMIC})
    g.output = g.add("union", [body, handle])
    return g


def twisted_column() -> Graph:
    g = Graph()
    base = g.placed("box", pos=(0, -0.8, 0), b=[0.45, 0.1, 0.45], attrs={"material": STONE})
    shaft_p = g.add("box", params={"b": [0.22, 0.7, 0.22], "round": [0.001]}, attrs={"material": STONE})
    shaft = g.add("transform", [g.add("twist", [shaft_p], params={"k": [1.6]})],
                  params={"t": [0, 0, 0], "r": [0, 0, 0], "s": [1.0]})
    cap = g.placed("box", pos=(0, 0.8, 0), b=[0.4, 0.1, 0.4], attrs={"material": STONE})
    g.output = g.add("union", [base, shaft, cap])
    return g


def fence(n_posts: int = 6) -> Graph:
    g = Graph()
    xs = [-0.85 + 1.7 * i / (n_posts - 1) for i in range(n_posts)]
    post = g.add("box", params={"b": [0.06, 0.55, 0.06], "round": [0.001]}, attrs={"material": WOOD})
    posts = g.add("instance", [post], params={"t": [[x, 0, 0] for x in xs], "r": [[0, 0, 0]] * n_posts,
                                              "s": [[1.0]] * n_posts, "logit": [3.0] * n_posts})
    rail = g.add("box", params={"b": [0.95, 0.05, 0.03], "round": [0.001]}, attrs={"material": WOOD})
    rails = g.add("instance", [rail], params={"t": [[0, 0.3, 0.08], [0, -0.2, 0.08]], "r": [[0, 0, 0]] * 2,
                                              "s": [[1.0]] * 2, "logit": [3.0] * 2})
    g.output = g.add("union", [posts, rails])
    return g


def bent_pipe() -> Graph:
    g = Graph()
    cap = g.add("capsule", params={"r": [0.16], "h": [0.7]}, attrs={"material": METAL})
    g.output = g.add("transform", [g.add("bend", [cap], params={"k": [1.3]})],
                     params={"t": [-0.25, 0, 0], "r": [0, 0, 0], "s": [1.0]})
    return g


def bracket() -> Graph:
    """Plate with a row of bolt holes: an instance node *inside* a subtract."""
    g = Graph()
    plate = g.placed("box", pos=(0, 0, 0), b=[0.85, 0.12, 0.35], attrs={"material": METAL})
    hole = g.add("cylinder", params={"r": [0.1], "h": [0.3]})
    holes = g.add("instance", [hole], params={"t": [[-0.55, 0, 0], [0, 0, 0], [0.55, 0, 0]], "r": [[0, 0, 0]] * 3,
                                              "s": [[1.0]] * 3, "logit": [3.0] * 3})
    g.output = g.add("subtract", [plate, holes])
    return g


def stool() -> Graph:
    """Splayed legs: per-instance rotation + per-instance taper (vertex-shader-style deformation)."""
    g = Graph()
    seat = g.placed("cylinder", pos=(0, 0.55, 0), r=[0.45], h=[0.06], attrs={"material": WOOD})
    leg = g.add("cylinder", params={"r": [0.06], "h": [0.55]}, attrs={"material": WOOD})
    ts, rs = [], []
    for i in range(3):
        a = 2 * math.pi * i / 3
        ts.append([0.32 * math.cos(a), -0.05, 0.32 * math.sin(a)])
        tilt = 0.22  # lean outward: rotate about the horizontal axis perpendicular to the radius
        rs.append([tilt * math.sin(a), 0.0, -tilt * math.cos(a)])
    legs = g.add("instance", [leg], params={"t": ts, "r": rs, "s": [[1.0]] * 3, "logit": [3.0] * 3,
                                            "dk": [[-0.5, 0, 0]] * 3})
    g.output = g.add("union", [seat, legs])
    return g


ALL = {
    "table": table,
    "mug": mug,
    "twisted_column": twisted_column,
    "fence": fence,
    "bent_pipe": bent_pipe,
    "bracket": bracket,
    "stool": stool,
}
