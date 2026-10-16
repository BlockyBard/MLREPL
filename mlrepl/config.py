"""JSON (de)serialization for the nested config dataclasses (SearchConfig, FitConfig, ImageFitConfig, ...).

    cfg = SearchConfig(max_evals=200)
    save_config(cfg, "run.json"); cfg2 = load_config(SearchConfig, "run.json")

Unknown keys are rejected (typos in a run config should fail loudly, not silently use defaults).
"""
from __future__ import annotations

import dataclasses
import json
import typing
from typing import Any, Type, TypeVar

T = TypeVar("T")


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if f.name == "library":  # mined macros carry graphs; serialized via their own to_dict
                out[f.name] = [m.to_dict() for m in v]
            else:
                out[f.name] = to_dict(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def from_dict(cls: Type[T], d: dict) -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(cls)
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kw = {}
    for k, v in d.items():
        t = hints.get(k)
        if k == "library":
            from .abstraction import Macro

            kw[k] = [Macro.from_dict(m) for m in v]
        elif dataclasses.is_dataclass(t) and isinstance(v, dict):
            kw[k] = from_dict(t, v)
        elif typing.get_origin(t) is tuple or (isinstance(t, type) and issubclass(t, tuple)) or (
                isinstance(getattr(cls, k, None), tuple)):
            kw[k] = tuple(v)
        else:
            kw[k] = v
    # tuple-typed defaults declared as plain `tuple` annotations
    for f in dataclasses.fields(cls):
        if f.name in kw and isinstance(f.default, tuple) and isinstance(kw[f.name], list):
            kw[f.name] = tuple(kw[f.name])
    return cls(**kw)


def save_config(cfg: Any, path: str):
    with open(path, "w") as f:
        json.dump(to_dict(cfg), f, indent=1)


def load_config(cls: Type[T], path: str) -> T:
    with open(path) as f:
        return from_dict(cls, json.load(f))
