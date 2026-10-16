import json
import math
import os

import pytest
import torch

from mlrepl import library
from mlrepl.config import load_config, save_config
from mlrepl.fit import FitConfig, clamp_params, fit, iou
from mlrepl.graph import Graph, GraphError
from mlrepl.ops import EvalCtx
from mlrepl.search import SearchConfig, search
from mlrepl.targets import from_graph

TINY = FitConfig(steps=30, nv=512, ns=128)


@pytest.fixture(scope="module")
def bracket_target():
    return from_graph(library.bracket(), n_eval=8000)


# ---------------------------------------------------------------- validation


def test_validate_accepts_library():
    for f in library.ALL.values():
        f().validate()


@pytest.mark.parametrize("mutate,msg", [
    (lambda g: g.nodes[g.output].inputs.append("nope"), "does not exist"),
    (lambda g: g.nodes[g.nodes[g.output].inputs[0]].inputs.append(g.output), "cycle"),
    (lambda g: g.nodes[g.output].inputs.append(g.nodes[g.output].inputs[0]) or
     setattr(g.nodes[g.output], "op", "subtract"), "takes 2"),
    (lambda g: g.nodes["box0"].raw.__setitem__("b", torch.zeros(2)), "shape"),
    (lambda g: g.nodes["box0"].raw.__setitem__("b", torch.full((3,), float("nan"))), "non-finite"),
])
def test_validate_rejects_broken_graphs(mutate, msg):
    g = library.bracket()
    mutate(g)
    with pytest.raises(GraphError, match=msg):
        g.validate()


def test_bad_json_fails_early():
    with pytest.raises(GraphError, match="unknown op"):
        Graph.from_dict({"output": "a", "nodes": [{"id": "a", "op": "teapot"}]})
    with pytest.raises(GraphError, match="unknown param"):
        Graph.from_dict({"output": "a", "nodes": [{"id": "a", "op": "sphere", "params": {"radius": [1]}}]})


# ---------------------------------------------------------------- robust fitting


def test_rotation_wrap_preserves_shape():
    g = library.bracket()
    tid = next(n for n in g.nodes if g.nodes[n].op == "transform")
    g.nodes[tid].set_value("r", torch.tensor([0.0, 2 * math.pi + 0.3, 0.0]))
    P = torch.rand(2000, 3) * 2 - 1
    before = g.sdf(P)
    clamp_params(g, 10.0)
    assert g.nodes[tid].value("r").norm() <= math.pi + 1e-5
    assert torch.allclose(before, g.sdf(P), atol=1e-4)


def test_fit_survives_nan_steps(bracket_target, monkeypatch):
    import mlrepl.fit as F

    real = F.loss_terms
    calls = {"n": 0}

    def flaky(*a, **k):
        loss, t = real(*a, **k)
        calls["n"] += 1
        if calls["n"] in (3, 4, 9):
            loss = loss * float("nan")
        return loss, t

    monkeypatch.setattr(F, "loss_terms", flaky)
    r = F.fit(library.bracket(), bracket_target, TINY)
    assert r.nan_events == 3
    assert all(torch.isfinite(t).all() for n in r.graph.nodes.values() for t in n.raw.values())
    assert r.iou > 0.9


def test_param_groups_and_early_stop(bracket_target):
    from mlrepl.fit import _make_opt

    opt = _make_opt(library.bracket(), FitConfig())
    names = {grp["name"] for grp in opt.param_groups}
    assert {"t", "r", "size"} <= names
    assert FitConfig().early_stop is False  # off by default (see FitConfig comment for the measurement)
    long = FitConfig(steps=300, nv=512, ns=128)
    r_full = fit(library.bracket(), bracket_target, long)
    r_es = fit(library.bracket(), bracket_target, FitConfig(**{**long.__dict__, "early_stop": True,
                                                              "plateau_window": 60, "plateau_tol": 0.0}))
    assert r_full.steps_run == 300 and r_es.steps_run <= 300
    assert r_es.iou > r_full.iou - 0.03


# ---------------------------------------------------------------- config / checkpoint / CLI


def test_config_roundtrip(tmp_path):
    cfg = SearchConfig(max_evals=17, fit=FitConfig(steps=11), weights={"add": 2.0})
    save_config(cfg, tmp_path / "c.json")
    assert load_config(SearchConfig, tmp_path / "c.json") == cfg
    with open(tmp_path / "bad.json", "w") as f:
        json.dump({"beem": 3}, f)
    with pytest.raises(ValueError, match="unknown"):
        load_config(SearchConfig, tmp_path / "bad.json")


def _tiny_search(**kw):
    return SearchConfig(beam=2, proposals_per_parent=2, patience=50, simplify=False, fit=TINY, final_fit=TINY, **kw)


def test_checkpoint_resume_matches_uninterrupted(bracket_target, tmp_path):
    full = search(bracket_target, _tiny_search(generations=4, max_evals=0), log=lambda *_: None)
    d = str(tmp_path / "run")
    search(bracket_target, _tiny_search(generations=2), log=lambda *_: None, checkpoint_dir=d)
    st = json.load(open(os.path.join(d, "state.json")))
    st["phase"] = "search"  # pretend the process was killed mid-search
    json.dump(st, open(os.path.join(d, "state.json"), "w"))
    res = search(bracket_target, _tiny_search(generations=4), log=lambda *_: None, checkpoint_dir=d, resume=True)
    assert res.evaluated == full.evaluated
    assert abs(res.best.score - full.best.score) < 0.02
    lines = [json.loads(x) for x in open(os.path.join(d, "log.jsonl"))]
    assert [x["gen"] for x in lines if "gen" in x] == [1, 2, 3, 4]


def test_resume_rejects_other_target(bracket_target, tmp_path):
    d = str(tmp_path / "run")
    search(bracket_target, _tiny_search(generations=1), log=lambda *_: None, checkpoint_dir=d)
    st = json.load(open(os.path.join(d, "state.json")))
    st["phase"] = "search"
    json.dump(st, open(os.path.join(d, "state.json"), "w"))
    with pytest.raises(ValueError, match="different target"):
        search(from_graph(library.mug(), n_eval=8000), _tiny_search(generations=2), log=lambda *_: None,
               checkpoint_dir=d, resume=True)


def test_cli_search_and_resume(tmp_path, capsys):
    from mlrepl.__main__ import main

    out = str(tmp_path / "cli")
    cfgp = str(tmp_path / "cfg.json")
    save_config(_tiny_search(generations=1), cfgp)
    main(["search", "library:bracket", "--out", out, "--config", cfgp, "--n-eval", "8000", "--no-preview",
          "--device", "cpu"])
    for f in ("best.json", "best.glb", "summary.json", "state.json", "log.jsonl", "target.pt", "config.json"):
        assert os.path.exists(os.path.join(out, f)), f
    main(["validate", os.path.join(out, "best.json")])
    main(["search", out, "--resume", "--generations", "2", "--no-preview", "--device", "cpu"])
    assert json.load(open(os.path.join(out, "summary.json")))["evaluated"] > 0


# ---------------------------------------------------------------- faster evaluation


def test_process_evaluator_matches_serial(bracket_target):
    from mlrepl.parallel import ProcessEvaluator

    cfg = _tiny_search(generations=1, max_evals=8)
    serial = search(bracket_target, cfg, log=lambda *_: None)
    ev = ProcessEvaluator(2)
    try:
        par = search(bracket_target, cfg, log=lambda *_: None, evaluator=ev)
    finally:
        ev.close()
    assert par.evaluated == serial.evaluated
    assert abs(par.best.score - serial.best.score) < 1e-3


def test_multistart_is_no_worse_than_its_first_start(bracket_target):
    from mlrepl.parallel import fit_multistart

    g = library.bracket()
    tid = next(n for n in g.nodes if g.nodes[n].op == "transform")
    g.nodes[tid].set_value("t", torch.tensor([0.25, 0.1, -0.1]))
    best, scores = fit_multistart(g, bracket_target, starts=4, cfg=FitConfig(steps=40, nv=512, ns=128, early_stop=False))
    assert len(scores) == 4 and best.score == min(scores)
    assert best.iou > 0.9
