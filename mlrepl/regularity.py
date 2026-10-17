"""Detector bank: proposals that rewrite geometry into regularity ops (array / radial / mirror) or
fit dominant planes. Detectors only *propose*; every proposal is fitted and accepted by the same
description-length score as everything else, so a detector that mis-fires costs search time, not
correctness. There is deliberately no man-made/organic switch here: regularity is kept exactly when
it shortens the program.

Detectors
  pattern_from_instance  instance poses on a line (1-D array), a grid (nested arrays) or a circle
                         (radial) -> pattern op + one prototype pose
  mirror_pairs           two same-structure parts reflected across an axis-aligned plane -> mirror
  symmetrize             target symmetric about an axis plane -> mirror a part across it
  plane_slab             RANSAC plane through target surface near missing volume -> thin box
  pattern_count          n +/- 1 on array / radial nodes
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .graph import Graph
from .ops import EvalCtx
from .targets import Target

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _np(t):
    return t.detach().cpu().double().numpy()


def _rotvec_to_mat(w):
    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def _mat_to_rotvec(R):
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def _mean_rotvec(ws):
    return Rotation.from_rotvec(np.asarray(ws, float)).mean().as_rotvec()


def _frame_y_to(n):
    """Rotation (axis-angle) mapping +Y onto unit vector n."""
    n = n / np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, n)
    s, c = np.linalg.norm(v), float(np.dot(y, n))
    if s < 1e-8:
        return np.zeros(3) if c > 0 else np.array([math.pi, 0.0, 0.0])
    return v / s * math.atan2(s, c)


def fit_line_pattern(T: np.ndarray, tol: float = 0.2):
    """Evenly spaced points on a line? Returns (center, d) with copies at center + (i-(n-1)/2) d."""
    K = len(T)
    if K < 2:
        return None
    c = T.mean(0)
    X = T - c
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    s = X @ u
    order = np.argsort(s)
    gaps = np.diff(s[order])
    step = float(np.median(gaps))
    if step < 1e-4:
        return None
    perp = np.linalg.norm(X - np.outer(s, u), axis=1).max()
    ideal = (np.arange(K) - (K - 1) / 2.0) * step
    dev = np.abs(s[order] - ideal).max()
    if perp > tol * step or dev > tol * step:
        return None
    return c, u * step


def fit_grid_pattern(T: np.ndarray, tol: float = 0.2):
    """a x b grid on two principal axes -> (center, d_outer, n_outer, d_inner, n_inner)."""
    K = len(T)
    c = T.mean(0)
    X = T - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    for a in range(2, K):
        if K % a or K // a < 2:
            continue
        b = K // a
        for u1, u2 in ((Vt[0], Vt[1]), (Vt[1], Vt[0])):
            s1, s2 = X @ u1, X @ u2
            l1, l2 = _levels(s1, a), _levels(s2, b)
            if l1 is None or l2 is None:
                continue
            st1, st2 = np.diff(l1).mean(), np.diff(l2).mean()
            if st1 < 1e-4 or st2 < 1e-4:
                continue
            if np.abs(np.diff(l1) - st1).max() > tol * st1 or np.abs(np.diff(l2) - st2).max() > tol * st2:
                continue
            return c, u1 * st1, a, u2 * st2, b
    return None


def _levels(s, k):
    """Cluster 1-D values into k equal-size levels; return sorted level means or None."""
    o = np.sort(s)
    n = len(o) // k
    groups = [o[i * n:(i + 1) * n] for i in range(k)]
    spread = max(g.max() - g.min() for g in groups)
    means = np.array([g.mean() for g in groups])
    if k > 1 and spread > 0.25 * np.diff(means).min():
        return None
    return means


def fit_circle_pattern(T: np.ndarray, tol: float = 0.2):
    """Evenly spaced points on a circle? Returns (center, normal, radius, angles) or None."""
    K = len(T)
    if K < 3:
        return None
    c0 = T.mean(0)
    X = T - c0
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    nrm = Vt[2]
    if S[2] > 0.15 * S[1]:  # not planar
        return None
    e1, e2 = Vt[0], Vt[1]
    P = np.stack([X @ e1, X @ e2], 1)
    # algebraic circle fit in the plane
    A = np.concatenate([2 * P, np.ones((K, 1))], 1)
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cc = sol[:2]
    R = math.sqrt(max(sol[2] + cc @ cc, 1e-12))
    rad = np.linalg.norm(P - cc, axis=1)
    if np.abs(rad - R).max() > tol * R:
        return None
    ang = np.sort(np.arctan2(P[:, 1] - cc[1], P[:, 0] - cc[0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi]))
    if np.abs(gaps - 2 * math.pi / K).max() > tol * 2 * math.pi / K:
        return None
    center = c0 + cc[0] * e1 + cc[1] * e2
    return center, nrm, R


# ---------------------------------------------------------------------------
# rewrite builders
# ---------------------------------------------------------------------------


def _replace(g: Graph, old: str, new: str):
    g.replace_refs(old, new)
    return g.prune_unreachable()


def _pose_lists(n):
    v = {k: _np(x) for k, x in n.values().items()}
    alive = v["logit"] >= 0
    return v["t"][alive], v["r"][alive], v["s"][alive][:, 0]


def pattern_rewrites(g: Graph, inst_id: str) -> list[tuple[str, Graph]]:
    """All pattern explanations of one instance node's poses (line / grid / circle)."""
    n = g.nodes[inst_id]
    T, Rv, S = _pose_lists(n)
    K = len(T)
    if K < 2:
        return []
    proto = n.inputs[0]
    out = []
    rot = _mean_rotvec(Rv)
    sc = float(np.exp(np.log(np.maximum(S, 1e-6)).mean()))

    def inner(h):
        if np.linalg.norm(rot) < 1e-3 and abs(sc - 1) < 1e-3:
            return proto
        return h.add("transform", [proto], params={"t": [0, 0, 0], "r": rot.tolist(), "s": [sc]})

    line = fit_line_pattern(T)
    if line is not None:
        c, d = line
        h = g.clone()
        arr = h.add("array", [inner(h)], params={"d": d.tolist()}, attrs={"n": K})
        top = h.add("transform", [arr], params={"t": c.tolist(), "r": [0, 0, 0], "s": [1.0]})
        out.append(("array", _replace(h, inst_id, top)))
    grid = fit_grid_pattern(T) if K >= 4 else None
    if grid is not None:
        c, d1, a, d2, b = grid
        h = g.clone()
        a_in = h.add("array", [inner(h)], params={"d": d2.tolist()}, attrs={"n": b})
        a_out = h.add("array", [a_in], params={"d": d1.tolist()}, attrs={"n": a})
        top = h.add("transform", [a_out], params={"t": c.tolist(), "r": [0, 0, 0], "s": [1.0]})
        out.append(("grid", _replace(h, inst_id, top)))
    circ = fit_circle_pattern(T) if K >= 3 else None
    if circ is not None:
        c, nrm, _R = circ
        w_out = _frame_y_to(nrm)
        R_out = _rotvec_to_mat(w_out)
        # prototype pose = instance 0 expressed in the radial frame; copies are its rotations about local Y
        t0 = R_out.T @ (T[0] - c)
        r0 = _mat_to_rotvec(R_out.T @ _rotvec_to_mat(Rv[0]))
        h = g.clone()
        pin = h.add("transform", [proto], params={"t": t0.tolist(), "r": r0.tolist(), "s": [float(S[0])]})
        rad = h.add("radial", [pin], attrs={"n": K})
        top = h.add("transform", [rad], params={"t": c.tolist(), "r": w_out.tolist(), "s": [1.0]})
        out.append(("radial", _replace(h, inst_id, top)))
    return out


def _world_pose(g: Graph, nid: str):
    """(prim/subgraph id, translation, rotvec) for a transform(...) part; None otherwise."""
    n = g.nodes[nid]
    if n.op != "transform":
        return None
    v = {k: _np(x) for k, x in n.values().items()}
    return n.inputs[0], v["t"], v["r"]


def mirror_pair_rewrites(g: Graph, rel_tol: float = 0.12) -> list[tuple[str, Graph]]:
    """Two top-level parts with the same structure whose positions are reflections across an
    axis-aligned plane -> mirror(one of them)."""
    from .search import _parts, signature

    root = g.nodes[g.output]
    if root.op != "union":
        return []
    parts = _parts(g)
    out = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = _world_pose(g, parts[i]), _world_pose(g, parts[j])
            if a is None or b is None:
                continue
            sa, sb = _sub(g, a[0]), _sub(g, b[0])
            if signature(sa) != signature(sb) or not _similar_params(sa, sb):
                continue
            ta, tb = a[1], b[1]
            diff = np.abs(ta - tb)
            ax = int(np.argmax(diff))
            scale = max(diff[ax], 1e-3)
            if np.delete(diff, ax).max() > rel_tol * scale + 0.02:
                continue
            c = 0.5 * (ta[ax] + tb[ax])
            h = g.clone()
            mir = h.add("mirror", [parts[i]], params={"c": [float(c)]}, attrs={"axis": ax})
            hr = h.nodes[h.output]
            hr.inputs = [mir if x == parts[i] else x for x in hr.inputs if x != parts[j]]
            if len(hr.inputs) == 1:
                h.replace_refs(hr.id, hr.inputs[0])
            out.append(("mirror", h.prune_unreachable()))
    return out


def _param_vector(g: Graph) -> np.ndarray:
    return np.concatenate([_np(v).ravel() for nid in g.topo() for k, v in g.nodes[nid].values().items()
                           if k != "color"] or [np.zeros(0)])


def _similar_params(a: Graph, b: Graph, rel: float = 0.15) -> bool:
    """Same-structure subgraphs whose parameters (sizes, inner poses) agree within `rel`."""
    va, vb = _param_vector(a), _param_vector(b)
    if va.shape != vb.shape:
        return False
    return bool(np.all(np.abs(va - vb) <= rel * np.maximum(np.abs(va), np.abs(vb)) + 0.02))


def _sub(g: Graph, root: str) -> Graph:
    return Graph([g.nodes[n].clone() for n in g.topo(root)], root, device=g.device)


def symmetry_planes(target: Target, thresh: float = 0.88, n: int = 3000, search: float = 0.12,
                    steps: int = 25) -> list[tuple[int, float]]:
    """Axis-aligned mirror planes of the target. For each axis, scan plane offsets around the
    occupancy centroid; a plane counts if reflected inside samples land inside (label of the nearest
    labelled sample, volume + near-surface pools) more than `thresh` of the time."""
    from scipy.spatial import cKDTree

    P = torch.cat([target.eval_points, target.points]).detach().cpu().numpy()
    occ = (torch.cat([target.eval_occ, target.occ]) > 0.5).detach().cpu().numpy()
    inside = P[occ]
    if len(inside) < 50:
        return []
    sel = inside[np.random.default_rng(0).choice(len(inside), min(n, len(inside)), replace=False)]
    tree = cKDTree(P)
    out = []
    for ax in range(3):
        c0 = float(inside[:, ax].mean())
        best = (0.0, c0)
        for c in np.linspace(c0 - search, c0 + search, steps):
            Q = sel.copy()
            Q[:, ax] = 2 * c - Q[:, ax]
            _, j = tree.query(Q)
            agree = float(occ[j].mean())
            if agree > best[0]:
                best = (agree, float(c))
        if best[0] > thresh:
            out.append((ax, best[1]))
    return out


def symmetrize_rewrites(g: Graph, planes: list) -> list[tuple[str, Graph]]:
    """Mirror a top-level part across a detected target symmetry plane, if the part sits on one side."""
    from .search import _parts

    out = []
    for pid in _parts(g):
        n = g.nodes[pid]
        pose = _world_pose(g, pid)
        for ax, c in planes:
            if pose is not None and abs(pose[1][ax] - c) < 0.05:  # straddles the plane: already symmetric
                continue
            h = g.clone()
            mir = h.add("mirror", [pid], params={"c": [c]}, attrs={"axis": ax})
            if h.output == pid:
                h.output = mir
            else:
                h.nodes[h.output].inputs = [mir if x == pid else x for x in h.nodes[h.output].inputs]
            out.append(("symmetrize", h))
    return out


def ransac_plane(S: np.ndarray, rng, iters: int = 128, thresh: float = 0.015):
    best = (0, None)
    if len(S) < 3:
        return None
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        inl = np.abs((S - a) @ nrm) < thresh
        if inl.sum() > best[0]:
            best = (int(inl.sum()), (a, nrm, inl))
    return best[1]


def plane_slab(g: Optional[Graph], part: np.ndarray, target: Target, rng) -> Optional[Graph]:
    """Missing region -> dominant plane of nearby target surface -> thin box spanning the plane's inliers."""
    lo, hi = part.min(0) - 0.05, part.max(0) + 0.05
    S = target.surface.detach().cpu().numpy()
    S = S[((S >= lo) & (S <= hi)).all(1)]
    if len(S) < 30:
        return None
    pl = ransac_plane(S, rng)
    if pl is None:
        return None
    a, nrm, inl = pl
    if inl.sum() < 20:
        return None
    Pin = S[inl]
    # thickness: extent of the missing volume along the normal
    h = (part - a) @ nrm
    thick = max(float(np.percentile(h, 95) - np.percentile(h, 5)) * 0.5, 0.01)
    mid = a + nrm * float(np.median(h))
    X = Pin - Pin.mean(0)
    X -= np.outer(X @ nrm, nrm)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0] - (Vt[0] @ nrm) * nrm
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    ext1 = 0.5 * float(np.ptp(X @ e1))
    ext2 = 0.5 * float(np.ptp(X @ e2))
    center = Pin.mean(0) - ((Pin.mean(0) - mid) @ nrm) * nrm
    M = np.stack([e1, nrm, e2], 1)
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1
    rv = Rotation.from_matrix(M).as_rotvec()
    from .search import union_with

    h = g.clone() if g is not None else Graph()
    box = h.placed("box", pos=center.tolist(), rot=rv.tolist(), b=[max(ext1, 0.01), thick, max(ext2, 0.01)],
                   round=[1e-3])
    if g is None:
        h.output = box
    else:
        union_with(h, box)
    return h


# ---------------------------------------------------------------------------
# Data-side pattern detection: repeated *missing* pieces -> pattern op with one fitted prototype.
# Works before any instance node exists, so the search doesn't have to stumble into a pattern-exact
# intermediate program first.
# ---------------------------------------------------------------------------


def _shape_feature(P: np.ndarray):
    c = P.mean(0)
    w = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((P - c).T) + 1e-9 * np.eye(3)), 0))[::-1]
    return c, w, len(P)


def component_groups(parts: list, rel: float = 0.35, min_size: int = 2) -> list[list[int]]:
    """Greedy grouping of point-cloud components with similar size (principal std devs, point count)."""
    feats = [_shape_feature(P) for P in parts]
    groups: list[list[int]] = []
    for i, (_, w, n) in enumerate(feats):
        for grp in groups:
            _, w0, n0 = feats[grp[0]]
            if np.all(np.abs(w - w0) <= rel * w0 + 0.01) and 0.5 <= n / max(n0, 1) <= 2.0:
                grp.append(i)
                break
        else:
            groups.append([i])
    return sorted([grp for grp in groups if len(grp) >= min_size], key=len, reverse=True)


def data_pattern_proposals(g: Optional[Graph], parts: list, target: Target, rng, kind: Optional[str] = None
                           ) -> list[tuple[str, Graph]]:
    """For groups of similar missing components (largest first): fit a line / grid / circle to their
    centroids and propose pattern(placed prototype) unioned into g. Returns the proposals of the first
    group that admits any pattern."""

    groups = component_groups(parts)
    if not groups:
        return []
    out_all = []
    for grp in groups[:3]:
        out_all += _group_proposals(g, [parts[i] for i in grp], target, rng, kind)
        if out_all:
            break
    return out_all


def _group_proposals(g, members, target, rng, kind):
    from .search import best_kind, prim_from_points, union_with

    C = np.array([m.mean(0) for m in members])
    K = len(members)
    rep = members[int(np.argsort([len(m) for m in members])[K // 2])]
    if kind is None:
        kind = best_kind(rep, target, rng, g) if g is not None else str(rng.choice(["box", "cylinder"]))
    params, _, rot = prim_from_points(kind, rep - rep.mean(0))
    out = []

    def new_graph():
        h = g.clone() if g is not None else Graph()
        p = h.add(kind, params=params)
        return h, p

    def finish(h, top, name):
        if g is None:
            h.output = top
        else:
            union_with(h, top)
        out.append((name, h))

    line = fit_line_pattern(C, tol=0.25)
    if line is not None:
        c, d = line
        h, p = new_graph()
        inner = h.add("transform", [p], params={"t": [0, 0, 0], "r": rot.tolist(), "s": [1.0]})
        arr = h.add("array", [inner], params={"d": d.tolist()}, attrs={"n": K})
        finish(h, h.add("transform", [arr], params={"t": c.tolist(), "r": [0, 0, 0], "s": [1.0]}), "data_array")
    grid = fit_grid_pattern(C, tol=0.25) if K >= 4 else None
    if grid is not None:
        c, d1, a, d2, b = grid
        h, p = new_graph()
        inner = h.add("transform", [p], params={"t": [0, 0, 0], "r": rot.tolist(), "s": [1.0]})
        a_in = h.add("array", [inner], params={"d": d2.tolist()}, attrs={"n": b})
        a_out = h.add("array", [a_in], params={"d": d1.tolist()}, attrs={"n": a})
        finish(h, h.add("transform", [a_out], params={"t": c.tolist(), "r": [0, 0, 0], "s": [1.0]}), "data_grid")
    circ = fit_circle_pattern(C, tol=0.25) if K >= 3 else None
    if circ is not None:
        c, nrm, _R = circ
        w_out = _frame_y_to(nrm)
        R_out = _rotvec_to_mat(w_out)
        _, _, rot0 = prim_from_points(kind, members[0] - members[0].mean(0))
        t0 = R_out.T @ (C[0] - c)
        r0 = _mat_to_rotvec(R_out.T @ _rotvec_to_mat(rot0))
        h, p = new_graph()
        pin = h.add("transform", [p], params={"t": t0.tolist(), "r": r0.tolist(), "s": [1.0]})
        rad = h.add("radial", [pin], attrs={"n": K})
        finish(h, h.add("transform", [rad], params={"t": c.tolist(), "r": w_out.tolist(), "s": [1.0]}),
               "data_radial")
    return out


def _periodic_peaks(x: np.ndarray, bins: int = 64, min_peaks: int = 2, tol: float = 0.3):
    """Evenly spaced density peaks of a 1-D sample (posts along a fence, teeth around a gear).
    Returns sorted peak positions, or None if the profile isn't periodic."""
    if len(x) < 30:
        return None
    hist, edges = np.histogram(x, bins=bins)
    h = np.convolve(hist, np.ones(3) / 3, mode="same")
    thr = h.mean() + 0.5 * h.std()
    pk = [i for i in range(1, bins - 1) if h[i] >= h[i - 1] and h[i] > h[i + 1] and h[i] > thr]
    if len(pk) < min_peaks:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])[pk]
    gaps = np.diff(centers)
    if len(gaps) and np.abs(gaps - np.median(gaps)).max() > tol * np.median(gaps):
        return None
    return centers


def split_periodic(part: np.ndarray, min_pts: int = 60) -> list:
    """Split one connected missing region into repeated pieces when its density along a principal axis
    (or its angle around the region's normal axis) is periodic: rails joining posts, a lintel joining
    columns, adjacent gear teeth merged by the residual grid. Returns [] when nothing periodic is found."""
    if len(part) < min_pts:
        return []
    c = part.mean(0)
    X = part - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    for ax in Vt[:2]:  # linear periodicity along the two main axes
        s = X @ ax
        pk = _periodic_peaks(s)
        if pk is None or len(pk) < 3:
            continue
        half = 0.3 * np.median(np.diff(pk))
        pieces = [part[np.abs(s - p) < half] for p in pk]
        pieces = [q for q in pieces if len(q) >= 8]
        if len(pieces) >= 3:
            return pieces
    # angular periodicity around the axis normal to the region's main plane
    nrm = Vt[2]
    e1, e2 = Vt[0], Vt[1]
    ang = np.arctan2(X @ e2, X @ e1)
    rad = np.linalg.norm(np.stack([X @ e1, X @ e2], 1), axis=1)
    for keep in (np.ones(len(part), bool), rad > np.percentile(rad, 40)):  # whole ring, then rim only
        pk = _periodic_peaks(ang[keep], bins=96, min_peaks=3)
        if pk is None or len(pk) < 3:
            continue
        half = 0.3 * np.median(np.diff(pk))
        sel, a = part[keep], ang[keep]
        pieces = [sel[np.abs(np.angle(np.exp(1j * (a - p)))) < half] for p in pk]
        pieces = [q for q in pieces if len(q) >= 8]
        if len(pieces) >= 3:
            return pieces
    return []


def candidate_piece_sets(parts: list) -> list:
    """Alternative decompositions of the missing volume, kept separate so pieces of different
    decompositions never get grouped together: the residual components themselves, then the periodic
    split of each component and of all components together (teeth merged pairwise by the residual
    grid only show their period as a whole)."""
    sets = [list(parts)] if len(parts) >= 2 else []
    regions = list(parts) + ([np.concatenate(parts)] if len(parts) > 1 else [])
    for P in regions:
        sp = split_periodic(P)
        if len(sp) >= 2:
            sets.append(sp)
    return sets


def candidate_pieces(parts: list) -> list:
    """Flat list of all candidate pieces (for inspection/tests)."""
    return [p for s in candidate_piece_sets(parts) for p in s]


def make_m_data_pattern(target: Target):
    def m(g, res, rng):
        props = []
        for pieces in candidate_piece_sets(res.fn_parts):
            props += data_pattern_proposals(g, pieces, target, rng)
        return props[rng.integers(len(props))][1] if props else None

    return m


# ---------------------------------------------------------------------------
# mutation entry points (signature: (g, residual, rng) -> Graph | None)
# ---------------------------------------------------------------------------


def m_pattern(g, res, rng):
    insts = [nid for nid in g.reachable() if g.nodes[nid].spec.category == "inst" and g.nodes[nid].k >= 2]
    rewrites = []
    for nid in insts:
        rewrites += pattern_rewrites(g, nid)
    if not rewrites:
        return None
    return rewrites[rng.integers(len(rewrites))][1]


def m_mirror(g, res, rng):
    rw = mirror_pair_rewrites(g)
    return rw[rng.integers(len(rw))][1] if rw else None


def make_m_symmetrize(target):
    planes = symmetry_planes(target)

    def m(g, res, rng):
        if not planes:
            return None
        rw = symmetrize_rewrites(g, planes)
        return rw[rng.integers(len(rw))][1] if rw else None

    m.planes = planes
    return m


def make_m_plane(target):
    def m(g, res, rng):
        if not res.fn_parts:
            return None
        part = res.fn_parts[0] if rng.random() < 0.6 else res.fn_parts[rng.integers(len(res.fn_parts))]
        return plane_slab(g, part, target, rng)

    return m


def m_count(g, res, rng):
    pats = [nid for nid in g.reachable() if g.nodes[nid].op in ("array", "radial")]
    if not pats:
        return None
    h = g.clone()
    n = h.nodes[pats[rng.integers(len(pats))]]
    k = int(n.attrs["n"]) + (1 if rng.random() < 0.5 else -1)
    if k < (2 if n.op == "radial" else 1):
        return None
    n.attrs = dict(n.attrs, n=k)
    return h


def deterministic_rewrites(g: Graph) -> list[tuple[str, Graph]]:
    """All pattern / mirror compressions of g (used by the final simplification sweep)."""
    out = []
    for nid in g.reachable():
        if g.nodes[nid].spec.category == "inst" and g.nodes[nid].k >= 2:
            out += pattern_rewrites(g, nid)
    return out + mirror_pair_rewrites(g)
