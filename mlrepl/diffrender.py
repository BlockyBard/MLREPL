"""Differentiable rendering of graph SDFs: silhouette, depth, normals and shaded color.

Per ray:
  1. sphere trace without gradients -> hit distance t*, plus the ray point where the SDF is smallest
     (sampling a window *inside* the surface for hit rays, so the minimum is negative there)
  2. soft silhouette  m = sigmoid(-f(x_min) / sigma)      -> gradients at and near silhouette edges
  3. hit point with implicit differentiation (IDR-style):
        t = t* - (f(x*) - sg[f(x*)]) / sg[grad f . d]     -> exact first-order dt/dtheta
     depth, normals (finite differences of f, differentiable) and albedo (color field) at x(t)
  4. shading: albedo * (ambient + (1 - ambient) * max(0, n . L))

Cameras follow OpenCV conventions (x right, y down, z forward); `render.look_at` builds them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .graph import Graph
from .ops import EvalCtx
from .render import intrinsics, look_at


@dataclass
class Camera:
    cam_to_world: np.ndarray
    K: np.ndarray
    H: int
    W: int

    def rays(self, idx: Optional[torch.Tensor] = None, device=None):
        """Return (origins, unit dirs, zscale) for flat pixel indices idx (default: all)."""
        if idx is None:
            idx = torch.arange(self.H * self.W)
        idx = idx.cpu()
        ys, xs = (idx // self.W).double(), (idx % self.W).double()
        K = torch.as_tensor(self.K)
        d = torch.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
        M = torch.as_tensor(self.cam_to_world)
        d = d @ M[:3, :3].T
        n = d.norm(dim=-1, keepdim=True)
        o = M[:3, 3].expand_as(d)
        dev = device or "cpu"
        return o.float().to(dev), (d / n).float().to(dev), (1.0 / n[:, 0]).float().to(dev)


def orbit_cameras(n: int, res: int = 64, radius: float = 3.2, elevations=(0.3, -0.45, 0.9), fov: float = 40.0,
                  offset: float = 0.3) -> list[Camera]:
    cams = []
    for i in range(n):
        a = offset + 2 * math.pi * i / n
        e = elevations[i % len(elevations)]
        eye = (radius * math.cos(a) * math.cos(e), radius * math.sin(e), radius * math.sin(a) * math.cos(e))
        cams.append(Camera(look_at(eye), intrinsics(res, fov), res, res))
    return cams


@dataclass
class RenderOpts:
    steps: int = 64
    far: float = 7.0
    eps_hit: float = 1e-3
    step_scale: float = 0.8
    inside_window: float = 0.3
    inside_samples: int = 6
    sigma: float = 0.01  # soft silhouette width (scene units)
    ambient: float = 0.3
    light: tuple = (0.4, 0.8, 0.45)
    fd_eps: float = 1e-3
    normal_grad: bool = False


@torch.no_grad()
def _trace(g: Graph, o, d, opts: RenderOpts, ctx: EvalCtx):
    N, dev = len(o), o.device
    t = torch.zeros(N, device=dev)
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    best_s = torch.full((N,), float("inf"), device=dev)
    best_t = torch.zeros(N, device=dev)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    for _ in range(opts.steps):
        idx = alive.nonzero()[:, 0]
        if len(idx) == 0:
            break
        s = g.sdf(o[idx] + d[idx] * t[idx, None], ctx)
        better = s < best_s[idx]
        best_s[idx] = torch.where(better, s, best_s[idx])
        best_t[idx] = torch.where(better, t[idx], best_t[idx])
        h = s < opts.eps_hit
        hit[idx[h]] = True
        t[idx] = torch.where(h, t[idx], t[idx] + opts.step_scale * s.clamp(min=1e-3))
        alive[idx] = ~h & (t[idx] < opts.far)
    # inside window for hit rays: the minimum along the ray is inside the solid
    hi = hit.nonzero()[:, 0]
    if len(hi):
        offs = torch.linspace(0, opts.inside_window, opts.inside_samples + 1, device=dev)[1:]
        T = t[hi, None] + offs[None]
        P = o[hi, None] + d[hi, None] * T[..., None]
        s = g.sdf(P.reshape(-1, 3), ctx).reshape(len(hi), -1)
        smin, j = s.min(1)
        better = smin < best_s[hi]
        best_t[hi] = torch.where(better, T[torch.arange(len(hi), device=dev), j], best_t[hi])
    return t, hit, best_t


def render(g: Graph, o, d, zscale, ctx: Optional[EvalCtx] = None, opts: Optional[RenderOpts] = None,
           want_color: bool = True) -> dict:
    """Differentiable w.r.t. graph parameters. Returns per-ray mask (soft), hit (bool), depth,
    normal, albedo, rgb. Values for non-hit rays are zero."""
    ctx = ctx or EvalCtx()
    opts = opts or RenderOpts()
    trace_ctx = EvalCtx(tau=ctx.tau, gate_scale=ctx.gate_scale, hard_gates=ctx.hard_gates)
    t_star, hit, t_best = _trace(g, o, d, opts, trace_ctx)
    N = len(o)
    out = {"hit": hit, "t": t_star}
    xb = (o + d * t_best[:, None]).detach()
    out["mask"] = torch.sigmoid(-g.sdf(xb, trace_ctx) / opts.sigma)
    dev = o.device
    depth = torch.zeros(N, device=dev)
    normal = torch.zeros(N, 3, device=dev)
    albedo = torch.zeros(N, 3, device=dev)
    rgb = torch.zeros(N, 3, device=dev)
    hi = hit.nonzero()[:, 0]
    if len(hi):
        oh, dh = o[hi], d[hi]
        xs = (oh + dh * t_star[hi, None]).detach()
        f = g.sdf(xs, trace_ctx)
        e = opts.fd_eps
        E = torch.eye(3, device=dev) * e
        with torch.no_grad():
            grad = torch.stack([g.sdf(xs + E[i], trace_ctx) - g.sdf(xs - E[i], trace_ctx) for i in range(3)], -1)
            grad = grad / (2 * e)
            nd = (grad * dh).sum(-1)
            nd = torch.where(nd < 0, torch.clamp(nd, max=-1e-2), torch.clamp(nd, min=1e-2))
        th = t_star[hi] - (f - f.detach()) / nd
        x = oh + dh * th[:, None]
        if opts.normal_grad:
            n = torch.stack([g.sdf(x + E[i], trace_ctx) - g.sdf(x - E[i], trace_ctx) for i in range(3)], -1)
        else:  # shading normals without gradient: 6 fewer differentiable graph evals per step
            n = grad
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
        depth = depth.index_put((hi,), th * zscale[hi])
        normal = normal.index_put((hi,), n)
        if want_color:
            cctx = EvalCtx(tau=ctx.tau, gate_scale=ctx.gate_scale, hard_gates=ctx.hard_gates, color=True)
            alb = g.sdf(x, cctx)[:, 1:]
            L = torch.tensor(opts.light, dtype=torch.float32, device=dev)
            L = L / L.norm()
            shade = opts.ambient + (1 - opts.ambient) * torch.clamp((n * L).sum(-1), min=0)
            albedo = albedo.index_put((hi,), alb)
            rgb = rgb.index_put((hi,), alb * shade[:, None])
    out.update(depth=depth, normal=normal, albedo=albedo, rgb=rgb)
    return out


# ---------------------------------------------------------------------------
# Multi-view image targets
# ---------------------------------------------------------------------------


@dataclass
class View:
    cam: Camera
    rgb: torch.Tensor  # (H*W, 3)
    mask: torch.Tensor  # (H*W,)
    depth: Optional[torch.Tensor] = None  # (H*W,), 0 = background


@dataclass
class ImageTarget:
    views: list
    background: tuple = (1.0, 1.0, 1.0)

    @property
    def device(self):
        return self.views[0].rgb.device

    def to(self, device) -> "ImageTarget":
        for v in self.views:
            v.rgb, v.mask = v.rgb.to(device), v.mask.to(device)
            if v.depth is not None:
                v.depth = v.depth.to(device)
        return self

    def sample(self, n: int, gen=None, fg_frac: float = 0.5):
        """Random rays across views, half drawn from foreground pixels (else thin parts are rarely hit)."""
        per = max(1, n // len(self.views))
        O, D, Z, RGB, M, DEP, HASD = [], [], [], [], [], [], []
        for v in self.views:
            HW = v.cam.H * v.cam.W
            nf = int(per * fg_frac)
            fg = v.mask.nonzero()[:, 0].cpu()
            idx = torch.randint(HW, (per - nf,), generator=gen)
            if len(fg):
                idx = torch.cat([idx, fg[torch.randint(len(fg), (nf,), generator=gen)]])
            dev = v.rgb.device
            o, d, z = v.cam.rays(idx, dev)
            idx = idx.to(dev)
            O.append(o), D.append(d), Z.append(z), RGB.append(v.rgb[idx]), M.append(v.mask[idx])
            if v.depth is not None:
                DEP.append(v.depth[idx]), HASD.append(torch.ones(len(idx), dtype=torch.bool, device=dev))
            else:
                DEP.append(torch.zeros(len(idx), device=dev)), HASD.append(torch.zeros(len(idx), dtype=torch.bool,
                                                                                          device=dev))
        cat = torch.cat
        return cat(O), cat(D), cat(Z), cat(RGB), cat(M), cat(DEP), cat(HASD)


@torch.no_grad()
def render_view(g: Graph, cam: Camera, opts: Optional[RenderOpts] = None, chunk: int = 4096) -> dict:
    outs = []
    o, d, z = cam.rays(device=g.device)
    ctx = EvalCtx(hard_gates=True)
    for s in range(0, len(o), chunk):
        outs.append(render(g, o[s:s + chunk], d[s:s + chunk], z[s:s + chunk], ctx, opts))
    return {k: torch.cat([x[k] for x in outs]) for k in outs[0]}


def make_views(g: Graph, cams: list[Camera], with_depth: bool = True, opts: Optional[RenderOpts] = None,
               background=(1.0, 1.0, 1.0)) -> ImageTarget:
    views = []
    bg = torch.tensor(background, device=g.device)
    for cam in cams:
        r = render_view(g, cam, opts)
        hit = r["hit"]
        rgb = torch.where(hit[:, None], r["rgb"], bg)
        views.append(View(cam, rgb, hit.float(), r["depth"] if with_depth else None))
    return ImageTarget(views, background)


def view_metrics(g: Graph, target: ImageTarget, opts: Optional[RenderOpts] = None) -> dict:
    """Mask IoU, depth MAE (on co-visible pixels), PSNR over all pixels (background composited)."""
    ious, maes, psnrs = [], [], []
    bg = torch.tensor(target.background, device=g.device)
    for v in target.views:
        r = render_view(g, v.cam, opts)
        pm, gm = r["hit"], v.mask > 0.5
        ious.append(float((pm & gm).sum() / max(int((pm | gm).sum()), 1)))
        if v.depth is not None and (pm & gm).any():
            maes.append(float((r["depth"] - v.depth)[pm & gm].abs().mean()))
        rgb = torch.where(pm[:, None], r["rgb"], bg)
        mse = float(((rgb - v.rgb) ** 2).mean())
        psnrs.append(10 * math.log10(1.0 / max(mse, 1e-10)))
    return {"mask_iou": float(np.mean(ious)), "depth_mae": float(np.mean(maes)) if maes else float("nan"),
            "psnr": float(np.mean(psnrs))}


# ---------------------------------------------------------------------------
# Views -> volumetric proxy target (for structure search) by carving
# ---------------------------------------------------------------------------


def carve(target: ImageTarget, lo=(-1.1, -1.1, -1.1), hi=(1.1, 1.1, 1.1), res: int = 72,
          depth_margin: float = 0.02) -> np.ndarray:
    """Occupancy grid: a point survives if every view that sees it projects it inside the mask, and (when
    depth is available) not in front of the observed surface. Masks alone give the visual hull
    (concavities are filled); depth carves visible concavities."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    occ = np.ones(len(P), bool)
    for v in target.views:
        M = v.cam.cam_to_world
        R, t = M[:3, :3], M[:3, 3]
        pc = (P - t) @ R  # world -> camera
        z = pc[:, 2]
        front = z > 1e-3
        u = v.cam.K[0, 0] * pc[:, 0] / np.where(front, z, 1) + v.cam.K[0, 2]
        w = v.cam.K[1, 1] * pc[:, 1] / np.where(front, z, 1) + v.cam.K[1, 2]
        inside = front & (u >= 0) & (u < v.cam.W) & (w >= 0) & (w < v.cam.H)
        ui, wi = np.clip(u.astype(int), 0, v.cam.W - 1), np.clip(w.astype(int), 0, v.cam.H - 1)
        pix = wi * v.cam.W + ui
        m = v.mask.detach().cpu().numpy()[pix] > 0.5
        keep = ~inside | m
        if v.depth is not None:
            dep = v.depth.detach().cpu().numpy()[pix]
            keep &= ~inside | (dep <= 0) | (z >= dep - depth_margin)
        occ &= keep
    return occ.reshape(res, res, res)


def target_from_views(target: ImageTarget, lo=(-1.1, -1.1, -1.1), hi=(1.1, 1.1, 1.1), res: int = 72, **kw):
    from scipy import ndimage

    from .targets import from_sdf

    occ = carve(target, lo, hi, res)
    # signed distance to the carved surface (voxel units -> scene units), smooth enough for marching cubes
    vox = (np.asarray(hi) - np.asarray(lo)) / (res - 1)
    sd = (ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)) * vox.mean()
    lo_a = np.asarray(lo)

    def fn(p):
        idx = np.clip(np.round((p - lo_a) / vox).astype(int), 0, res - 1)
        return sd[idx[:, 0], idx[:, 1], idx[:, 2]]

    return from_sdf(fn, lo, hi, res=res, **kw)


# ---------------------------------------------------------------------------
# Render-based refinement: geometry + per-primitive base color from images
# ---------------------------------------------------------------------------


@dataclass
class ImageFitConfig:
    steps: int = 300
    rays: int = 1536
    lr: float = 0.004
    lr_color: float = 0.05
    color_warmup: float = 0.3  # first fraction of steps: colors only (geometry frozen), so a wrong
    #                            initial albedo can't push geometry around through the rgb loss
    w_mask: float = 1.0
    w_depth: float = 2.0
    w_rgb: float = 1.0
    sigma0: float = 0.04
    sigma1: float = 0.006
    tau0: float = 0.0  # structure is already hard; keep exact CSG semantics by default
    fit_color: bool = True
    fit_geometry: bool = True
    guard: bool = True  # early-stop geometry on held-out views (keeps the input geometry if fitting never helps)
    holdout_views: int = 1  # views withheld from fitting and used only for the guard
    val_every: int = 25
    seed: int = 0
    opts: RenderOpts = field(default_factory=RenderOpts)


def add_colors(g: Graph, value=0.7) -> Graph:
    for nid in g.reachable():
        n = g.nodes[nid]
        if n.spec.category == "prim" and "color" not in n.raw:
            n.set_value("color", torch.full((3,), float(value)))  # set_value keeps the node's device
    return g


def fit_images(g: Graph, target: ImageTarget, cfg: Optional[ImageFitConfig] = None, verbose=False):
    cfg = cfg or ImageFitConfig()
    g = g.clone().to(target.device)
    g = add_colors(g) if cfg.fit_color else g
    g0 = g.clone()
    g.requires_grad_(True)
    geo, col = [], []
    for nid in g.reachable():
        n = g.nodes[nid]
        for k, r in n.raw.items():
            if k == "logit" or (n.op == "transform" and k == "s"):
                r.requires_grad_(False)
            elif k == "color":
                (col if cfg.fit_color else []).append(r)
                r.requires_grad_(cfg.fit_color)
            else:
                (geo if cfg.fit_geometry else []).append(r)
                r.requires_grad_(cfg.fit_geometry)
    groups = ([{"params": geo, "lr": cfg.lr}] if geo else []) + ([{"params": col, "lr": cfg.lr_color}] if col else [])
    opt = torch.optim.Adam(groups)
    gen = torch.Generator().manual_seed(cfg.seed)
    hist = []
    full = target
    use_val = cfg.fit_geometry and cfg.guard and cfg.holdout_views > 0 and len(target.views) > 3
    if use_val:  # validation views never contribute gradients; they choose which geometry to keep
        target = ImageTarget(full.views[:-cfg.holdout_views], full.background)
        val = ImageTarget(full.views[-cfg.holdout_views:], full.background)
        val_rays = val.sample(4096, torch.Generator().manual_seed(cfg.seed + 1))
    elif cfg.fit_geometry and cfg.guard:
        val_rays = full.sample(4096, torch.Generator().manual_seed(cfg.seed + 1))
    best = None

    def snapshot():
        return {nid: {k: r.detach().clone() for k, r in n.raw.items() if k != "color"} for nid, n in g.nodes.items()}

    def val_score():
        i, m = _ray_metrics(g, val_rays, cfg.opts)
        return i - 2.0 * m

    if cfg.fit_geometry and cfg.guard:
        best = (val_score(), snapshot(), 0)
    for step in range(cfg.steps):
        f = step / max(cfg.steps - 1, 1)
        opts = RenderOpts(**{**cfg.opts.__dict__,
                             "sigma": math.exp((1 - f) * math.log(cfg.sigma0) + f * math.log(cfg.sigma1))})
        ctx = EvalCtx(tau=cfg.tau0 * (1 - f), hard_gates=True)
        warm = cfg.fit_color and f < cfg.color_warmup
        if geo:
            opt.param_groups[0]["lr"] = 0.0 if warm else cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f)))
        o, d, z, rgb_gt, m_gt, dep_gt, has_d = target.sample(cfg.rays, gen)
        r = render(g, o, d, z, ctx, opts, want_color=cfg.fit_color)
        # IDR-style: silhouette loss only on rays whose hit/miss disagrees with the mask. A soft mask on
        # agreeing rays is biased (thin parts never reach sigmoid(-f/sigma) = 1 and get inflated).
        wrong = (r["hit"] != (m_gt > 0.5)).float()
        bce = F.binary_cross_entropy(r["mask"].clamp(1e-5, 1 - 1e-5), m_gt, reduction="none")
        loss = cfg.w_mask * (bce * wrong).sum() / len(m_gt)
        both = r["hit"] & (m_gt > 0.5)
        if both.any():
            if cfg.fit_color:
                loss = loss + cfg.w_rgb * (r["rgb"][both] - rgb_gt[both]).abs().mean()
            dm = both & has_d
            if dm.any():
                loss = loss + cfg.w_depth * (r["depth"][dm] - dep_gt[dm]).abs().mean()
        opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
            for grp in opt.param_groups:
                for p in grp["params"]:
                    if p.grad is not None:
                        p.grad.nan_to_num_(0.0)
            opt.step()
        hist.append(loss.item())
        if best is not None and not warm and ((step + 1) % cfg.val_every == 0 or step == cfg.steps - 1):
            v = val_score()
            if v > best[0] + 1e-4:
                best = (v, snapshot(), step + 1)
        if verbose and step % 50 == 0:
            print(f"  img step {step:4d} loss {float(loss):.4f} sigma {opts.sigma:.4f}")
    g.requires_grad_(False)
    if best is not None:  # restore the best-validated geometry, keep the final colors
        for nid, raws in best[1].items():
            for k, r in raws.items():
                g.nodes[nid].raw[k] = r
        if verbose:
            print(f"  guard: kept geometry from step {best[2]} (val score {best[0]:.4f})")
    return g, hist


@torch.no_grad()
def _ray_metrics(g: Graph, rays, opts: RenderOpts) -> tuple[float, float]:
    o, d, z, rgb, m, dep, has_d = rays
    r = render(g, o, d, z, EvalCtx(hard_gates=True), opts, want_color=False)
    gm = m > 0.5
    iou_ = float((r["hit"] & gm).sum() / max(int((r["hit"] | gm).sum()), 1))
    both = r["hit"] & gm & has_d
    mae = float((r["depth"] - dep)[both].abs().mean()) if both.any() else 0.0
    return iou_, mae
