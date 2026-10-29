"""Grid search under a false-motion budget, so every controller is tuned the same way at each accuracy.

Two numbers matter: how long a reach takes, and how far the arm wanders while the person rests. A
controller that moves on every noisy window reaches fast but is useless when you stop thinking about it,
so each candidate is scored on both and the pick is the fastest one inside the idle budget.
"""
import itertools
from multiprocessing import get_context

import numpy as np

from .controllers import CONTROLLERS
from .decoder import GenerativeDecoder
from .episode import evaluate, idle_drift

GRIDS = {
    "direct": {},
    "evidence": dict(alpha=(0.3, 0.5, 0.8), theta=(0.4, 0.5, 0.6, 0.7), hold=(1, 2, 3), refractory=(0, 1, 2, 3)),
    "proportional": dict(alpha=(0.3, 0.5, 0.7), theta=(0.4, 0.5, 0.6, 0.7), gamma=(0.5, 1.0, 2.0),
                         vmax=(0.2, 0.3, 0.4), hold=(1, 2)),
    "shared": dict(kappa=(0.15, 0.3, 0.6), tau=(0.8, 0.9, 0.97)),
}
IDLE_BUDGET = 0.05          # units/min the arm may wander at rest: one false command of the standard step per minute


def configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())] or [{}]


def measure(args):
    cfg, name, params, decoder_kw, reach_seeds, idle_seeds = args
    factory = lambda rng: GenerativeDecoder(cfg, rng=rng, **decoder_kw)
    ctrl = CONTROLLERS[name](cfg, **params)
    r = evaluate(cfg, ctrl, factory, reach_seeds)
    return dict(r, params=params, time=r["time_all"], idle=idle_drift(cfg, ctrl, factory, idle_seeds))


def pareto(points):
    """Indices of points not beaten on both (idle, time)."""
    keep = []
    for i, (a, b) in enumerate(points):
        if not any((c <= a and d <= b) and (c < a or d < b) for k, (c, d) in enumerate(points) if k != i):
            keep.append(i)
    return keep


def pick(results, budget):
    ok = [r for r in results if r["idle"] <= budget]
    return min(ok, key=lambda r: r["time"]) if ok else min(results, key=lambda r: r["idle"])


def tune(cfg, name, accuracy, base_params=None, budgets=(IDLE_BUDGET,), workers=4, seed0=1000, **decoder_kw):
    """Returns ({budget: best params}, stage-2 results of the candidates on the idle/time frontier)."""
    base = base_params or {}
    grid = [{**base, **c} for c in configs(GRIDS[name])]
    kw = dict(accuracy=accuracy, **decoder_kw)
    pool = get_context("fork").Pool(workers)
    try:
        def run(cands, n_reach, n_idle, off):
            reach = list(range(seed0 + off, seed0 + off + n_reach))
            idle = list(range(seed0 + off + 50_000, seed0 + off + 50_000 + n_idle))
            return pool.map(measure, [(cfg, name, c, kw, reach, idle) for c in cands])

        if len(grid) == 1:
            res = run(grid, 120, 20, 10_000)
            return {b: grid[0] for b in budgets}, res
        s1 = run(grid, 30, 6, 0)
        idx = pareto([(r["idle"], r["time"]) for r in s1])
        near = {i for b in budgets for i in sorted(range(len(s1)), key=lambda i: (s1[i]["idle"] > b * 3, s1[i]["time"]))[:4]}
        cand = sorted(set(idx) | near, key=lambda i: s1[i]["idle"])
        if len(cand) > 14:
            cand = [cand[i] for i in np.linspace(0, len(cand) - 1, 14).astype(int)]
        s2 = run([grid[i] for i in cand], 120, 20, 10_000)
        return {b: pick(s2, b)["params"] for b in budgets}, s2
    finally:
        pool.close()
        pool.join()
