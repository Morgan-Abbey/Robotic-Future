"""Render one simulated reach as a GIF: arm, goal, decoder probabilities and what the interpreter does.

    python make_animation.py --controller proportional --accuracy 0.75 --budget 0.25 --out ../reports/arm_reach.gif

The arm is a stylised planar 3-link chain (joint i bends by q_i * 90 degrees), not your arm's geometry.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from armsim import CONTROLLERS, TARGETS, Episode, GenerativeDecoder, SimConfig

INK, MUTED, SURFACE, BLUE, ORANGE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834", "#d9d8d3"
LINKS = np.array([0.45, 0.35, 0.25])
RESULTS = Path(__file__).resolve().parents[1] / "reports" / "arm_sim_results.json"


def chain(q):
    ang = np.cumsum(np.concatenate([[np.pi / 2 + q[0] * np.pi / 2], q[1:] * np.pi / 2]))
    pts = np.vstack([[0, 0], np.cumsum(np.c_[LINKS * np.cos(ang), LINKS * np.sin(ang)], axis=0)])
    return pts


def record(cfg, controller, accuracy, goal, seed):
    ep = Episode(cfg, GenerativeDecoder(cfg, accuracy, np.random.default_rng(seed)), goal)
    controller.reset()
    frames = []
    while not ep.done:
        probs, jaw, obs = ep.sense()
        a = controller.step(probs, jaw, obs)
        frames.append(dict(q=ep.q.copy(), j=ep.j, probs=probs, jaw=jaw, t=ep.t * cfg.dt, auto=a.auto_target, dq=a.dq))
        ep.act(a)
    frames.append(dict(q=ep.q.copy(), j=ep.j, probs=probs, jaw=False, t=ep.t * cfg.dt, auto=None, dq=0.0))
    return frames, ep.success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="proportional", choices=list(CONTROLLERS))
    ap.add_argument("--accuracy", type=float, default=0.75)
    ap.add_argument("--budget", default="0.25")
    ap.add_argument("--goal", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(RESULTS.parent / "arm_reach.gif"))
    a = ap.parse_args()
    cfg = SimConfig()
    params = {}
    if RESULTS.exists() and a.controller != "direct":
        params = json.loads(RESULTS.read_text())["tuned"].get(str(a.accuracy), {}).get(a.controller, {}).get(a.budget, {})
    frames, ok = record(cfg, CONTROLLERS[a.controller](cfg, **params), a.accuracy, a.goal, a.seed)
    frames += [frames[-1]] * 6

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]), facecolor=SURFACE)
    for x in (ax, bx):
        x.set_facecolor(SURFACE)
    tips = np.array([chain(t)[-1] for t in TARGETS])

    def draw(i):
        f = frames[i]
        ax.clear(); bx.clear()
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.2, 1.15); ax.set_aspect("equal"); ax.axis("off")
        ax.scatter(*tips.T, s=70, color=GRID, zorder=1)
        ax.scatter(*tips[a.goal], s=120, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=2)
        for k, t in enumerate(tips):
            ax.text(t[0], t[1] + 0.07, "goal" if k == a.goal else "", color=ORANGE, ha="center", fontsize=8)
        pts = chain(f["q"])
        ax.plot(pts[:, 0], pts[:, 1], color=BLUE, linewidth=3, solid_capstyle="round", zorder=3)
        ax.scatter(pts[:, 0], pts[:, 1], s=[40, 40, 40, 70], color=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4)
        ax.scatter(*pts[1 + f["j"]], s=150, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5)
        ax.text(0, -0.15, "t = %.1f s   selected joint %d%s%s" % (f["t"], f["j"] + 1, "   jaw clench" if f["jaw"] else "",
                                                                  "   auto-reach" if f["auto"] is not None else ""),
                ha="center", color=INK, fontsize=9)
        bx.barh([0, 1, 2], f["probs"][::-1], color=BLUE, height=0.55)
        bx.set_yticks([0, 1, 2]); bx.set_yticklabels(["right", "left", "rest"], color=MUTED, fontsize=9)
        bx.set_xlim(0, 1); bx.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            bx.spines[s].set_color(GRID)
        bx.set_title("decoder output (%d%% accuracy)" % round(a.accuracy * 100), color=INK, fontsize=9, loc="left")
        ax.set_title("%s controller, %s" % (a.controller, "reached" if ok and i >= len(frames) - 7 else "reaching"),
                     color=INK, fontsize=9, loc="left")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=125)
    anim.save(a.out, writer=PillowWriter(fps=8), dpi=90)
    print("saved", a.out, "frames", len(frames), "reached" if ok else "timed out")


if __name__ == "__main__":
    main()
