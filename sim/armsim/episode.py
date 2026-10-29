from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import REST, TARGETS, SimConfig
from .user import ScriptedUser


@dataclass
class Obs:
    q: np.ndarray
    j: int
    t: int


@dataclass
class Action:
    dq: float = 0.0                          # displacement of the selected joint
    switch: bool = False                     # select the next joint
    auto: Optional[np.ndarray] = None        # per-joint displacement of an autonomous reach
    auto_target: Optional[int] = None        # which candidate the arm is heading for (shown to the user)


class Episode:
    """One reach. `sense()` runs the user and decoder for this decision, `act()` applies the controller's move."""

    def __init__(self, cfg, decoder, goal_idx, targets=TARGETS, forced_intent=None):
        self.cfg, self.decoder, self.targets, self.goal_idx = cfg, decoder, targets, goal_idx
        self.forced_intent = forced_intent         # idle test: the person only rests, whatever the arm does
        self.goal = targets[goal_idx]
        self.q, self.j, self.t = np.zeros(3), 0, 0
        self.user = ScriptedUser(cfg, self.goal, goal_idx)
        self.auto_target, self.in_tol, self.done, self.success = None, 0, False, False
        self.stats = dict(moved=0, wrong=0, switches=0, auto_commits=0, wrong_commits=0)

    def sense(self):
        self.user.observe(self.q, self.j)
        intent = self.user.intent(self.auto_target) if self.forced_intent is None else self.forced_intent
        probs, jaw = self.decoder.step(intent)
        return probs, jaw, Obs(self.q.copy(), self.j, self.t)

    def act(self, a):
        cfg, s = self.cfg, self.stats
        if a.auto_target is not None and self.auto_target is None:
            s["auto_commits"] += 1
            s["wrong_commits"] += a.auto_target != self.goal_idx
        if a.auto is not None:
            self.q = np.clip(self.q + np.clip(a.auto, -cfg.max_dq, cfg.max_dq), -cfg.q_limit, cfg.q_limit)
        elif a.switch:
            self.j = (self.j + 1) % 3
            s["switches"] += 1
        elif a.dq:
            old = abs(self.goal[self.j] - self.q[self.j])
            self.q[self.j] = np.clip(self.q[self.j] + np.clip(a.dq, -cfg.max_dq, cfg.max_dq),
                                     -cfg.q_limit, cfg.q_limit)
            s["moved"] += 1
            s["wrong"] += abs(self.goal[self.j] - self.q[self.j]) > old + 1e-9
        self.auto_target = a.auto_target
        self.t += 1
        self.in_tol = self.in_tol + 1 if (np.abs(self.goal - self.q) <= cfg.tol).all() else 0
        if self.in_tol >= cfg.dwell:
            self.done = self.success = True
        elif self.t >= cfg.n_steps:
            self.done = True
        return self.done

    def result(self):
        out = dict(self.stats, success=self.success, time=self.t * self.cfg.dt)
        out["wrong_frac"] = out["wrong"] / out["moved"] if out["moved"] else 0.0
        return out


def run_episode(cfg, controller, decoder_factory, goal_idx, seed):
    rng = np.random.default_rng(seed)
    ep = Episode(cfg, decoder_factory(rng), goal_idx)
    controller.reset()
    while not ep.done:
        probs, jaw, obs = ep.sense()
        ep.act(controller.step(probs, jaw, obs))
    return ep.result()


def run_idle(cfg, controller, decoder_factory, seed, minutes=2.0):
    """The person rests with the arm parked on a target. Returns how far the arm wandered, in units/min."""
    rng = np.random.default_rng(seed)
    ep = Episode(cfg, decoder_factory(rng), int(seed) % len(TARGETS), forced_intent=REST)
    ep.q = ep.goal.copy()
    controller.reset()
    moved = 0.0
    for _ in range(int(round(minutes * 60 / cfg.dt))):
        probs, jaw, obs = ep.sense()
        before = ep.q.copy()
        ep.act(controller.step(probs, jaw, obs))
        moved += np.abs(ep.q - before).sum()
    return moved / minutes


def idle_drift(cfg, controller, decoder_factory, seeds, minutes=2.0):
    return float(np.mean([run_idle(cfg, controller, decoder_factory, int(s), minutes) for s in seeds]))


def evaluate(cfg, controller, decoder_factory, seeds):
    """Run one episode per seed (goal cycles through the candidates) and aggregate."""
    rs = [run_episode(cfg, controller, decoder_factory, int(s) % len(TARGETS), int(s)) for s in seeds]
    ok = np.array([r["success"] for r in rs])
    time = np.array([r["time"] for r in rs])
    n = len(rs)
    return dict(
        n=n, success=float(ok.mean()), time_all=float(time.mean()),
        time_all_se=float(time.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        time_ok=float(np.median(time[ok])) if ok.any() else float("nan"),
        wrong_frac=float(np.mean([r["wrong_frac"] for r in rs])),
        switches=float(np.mean([r["switches"] for r in rs])),
        wrong_commits=float(np.mean([r["wrong_commits"] for r in rs])),
        auto_commits=float(np.mean([r["auto_commits"] for r in rs])),
    )
