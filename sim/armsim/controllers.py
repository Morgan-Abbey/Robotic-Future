"""Interpreters: turn the decoder's probabilities (plus a jaw-clench event) into arm motion.

All of them share the same joint switching: a detected clench selects the next joint. They differ in how
left/right evidence becomes motion on the selected joint. None of them is told the goal.
"""
import numpy as np

from .config import LEFT, REST, RIGHT, TARGETS
from .episode import Action


class Controller:
    name = "base"
    defaults = {}

    def __init__(self, cfg, **params):
        self.cfg = cfg
        self.p = {**self.defaults, **params}
        self.reset()

    def reset(self):
        self.last_switch = -10 ** 9
        self.pt = None                                   # smoothed probabilities

    def smooth(self, probs):
        a = self.p.get("alpha", 1.0)
        self.pt = probs.copy() if self.pt is None else (1 - a) * self.pt + a * probs
        return self.pt

    def step(self, probs, jaw, obs):
        if jaw and obs.t - self.last_switch >= self.cfg.switch_refractory:
            self.last_switch = obs.t
            self.on_switch()
            return Action(switch=True)
        return self.move(probs, obs)

    def on_switch(self):
        pass

    def move(self, probs, obs):
        raise NotImplementedError


def sign(c):
    return -1.0 if c == LEFT else 1.0


class DirectArgmax(Controller):
    """Whatever class wins this window moves the joint by one step. No smoothing, nothing to tune."""
    name = "direct"

    def move(self, probs, obs):
        c = int(np.argmax(probs))
        return Action() if c == REST else Action(dq=sign(c) * self.cfg.step)


class Evidence(Controller):
    """Smooth, then issue one fixed-size command once a class has stayed above a threshold for `hold`
    windows, then pause for `refractory` windows. Modelled on the command layer described in the desktop
    README (evidence smoothing + threshold + hold time + refractory); it is not that code."""
    name = "evidence"
    defaults = dict(alpha=0.5, theta=0.5, hold=2, refractory=2)

    def reset(self):
        super().reset()
        self.cls, self.count, self.cool = None, 0, 0

    def on_switch(self):
        self.cls, self.count = None, 0

    def move(self, probs, obs):
        pt, p = self.smooth(probs), self.p
        if self.cool > 0:
            self.cool -= 1
            return Action()
        c = int(np.argmax(pt))
        if c == REST or pt[c] <= p["theta"]:
            self.cls, self.count = None, 0
            return Action()
        self.count = self.count + 1 if c == self.cls else 1
        self.cls = c
        if self.count < p["hold"]:
            return Action()
        self.cls, self.count, self.cool = None, 0, p["refractory"]
        return Action(dq=sign(c) * self.cfg.step)


class Proportional(Controller):
    """Speed grows with how far the smoothed confidence is above a threshold; rest or low confidence holds."""
    name = "proportional"
    defaults = dict(alpha=0.5, theta=0.5, gamma=1.0, vmax=0.3, hold=1)

    def reset(self):
        super().reset()
        self.cls, self.count = None, 0

    def on_switch(self):
        self.cls, self.count = None, 0

    def move(self, probs, obs):
        pt, p = self.smooth(probs), self.p
        c = int(np.argmax(pt))
        if c == REST or pt[c] < p["theta"]:
            self.cls, self.count = None, 0
            return Action()
        self.count = self.count + 1 if c == self.cls else 1
        self.cls = c
        if self.count < p["hold"]:
            return Action()
        gain = ((pt[c] - p["theta"]) / (1 - p["theta"])) ** p["gamma"]
        dq = sign(c) * p["vmax"] * gain * self.cfg.dt
        return Action(dq=dq) if abs(dq) >= 0.005 else Action()


class SharedAutonomy(Proportional):
    """Proportional control plus intent inference: a belief over the candidate goals is updated from the
    left/right/rest evidence, and once one goal is confident enough the arm finishes the reach itself.

    With `gate` on, the belief only learns from windows where the proportional layer is itself moving the
    joint, so idle noise can't build up into a commit."""
    name = "shared"
    defaults = dict(Proportional.defaults, kappa=0.3, tau=0.9, v_auto=0.3, eps=0.05, gate=1)

    def __init__(self, cfg, targets=TARGETS, **params):
        self.targets = targets
        super().__init__(cfg, **params)

    def reset(self):
        super().reset()
        self.b = np.full(len(self.targets), 1.0 / len(self.targets))
        self.auto = None

    def expected(self, obs):
        err = self.targets[:, obs.j] - obs.q[obs.j]
        return np.where(np.abs(err) <= self.cfg.tol, REST, np.where(err < 0, LEFT, RIGHT))

    def step(self, probs, jaw, obs):
        cfg, p = self.cfg, self.p
        if self.auto is not None:
            d = self.targets[self.auto] - obs.q
            if (np.abs(d) <= cfg.tol / 2).all():
                self.b = np.full(len(self.targets), 1.0)
                self.b[self.auto] = 0.1                  # the user is still asking for something else
                self.b /= self.b.sum()
                self.auto, self.pt = None, None
            else:
                return Action(auto=np.clip(d, -p["v_auto"] * cfg.dt, p["v_auto"] * cfg.dt), auto_target=self.auto)
        action = super().step(probs, jaw, obs)
        if not action.switch and (action.dq != 0 or not p["gate"]):
            like = p["eps"] + probs[self.expected(obs)]
            self.b = np.maximum(self.b * like ** p["kappa"], 1e-6)
            self.b /= self.b.sum()
            if self.b.max() > p["tau"]:
                self.auto = int(np.argmax(self.b))
                d = self.targets[self.auto] - obs.q
                return Action(auto=np.clip(d, -p["v_auto"] * cfg.dt, p["v_auto"] * cfg.dt), auto_target=self.auto)
        return action


CONTROLLERS = {c.name: c for c in (DirectArgmax, Evidence, Proportional, SharedAutonomy)}
