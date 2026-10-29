"""Joint pose integrator and interpreter factory shared by the bridge and its tests."""
import json
from pathlib import Path

import numpy as np

from armsim import CONTROLLERS, SimConfig
from armsim.episode import Obs

ACCS = (45, 55, 65, 75, 85)
RESULTS = Path(__file__).resolve().parents[2] / "reports" / "arm_sim_results.json"


def tuned_params(name, accuracy, budget, path=RESULTS):
    """Interpreter settings found by sim/run_sweep.py for the nearest tuned accuracy."""
    if name == "direct":
        return {}
    nearest = min(ACCS, key=lambda a: abs(a - accuracy))
    table = json.loads(Path(path).read_text())["tuned"][str(nearest / 100)]
    return dict(table[name][str(float(budget)) if str(budget) in ("1", "1.0") else str(budget)])


def make_controller(name, dt, accuracy, budget, targets):
    cfg = SimConfig(dt=dt)
    params = tuned_params(name, accuracy, budget)
    cls = CONTROLLERS[name]
    ctrl = cls(cfg, targets=np.array(targets), **params) if name == "shared" else cls(cfg, **params)
    return cfg, ctrl


class Plant:
    """Dead-reckoned joint pose (units -1..1) and the selected joint."""

    def __init__(self, cfg):
        self.cfg, self.q, self.j, self.t = cfg, np.zeros(3), 0, 0

    def zero(self):
        self.q = np.zeros(3)

    def obs(self):
        return Obs(self.q.copy(), self.j, self.t)

    def apply(self, a):
        """Apply an interpreter Action; returns a short description of the move."""
        cfg, lim = self.cfg, self.cfg.q_limit
        self.t += 1
        if a.auto is not None:
            self.q = np.clip(self.q + np.clip(a.auto, -cfg.max_dq, cfg.max_dq), -lim, lim)
            return "auto-reach"
        if a.switch:
            self.j = (self.j + 1) % 3
            return "switch joint"
        if a.dq:
            self.q[self.j] = np.clip(self.q[self.j] + np.clip(a.dq, -cfg.max_dq, cfg.max_dq), -lim, lim)
            return "left" if a.dq < 0 else "right"
        return "hold"
