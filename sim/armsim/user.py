from collections import deque

import numpy as np

from .config import LEFT, REST, RIGHT, SWITCH


class ScriptedUser:
    """Someone who wants the arm at `goal`: imagines left/right for the selected joint, clenches to switch.

    They see the arm `delay` decisions late (optionally extrapolating its motion), stop imagining once the
    joint looks close enough, and wait for a joint switch to show up before asking for another. This is a
    model of a person, not a measurement: real users also adapt to the controller, which this does not.
    """

    def __init__(self, cfg, goal, goal_idx):
        self.cfg, self.goal, self.goal_idx = cfg, np.asarray(goal, float), goal_idx
        self.hist = deque(maxlen=cfg.delay + 1)
        self.wait = 0

    def observe(self, q, j):
        if not self.hist:
            self.hist.extend([(q.copy(), j)] * self.hist.maxlen)
        else:
            self.hist.append((q.copy(), j))

    def intent(self, auto_target=None):
        cfg = self.cfg
        if auto_target is not None and auto_target == self.goal_idx:
            return REST                                  # the arm is heading to the right place
        if self.wait > 0:
            self.wait -= 1
            return REST
        (q_old, j_obs), (q_next, _) = self.hist[0], self.hist[1] if len(self.hist) > 1 else self.hist[0]
        q = q_old + cfg.delay * (q_next - q_old) if cfg.anticipate else q_old
        err = self.goal - q
        far = np.abs(err) > cfg.tol
        if not far.any():
            return REST
        if far[j_obs]:
            return LEFT if err[j_obs] < 0 else RIGHT
        self.wait = cfg.delay                            # let the switch become visible first
        return SWITCH
