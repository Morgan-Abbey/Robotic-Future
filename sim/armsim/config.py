from dataclasses import dataclass

import numpy as np

REST, LEFT, RIGHT, SWITCH = 0, 1, 2, 3     # the decoder knows 0-2; SWITCH is the jaw clench

# Candidate goals in normalised joint units (each joint spans -1..1). Every pair differs in sign on at
# least one joint, so a few left/right commands can tell them apart. Placeholder geometry, not your arm.
TARGETS = np.array([[0.6, -0.4, 0.3], [-0.5, 0.5, 0.6], [0.3, 0.7, -0.5], [-0.6, -0.5, -0.3]])


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.25               # one decoder decision per 0.25 s (assumed; set to your window hop)
    q_limit: float = 1.0
    tol: float = 0.06              # a joint counts as "there" within this distance of the goal
    dwell: int = 2                 # decisions the arm must stay within tolerance to count as reached
    step: float = 0.05             # joint displacement of one issued command
    max_dq: float = 0.10           # fastest the arm moves per decision (0.4 units/s)
    t_max: float = 90.0
    delay: int = 2                 # the user sees the arm 2 decisions (0.5 s) late
    anticipate: bool = True        # the user extrapolates the arm's motion across that delay
    switch_refractory: int = 2     # decisions between two jaw-triggered joint switches
    jaw_hit: float = 0.9           # chance an intended clench is detected
    jaw_false_per_min: float = 0.3 # chance rate of accidental joint switches
    rho: float = 0.85              # decoder output correlation between successive windows (overlap)
    temperature: float = 1.0       # >1 under-confident decoder, <1 over-confident (1 = calibrated)

    @property
    def n_steps(self):
        return int(round(self.t_max / self.dt))
