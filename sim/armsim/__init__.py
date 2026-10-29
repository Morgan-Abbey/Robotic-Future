"""Closed-loop simulator for a BCI-driven 3-joint arm: scripted user -> decoder model -> interpreter -> arm."""
from .config import LEFT, REST, RIGHT, SWITCH, TARGETS, SimConfig
from .decoder import GenerativeDecoder, ReplayDecoder, accuracy_to_signal
from .episode import Action, Episode, Obs, evaluate, idle_drift, run_episode, run_idle
from .controllers import CONTROLLERS, DirectArgmax, Evidence, Proportional, SharedAutonomy

__all__ = [
    "LEFT", "REST", "RIGHT", "SWITCH", "TARGETS", "SimConfig", "GenerativeDecoder", "ReplayDecoder",
    "accuracy_to_signal", "Action", "Episode", "Obs", "evaluate", "idle_drift", "run_episode", "run_idle", "CONTROLLERS",
    "DirectArgmax", "Evidence", "Proportional", "SharedAutonomy",
]
