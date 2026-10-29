from functools import lru_cache

import numpy as np

from .config import REST, SWITCH


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@lru_cache(maxsize=None)
def accuracy_to_signal(acc, n_classes=3):
    """Signal strength s such that argmax(s*onehot + N(0, I)) is correct with probability `acc`."""
    if acc <= 1.0 / n_classes:
        return 0.0
    z = np.random.default_rng(0).standard_normal((200_000, n_classes))
    lo, hi = 0.0, 10.0
    for _ in range(40):
        s = (lo + hi) / 2
        zz = z.copy()
        zz[:, 0] += s
        lo, hi = (s, hi) if (zz.argmax(1) == 0).mean() < acc else (lo, s)
    return (lo + hi) / 2


class GenerativeDecoder:
    """Stand-in for the EEG decoder: noisy class logits that are correlated across successive windows.

    Logits are s*onehot(class) + AR(1) noise, and p = softmax(s*logits / T). With T=1 that is the exact
    posterior, i.e. a calibrated decoder: a 0.8 confidence is right 80% of the time.
    """

    def __init__(self, cfg, accuracy, rng):
        self.cfg, self.rng = cfg, rng
        self.s = accuracy_to_signal(round(float(accuracy), 4))
        self.z = rng.standard_normal(3)

    def step(self, intent):
        cfg = self.cfg
        self.z = cfg.rho * self.z + np.sqrt(1 - cfg.rho ** 2) * self.rng.standard_normal(3)
        logits = self.z.copy()
        logits[REST if intent == SWITCH else intent] += self.s
        probs = softmax(self.s * logits / cfg.temperature)
        jaw = (intent == SWITCH and self.rng.random() < cfg.jaw_hit) or \
            self.rng.random() < cfg.jaw_false_per_min * cfg.dt / 60.0
        return probs, bool(jaw)


class ReplayDecoder:
    """Replays recorded decoder output, so errors keep their real streaks and the real calibration.

    `probs` is (N, 3) with columns rest/left/right and `intent` is (N,) with the class the person was
    cued to imagine at each window. The recording is cut into runs of constant intent; during a simulated
    episode a random run of the wanted class is played back from its start.
    """

    def __init__(self, cfg, probs, intent, rng):
        self.cfg, self.rng = cfg, rng
        probs, intent = np.asarray(probs, float), np.asarray(intent, int)
        cuts = np.flatnonzero(np.diff(intent)) + 1
        self.runs = {c: [] for c in range(3)}
        for seg_p, seg_i in zip(np.split(probs, cuts), np.split(intent, cuts)):
            self.runs[int(seg_i[0])].append(seg_p)
        missing = [c for c, r in self.runs.items() if not r]
        if missing:
            raise ValueError("no recorded windows for class(es) %s" % missing)
        self.cls, self.run, self.pos = None, None, 0

    @classmethod
    def from_npz(cls, cfg, path, rng):
        d = np.load(path)
        return cls(cfg, d["probs"], d["intent"], rng)

    def step(self, intent):
        c = REST if intent == SWITCH else intent
        if c != self.cls or self.pos >= len(self.run):
            self.cls = c
            runs = self.runs[c]
            self.run, self.pos = runs[self.rng.integers(len(runs))], 0
        probs = self.run[self.pos]
        self.pos += 1
        jaw = (intent == SWITCH and self.rng.random() < self.cfg.jaw_hit) or \
            self.rng.random() < self.cfg.jaw_false_per_min * self.cfg.dt / 60.0
        return probs, bool(jaw)
