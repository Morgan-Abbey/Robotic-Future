"""Ground-truth EEG + eye activity generator, to check the cleaner against a known clean answer.

Everything about the head is an assumption (propagation factors, EOG leakage, spectra); this checks that the
algorithm does what it claims under stated conditions. It says nothing about a real head.
"""
import numpy as np
from scipy.signal import butter, sosfilt

# amplitude of vertical / horizontal eye activity on each scalp site (fraction of the EOG signal); assumed,
# in the range reported for blink propagation: large frontal, small central.
PROPAGATION = {"FCz": (0.25, 0.08), "Cz": (0.08, 0.02), "C3": (0.06, 0.05), "C4": (0.06, -0.05),
               "CP3": (0.04, 0.03), "CP4": (0.04, -0.03)}
LAYOUTS = {"3 EEG (C3 Cz C4)": ("C3", "Cz", "C4"), "6 EEG (+CP3 CP4 FCz)": ("C3", "Cz", "C4", "CP3", "CP4", "FCz")}


def colored(rng, n, fs, size):
    f = np.fft.rfftfreq(size, 1 / fs)
    spec = (rng.standard_normal((n, f.size)) + 1j * rng.standard_normal((n, f.size))) / np.maximum(f, 1.0) ** 1.0
    return np.fft.irfft(spec, size, axis=1)


def person(rng, layout, n_src=8):
    """What stays fixed about a head between sessions: how brain sources mix onto the chosen sites."""
    names = LAYOUTS[layout]
    return dict(layout=layout, names=names,
                mix=rng.standard_normal((len(names), n_src)) + 0.8 * rng.standard_normal((len(names), 1)))


def brain(rng, mix, fs, size, drift=0.1):
    """1/f background plus mu (10 Hz) and beta (20 Hz) bursts, mixed over channels; about 10 uV rms.
    `drift` perturbs the mixing a little, as electrode placement and impedance do from day to day."""
    n_src = mix.shape[1]
    t = np.arange(size) / fs
    src = colored(rng, n_src, fs, size)
    for k in range(n_src):
        for f0, amp in ((10, 1.0), (20, 0.6)):
            env = np.clip(sosfilt(butter(2, 0.3, fs=fs, output="sos"), rng.standard_normal(size)) * 6 + 0.7, 0.05, None)
            src[k] += amp * env * np.sin(2 * np.pi * (f0 + rng.normal(0, 0.3)) * t + rng.uniform(0, 6.28)) * src[k].std()
    src /= src.std(1, keepdims=True)
    x = (mix + drift * rng.standard_normal(mix.shape)) @ src
    return x * 10.0 / np.sqrt((mix ** 2).sum(1).mean())


def eye_sources(rng, fs, size, rate_blink, rate_vert, rate_horiz):
    """Vertical source (blinks + vertical looks) and horizontal source (looks), in uV."""
    v, h = np.zeros(size), np.zeros(size)
    t = np.arange(size)
    events = []

    def add(sig, centre, amp, width_s, kind):
        if kind == "pulse":
            events.append((int(centre), "blink", abs(float(amp))))
        w = int(width_s * fs)
        lo, hi = max(0, centre - 3 * w), min(size, centre + 3 * w)
        if kind == "pulse":
            sig[lo:hi] += amp * np.exp(-0.5 * ((t[lo:hi] - centre) / w) ** 2)
        else:                                            # a look: step up, hold ~1 s, step back
            d = int(rng.uniform(0.6, 1.4) * fs)
            events.extend([(int(centre), "look", abs(float(amp))), (int(centre) + d, "look", abs(float(amp)))])   # onset and return
            ramp = lambda c: 1 / (1 + np.exp(-(t[lo:min(size, hi + d)] - c) / (0.03 * fs)))
            seg = slice(lo, min(size, hi + d))
            sig[seg] += amp * (ramp(centre) - ramp(centre + d))

    for c in rng.uniform(0, size, rng.poisson(rate_blink * size / fs)).astype(int):
        add(v, c, rng.uniform(80, 160), 0.07, "pulse")
    for c in rng.uniform(0, size, rng.poisson(rate_vert * size / fs)).astype(int):
        add(v, c, rng.choice([-1, 1]) * rng.uniform(30, 70), 0, "step")
    for c in rng.uniform(0, size, rng.poisson(rate_horiz * size / fs)).astype(int):
        add(h, c, rng.choice([-1, 1]) * rng.uniform(30, 70), 0, "step")
    return v, h, sorted(events)


def session(rng, who, fs=250, seconds=180, rates=(0.25, 0.1, 0.1), prop_scale=1.0, drift=0.1):
    size = int(seconds * fs)
    names = who["names"]
    hp = butter(2, 1.0, btype="high", fs=fs, output="sos")             # the desktop front end high-passes at 1 Hz
    b = sosfilt(hp, brain(rng, who["mix"], fs, size, drift), axis=1)
    v, h, events = eye_sources(rng, fs, size, *rates)
    v, h = sosfilt(hp, v), sosfilt(hp, h)
    A = prop_scale * np.array([PROPAGATION[n] for n in names])
    eeg = b + A @ np.vstack([v, h])
    leak = b.mean(0)                                                     # EOG electrodes also pick up some brain activity
    eog = np.vstack([0.5 * v + 1.0 * h, 1.0 * v, 0.5 * v - 1.0 * h]) + 0.3 * leak + rng.normal(0, 4.0, (3, size))
    return dict(eeg=eeg, eog=eog, brain=b, v=v, h=h, names=names, events=events, fs=fs)
