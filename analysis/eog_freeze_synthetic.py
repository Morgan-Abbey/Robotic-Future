"""Synthetic check of the EOG-calibrate-then-freeze cleaner.

    python eog_freeze_synthetic.py        # writes ../reports/eog_freeze_synthetic.md
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eog_freeze import METHODS, FrozenEyeCleaner
from synthetic import LAYOUTS, person, session

FS = 250
OUT = Path(__file__).resolve().parents[1] / "reports" / "eog_freeze_synthetic.md"
LABEL = {"none": "no cleaning", "eog_regression": "EOG regression (needs EOG at run time)",
         "proj_orth": "frozen: orthogonal projection, always on", "gls": "frozen: GLS, always on",
         "gls_lp": "frozen: GLS, slow (<7 Hz) only", "gls_gate": "frozen: GLS, broadband, gated",
         "gls_lp_gate": "frozen: GLS, slow only, gated"}
MU_BETA = butter(4, [8, 30], btype="band", fs=FS, output="sos")


def rrmse(x, truth, mask=None):
    if mask is not None:
        x, truth = x[:, mask], truth[:, mask]
    return float(np.sqrt(np.mean((x - truth) ** 2) / np.mean(truth ** 2)))


def score(clean, s):
    active = np.convolve((np.abs(s["v"]) > 15) | (np.abs(s["h"]) > 15), np.ones(FS // 2), "same") > 0
    yb, tb = sosfiltfilt(MU_BETA, clean, axis=1), sosfiltfilt(MU_BETA, s["brain"], axis=1)
    corr = lambda ref: float(np.mean([abs(np.corrcoef(c, ref)[0, 1]) for c in clean]))
    return dict(err_eye=rrmse(clean, s["brain"], active), err_quiet=rrmse(clean, s["brain"], ~active),
                err_mubeta=rrmse(yb, tb), corr_v=corr(s["v"]), corr_h=corr(s["h"]))


def run(layout, seeds, prop_scale=1.0, **kw):
    rows = {m: [] for m in ("none",) + METHODS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        who = person(rng, layout)
        cal = session(rng, who, seconds=180, rates=(0.25, 0.15, 0.15), prop_scale=prop_scale)   # deliberate blinks and looks
        test = session(rng, who, seconds=300, rates=(0.2, 0.05, 0.05), prop_scale=prop_scale)   # everyday use: mostly blinks
        cleaner = FrozenEyeCleaner(FS, **kw).fit(cal["eeg"], cal["eog"])
        rows["none"].append(score(test["eeg"], test))
        for m in METHODS:
            rows[m].append(score(cleaner.transform(test["eeg"], m, eog=test["eog"]), test))
    return {m: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for m, rs in rows.items()}


def detection(cleaner, test, fs=FS):
    """Event-level scores of the EEG-only detector: hit rate per event type, false alarms per minute, time gated."""
    gate = cleaner.detect(test["eeg"])
    hits = {"blink": [], "look": []}
    for t, kind, amp in test["events"]:
        lo, hi = max(0, t - int(0.1 * fs)), min(gate.size, t + int(0.6 * fs))
        hits[kind].append(bool(gate[lo:hi].any()))
    onsets = np.flatnonzero(np.diff(gate.astype(int)) == 1) + 1
    times = np.array([e[0] for e in test["events"]])
    false = sum(not ((times > o - int(0.3 * fs)) & (times < o + int(0.8 * fs))).any() for o in onsets)
    return dict(blink=float(np.mean(hits["blink"])), look=float(np.mean(hits["look"])),
                false_per_min=false / (gate.size / fs / 60), gated=float(gate.mean()))


def run_detection(layout, seeds, ks, prop_scale=1.0):
    acc = {k: [] for k in ks}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        who = person(rng, layout)
        cal = session(rng, who, seconds=180, rates=(0.25, 0.15, 0.15), prop_scale=prop_scale)
        test = session(rng, who, seconds=300, rates=(0.2, 0.05, 0.05), prop_scale=prop_scale)
        cleaner = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
        for k in ks:
            cleaner.gate_k = k
            acc[k].append(detection(cleaner, test))
    return {k: {m: float(np.mean([r[m] for r in rs])) for m in rs[0]} for k, rs in acc.items()}


def main(seeds=range(8)):
    L = ["# EOG for calibration only: synthetic check", "",
         "Ground-truth simulation (`analysis/synthetic.py`): 1/f + mu/beta EEG, blinks and looks, assumed",
         "propagation factors (large frontal, small central), EOG electrodes that also pick up some brain activity.",
         "Fit on 3 min of deliberate eye movements, test on 5 min of mostly blinks, mean of %d simulated people." % len(seeds),
         "**This checks the algorithm under stated assumptions. It is not evidence about a real head or the",
         "BNCI data; that test is still to be run (see `analysis/README.md`).**", "",
         "Columns: error relative to the true brain signal while the eye is active, while it is quiet (damage to",
         "clean data), and in the 8-30 Hz band the decoder uses; then leftover correlation with the true vertical",
         "and horizontal eye signals (before cleaning these are what the desktop evals report as 0.49 / 0.28).", ""]
    for layout in LAYOUTS:
        res = run(layout, seeds)
        L += ["## %s" % layout, "", "| method | error, eye active | error, eye quiet | error, 8-30 Hz | leftover eye corr (up-down / left-right) |",
              "|---|---|---|---|---|"]
        for m in ("none",) + METHODS:
            r = res[m]
            L.append("| %s | %.3f | %.3f | %.3f | %.2f / %.2f |" % (LABEL.get(m, m), r["err_eye"], r["err_quiet"], r["err_mubeta"], r["corr_v"], r["corr_h"]))
        L.append("")
        det = run_detection(layout, seeds, (3.0, 4.0, 6.0, 8.0))
        L += ["Detect-and-hold instead of subtracting: an EEG-only eye-event detector (blinks %d-%d uV, looks %d-%d uV).",
              "", "| threshold (x resting sd) | blinks caught | looks caught | false alarms / min | time flagged |", "|---|---|---|---|---|"]
        L[-4] = L[-4] % (80, 160, 30, 70)
        for k, d in det.items():
            L.append("| %g | %.0f%% | %.0f%% | %.1f | %.0f%% |" % (k, 100 * d["blink"], 100 * d["look"], d["false_per_min"], 100 * d["gated"]))
        L.append("")
    L += ["## How strong must the eye signal be on the scalp?", "",
          "6-channel layout again, with every propagation factor multiplied by the scale below (1 = the assumed values,",
          "e.g. 0.25 / 0.08 at FCz and 0.08 at Cz). Frozen cleaner = GLS, broadband, gated. The desktop evals report a",
          "leftover-eye correlation of about 0.49 / 0.28 (up-down / left-right) on real data before cleaning; compare it with the",
          "'uncleaned corr' column to see which row looks most like the real data (correlations are measured slightly differently).", "",
          "| scale | uncleaned corr | error, eye active (none -> cleaned) | error, eye quiet | error, 8-30 Hz (none -> cleaned) | corr after | slow-only variant: active / 8-30 Hz | blinks caught at 4x | false alarms / min |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sc in (1, 2, 3, 4, 8):
        res, det = run("6 EEG (+CP3 CP4 FCz)", seeds, prop_scale=sc), run_detection("6 EEG (+CP3 CP4 FCz)", seeds, (4.0,), prop_scale=sc)[4.0]
        a, b, c = res["none"], res["gls_gate"], res["gls_lp_gate"]
        L.append("| %g | %.2f / %.2f | %.2f -> %.2f | %.2f | %.3f -> %.3f | %.2f / %.2f | %.2f / %.3f | %.0f%% | %.1f |" % (
            sc, a["corr_v"], a["corr_h"], a["err_eye"], b["err_eye"], b["err_quiet"], a["err_mubeta"], b["err_mubeta"],
            b["corr_v"], b["corr_h"], c["err_eye"], c["err_mubeta"], 100 * det["blink"], det["false_per_min"]))
    L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
