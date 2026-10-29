# EOG for calibration only: frozen eye-artifact removal

Idea: record the eye (EOG) electrodes only during a short calibration, learn how eye activity shows up on each EEG
channel, then clean the EEG without any eye electrode. This folder has the method, a ground-truth simulator, and a
check. **Everything here was run on simulated data. The real-data test (BNCI 2014-001, day 1 → day 2) has not been
run: the cloud session had neither the desktop code nor the dataset (the download hosts are blocked by its network
policy).** Full numbers: [`../reports/eog_freeze_synthetic.md`](../reports/eog_freeze_synthetic.md).

| file | what |
|---|---|
| `eog_freeze.py` | `FrozenEyeCleaner`: `fit(eeg_cal, eog_cal)`, then `transform(eeg, method)` and `detect(eeg)` with no EOG |
| `synthetic.py` | simulated people: 1/f + mu/beta EEG, blinks and looks, propagation factors, EOG leakage, 1 Hz high-pass |
| `eog_freeze_synthetic.py` | the experiment; regenerates the report |
| `test_eog_freeze.py` | 10 tests (`pytest`), including causality and the rank argument below |

```
pip install numpy scipy pytest
python -m pytest test_eog_freeze.py          # ~3 s
python eog_freeze_synthetic.py               # ~1 min, rewrites ../reports/eog_freeze_synthetic.md
```

## How it works

Calibration: take the two main directions of the EOG (vertical, horizontal), regress each EEG channel on them to get
the propagation factors `B` (channels × 2), and measure the covariance of what is left (brain + noise), `Σ`.
Run time, EEG only: estimate the eye amplitude by generalised least squares, `ŝ = (BᵀΣ⁻¹B)⁻¹BᵀΣ⁻¹x`, and subtract
`B·ŝ`. Optional: low-pass `ŝ` below 7 Hz so the mu/beta band is untouched, and gate the subtraction so it only runs
while `ŝ` is clearly above its resting level (causal, with a 0.3 s hold). `detect()` exposes the gate on its own, for
"hold the last command during a blink" instead of subtracting.

## What the simulation says

(Under stated assumptions, 8 simulated people; the assumptions are the weak point, see below.)

1. **Fixed linear removal always costs brain signal.** Removing the eye directions also removes whatever brain activity
   lies along them: orthogonal projection of two eye directions out of 3 channels leaves one dimension (error 0.84 of the
   signal, worse than not cleaning); out of 6 channels, 0.61. A test pins this (`rank(P) == 1`).
2. **EEG-only eye detection is limited by how strongly the eye reaches the scalp.** With the assumed weak propagation
   (large at FCz, small central), 3 channels catch 0-7% of blinks; 6 channels catch 44% at 0 false alarms/min and 77% at
   2 /min. If the eye signal on the central channels is about 4× larger (which makes the *uncleaned* vertical correlation
   0.49, the value the desktop evals report on real data), the gated cleaner catches essentially every blink at 0.2 false
   alarms/min and cuts the eye-active error from 1.58 to 0.53 (vertical correlation 0.49 → 0.05).
3. **It does not improve the 8-30 Hz band the decoder uses.** In this model the eye contributes little there (1-5% error
   before cleaning) and the cleaner adds more (broadband gated: 0.05 → 0.21 at the 4× setting; slow-only gated: 0.085, but it
   removes less of the artifact because the causal low-pass delays the correction by about one blink width). So its value
   would be detection, hold-off and protecting broadband stages, not decoding accuracy. Real saccade spike potentials
   are broadband and are not in the simulation, which could change this.
4. **Regression with EOG available at run time still works best** (eye-active error 0.11-0.12 against 0.23-0.39 uncleaned), as expected; that is the reference to beat.

## Caveats

- The propagation factors (`synthetic.PROPAGATION`), EOG-electrode leakage and spectra are assumptions; the simulator found
  and fixed three of my own modelling errors along the way (a different random head every session, a missing 1 Hz high-pass,
  and a look's return edge not counted as an event), which shows how much the result depends on them.
- The cleaner itself uses no propagation assumption, only what calibration measures.

## Running it on the real data (not done)

For each BNCI 2014-001 subject, at 250 Hz after the same front end as the desktop pipeline (1 Hz high-pass, notches):

```python
from eog_freeze import FrozenEyeCleaner
cl = FrozenEyeCleaner(250).fit(day1_eeg, day1_eog)          # sites x samples, 3 EOG channels, day 1 only
clean = cl.transform(day2_eeg, "gls_gate")                  # no EOG from here on
flags = cl.detect(day2_eeg)                                 # EEG-only eye-event flag
```

Compare against no cleaning, the existing EOG regression and ICA, using the metrics the desktop already has: leftover
correlation with day 2's EOG channels, 4-class accuracy through the existing decoder, and agreement of `flags` with the
dataset's expert eye-artifact labels (part C of `evaluate_public.py`). The glue to the desktop `bci` package is untested
because that code was not available here. Until that passes, keep at least one eye electrode.
