# EOG for calibration only: synthetic check

Ground-truth simulation (`analysis/synthetic.py`): 1/f + mu/beta EEG, blinks and looks, assumed
propagation factors (large frontal, small central), EOG electrodes that also pick up some brain activity.
Fit on 3 min of deliberate eye movements, test on 5 min of mostly blinks, mean of 8 simulated people.
**This checks the algorithm under stated assumptions. It is not evidence about a real head or the
BNCI data; that test is still to be run (see `analysis/README.md`).**

Columns: error relative to the true brain signal while the eye is active, while it is quiet (damage to
clean data), and in the 8-30 Hz band the decoder uses; then leftover correlation with the true vertical
and horizontal eye signals (before cleaning these are what the desktop evals report as 0.49 / 0.28).

## 3 EEG (C3 Cz C4)

| method | error, eye active | error, eye quiet | error, 8-30 Hz | leftover eye corr (up-down / left-right) |
|---|---|---|---|---|
| no cleaning | 0.229 | 0.021 | 0.007 | 0.14 / 0.02 |
| EOG regression (needs EOG at run time) | 0.122 | 0.047 | 0.031 | 0.05 / 0.01 |
| frozen: orthogonal projection, always on | 0.844 | 0.843 | 0.842 | 0.03 / 0.01 |
| frozen: GLS, always on | 0.745 | 0.745 | 0.745 | 0.03 / 0.01 |
| frozen: GLS, slow (<7 Hz) only | 0.551 | 0.517 | 0.141 | 0.05 / 0.01 |
| frozen: GLS, broadband, gated | 0.250 | 0.087 | 0.079 | 0.14 / 0.02 |
| frozen: GLS, slow only, gated | 0.251 | 0.085 | 0.035 | 0.14 / 0.02 |

Detect-and-hold instead of subtracting: an EEG-only eye-event detector (blinks 80-160 uV, looks 30-70 uV).

| threshold (x resting sd) | blinks caught | looks caught | false alarms / min | time flagged |
|---|---|---|---|---|
| 3 | 7% | 4% | 1.2 | 1% |
| 4 | 0% | 0% | 0.0 | 0% |
| 6 | 0% | 0% | 0.0 | 0% |
| 8 | 0% | 0% | 0.0 | 0% |

## 6 EEG (+CP3 CP4 FCz)

| method | error, eye active | error, eye quiet | error, 8-30 Hz | leftover eye corr (up-down / left-right) |
|---|---|---|---|---|
| no cleaning | 0.394 | 0.035 | 0.012 | 0.17 / 0.02 |
| EOG regression (needs EOG at run time) | 0.107 | 0.058 | 0.037 | 0.04 / 0.01 |
| frozen: orthogonal projection, always on | 0.617 | 0.612 | 0.612 | 0.04 / 0.02 |
| frozen: GLS, always on | 0.379 | 0.366 | 0.371 | 0.04 / 0.02 |
| frozen: GLS, slow (<7 Hz) only | 0.424 | 0.252 | 0.074 | 0.05 / 0.01 |
| frozen: GLS, broadband, gated | 0.346 | 0.071 | 0.118 | 0.08 / 0.02 |
| frozen: GLS, slow only, gated | 0.407 | 0.065 | 0.047 | 0.10 / 0.02 |

Detect-and-hold instead of subtracting: an EEG-only eye-event detector (blinks 80-160 uV, looks 30-70 uV).

| threshold (x resting sd) | blinks caught | looks caught | false alarms / min | time flagged |
|---|---|---|---|---|
| 3 | 77% | 28% | 2.2 | 10% |
| 4 | 44% | 9% | 0.0 | 4% |
| 6 | 15% | 2% | 0.0 | 1% |
| 8 | 2% | 0% | 0.0 | 0% |

## How strong must the eye signal be on the scalp?

6-channel layout again, with every propagation factor multiplied by the scale below (1 = the assumed values,
e.g. 0.25 / 0.08 at FCz and 0.08 at Cz). Frozen cleaner = GLS, broadband, gated. The desktop evals report a
leftover-eye correlation of about 0.49 / 0.28 (up-down / left-right) on real data before cleaning; compare it with the
'uncleaned corr' column to see which row looks most like the real data (correlations are measured slightly differently).

| scale | uncleaned corr | error, eye active (none -> cleaned) | error, eye quiet | error, 8-30 Hz (none -> cleaned) | corr after | slow-only variant: active / 8-30 Hz | blinks caught at 4x | false alarms / min |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.17 / 0.02 | 0.39 -> 0.35 | 0.07 | 0.012 -> 0.118 | 0.08 / 0.02 | 0.41 / 0.047 | 44% | 0.0 |
| 2 | 0.31 / 0.04 | 0.79 -> 0.43 | 0.10 | 0.025 -> 0.164 | 0.06 / 0.03 | 0.71 / 0.063 | 95% | 0.1 |
| 3 | 0.41 / 0.05 | 1.18 -> 0.48 | 0.12 | 0.037 -> 0.190 | 0.06 / 0.04 | 1.01 / 0.073 | 100% | 0.2 |
| 4 | 0.49 / 0.06 | 1.58 -> 0.53 | 0.13 | 0.050 -> 0.211 | 0.05 / 0.05 | 1.31 / 0.085 | 100% | 0.2 |
| 8 | 0.70 / 0.09 | 3.15 -> 0.68 | 0.16 | 0.099 -> 0.279 | 0.05 / 0.06 | 2.54 / 0.131 | 100% | 0.1 |
