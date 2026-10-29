# EEG BCI: status and to-do

Last updated 2026-10-29 from a cloud session. The pipeline itself lives on the desktop
(`C:\Users\morga\OneDrive\Desktop\Claude Code\openbci-cyton`) and is **not in this repo**; the cloud session could not
reach it, and its network policy blocks the public-data hosts (`bnci-horizon-2020.eu`, `www.bbci.de`, `zenodo.org`).
To let a cloud session run the real-data work: push the pipeline folder here (without raw recordings or the unrelated
`college-essays` folder) and allow those three hosts under the environment's Network access (cloud environment menu →
Edit → Allowed domains; steps at https://code.claude.com/docs/en/cloud-environments#network-access).

## Done in the cloud session (all on simulated data)

- **Arm simulator** (`sim/`, report `reports/arm_sim_report.md`). Compares ways to turn rest/left/right decoder output into
  arm motion, scored on reach time *and* false motion while resting. Main results: argmax is fast but wanders 31-113 false
  steps/min; proportional-to-confidence beats command-style evidence at matched false motion from 65% accuracy up;
  shared autonomy (arm finishes the reach) is fastest and the only one that works at 1 false step/min with a 65% decoder;
  PPO did not beat the hand-tuned controllers; calibration of decoder probabilities matters most (under-confident output
  drops three of four controllers to ~0% of goals reached). 33 tests, numpy-only PPO included.
- **EOG for calibration only** (`analysis/`, report `reports/eog_freeze_synthetic.md`). Method, ground-truth simulator and
  10 tests. Simulation says: fixed projection fails with few channels; an EEG-only gated cleaner works only if the eye
  signal reaches the central channels strongly (about the strength implied by the desktop's 0.49 uncleaned correlation),
  and then helps eye-artifact removal and detection but not the 8-30 Hz band. **Not tested on real data.**

## Corrections to what I said earlier in the conversation

- I listed a "simulated online run" as a gap. The desktop pipeline already has one (part E of `evaluate_public.py`, the full
  live pipeline replayed over day 2 and scored as commands), plus a command layer with smoothing, threshold, hold time and
  refractory period.
- I suggested feet imagery to switch joints. The desktop already switches joints with a short jaw clench, which saves a class.
- "Fix config and README drift" had no basis; I hadn't read the code. Dropped.
- I said the decoder was "trained on cued trials, not a continuous stream". The desktop code is causal and streaming; I had
  only seen part of it.
- On PPO: it was in your plan and I anchored on it. The desktop README already says PPO is not for reading EEG and might
  suit the arm controller; the simulation found no evidence it beats simpler controllers.

## To do next (needs the desktop code or data)

1. **Export replay data from part E** (`probs` N×3 rest/left/right and cue `intent` N, per person) and run
   `sim/` with `ReplayDecoder` to tune and compare interpreters on real error patterns (format in `sim/README.md`).
2. **Check decoder calibration** (reliability curve, temperature scaling) before using confidence as speed. The simulator
   says this matters more than the choice of controller.
3. **Real-data test of EOG-calibration-only** (spec at the end of `analysis/README.md`). Keep an eye electrode until it passes.
4. **Port the proportional interpreter** (and shared autonomy if the replay result supports it) into the command layer
   (`bci/commands.py`) and compare against the current one in `evaluate_public.py` part E.
5. **Riemannian ASR** (Blum 2019) in the same day 1 → day 2 harness. Expected gain is small.
6. **Decoder improvements on `6eeg`**: filter-bank Riemannian, MDM vs tangent-space + SVM, adaptive online re-centring.
7. **Recording tooling**: eye-calibration routine in `record_session.py` (if item 3 pans out); check whether electrode-pop,
   dropout and impedance checks already exist before adding any.
8. **Real arm**: verify the serial protocol (3 integers per line at 9600 baud, inferred from the Dec 2024 WinForms app)
   against the lost ESP32/TB6600 firmware before enabling `--send`.
9. Optional: `check_bci.py` → pytest + CI.

## Decisions and notes so far

- Control is proportional to decoder confidence first; shared autonomy ("finish the movement") is the next interpreter, on
  the same interface (`sim/armsim/controllers.py`).
- Imagery assignments must stay the same between any cursor training and the arm. Pick the pair of commands the user's own
  decoder separates best (left vs. right hand was strongest).
- Four classes is a soft limit of scalp EEG; more controllable axes come from mode switching (jaw clench), not more classes.
- The Cyton has 8 inputs and `6eeg` already uses all of them. More EEG channels means dropping the eye electrodes (item 3)
  or adding the Daisy board.
- Unity PPO cannot learn to read EEG. PPO's only role would be turning decoder commands plus arm state into smooth movement,
  and only if it beats simpler controllers (so far it doesn't).
