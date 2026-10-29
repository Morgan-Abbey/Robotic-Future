# EEG BCI: to-do list for future sessions

## Before a cloud session can do any of this
- The code is on the desktop (`C:\Users\morga\OneDrive\Desktop\Claude Code\openbci-cyton`), not in this repo.
  Push that folder (code, README, `reports/`) to this repo first. Leave out raw recordings and unrelated folders such as `college-essays`.
- Check whether the cloud container can download the public 9-subject dataset. If it can't, new code can be written and unit-tested here, but evals have to run on the desktop.

## Where things stand (from the desktop session, 2026-10-29)
- Riemannian decoder: 70% left/right and 52% 4-class with 3 EEG channels, against 62% / 38% for the PSD baseline.
- `--layout 6eeg` (C3, Cz, C4, CP3, CP4, FCz + 2 eye electrodes): 57.8% 4-class.
- 8 EEG channels with no eye electrodes: about 60% 4-class, but the eye signal is left in.
- ICA eye removal leaves a residual eye correlation of about 0.06 to 0.11. MCA was tested and dropped (about 20% removal against about 90% for ICA).
- ASR variants change decoding by under 1%.
- Full numbers are in the desktop README and `reports/public_eval.md`.

## To do, in rough priority order
1. **EOG only for calibration (calibrate, freeze, run without eye electrodes).**
   - Fit the eye removal using the EOG on day 1 and freeze it. Run day 2 with the EOG hidden.
   - Compare against no cleaning, the current ICA with EOG at runtime, and MCA, on 4-class accuracy and leftover eye correlation against the held-out EOG.
   - Without EOG at runtime the cleaner doesn't know when the eye moves. Try (a) a fixed projection that removes the learned eye directions, and (b) a detector trained on EEG-only features with EOG as labels, followed by subtract-or-hold during eye events.
   - With only 3 EEG channels, a projection that removes two eye directions leaves one, so (b) is the realistic option there.
   - Check whether the current ICA step already fits on calibration data and applies frozen online. If it does, the remaining work is removing the need for EOG at runtime.
   - If it works: add a 1-minute eye calibration routine to `record_session.py` (blinks, up/down, left/right). Expect to repeat it each session.
2. **Simulated online run.** Stream the public data through the pipeline in sliding windows with a confidence threshold and a rest gate. Report latency, false-trigger rate and usable accuracy.
3. **Arm simulator in Python (Gymnasium) under `sim/`.**
   - Swappable interpreter: decoder probabilities and arm state in, joint velocities out.
   - Build the proportional-to-confidence interpreter first: smoothed confidence above a threshold sets speed, with a deadband, hysteresis and speed/acceleration limits.
   - Check that the decoder's probabilities are calibrated before using confidence as speed.
   - Commands come from the measured confusion matrix and from replayed per-trial decoder predictions. The simulated user is scripted ("imagine toward target, rest on arrival").
   - Later: add the shared-autonomy "finish the movement" interpreter. Add PPO only as a comparison against the simpler controllers, and drop it if it doesn't clearly win.
   - Model axis switching for a 3-axis arm (one axis at a time, a separate command to cycle the axis) and measure how much slower it is.
   - Output: an animation, plus a report of reach rate and time to reach at several decoder accuracies.
4. **Riemannian ASR (Blum 2019).** Test it in the same 9-subject day 1 to day 2 harness. Expected gain is small.
5. **Decoder improvements on `6eeg`.** Filter-bank Riemannian, MDM vs. tangent-space plus SVM, adaptive online re-centring.
6. **Recording tooling and safety.** Electrode-pop and dropout detection, a pre-session impedance and quality check, and a written recording protocol with cue timing and the `6eeg` placement guide.
7. **Tests and cleanup.** Turn the self-checks into a pytest suite, add CI, fix config and README drift.

## Decisions and notes so far
- Control is proportional to decoder confidence first. Shared autonomy comes later on the same interface.
- Imagery assignments must stay the same between any cursor training and the arm, so the decoder and the user's practice carry over. Pick the pair of commands the user's own decoder separates best (left vs. right hand was strongest).
- Four classes is a soft limit of scalp EEG. More controllable axes come from mode switching, not more classes.
- Cyton has 8 inputs, and `6eeg` already uses all of them. More EEG channels means dropping the eye electrodes (item 1) or adding the Daisy board.
- Unity PPO can't learn to read EEG. PPO's only role would be turning decoder commands plus arm state into smooth movement, and only if it beats simpler controllers.
