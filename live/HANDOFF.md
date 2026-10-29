# Handoff for the desktop session (the one that has the pipeline code and the Cyton)

Paste the section below into the Claude session that works in `C:\Users\morga\OneDrive\Desktop\Claude Code\openbci-cyton`.
The cloud session could not see that code or any hardware, so these steps are written against what its notes say, not its source.

---

**Task: connect the pipeline to the live bridge. Do not enable or move the arm.**

The repo `Morgan-Abbey/Robotic-Future`, branch `ccr-2d566f6d-ejxwdd`, has `live/` (a local bridge + browser UI), `sim/` (interpreters,
simulator) and `reports/`. Get it with `git clone` or `git pull`, then:

1. Copy `live/bci_emit.py` into the pipeline folder. It sends small JSON datagrams to 127.0.0.1:9871 and never raises.
2. Add three hooks, each a single call, in the places where the information already exists:
   - `run_bci.py` (live loop), once per decoder window, after the class probabilities and jaw detector are computed:
     `emit({"type": "frame", "probs": [p_rest, p_left, p_right], "jaw": bool(jaw_event)})`.
     Use the same classes and order as the live commands (rest, left, right); if the decoder has a feet class, drop it and let the bridge renormalise.
     Emit the probabilities the command layer uses (after the decoder, before thresholds). Tell me whether they are calibrated.
   - `record_session.py`, at each cue: `emit({"type":"cue","class":"left"|"right"|"rest","phase":"start","trial":i,"n":N,"dur":seconds})`
     and the same with `"phase":"end"` when the cue ends.
   - `train.py`, when finished: `emit({"type":"trained","cv_acc":<cross-validated accuracy 0..1>})`.
3. Copy `live/config.example.json` to `live/config.json`. Replace the three `scripts` commands with the real ones from this
   README (board/port flags for the Cyton, layout, output paths). Set `dt` to the decoder window hop in seconds.
4. Check on the synthetic board first: run `python live/run_live.py --engine udp --config live/config.json`, open the page,
   press Record training session, then Train model, then Start live decode. Confirm cues appear, the log fills, and the decoder bars move.
5. Then with the real Cyton (arm disconnected, `--send` not passed): record a short session, train, run live, and report: the measured
   per-window accuracy, whether the probabilities look calibrated, and the end-to-end delay from EEG to the page.
6. Make sure `run_bci.py` is run **without** `--send` and `--arm-port` whenever the bridge is used. The bridge owns the serial port.

Do not set `"send": true`, do not pass `--send`, and do not press Enable. Arm commissioning is a separate step done by the user with the arm
clear of obstacles: find the firmware's real protocol, calibrate `steps_per_unit`, then use Jog at low steps.

Report what you changed, anything that did not match these assumptions, and the test results.
---
