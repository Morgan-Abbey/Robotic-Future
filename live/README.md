# Live bridge: train, decode and drive the arm from one window

```
  your pipeline (openbci-cyton)                    this bridge (live/run_live.py)                  browser (sim/web page)
  record_session.py ── cue events ──UDP──►                                                   ┌─ Training: cues, log, buttons
  train.py ───────── "trained" ─────UDP──►  engine ─► interpreter ─► arm output (serial) ◄───┤─ Arm output: Enable, E-STOP, jog
  run_bci.py ─────── decoder frames ─UDP──►         (sim/armsim)    safety limits            └─ 3D arm showing the commanded pose
```

Your existing pipeline stays the engine: it reads the Cyton, cleans, decodes and records. The bridge adds what it did not
have: a window that walks you through training, shows the decoder live, runs the interpreters from `sim/` (proportional
control, shared autonomy) and owns the arm connection with safety limits. The page is only a display and controls; the
interpreter and the arm output run in the bridge, so a frozen or closed tab cannot leave the arm driven.

## Try it now, no hardware

```
pip install -r live/requirements.txt
python live/run_live.py                    # simulated decoder, arm output OFF; opens http://127.0.0.1:8765/
```

The Live tab drives the 3D arm from the bridge. Hold `A`/`D` (or ←/→) for left/right hand and `Space` for the jaw clench: the
simulated decoder follows, the interpreter moves the commanded pose. Record training session runs a short demo with cues.
`--arm-port file:arm.log --send` (and `steps_per_unit` in a config) lets you watch the exact lines that would go to the arm.

## Connect your pipeline

1. Copy `live/bci_emit.py` next to your pipeline's scripts. Add three one-line hooks (details in `HANDOFF.md`):
   `emit({"type":"frame","probs":[rest,left,right],"jaw":jaw})` per decoder window in `run_bci.py`;
   `emit({"type":"cue",...})` at each cue in `record_session.py`; `emit({"type":"trained","cv_acc":...})` in `train.py`.
2. Copy `live/config.example.json` to `live/config.json` and check the three `scripts` against your README (the defaults
   are guesses). Set `dt` to your decoder's window hop.
3. `python live/run_live.py --engine udp --config live/config.json`. Record, Train and Start live decode now run your scripts
   and stream their output into the log. Run your pipeline's decode **without** `--send` or `--arm-port`: the bridge is
   the only thing that may write to the arm's port.
4. Before trusting it: with real EEG the interpreter settings (tuned on a simulated decoder at 45-85% accuracy) need
   checking against your decoder's measured accuracy and calibration. See `reports/arm_sim_report.md`, finding 6.

## Arm safety model

Nothing is ever written to the arm unless all of these hold: the bridge was started with `--send`; the config has `"send": true`,
a serial `port`, and a calibrated `steps_per_unit` for all three joints; you pressed Enable and then Confirm in the page
(two deliberate clicks); and no e-stop is latched. Then:

- **E-STOP** (red button, always on screen, or Esc anywhere in the page) disables output immediately and latches until Reset e-stop.
- **Watchdogs:** no browser message for `heartbeat_timeout_s` disables the arm; no decoder frames for `frame_timeout_s` stops driving it
  (and sends `stop_line` if your firmware has one). A failed serial write disables the arm.
- **Limits:** travel is clamped to `limit_units` either side of zero, each message to `max_steps_per_message`, the rate to `max_rate_hz`.
- **Dead reckoning:** the bridge only knows the steps it sent. Home the arm by hand, then press Set zero here. If steps are
  lost or the arm is moved by hand, the drawn pose and the real pose diverge.
- **The serial protocol is unverified.** `format` defaults to three integers per line (relative steps), which a session note says the
  Dec 2024 WinForms app sent, but the ESP32/TB6600 firmware was lost, so whether the numbers mean relative steps, absolute steps
  or speeds is unknown. Use `"port": "file:arm.log"` to see the lines, check them against the firmware, then Jog a few steps with
  the arm powered and clear of obstacles before enabling real motion.
- The bridge listens on 127.0.0.1 only. The browser page needs a one-time token from the page itself and a matching origin, so
  other websites open in your browser cannot talk to it. Any program on your computer can still send frames to the UDP port.

## What was tested, and what was not

Tested (automated, with the simulated decoder and a file standing in for the serial port): the arm safety rules (22 Python tests,
including a real WebSocket client for token and origin refusal, e-stop, heartbeat shutdown, malformed input, UDP parsing, script
launching, the demo training flow) and the whole stack in headless Chromium (`e2e_check.js`: live tab, two-step enable, keyboard
driving moving the pose and writing lines, Esc stopping all writes, cue display, a closed tab leaving the arm disabled).

**Not tested, because it needs your hardware or your code:** the three hooks in your pipeline; the default script commands;
a real Cyton stream; real serial ports and the arm; Windows-specific behaviour; end-to-end latency with a real decoder. Treat the
first run with the real arm as a commissioning step, with the arm clear of anything it could hit and a hand on Esc.

## Run the tests

```
cd live && python -m pytest                                   # 22 tests, ~20 s
python run_live.py --config <cfg with arm port "file:arm.log", send true, steps_per_unit set> --send --no-browser &
node e2e_check.js <three.min.js r128> http://127.0.0.1:8765/ arm.log shots     # needs `npm i playwright`
```
