# Arm simulator for the EEG BCI

A closed-loop simulation of the last stage of the BCI: **person → decoder → interpreter → arm**. It exists to
answer "how should decoder output become arm motion, and how good does the decoder have to be?" before any of
that is built in Unity or on the real arm. Results and discussion: [`../reports/arm_sim_report.md`](../reports/arm_sim_report.md).

It is a stand-in, not your arm and not your decoder:

- The arm is three joints in normalised units (−1…1), one selected at a time, as in the desktop controller
  (left/right imagery moves the selected joint −/+, a jaw clench selects the next one). Placeholder geometry.
- The decoder is a model: noisy rest/left/right probabilities at a chosen accuracy, correlated between
  successive windows. Or it replays real decoder output (below).
- The person is a script: they want the arm at a goal, imagine the matching command, switch joints with a
  clench, see the arm half a second late and anticipate its motion. Real people also adapt to the controller; this
  does not.

## Layout

| path | what |
|---|---|
| `armsim/config.py` | `SimConfig` (all timing, tolerance and decoder-model settings) and the four candidate goals |
| `armsim/decoder.py` | `GenerativeDecoder` (accuracy-parametrised, calibrated) and `ReplayDecoder` (real recorded output) |
| `armsim/user.py` | `ScriptedUser` |
| `armsim/episode.py` | one reach (`Episode`), `evaluate` (reach metrics), `idle_drift` (false motion while resting) |
| `armsim/controllers.py` | the interpreters: `DirectArgmax`, `Evidence`, `Proportional`, `SharedAutonomy` |
| `armsim/tuning.py` | grid search under a false-motion budget |
| `armsim/ppo_env.py`, `armsim/ppo.py` | Gymnasium environment for a learned interpreter, and a small numpy PPO |
| `run_sweep.py`, `run_ppo_grid.py`, `make_report.py`, `make_animation.py` | reproduce everything in `../reports/` |
| `models/` | the trained PPO policies behind the report (a few KB each) |
| `web/` | the interactive 3D arm and cursor demo (`template.html` + `build.py` → `bci_demo.html`) and its browser checks |
| `tests/` | 33 tests (`pytest`) |

## Interactive demo (3D arm and cursor)

`web/bci_demo.html` is a single-page, in-browser version of this simulator: a 3D arm (three.js) and a cursor field
(left/right, or left/right plus up/down with the jaw clench switching axis), both driven through the same decoder model
and interpreters. It runs the scripted user (Watch) or you (hold A/← and D/→ to imagine left and right hand, Space or J for
the jaw clench; on a phone, hold the on-screen buttons). It exposes the decoder accuracy, the interpreter, its strictness,
and over/under-confident decoder output, and shows the decoder probabilities live.

- The interpreter settings and decoder-signal table are baked in from `../reports/arm_sim_results.json`; rebuild after a new
  sweep with `python web/build.py` (writes `web/bci_demo.html` for the Artifact page and `web/bci_demo_standalone.html`
  that opens in any browser). Edit `web/template.html`, not the generated files.
- The JavaScript is a port of `armsim`. `web/check_demo.js` (Playwright) runs it headless against the Python numbers,
  takes screenshots and exercises keyboard and pointer input:
  `node web/check_demo.js <three.min.js r128> <screenshot dir> [fidelity|shots|interact|all]`.
  Fidelity on the eight cells compared: success rate and mean reach time within about 1-2 s of the Python results, false
  motion at rest within noise; one cell (evidence, 75%) fails 1.4% of reaches against 0.6% in Python over 1500 runs.
- When the page is served by the live bridge (`../live/`) a Live tab appears: the same 3D arm shows the pose commanded from real or simulated decoder frames, with training controls and arm output. See `../live/README.md`.
- The arm geometry (`armAngles`, link lengths) is a placeholder. The cursor uses a wider tolerance (0.1) than the arm (0.06).

## Run it

```
pip install numpy scipy matplotlib gymnasium pytest        # no deep-learning library needed
cd sim
python -m pytest tests                                      # ~10 s
python run_sweep.py                                         # ~8 min on 4 cores  -> ../reports/arm_sim_results.json
OMP_NUM_THREADS=1 python run_ppo_grid.py                    # ~12 min on 4 cores -> ../reports/arm_sim_ppo.json, models/
python make_report.py                                       # -> ../reports/arm_sim_report.md + arm_sim_frontier.png
python make_animation.py --controller proportional --accuracy 0.75 --budget 0.25   # -> ../reports/arm_reach.gif
```

## What "tuned under a false-motion budget" means

Each interpreter is judged on two things: how long a reach takes, and how far the arm wanders while the person
only rests (the idle test, in units per minute; one standard command = 0.05 units, so "1 false step per minute"
= 0.05). A controller that moves on every noisy window reaches fast and is useless when you stop thinking about
it, so speed alone is the wrong score. Every controller gets the same small grid search at each accuracy, and
the pick is the fastest setting inside each budget. The report plots whole frontiers, not single winners.

## Using real decoder output

`ReplayDecoder` plays back what your decoder actually produced, with its real streaks and real calibration. It
needs an `.npz` with:

- `probs`: (N, 3) float, columns `[rest, left, right]`, one row per decoder window in time order
- `intent`: (N,) int, the class the person was cued to imagine at each window (0 rest, 1 left, 2 right)

```python
from armsim import ReplayDecoder, SimConfig, evaluate, Proportional
cfg = SimConfig()                                           # set dt to your window hop
f = lambda rng: ReplayDecoder.from_npz(cfg, "replay.npz", rng)
print(evaluate(cfg, Proportional(cfg), f, range(200)))
```

The desktop pipeline's pseudo-online replay (part E of `evaluate_public.py`) already produces per-window
decisions; exporting the probabilities and cues from it in this format is the missing step. Per-person files
let you tune the interpreter on each person's own error pattern.

## Writing your own interpreter

Subclass `Controller` and implement `move(probs, obs) -> Action(dq=...)`. `probs` is the decoder's
(rest, left, right); `obs` has the joint angles `q`, the selected joint `j` and the step index `t`. Jaw-clench
switching is handled for you. Anything that is not told the goal is a fair competitor; `SharedAutonomy` shows
how to use the candidate goals as scene knowledge.

## For the real arm and Unity

- The sim's step is 0.05 units per issued command, capped at 0.1 per decision. Map these to the arm's serial
  protocol (3 integers per line at 9600 baud, inferred from the Dec 2024 WinForms app) and to your step counts.
  That mapping is not modelled and the protocol is unverified.
- `armsim.ppo_env.ArmBCIEnv` is a standard Gymnasium environment (discrete actions: joint speed levels). To train
  with stable-baselines3 instead of the bundled numpy PPO: `PPO("MlpPolicy", ArmBCIEnv(accuracy=0.65)).learn(...)`.
- The environment, reward and command interface carry over to Unity; a policy trained here would not (different
  dynamics), so it would be retrained there.

## Assumptions to know about

Decoder windows hop every 0.25 s; successive outputs are correlated (ρ = 0.85, from window overlap); decoder
probabilities are calibrated unless `temperature` says otherwise; the person sees the arm 0.5 s late; a jaw clench
is detected 90% of the time with 0.3 false switches per minute; the evidence controller is modelled on the
command layer described in the desktop README (smoothing + threshold + hold + refractory), not on its code.
The sensitivity section of the report varies the four most important ones.
