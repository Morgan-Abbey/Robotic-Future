"""Build reports/arm_sim_report.md and its figure from the sweep and PPO result files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = Path(__file__).resolve().parents[1] / "reports"
STEP = 0.05                                    # units of one standard command; idle drift / STEP = false steps per minute
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3de"
STYLE = {"direct": ("#8a8984", "s", "argmax, no smoothing"), "evidence": ("#2a78d6", "o", "evidence + hold (command style)"),
         "proportional": ("#eb6834", "^", "proportional to confidence"), "shared": ("#1baf7a", "D", "shared autonomy (finishes the reach)"),
         "ppo": ("#eda100", "P", "PPO (learned)")}
ORDER = ("direct", "evidence", "proportional", "shared", "ppo")


def load():
    res = json.loads((REP / "arm_sim_results.json").read_text())
    ppo_path = REP / "arm_sim_ppo.json"
    ppo = json.loads(ppo_path.read_text()) if ppo_path.exists() else None
    return res, ppo


def ppo_points(ppo, acc):
    """Mean over seeds for each idle penalty at one accuracy."""
    out = []
    for pen in ppo["penalties"]:
        rs = [r for r in ppo["runs"] if abs(r["accuracy"] - acc) < 1e-9 and r["penalty"] == pen]
        out.append(dict(penalty=pen, n=len(rs), success=np.mean([r["success"] for r in rs]), time=np.mean([r["time_all"] for r in rs]),
                        idle=np.mean([r["idle"] for r in rs]), wrong=np.mean([r["wrong_frac"] for r in rs]),
                        time_sd=np.std([r["time_all"] for r in rs]), idle_sd=np.std([r["idle"] for r in rs])))
    return out


def points(res, ppo, acc):
    row = res["by_accuracy"][str(acc)]
    pts = {"direct": [(row["direct"]["all"]["idle"], row["direct"]["all"]["time_all"])]}
    for name in ("evidence", "proportional", "shared"):
        pts[name] = [(row[name][str(b)]["idle"], row[name][str(b)]["time_all"]) for b in res["budgets"]]
    if ppo:
        pts["ppo"] = [(p["idle"], p["time"]) for p in ppo_points(ppo, acc)]
    return pts


def figure(res, ppo):
    accs = res["accuracies"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), facecolor=SURFACE, sharey=True)
    for ax, acc in zip(axes.flat, accs):
        pts = points(res, ppo, acc)
        for name in ORDER:
            if name not in pts:
                continue
            col, mk, _ = STYLE[name]
            x = np.maximum(np.array([p[0] for p in pts[name]]) / STEP, 0.1)
            y = np.array([p[1] for p in pts[name]])
            order = np.argsort(x)
            ax.plot(x[order], y[order], color=col, linewidth=2, zorder=2, linestyle="--" if name == "direct" else "-")
            ax.scatter(x, y, s=70, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.08, 200); ax.set_ylim(0, 95)
        ax.set_facecolor(SURFACE)
        ax.set_title("decoder accuracy %d%%" % round(acc * 100), color=INK, fontsize=10, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["≤0.1", "1", "10", "100"])
        ax.axhline(90, color=GRID, linewidth=1)
    for ax in axes[1]:
        ax.set_xlabel("false motion at rest (steps per minute, log scale)", color=MUTED, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean time to reach goal (s)", color=MUTED, fontsize=8)
    leg = axes.flat[-1]
    leg.axis("off")
    for k, name in enumerate(n for n in ORDER if n != "ppo" or ppo):
        col, mk, label = STYLE[name]
        y = 0.9 - k * 0.14
        leg.plot([0.02, 0.14], [y, y], color=col, linewidth=2, linestyle="--" if name == "direct" else "-", transform=leg.transAxes)
        leg.scatter([0.08], [y], s=70, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.5, transform=leg.transAxes, zorder=3)
        leg.text(0.18, y, label, transform=leg.transAxes, va="center", color=INK, fontsize=9)
    leg.text(0.02, 0.1, "Down and to the left is better.\nEach line joins one controller's settings:\ntuned for about 1, 5 and 20 false\nsteps/min (PPO: three idle penalties).\nFailures count as 90 s.",
             transform=leg.transAxes, va="top", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(REP / "arm_sim_frontier.png", dpi=130, facecolor=SURFACE)


def fmt(r):
    return "%.0f%% | %.1f ± %.1f | %.1f" % (100 * r["success"], r["time_all"], 1.96 * r["time_all_se"], r["idle"] / STEP)


def tables(res, ppo):
    L = []
    for acc in res["accuracies"]:
        row = res["by_accuracy"][str(acc)]
        L += ["### Decoder accuracy %d%%" % round(acc * 100), "",
              "| controller | tuned for (false steps/min) | reached goal | time to goal, s (95% CI; failures = 90 s) | measured false steps/min | wrong-way share of moves | auto-reaches (wrong) |",
              "|---|---|---|---|---|---|---|"]
        d = row["direct"]["all"]
        L.append("| argmax, no smoothing | n/a | %s | %.0f%% | 0 |" % (fmt(d), 100 * d["wrong_frac"]))
        for name in ("evidence", "proportional", "shared"):
            for b in res["budgets"]:
                r = row[name][str(b)]
                L.append("| %s | %g | %s | %.0f%% | %s |" % (STYLE[name][2], b / STEP, fmt(r), 100 * r["wrong_frac"],
                                                            "%.1f (%.1f)" % (r["auto_commits"], r["wrong_commits"]) if name == "shared" else "n/a"))
        if ppo:
            for p in ppo_points(ppo, acc):
                L.append("| PPO (penalty %g, %d seeds) | n/a | %.0f%% | %.1f ± %.1f (sd over seeds) | %.1f | %.0f%% | n/a |" % (
                    p["penalty"], p["n"], 100 * p["success"], p["time"], p["time_sd"], p["idle"] / STEP, 100 * p["wrong"]))
        L.append("")
    return L


def sensitivity(res):
    s = res["sensitivity"]
    L = ["Parameters were tuned at 65% accuracy and the 5 false steps/min budget, then the world was changed without retuning.",
         "Each cell is reached goal / time to goal (s) / false steps per minute.", ""]
    names = {"temperature": "Decoder confidence (1 = calibrated; <1 over-confident, >1 under-confident)", "rho": "Correlation between successive decoder windows",
             "delay": "User's view of the arm is late by this many decisions (0.25 s each)", "jaw_hit": "Chance an intended jaw clench is detected"}
    for key, variants in s["variants"].items():
        vals = list(variants)
        L += ["**%s**" % names[key], "", "| controller | " + " | ".join("%s = %s" % (key, v) for v in vals) + " |", "|---|" + "---|" * len(vals)]
        for name in ("direct", "evidence", "proportional", "shared"):
            cells = []
            for v in vals:
                r = variants[v][name]
                cells.append("%.0f%% / %.0f / %.1f" % (100 * r["success"], r["time_all"], r["idle"] / STEP))
            L.append("| %s | %s |" % (STYLE[name][2], " | ".join(cells)))
        L.append("")
    return L


def appendix(res):
    L = ["| accuracy | controller | tuned for (false steps/min) | parameters |", "|---|---|---|---|"]
    for acc, d in res["tuned"].items():
        for name, by_b in d.items():
            for b, p in by_b.items():
                if p:
                    L.append("| %s%% | %s | %g | %s |" % (round(float(acc) * 100), name, float(b) / STEP, ", ".join("%s=%g" % kv for kv in p.items())))
    return L


def main(intro="", findings="", limits=""):
    res, ppo = load()
    figure(res, ppo)
    cfg = res["config"]
    L = ["# Arm simulator: how the interpreter turns decoder output into arm motion", "", intro, "",
         "![time to reach vs false motion at rest](arm_sim_frontier.png)", "", findings, "",
         "## Results by decoder accuracy", ""] + tables(res, ppo) + ["## Sensitivity", ""] + sensitivity(res) + \
        ["## What this does and does not tell you", "", limits, "", "## Settings", "",
         "| setting | value |", "|---|---|"] + ["| %s | %s |" % (k, v) for k, v in cfg.items()] + \
        ["", "Goals: 4 fixed joint-space targets (`armsim/config.py`). Reach episodes: 300 per cell (fresh seeds, never used for tuning); idle test: 40 two-minute runs per cell.",
         "", "## Tuned parameters", ""] + appendix(res)
    (REP / "arm_sim_report.md").write_text("\n".join(L))
    print("wrote", REP / "arm_sim_report.md")


INTRO = """This compares ways of turning the decoder's rest/left/right output into arm motion, on a simulated 3-joint arm
with a scripted person (see `../sim/README.md`). Each interpreter is judged on two things: how long a reach takes, and
how far the arm wanders while the person only rests, measured in "false steps per minute" (one step = one standard
command). All numbers are simulation results at assumed per-window 3-class decoder accuracies. They are not
measurements of your system; map your decoder to the closest column, or replay its real output (`ReplayDecoder`)."""

FINDINGS = """## Findings

1. **Speed alone is the wrong score.** Plain argmax (move whenever a window says left/right) reaches the goal quickly
   (21 s at 65% accuracy) but wanders 72 false steps per minute while the person rests (31 at 85%, 113 at 45%).
   Smoothing, thresholds and hold times exist to stop that, and they cost speed. Compare controllers along the
   frontier in the figure, not by one number.
2. **Proportional-to-confidence beats command-style evidence at the same false-motion level, from 65% accuracy up.**
   At 75% accuracy and about 3 false steps/min: 29 s against 47 s. At 85% and about 1 false step/min: 29 s against 54 s.
   At 55% accuracy neither finishes reliably: even at 8-12 false steps/min they reach only about 75% of goals within 90 s.
3. **At about 1 false step per minute the ordinary interpreters need roughly 75% accuracy or better.** At 65% the
   evidence controller finishes 1% of reaches and the proportional one 22%; at 75% they finish 71% and 87%; at 85% both
   finish everything (54 s and 29 s). This is the same direction as the desktop's live-command result (about 1 false
   command/min but only 23% hits), though the two are not directly comparable.
4. **Shared autonomy (the arm finishes the reach once the goal is clear) is the fastest at every accuracy**, and the only
   interpreter that works at 1 false step/min with a 65% decoder (89% of reaches, 45 s; 100% and 25 s at 5/min). The costs:
   it needs the candidate goals to be known (4 assumed here); about 1 in 10 auto-reaches heads for the wrong goal at 65%
   (1 in 5 at 45%) and the person has to correct it; and its false motion at rest comes as rare large jumps, which the
   average steps/min understates.
5. **PPO did not beat the hand-tuned interpreters.** At 85% it lands in the same range as proportional (11-26 s at 2-12 false
   steps/min across settings and seeds); at 65-75% it is no faster than proportional at similar false motion; at 55% and below it mostly
   fails to learn. Seeds differ a lot (65% accuracy, penalty 160: one seed reaches 74% of goals, the other 0%). With only 400k
   training steps, 2 seeds and no hyperparameter search this is "no evidence it helps here", not "it cannot".
6. **Calibration matters more than the choice of controller.** Thresholds tuned on calibrated probabilities collapse when the
   decoder is under-confident: at temperature 2, evidence and proportional reach 0% of goals and shared autonomy 11%.
   Over-confident output (temperature 0.5) is fast but moves about 4-5 times more at rest (proportional: 13.6 against 2.8
   false steps/min; shared autonomy: 18.4 against 3.8). Check calibration, or retune thresholds, per person and per session before using confidence as speed.
7. **Slower user feedback hurts the command-style controllers; jaw detection quality barely matters.** Going from no delay
   to 1 s (4 decisions) in the user's view of the arm cuts evidence from 89% to 53% of goals reached and proportional from
   94% to 78%; shared autonomy stays at 99-100%. A jaw-clench detection rate anywhere from 70% to 98% changes little."""

LIMITS = """- **The person is a script.** They always imagine the right command, anticipate the arm's motion and never get tired. Real
  users adapt to the controller and to feedback (better or worse), which only real sessions can show.
- **The decoder is a model.** Gaussian logits, successive windows correlated at 0.85, probabilities calibrated unless
  stated. Real decoders have person-specific error patterns (the desktop reports 3 of 9 people at 38-64% live hits),
  session drift and uncalibrated confidence. `ReplayDecoder` exists to use real output instead; none was available here.
- **Accuracy columns are per-window 3-class accuracy (rest/left/right).** The desktop's 70% left/right and 52% 4-class are
  different class sets, so the mapping to a column is approximate.
- **The arm is three joints in normalised units** with instant response, placeholder geometry and no serial protocol.
- **The evidence controller is a model of the command layer described in the desktop README** (smoothing + threshold +
  hold + refractory), not that code, which also tunes per person and may do better than this grid search.
- **Budgets are tuning targets, not guarantees.** Settings are picked on tuning seeds and measured on fresh ones, so the
  measured false steps/min can differ from the "tuned for" label. Compare on the frontier.
- **PPO was given a modest budget** (400k steps, 2 seeds per cell, one network size, no tuning). One goal set of 4 targets."""

if __name__ == "__main__":
    main(INTRO, FINDINGS, LIMITS)
