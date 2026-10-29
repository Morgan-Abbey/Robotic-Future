"""Tune the hand-written interpreters at each decoder accuracy and evaluate them on fresh episodes.

    python run_sweep.py                 # writes ../reports/arm_sim_results.json (about 10 minutes on 4 cores)
"""
import json
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

from armsim import SimConfig
from armsim.tuning import IDLE_BUDGET, measure, tune

ACCURACIES = (0.45, 0.55, 0.65, 0.75, 0.85)
BUDGETS = (IDLE_BUDGET, 0.25, 1.0)             # units/min at rest: about 1, 5 and 20 false steps per minute
REACH_SEEDS = list(range(100_000, 100_300))
IDLE_SEEDS = list(range(200_000, 200_040))
OUT = Path(__file__).resolve().parents[1] / "reports" / "arm_sim_results.json"


def final(pool, cfg, name, accuracy, param_list):
    jobs = [(cfg, name, p, dict(accuracy=accuracy), REACH_SEEDS, IDLE_SEEDS) for p in param_list]
    return pool.map(measure, jobs)


def main():
    cfg = SimConfig()
    t0 = time.time()
    results = {"config": cfg.__dict__, "budgets": BUDGETS, "accuracies": ACCURACIES, "by_accuracy": {}}
    tuned = {}
    for acc in ACCURACIES:
        row, tuned[acc] = {}, {}
        pool = get_context("fork").Pool(4)
        try:
            row["direct"] = {"all": final(pool, cfg, "direct", acc, [{}])[0]}
        finally:
            pool.close(); pool.join()
        for name in ("evidence", "proportional"):
            picks, frontier = tune(cfg, name, acc, budgets=BUDGETS)
            tuned[acc][name] = picks
            pool = get_context("fork").Pool(4)
            try:
                res = final(pool, cfg, name, acc, [picks[b] for b in BUDGETS])
            finally:
                pool.close(); pool.join()
            row[name] = {str(b): r for b, r in zip(BUDGETS, res)}
            row[name]["frontier"] = frontier
            print("acc %.2f %-12s %.0fs" % (acc, name, time.time() - t0), flush=True)
        shared, frontier = {}, []
        for b in BUDGETS:
            picks, fr = tune(cfg, "shared", acc, base_params=tuned[acc]["proportional"][b], budgets=(b,))
            shared[b] = picks[b]
            frontier += fr
        tuned[acc]["shared"] = shared
        pool = get_context("fork").Pool(4)
        try:
            res = final(pool, cfg, "shared", acc, [shared[b] for b in BUDGETS])
        finally:
            pool.close(); pool.join()
        row["shared"] = {str(b): r for b, r in zip(BUDGETS, res)}
        row["shared"]["frontier"] = frontier
        print("acc %.2f shared       %.0fs" % (acc, time.time() - t0), flush=True)
        results["by_accuracy"][str(acc)] = row

    # Sensitivity: keep the parameters tuned at 65% accuracy / 5 false steps per minute and change the world.
    acc, b = 0.65, 0.25
    sens = {}
    variants = {"temperature": (0.5, 1.0, 2.0), "rho": (0.5, 0.85, 0.95), "delay": (0, 2, 4),
                "jaw_hit": (0.7, 0.9, 0.98)}
    pool = get_context("fork").Pool(4)
    try:
        for key, values in variants.items():
            sens[key] = {}
            for v in values:
                c2 = replace(cfg, **{key: v})
                sens[key][str(v)] = {}
                for name in ("direct", "evidence", "proportional", "shared"):
                    p = {} if name == "direct" else tuned[acc][name][b]
                    jobs = [(c2, name, p, dict(accuracy=acc), REACH_SEEDS[:150], IDLE_SEEDS[:20])]
                    sens[key][str(v)][name] = pool.map(measure, jobs)[0]
            print("sensitivity", key, "%.0fs" % (time.time() - t0), flush=True)
    finally:
        pool.close(); pool.join()
    results["sensitivity"] = {"accuracy": acc, "budget": b, "variants": sens}
    results["tuned"] = {str(a): {n: {str(k): v for k, v in d.items()} for n, d in t.items()} for a, t in tuned.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, default=float))
    print("done in %.0fs -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
