"""Train and evaluate the PPO interpreter over decoder accuracy x idle penalty x seed.

    OMP_NUM_THREADS=1 python run_ppo_grid.py          # about 25 minutes on 4 cores
"""
import json
import time
from multiprocessing import get_context
from pathlib import Path

from armsim import GenerativeDecoder, SimConfig, evaluate, idle_drift
from armsim.ppo import PPO
from armsim.ppo_env import PolicyController
from run_sweep import ACCURACIES, IDLE_SEEDS, REACH_SEEDS
from train_ppo import train

PENALTIES = (10.0, 40.0, 160.0)      # reward cost per unit of arm motion while the person rests
SEEDS = (0, 1)
STEPS = 400_000
MODELS = Path(__file__).resolve().parent / "models"
OUT = Path(__file__).resolve().parents[1] / "reports" / "arm_sim_ppo.json"


def job(args):
    acc, pen, seed = args
    cfg = SimConfig()
    path = MODELS / ("ppo_acc%d_pen%d_s%d.npz" % (round(acc * 100), pen, seed))
    t0 = time.time()
    train(acc, pen, STEPS, seed=seed, out=path, log=lambda m: None, cfg=cfg)
    ctrl = PolicyController(cfg, PPO.load_policy(path))
    f = lambda rng: GenerativeDecoder(cfg, acc, rng)
    r = evaluate(cfg, ctrl, f, REACH_SEEDS)
    r.update(accuracy=acc, penalty=pen, seed=seed, time=r["time_all"], idle=idle_drift(cfg, ctrl, f, IDLE_SEEDS),
             model=str(path.relative_to(MODELS.parent)), train_s=time.time() - t0)
    print("acc %.2f pen %5g seed %d  success %.2f  time %.1f  idle %.3f  (%.0fs)" % (
        acc, pen, seed, r["success"], r["time"], r["idle"], r["train_s"]), flush=True)
    return r


if __name__ == "__main__":
    jobs = [(a, p, s) for a in ACCURACIES for p in PENALTIES for s in SEEDS]
    with get_context("fork").Pool(4) as pool:
        results = pool.map(job, jobs, chunksize=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(steps=STEPS, penalties=PENALTIES, seeds=SEEDS, runs=results), indent=1, default=float))
    print("wrote", OUT)
