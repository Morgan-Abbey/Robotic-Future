"""Train the PPO interpreter for one decoder accuracy and idle penalty.

    python train_ppo.py --accuracy 0.65 --idle-penalty 4 --steps 300000 --out models/ppo_acc65_pen4.npz
"""
import argparse
import time
from pathlib import Path

from armsim import SimConfig
from armsim.ppo import PPO
from armsim.ppo_env import OBS_DIM, ArmBCIEnv


def train(accuracy, idle_penalty, steps, seed=0, out=None, log=print, cfg=None):
    cfg = cfg or SimConfig()
    ppo = PPO(OBS_DIM, 5, seed=seed)
    t0 = time.time()
    ppo.train(lambda: ArmBCIEnv(cfg, accuracy=accuracy, idle_penalty=idle_penalty), steps, seed=seed * 100,
              log=lambda m: log("[acc %.2f pen %g] %s (%.0fs)" % (accuracy, idle_penalty, m, time.time() - t0)))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        ppo.save(out)
    return ppo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy", type=float, default=0.65)
    ap.add_argument("--idle-penalty", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train(a.accuracy, a.idle_penalty, a.steps, a.seed, a.out)
