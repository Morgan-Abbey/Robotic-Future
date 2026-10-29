import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armsim import SimConfig, evaluate, GenerativeDecoder
from armsim.ppo import MLP, PPO, softmax
from armsim.ppo_env import ArmBCIEnv, OBS_DIM, PolicyController, levels


class Bandit:
    """Observe a one-hot context, get +1 for choosing the matching action."""
    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.c = int(self.rng.integers(4))
        return np.eye(4, dtype=np.float32)[self.c], {}

    def step(self, a):
        r = float(a == self.c)
        self.c = int(self.rng.integers(4))
        return np.eye(4, dtype=np.float32)[self.c], r, False, False, {"idle": False, "success": False}


def test_mlp_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 6, 3], rng)
    x, w = rng.standard_normal((4, 5)), rng.standard_normal((4, 3))
    net.forward(x)
    grads = net.backward(w)                               # gradient of sum(w * output)
    for layer in (0, 2):
        for which, arr in ((0, net.W[layer]), (1, net.b[layer])):
            idx = tuple(0 for _ in arr.shape)
            old = arr[idx]
            arr[idx] = old + 1e-6
            hi = (net.forward(x) * w).sum()
            arr[idx] = old - 1e-6
            lo = (net.forward(x) * w).sum()
            arr[idx] = old
            assert abs((hi - lo) / 2e-6 - grads[layer][which][idx]) < 1e-5


def test_ppo_solves_a_contextual_bandit():
    ppo = PPO(4, 4, seed=0, hidden=32, lr=3e-3, minibatch=128)
    ppo.train(Bandit, total_steps=60_000, n_envs=4, n_steps=128)
    obs = np.eye(4, dtype=np.float32)
    assert (ppo.act(obs, greedy=True) == np.arange(4)).all()
    p = softmax(ppo.pi.forward(obs))
    assert p[np.arange(4), np.arange(4)].min() > 0.8


def test_env_api_and_reward_signs():
    env = ArmBCIEnv(accuracy=0.9, p_idle=0.0)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    goal = env.ep.goal
    toward = 4 if goal[env.ep.j] > 0 else 0               # biggest step in the right direction
    away = 0 if toward == 4 else 4
    env.pending = (False, env.pending[1])                 # no jaw event this step
    _, r_toward, *_ = env.step(toward)
    env.reset(seed=1)
    env.pending = (False, env.pending[1])
    _, r_away, *_ = env.step(away)
    assert r_toward > 0 > r_away


def test_idle_episode_penalises_motion_and_ends_on_time():
    env = ArmBCIEnv(accuracy=0.9, p_idle=1.0, idle_steps=10, idle_penalty=4.0)
    env.reset(seed=0)
    assert env.idle
    env.pending = (False, env.pending[1])
    _, r_move, *_ = env.step(4)
    env.pending = (False, env.pending[1])
    _, r_hold, term, trunc, _ = env.step(2)
    assert r_move < 0 and r_hold == 0 and not term
    for _ in range(20):
        env.pending = (False, env.pending[1])
        _, _, term, trunc, _ = env.step(2)
        if trunc:
            break
    assert trunc and not term


def test_policy_controller_runs_through_the_normal_loop_and_save_load(tmp_path):
    cfg = SimConfig()
    ppo = PPO(OBS_DIM, 5, seed=0, hidden=16)
    path = tmp_path / "p.npz"
    ppo.save(path)
    pol = PPO.load_policy(path)
    x = np.random.default_rng(0).standard_normal((3, OBS_DIM))
    assert (pol.act(x, greedy=True) == ppo.act(x, greedy=True)).all()
    r = evaluate(cfg, PolicyController(cfg, pol), lambda rng: GenerativeDecoder(cfg, 0.8, rng), range(3))
    assert 0 <= r["success"] <= 1 and levels(cfg)[2] == 0.0
