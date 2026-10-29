"""Gymnasium environment for learning the interpreter with RL (the PPO comparison).

The policy sees the last few decoder outputs, the jaw events, the joint angles and the selected joint. It
does not see the goal. Reward is progress toward the goal, a success bonus and a small time cost on reach
episodes, and a penalty on any motion on idle episodes (the person only rests), which is what stops it
from twitching on every noisy window.
"""
import gymnasium as gym
import numpy as np

from .config import REST, TARGETS, SimConfig
from .controllers import Controller
from .decoder import GenerativeDecoder
from .episode import Action, Episode

HISTORY = 8
OBS_DIM = HISTORY * 4 + 3 + 3


def levels(cfg):
    return np.array([-cfg.max_dq, -cfg.step, 0.0, cfg.step, cfg.max_dq])


class ObsBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buf = np.zeros((HISTORY, 4))

    def push(self, probs, jaw):
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1, :3], self.buf[-1, 3] = probs, float(jaw)

    def vector(self, q, j):
        onehot = np.zeros(3)
        onehot[j] = 1.0
        return np.concatenate([self.buf.ravel(), q, onehot]).astype(np.float32)


class PolicyController(Controller):
    """Runs a trained policy through the same episode loop as the hand-written controllers."""
    name = "ppo"

    def __init__(self, cfg, policy):
        self.policy, self.levels = policy, levels(cfg)
        super().__init__(cfg)

    def reset(self):
        super().reset()
        self.ob = ObsBuilder()

    def step(self, probs, jaw, obs):
        self.ob.push(probs, jaw)
        return super().step(probs, jaw, obs)

    def move(self, probs, obs):
        a = int(self.policy.act(self.ob.vector(obs.q, obs.j)[None], greedy=True)[0])
        return Action(dq=float(self.levels[a]))


class ArmBCIEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, accuracy=0.65, idle_penalty=4.0, p_idle=0.3, idle_steps=120):
        self.cfg = cfg or SimConfig()
        self.accuracy, self.idle_penalty, self.p_idle, self.idle_steps = accuracy, idle_penalty, p_idle, idle_steps
        self.levels = levels(self.cfg)
        self.action_space = gym.spaces.Discrete(len(self.levels))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.ob = ObsBuilder()

    def _sense(self):
        probs, jaw, o = self.ep.sense()
        self.pending = (jaw, o)
        self.ob.push(probs, jaw)
        return self.ob.vector(o.q, o.j)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        goal = int(rng.integers(len(TARGETS)))
        self.idle = bool(rng.random() < self.p_idle)
        dec = GenerativeDecoder(self.cfg, self.accuracy, rng)
        self.ep = Episode(self.cfg, dec, goal, forced_intent=REST if self.idle else None)
        if self.idle:
            self.ep.q = self.ep.goal.copy()
        self.ob.reset()
        self.last_switch = -10 ** 9
        self.d0 = max(float(np.abs(self.ep.goal - self.ep.q).sum()), 1e-6)
        return self._sense(), {}

    def step(self, a):
        cfg, ep = self.cfg, self.ep
        jaw, o = self.pending
        if jaw and o.t - self.last_switch >= cfg.switch_refractory:
            self.last_switch = o.t
            action = Action(switch=True)
        else:
            action = Action(dq=float(self.levels[int(a)]))
        q0 = ep.q.copy()
        e0 = float(np.abs(ep.goal - q0).sum())
        ep.act(action)
        moved = float(np.abs(ep.q - q0).sum())
        if self.idle:
            reward = -self.idle_penalty * moved
            terminated, truncated = False, ep.t >= self.idle_steps
        else:
            reward = (e0 - float(np.abs(ep.goal - ep.q).sum())) / self.d0 - 0.002
            terminated, truncated = ep.success, ep.done and not ep.success
            reward += 1.0 if terminated else 0.0
        info = dict(idle=self.idle, success=bool(ep.success and not self.idle), moved=moved)
        return self._sense(), reward, terminated, truncated, info
