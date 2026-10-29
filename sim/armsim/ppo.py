"""Proximal policy optimisation in plain numpy (two small tanh MLPs, manual backprop, Adam).

Written so the simulator has no deep-learning dependency. The environment is a standard Gymnasium env, so
for anything bigger use stable-baselines3 on `ArmBCIEnv` instead.
"""
import numpy as np


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class MLP:
    def __init__(self, sizes, rng, last_gain=1.0):
        self.W = [rng.standard_normal((a, b)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= last_gain
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        self.cache = [x]
        h = x
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if k < len(self.W) - 1:
                h = np.tanh(h)
            self.cache.append(h)
        return h

    def backward(self, dout):
        grads, d = [], dout
        for k in reversed(range(len(self.W))):
            grads.append((self.cache[k].T @ d, d.sum(0)))
            if k:
                d = (d @ self.W[k].T) * (1 - self.cache[k] ** 2)
        return grads[::-1]

    def params(self):
        return self.W + self.b


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]

    def step(self, grads, max_norm=0.5):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        scale = min(1.0, max_norm / (norm + 1e-8))
        self.t += 1
        for x, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * scale
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            x -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


class PPO:
    def __init__(self, obs_dim, n_actions, seed=0, hidden=64, lr=1e-3, gamma=0.99, lam=0.95, clip=0.2,
                 ent=0.01, epochs=4, minibatch=256):
        self.rng = np.random.default_rng(seed)
        self.pi = MLP([obs_dim, hidden, hidden, n_actions], self.rng, last_gain=0.01)
        self.vf = MLP([obs_dim, hidden, hidden, 1], self.rng)
        self.opt = Adam(self.pi.params() + self.vf.params(), lr)
        self.gamma, self.lam, self.clip, self.ent, self.epochs, self.mb = gamma, lam, clip, ent, epochs, minibatch
        self.lr0 = lr

    def act(self, obs, greedy=False):
        p = softmax(self.pi.forward(np.asarray(obs, float)))
        if greedy:
            return p.argmax(-1)
        return (p.cumsum(-1) > self.rng.random((len(p), 1))).argmax(-1)

    def value(self, obs):
        return self.vf.forward(np.asarray(obs, float))[:, 0]

    def update(self, obs, act, logp_old, adv, ret):
        n = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = []
        for _ in range(self.epochs):
            order = self.rng.permutation(n)
            for s in range(0, n, self.mb):
                i = order[s:s + self.mb]
                m = len(i)
                p = softmax(self.pi.forward(obs[i]))
                logp = np.log(p[np.arange(m), act[i]] + 1e-10)
                ratio = np.exp(logp - logp_old[i])
                a = adv[i]
                active = ((a >= 0) & (ratio <= 1 + self.clip)) | ((a < 0) & (ratio >= 1 - self.clip))
                dlogp = np.where(active, -a * ratio, 0.0) / m
                dz = -p * dlogp[:, None]
                dz[np.arange(m), act[i]] += dlogp
                H = -(p * np.log(p + 1e-10)).sum(-1, keepdims=True)
                dz += self.ent * p * (np.log(p + 1e-10) + H) / m
                gpi = self.pi.backward(dz)
                v = self.vf.forward(obs[i])
                gvf = self.vf.backward(0.5 * (v - ret[i][:, None]) / m)
                self.opt.step([g[0] for g in gpi] + [g[1] for g in gpi] + [g[0] for g in gvf] + [g[1] for g in gvf])
                stats.append(float(H.mean()))
        return float(np.mean(stats))

    def train(self, make_env, total_steps, n_envs=8, n_steps=256, seed=0, log=None):
        envs = [make_env() for _ in range(n_envs)]
        obs = np.stack([e.reset(seed=seed + k)[0] for k, e in enumerate(envs)])
        n_updates = max(1, total_steps // (n_envs * n_steps))
        for u in range(n_updates):
            self.opt.lr = self.lr0 * (1 - 0.9 * u / n_updates)
            O, A, L, R, D, V = [], [], [], [], [], []
            ok = idle_moved = n_reach = n_idle_eps = 0
            for _ in range(n_steps):
                p = softmax(self.pi.forward(obs))
                a = (p.cumsum(-1) > self.rng.random((n_envs, 1))).argmax(-1)
                v = self.value(obs)
                O.append(obs); A.append(a); V.append(v)
                L.append(np.log(p[np.arange(n_envs), a] + 1e-10))
                r, d, nxt = np.zeros(n_envs), np.zeros(n_envs), []
                for k, e in enumerate(envs):
                    o2, rk, term, trunc, info = e.step(a[k])
                    if trunc and not term:
                        rk += self.gamma * float(self.value(o2[None])[0])
                    if term or trunc:
                        if not info["idle"]:
                            n_reach += 1
                            ok += info["success"]
                        o2, _ = e.reset()
                    r[k], d[k] = rk, float(term or trunc)
                    nxt.append(o2)
                R.append(r); D.append(d)
                obs = np.stack(nxt)
            O, A, L, R, D, V = map(np.array, (O, A, L, R, D, V))
            last = self.value(obs)
            adv, g = np.zeros_like(R), np.zeros(n_envs)
            for t in reversed(range(n_steps)):
                nv = last if t == n_steps - 1 else V[t + 1]
                delta = R[t] + self.gamma * nv * (1 - D[t]) - V[t]
                g = delta + self.gamma * self.lam * (1 - D[t]) * g
                adv[t] = g
            ret = adv + V
            ent = self.update(O.reshape(-1, O.shape[-1]), A.reshape(-1), L.reshape(-1), adv.reshape(-1), ret.reshape(-1))
            if log and (u % 20 == 0 or u == n_updates - 1):
                log("update %d/%d  reach success %.2f (%d eps)  entropy %.2f" % (u + 1, n_updates, ok / max(n_reach, 1), n_reach, ent))
        return self

    def save(self, path):
        np.savez(path, **{"pi_W%d" % k: w for k, w in enumerate(self.pi.W)}, **{"pi_b%d" % k: b for k, b in enumerate(self.pi.b)})

    @classmethod
    def load_policy(cls, path):
        d = np.load(path)
        n = len([k for k in d if k.startswith("pi_W")])
        pol = cls.__new__(cls)
        pol.pi = MLP.__new__(MLP)
        pol.pi.W = [d["pi_W%d" % k] for k in range(n)]
        pol.pi.b = [d["pi_b%d" % k] for k in range(n)]
        return pol
