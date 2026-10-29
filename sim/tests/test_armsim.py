import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armsim import (CONTROLLERS, LEFT, REST, RIGHT, SWITCH, TARGETS, Episode, GenerativeDecoder, ReplayDecoder,
                    SimConfig, accuracy_to_signal, evaluate, idle_drift, run_episode)
from armsim.controllers import Evidence, Proportional, SharedAutonomy
from armsim.episode import Action, Obs
from armsim.tuning import configs, GRIDS, pareto, pick

CFG = SimConfig()


def decoder(acc, seed=0, cfg=CFG):
    return GenerativeDecoder(cfg, acc, np.random.default_rng(seed))


@pytest.mark.parametrize("acc", [0.5, 0.7, 0.9])
def test_decoder_hits_target_accuracy(acc):
    d = decoder(acc)
    hits = []
    for _ in range(40):                                  # many independent streams, correlated within each
        d = decoder(acc, seed=len(hits))
        for c in (REST, LEFT, RIGHT):
            hits += [int(np.argmax(d.step(c)[0])) == c for _ in range(300)]
    assert abs(np.mean(hits) - acc) < 0.03


def test_decoder_probabilities_are_calibrated():
    d = decoder(0.7)
    conf, ok = [], []
    for i in range(30000):
        c = (REST, LEFT, RIGHT)[i % 3]
        p, _ = d.step(c)
        conf.append(p.max())
        ok.append(int(np.argmax(p)) == c)
    conf, ok = np.array(conf), np.array(ok)
    for lo, hi in ((0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        m = (conf >= lo) & (conf < hi)
        assert abs(ok[m].mean() - conf[m].mean()) < 0.06


def test_temperature_changes_confidence_not_decisions():
    cold = GenerativeDecoder(SimConfig(temperature=0.5), 0.7, np.random.default_rng(1))
    hot = GenerativeDecoder(SimConfig(temperature=2.0), 0.7, np.random.default_rng(1))
    for _ in range(50):
        pc, ph = cold.step(LEFT)[0], hot.step(LEFT)[0]
        assert np.argmax(pc) == np.argmax(ph) and pc.max() >= ph.max()


def test_signal_grows_with_accuracy():
    assert accuracy_to_signal(0.3) == 0.0
    assert accuracy_to_signal(0.5) < accuracy_to_signal(0.7) < accuracy_to_signal(0.9)


def test_jaw_clench_detected_at_configured_rate():
    d = decoder(0.7)
    assert abs(np.mean([d.step(SWITCH)[1] for _ in range(4000)]) - CFG.jaw_hit) < 0.03
    d = decoder(0.7)
    per_min = np.mean([d.step(REST)[1] for _ in range(40000)]) * 60 / CFG.dt
    assert abs(per_min - CFG.jaw_false_per_min) < 0.15


@pytest.mark.parametrize("name", list(CONTROLLERS))
def test_every_controller_reaches_every_goal_with_a_near_perfect_decoder(name):
    for goal in range(len(TARGETS)):
        r = run_episode(CFG, CONTROLLERS[name](CFG), lambda rng: GenerativeDecoder(CFG, 0.995, rng), goal, 7 + goal)
        assert r["success"], (name, goal)


def test_shared_autonomy_is_fastest_with_a_good_decoder():
    f = lambda rng: GenerativeDecoder(CFG, 0.9, rng)
    t = {n: evaluate(CFG, CONTROLLERS[n](CFG), f, range(40))["time_all"] for n in ("proportional", "evidence", "shared")}
    assert t["shared"] < t["proportional"] < t["evidence"]


def test_arm_respects_joint_limits_and_speed_limit():
    ep = Episode(CFG, decoder(0.9), 0)
    for _ in range(200):
        before = ep.q.copy()
        ep.act(Action(dq=5.0))
        assert np.all(np.abs(ep.q) <= CFG.q_limit) and np.abs(ep.q - before).max() <= CFG.max_dq + 1e-12
    ep = Episode(CFG, decoder(0.9), 0)
    ep.act(Action(auto=np.array([1.0, -1.0, 1.0])))
    assert np.abs(ep.q).max() <= CFG.max_dq + 1e-12


def test_success_needs_the_arm_to_stay_put():
    ep = Episode(CFG, decoder(0.9), 0)
    ep.q = ep.goal.copy()
    ep.act(Action())
    assert not ep.done
    ep.act(Action())
    assert ep.done and ep.success
    ep = Episode(CFG, decoder(0.9), 0)
    ep.q = ep.goal.copy()
    ep.act(Action())
    ep.q = ep.goal + 0.5                                  # leaves before the dwell is over
    ep.act(Action())
    assert not ep.done


def test_timeout_without_success():
    ep = Episode(CFG, decoder(0.9), 0)
    for _ in range(CFG.n_steps):
        ep.act(Action())
    assert ep.done and not ep.success and ep.result()["time"] == CFG.t_max


def test_user_switches_joint_when_the_current_one_is_done():
    ep = Episode(CFG, decoder(0.99), 0)
    ep.q = np.array([TARGETS[0][0], 0.0, 0.0])           # joint 0 already there, joint 1 is not
    intents = []
    for _ in range(6):
        ep.user.observe(ep.q, ep.j)
        intents.append(ep.user.intent())
    assert intents[0] == SWITCH and intents[1] == REST    # then waits for the switch to become visible


def test_user_stays_quiet_when_the_arm_is_heading_for_the_goal():
    ep = Episode(CFG, decoder(0.99), 2)
    ep.user.observe(ep.q, ep.j)
    assert ep.user.intent(auto_target=2) == REST
    assert ep.user.intent(auto_target=1) != REST


def test_evidence_needs_hold_and_respects_refractory():
    c = Evidence(CFG, alpha=1.0, theta=0.5, hold=3, refractory=2)
    left = np.array([0.1, 0.8, 0.1])
    out = [c.step(left, False, Obs(np.zeros(3), 0, t)).dq for t in range(8)]
    assert out == [0, 0, -CFG.step, 0, 0, 0, 0, -CFG.step]


def test_proportional_speed_follows_confidence_and_rest_holds():
    c = Proportional(CFG, alpha=1.0, theta=0.5, gamma=1.0, vmax=0.4, hold=1)
    weak = c.step(np.array([0.2, 0.0 + 0.55, 0.25]), False, Obs(np.zeros(3), 0, 0)).dq
    c.reset()
    strong = c.step(np.array([0.05, 0.0, 0.95]), False, Obs(np.zeros(3), 0, 0)).dq
    c.reset()
    held = c.step(np.array([0.7, 0.2, 0.1]), False, Obs(np.zeros(3), 0, 0)).dq
    assert weak < 0 < strong and abs(strong) > abs(weak) and held == 0


def test_jaw_switch_respects_refractory_in_every_controller():
    for C in CONTROLLERS.values():
        c = C(CFG)
        a = [c.step(np.array([1.0, 0, 0]), True, Obs(np.zeros(3), 0, t)).switch for t in range(4)]
        assert a == [True, False, True, False]


def test_shared_autonomy_commits_to_the_goal_the_user_is_heading_for():
    c = SharedAutonomy(CFG, kappa=0.6, tau=0.9)
    q, t = np.zeros(3), 0
    for _ in range(30):                       # "right" on joint 0 fits goals 0 and 2 but not 1 or 3
        assert c.step(np.array([0.05, 0.05, 0.9]), False, Obs(q, 0, t)).auto_target is None
        t += 1
    assert c.b[[0, 2]].sum() > 0.95 and abs(c.b[0] - c.b[2]) < 0.2
    commit = None
    for _ in range(30):                       # "left" on joint 1 separates goal 0 (-0.4) from goal 2 (+0.7)
        a = c.step(np.array([0.05, 0.9, 0.05]), False, Obs(q, 1, t))
        t += 1
        if a.auto_target is not None:
            commit = a.auto_target
            break
    assert commit == 0


def test_shared_autonomy_parked_on_a_target_and_resting_does_not_move():
    c = SharedAutonomy(CFG, kappa=0.3, tau=0.9)
    q = TARGETS[0].copy()
    for t in range(400):
        a = c.step(np.array([0.9, 0.05, 0.05]), False, Obs(q, 0, t))
        assert a.dq == 0 and (a.auto is None or np.abs(a.auto).max() == 0)


def test_idle_drift_is_zero_for_a_perfect_resting_decoder():
    class Quiet:
        def step(self, intent):
            return np.array([1.0, 0, 0]), False
    ctrl = Proportional(CFG)
    assert idle_drift(CFG, ctrl, lambda rng: Quiet(), range(3)) == 0.0


def test_replay_decoder_replays_runs_of_the_asked_class():
    rng = np.random.default_rng(0)
    probs = np.concatenate([np.tile([0.9, 0.05, 0.05], (20, 1)), np.tile([0.1, 0.8, 0.1], (20, 1)),
                            np.tile([0.1, 0.1, 0.8], (20, 1))])
    intent = np.repeat([REST, LEFT, RIGHT], 20)
    d = ReplayDecoder(CFG, probs, intent, rng)
    assert np.argmax(d.step(LEFT)[0]) == LEFT and np.argmax(d.step(LEFT)[0]) == LEFT
    assert np.argmax(d.step(RIGHT)[0]) == RIGHT
    assert np.argmax(d.step(SWITCH)[0]) == REST           # a clench is rest to the EEG decoder
    for _ in range(100):                                  # runs are reused when they run out
        d.step(RIGHT)
    with pytest.raises(ValueError):
        ReplayDecoder(CFG, probs[:20], intent[:20], rng)


def test_replay_decoder_loads_npz(tmp_path):
    f = tmp_path / "r.npz"
    np.savez(f, probs=np.tile([0.4, 0.3, 0.3], (30, 1)), intent=np.repeat([0, 1, 2], 10))
    assert ReplayDecoder.from_npz(CFG, f, np.random.default_rng(0)).step(RIGHT)[0].shape == (3,)


def test_pareto_and_pick():
    pts = [(0.0, 50), (0.1, 30), (0.1, 40), (0.5, 10), (0.6, 20)]
    assert pareto(pts) == [0, 1, 3]
    res = [dict(idle=i, time=t, params=k) for k, (i, t) in enumerate(pts)]
    assert pick(res, 0.1)["params"] == 1
    assert pick(res, 0.0)["params"] == 0
    assert pick([dict(idle=0.3, time=1, params="a"), dict(idle=0.2, time=9, params="b")], 0.05)["params"] == "b"


def test_shared_autonomy_gate_ignores_noise_while_idle():
    weak_right, weak_left = np.array([0.25, 0.15, 0.6]), np.array([0.25, 0.6, 0.15])
    for gate, expect_commit in ((1, False), (0, True)):
        c = SharedAutonomy(CFG, kappa=0.6, tau=0.9, theta=0.8, gate=gate)
        q, commit = np.zeros(3), False
        for t in range(300):                      # never confident enough to move; joint 0 "right", joint 1 "left"
            commit |= c.step(weak_right if t < 150 else weak_left, False, Obs(q, 0 if t < 150 else 1, t)).auto_target is not None
        assert commit == expect_commit


def test_grids_are_complete():
    assert len(configs(GRIDS["evidence"])) == 144 and configs(GRIDS["direct"]) == [{}]
    for name, grid in GRIDS.items():
        for cfg in configs(grid):
            CONTROLLERS[name](CFG, **cfg)
