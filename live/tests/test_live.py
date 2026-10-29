import asyncio
import json
import socket
import urllib.request

import numpy as np
import pytest
import websockets
from websockets.asyncio.client import connect

from livebci import config
from livebci.arm import ArmError, ArmPort
from livebci.engine import MockEngine, UdpEngine, normalise
from livebci.plant import Plant, make_controller, tuned_params
from livebci.server import Bridge


def free_port(udp=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM if udp else socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Sink:
    def __init__(self):
        self.lines, self.fail = [], False

    def write(self, data):
        if self.fail:
            raise OSError("port gone")
        self.lines.append(data.decode())

    def close(self):
        pass


def arm_cfg(**kw):
    c = config.load(None, {"arm": dict(port="COMX", send=True, steps_per_unit=[100, 200, 300])})["arm"]
    c.update(kw)
    return c


def make_arm(allow=True, sink=None, **kw):
    sink = sink or Sink()
    clock = {"t": 0.0}
    arm = ArmPort(arm_cfg(**kw), allow, lambda *_: sink, lambda: clock["t"])
    return arm, sink, clock


# ---------------------------------------------------------------------- arm safety
def test_arm_blocked_by_default():
    arm = ArmPort(config.load()["arm"])
    assert len(arm.blockers()) == 4
    with pytest.raises(ArmError, match="cannot enable"):
        arm.enable()
    assert not arm.enabled


def test_arm_needs_every_condition():
    for kw in (dict(allow=False), dict(send=False), dict(port=None), dict(steps_per_unit=[100, 0, 300])):
        allow = kw.pop("allow", True)
        arm, _, _ = make_arm(allow=allow, **kw)
        assert arm.blockers(), kw
        with pytest.raises(ArmError):
            arm.enable()
    arm, _, _ = make_arm()
    arm.enable()
    assert arm.enabled and not arm.blockers()


def test_dry_run_never_writes_but_counts():
    arm, sink, clock = make_arm()
    arm.update(np.array([0.1, 0.0, 0.0]))
    assert sink.lines == [] and arm.dry == [10, 0, 0] and arm.n_dry == 1 and arm.n_sent == 0


def test_enabled_update_sends_scaled_steps_and_tracks_position():
    arm, sink, clock = make_arm()
    arm.enable()
    arm.update(np.array([0.1, -0.05, 0.02]))
    assert sink.lines == ["10 -10 6\n"] and arm.sent == [10, -10, 6]
    clock["t"] += 1
    arm.update(np.array([0.1, -0.05, 0.02]))
    assert len(sink.lines) == 1                       # already there: nothing to send


def test_rate_limit_and_per_message_cap_keep_the_remainder():
    arm, sink, clock = make_arm(max_steps_per_message=30, max_rate_hz=2.0)
    arm.enable()
    arm.update(np.array([0.9, 0, 0]))                 # wants 90 steps
    assert sink.lines == ["30 0 0\n"]
    arm.update(np.array([0.9, 0, 0]))                 # too soon
    assert len(sink.lines) == 1
    clock["t"] += 0.6
    arm.update(np.array([0.9, 0, 0]))
    assert sink.lines[-1] == "30 0 0\n" and arm.pending(np.array([0.9, 0, 0])) == [30, 0, 0]


def test_travel_limit_and_inversion():
    arm, sink, clock = make_arm(limit_units=0.5, invert=[True, False, False])
    arm.enable()
    arm.update(np.array([1.0, 1.0, 0.0]))
    assert arm.sent == [-40, 40, 0]                   # clamped to 0.5 units, and the first joint is inverted
    for _ in range(10):
        clock["t"] += 1
        arm.update(np.array([1.0, 1.0, 0.0]))
    assert arm.sent == [-50, 100, 0]


def test_estop_latches_and_stops():
    arm, sink, clock = make_arm(stop_line="STOP\n")
    arm.enable()
    arm.estop()
    assert not arm.enabled and arm.estopped and sink.lines[-1] == "STOP\n"
    with pytest.raises(ArmError, match="e-stop"):
        arm.enable()
    arm.reset_estop()
    arm.enable()
    assert arm.enabled


def test_zero_redefines_the_origin():
    arm, sink, clock = make_arm()
    arm.enable()
    arm.update(np.array([0.2, 0, 0]))
    arm.zero()
    assert arm.sent == [0, 0, 0]
    clock["t"] += 1
    arm.update(np.array([0.2, 0, 0]))
    assert sink.lines[-1] == "20 0 0\n"               # the pose is still 0.2 units, so it moves another 20 from the new zero


def test_jog_needs_enable_and_is_capped():
    arm, sink, clock = make_arm(max_jog_steps=5)
    with pytest.raises(ArmError):
        arm.jog(0, 3)
    arm.enable()
    assert arm.jog(2, 50) == [0, 0, 5] and sink.lines[-1] == "0 0 5\n"


def test_write_failure_disables_the_arm():
    arm, sink, clock = make_arm()
    arm.enable()
    sink.fail = True
    with pytest.raises(ArmError, match="write failed"):
        arm.update(np.array([0.5, 0, 0]))
    assert not arm.enabled and "write failed" in arm.status()["error"]


# ---------------------------------------------------------------------- plant and interpreters
def test_tuned_params_and_controllers_for_every_interpreter():
    for name in ("direct", "evidence", "proportional", "shared"):
        for budget in ("0.05", "0.25", "1.0"):
            cfg, ctrl = make_controller(name, 0.25, 65, budget, config.DEFAULTS["targets"])
            assert ctrl is not None
    assert tuned_params("proportional", 83, "0.25") == tuned_params("proportional", 85, "0.25")
    assert tuned_params("direct", 65, "0.25") == {}


def test_plant_applies_interpreter_actions_within_limits():
    cfg, ctrl = make_controller("proportional", 0.25, 65, "0.25", config.DEFAULTS["targets"])
    plant = Plant(cfg)
    probs = np.array([0.02, 0.01, 0.97])
    moves = [plant.apply(ctrl.step(probs, False, plant.obs())) for _ in range(60)]
    assert "right" in moves and 0 < plant.q[0] <= 1.0 and plant.q[1] == 0
    jaw = plant.apply(ctrl.step(probs, True, plant.obs()))
    assert jaw == "switch joint" and plant.j == 1


# ---------------------------------------------------------------------- engines
async def test_mock_engine_follows_intent_and_runs_a_demo_session():
    eng = MockEngine(dt=0.02, accuracy=0.95, seed=1)
    frames, events = [], []
    task = asyncio.create_task(eng.run(frames.append, events.append))
    eng.set_intent(2)
    await asyncio.sleep(0.5)
    right = [int(np.argmax(f["probs"])) for f in frames[-15:]]
    assert right.count(2) >= 10
    eng.start_demo_session(trials_per_class=1, cue_s=0.1, rest_s=0.05)
    await asyncio.sleep(0.8)
    task.cancel()
    kinds = [(e["type"], e.get("phase"), e.get("class")) for e in events]
    assert ("cue", "start", "left") in kinds and ("cue", "start", "right") in kinds and any(k[0] == "trained" for k in kinds)


def test_normalise_handles_bad_probabilities():
    assert normalise([0, 0, 0]) == pytest.approx([1 / 3] * 3)
    assert normalise([2, 1, 1, 9]) == pytest.approx([0.5, 0.25, 0.25])
    assert normalise([-1, 1, 1]) == pytest.approx([0, 0.5, 0.5])


async def test_udp_engine_parses_frames_and_events_and_ignores_junk():
    port = free_port(udp=True)
    eng = UdpEngine("127.0.0.1", port)
    frames, events = [], []
    task = asyncio.create_task(eng.run(frames.append, events.append))
    await asyncio.sleep(0.1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for d in (b'{"type":"frame","probs":[0.2,0.3,0.5,0.0],"jaw":true}', b"not json", b'{"type":"frame","probs":[1]}', b'[1,2]',
              b'{"type":"cue","class":"left","phase":"start"}', b'{"type":"nope"}'):
        s.sendto(d, ("127.0.0.1", port))
    await asyncio.sleep(0.2)
    eng.close()
    task.cancel()
    assert len(frames) == 1 and frames[0]["probs"] == pytest.approx([0.2, 0.3, 0.5]) and frames[0]["jaw"] is True
    assert [e["type"] for e in events] == ["cue"]


# ---------------------------------------------------------------------- the bridge end to end
class Running:
    def __init__(self, tmp_path, **over):
        self.tmp = tmp_path
        self.log = tmp_path / "arm.log"
        base = dict(http_port=free_port(), ws_port=free_port(), udp_port=free_port(udp=True), dt=0.03, heartbeat_timeout_s=0.6, frame_timeout_s=0.3,
                    arm=dict(port="file:%s" % self.log, send=True, steps_per_unit=[100, 100, 100], max_rate_hz=100.0))
        base.update(over)
        self.cfg = config.load(None, base)
        self.html = tmp_path / "page.html"
        self.html.write_text('<html><script src="https://cdnjs.cloudflare.com/x.js"></script></html>')

    async def __aenter__(self):
        self.bridge = Bridge(self.cfg, allow_send=True, html_path=self.html)
        self.ready = asyncio.Event()
        self.task = asyncio.create_task(self.bridge.run(self.ready))
        await asyncio.wait_for(self.ready.wait(), 5)
        return self

    async def __aexit__(self, *a):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass

    @property
    def uri(self):
        return "ws://127.0.0.1:%d/" % self.cfg["ws_port"]

    async def client(self, token=None, **kw):
        ws = await connect(self.uri, **kw)
        await ws.send(json.dumps({"type": "hello", "token": token or self.bridge.token}))
        return ws


async def collect(ws, want, timeout=3.0):
    """Read messages until one of type `want` arrives; returns it."""
    end = asyncio.get_running_loop().time() + timeout
    while True:
        raw = await asyncio.wait_for(ws.recv(), max(0.01, end - asyncio.get_running_loop().time()))
        m = json.loads(raw)
        if m["type"] == want:
            return m


async def state_where(ws, pred, timeout=4.0):
    """Read state frames (skipping any backlog) until one satisfies pred."""
    end = asyncio.get_running_loop().time() + timeout
    while True:
        m = await collect(ws, "state", max(0.01, end - asyncio.get_running_loop().time()))
        if pred(m):
            return m


async def status_where(ws, pred, timeout=3.0):
    """Read status messages until one satisfies pred (the greeting status predates any action)."""
    end = asyncio.get_running_loop().time() + timeout
    while True:
        m = await collect(ws, "status", max(0.01, end - asyncio.get_running_loop().time()))
        if pred(m):
            return m


async def test_page_has_token_and_health(tmp_path):
    async with Running(tmp_path) as r:
        loop = asyncio.get_running_loop()
        get = lambda p: loop.run_in_executor(None, lambda: urllib.request.urlopen("http://127.0.0.1:%d%s" % (r.cfg["http_port"], p)).read().decode())
        page = await get("/")
        assert "window.BCI_LIVE" in page and r.bridge.token in page and page.index("BCI_LIVE") < page.index("cdnjs")
        assert await get("/healthz") == "ok"


async def test_bad_token_and_foreign_origin_are_refused(tmp_path):
    async with Running(tmp_path) as r:
        ws = await r.client(token="wrong")
        with pytest.raises(websockets.ConnectionClosed):
            await asyncio.wait_for(ws.recv(), 2)
        assert ws.close_code == 4003
        with pytest.raises(Exception):
            await connect(r.uri, origin="http://evil.example")


async def test_state_stream_settings_and_intent(tmp_path):
    async with Running(tmp_path) as r:
        ws = await r.client()
        assert (await collect(ws, "hello_ok"))["engine"] == "mock"
        st = await collect(ws, "status")
        assert st["settings"]["interp"] == "proportional" and not st["arm"]["enabled"]
        await ws.send(json.dumps({"type": "settings", "interp": "direct"}))
        st = await status_where(ws, lambda s: s["settings"]["interp"] == "direct")
        assert st["settings"]["interp"] == "direct"
        await ws.send(json.dumps({"type": "intent", "value": 2}))
        m = await state_where(ws, lambda m: m["q"][0] > 0.02)      # holding "right" moves joint 0
        assert len(m["probs"]) == 3 and len(m["q"]) == 3 and m["j"] == 0
        await ws.close()


async def test_arm_gating_enable_move_estop_and_heartbeat(tmp_path):
    async with Running(tmp_path) as r:
        ws = await r.client()
        await collect(ws, "hello_ok")
        await ws.send(json.dumps({"type": "arm", "action": "enable"}))
        await status_where(ws, lambda s: s["arm"]["enabled"])
        await ws.send(json.dumps({"type": "settings", "interp": "direct"}))
        await ws.send(json.dumps({"type": "intent", "value": 2}))
        await asyncio.sleep(1.0)
        lines = r.log.read_text().split("\n")
        assert any(l.strip() and int(l.split()[0]) > 0 for l in lines)     # real motion lines reached the sink
        await ws.send(json.dumps({"type": "arm", "action": "estop"}))
        st = await status_where(ws, lambda s: s["arm"]["estopped"])
        assert st["arm"]["estopped"] and not st["arm"]["enabled"]
        n = len(r.log.read_text())
        await asyncio.sleep(0.4)
        assert len(r.log.read_text()) == n                        # nothing more after the e-stop
        await ws.send(json.dumps({"type": "arm", "action": "enable"}))
        st = await status_where(ws, lambda s: s["notice"] and "e-stop" in s["notice"])
        await ws.send(json.dumps({"type": "arm", "action": "reset_estop"}))
        await ws.send(json.dumps({"type": "arm", "action": "enable"}))
        await status_where(ws, lambda s: s["arm"]["enabled"])
        await asyncio.sleep(r.cfg["heartbeat_timeout_s"] + 0.8)    # a silent browser: no pings
        assert not r.bridge.arm.enabled and "not heard from" in (r.bridge.notice or "")
        await ws.close()


async def test_enable_refused_without_calibration(tmp_path):
    async with Running(tmp_path, arm=dict(steps_per_unit=[0, 0, 0])) as r:
        ws = await r.client()
        await collect(ws, "hello_ok")
        await ws.send(json.dumps({"type": "arm", "action": "enable"}))
        st = await status_where(ws, lambda s: s["notice"])
        assert not st["arm"]["enabled"] and "steps_per_unit" in st["notice"]


async def test_demo_training_session_and_script_runner(tmp_path):
    scripts = {"train": {"cwd": ".", "cmd": ["python", "-c", "print('hello from train'); print('done')"]}}
    async with Running(tmp_path, scripts=scripts) as r:
        ws = await r.client()
        await collect(ws, "hello_ok")
        await ws.send(json.dumps({"type": "script", "name": "record", "action": "start"}))
        cue = await collect(ws, "cue", 4)
        assert cue["phase"] == "start" and cue["class"] in ("left", "right")
        await ws.send(json.dumps({"type": "script", "name": "train", "action": "start"}))
        seen = []
        while "done" not in seen:
            m = await collect(ws, "log", 5)
            if m["name"] == "train":
                seen.append(m["line"])
        assert seen[0] == "hello from train"


async def test_malformed_messages_do_not_kill_the_connection(tmp_path):
    async with Running(tmp_path) as r:
        ws = await r.client()
        await collect(ws, "hello_ok")
        for bad in ("not json", json.dumps([1, 2]), json.dumps({"type": "arm", "action": "jog", "joint": "x"}), json.dumps({"type": "zzz"})):
            await ws.send(bad)
        await ws.send(json.dumps({"type": "ping"}))
        assert (await collect(ws, "state"))["type"] == "state"
