"""Sources of decoder frames.

MockEngine plays the simulator's decoder, steered by the browser's keys, so the whole bridge can be tried without
EEG. UdpEngine listens for JSON datagrams from your real pipeline (see bci_emit.py):

    {"type": "frame", "probs": [rest, left, right], "jaw": false}      one per decoder window
    {"type": "cue", "class": "left", "phase": "start", "trial": 3, "n": 40, "dur": 4.0}
    {"type": "trained", "cv_acc": 0.71}      {"type": "log", "line": "..."}
"""
import asyncio
import json
import time

import numpy as np

from armsim import GenerativeDecoder, SimConfig
from armsim.config import LEFT, REST, RIGHT, SWITCH

CLASSES = {"rest": REST, "left": LEFT, "right": RIGHT}


def normalise(probs):
    p = np.clip(np.asarray(probs, float)[:3], 0.0, None)
    s = p.sum()
    return (p / s if s > 0 else np.full(3, 1 / 3)).tolist()


class Engine:
    name = "base"

    async def run(self, on_frame, on_event):
        raise NotImplementedError

    def set_intent(self, value):
        pass

    def start_demo_session(self, **kw):
        return False

    def close(self):
        pass


class MockEngine(Engine):
    """Generative decoder at a fixed accuracy; the intent comes from the browser, or from a demo training session."""
    name = "mock"

    def __init__(self, dt=0.25, accuracy=0.7, seed=None):
        self.cfg, self.accuracy = SimConfig(dt=dt), accuracy
        self.decoder = GenerativeDecoder(self.cfg, accuracy, np.random.default_rng(seed))
        self.intent, self.demo, self._pulse = REST, None, 0

    def set_intent(self, value):
        if value == SWITCH:
            self._pulse = 1                               # a clench lasts one frame
        else:
            self.intent = int(value)

    def start_demo_session(self, trials_per_class=3, cue_s=2.0, rest_s=1.0):
        order = [c for _ in range(trials_per_class) for c in ("left", "right")]
        self.demo = dict(order=order, i=0, phase="rest", until=0.0, cue_s=cue_s, rest_s=rest_s, n=len(order))
        return True

    def _demo_tick(self, now):
        """Advance the demo training session; returns the events to send this frame."""
        d, ev = self.demo, []
        if now < d["until"]:
            return ev
        if d["phase"] == "rest":
            if d["i"] >= d["n"]:
                ev.append({"type": "trained", "cv_acc": round(self.accuracy, 3), "model": "mock"})
                self.demo, self.intent = None, REST
                return ev
            cls = d["order"][d["i"]]
            d.update(phase="cue", until=now + d["cue_s"])
            self.intent = CLASSES[cls]
            ev.append({"type": "cue", "class": cls, "phase": "start", "trial": d["i"] + 1, "n": d["n"], "dur": d["cue_s"]})
        else:
            ev.append({"type": "cue", "class": d["order"][d["i"]], "phase": "end", "trial": d["i"] + 1, "n": d["n"]})
            d.update(phase="rest", until=now + d["rest_s"], i=d["i"] + 1)
            self.intent = REST
        return ev

    async def run(self, on_frame, on_event):
        next_t = time.monotonic()
        while True:
            now = time.monotonic()
            if self.demo:
                for e in self._demo_tick(now):
                    on_event(e)
            intent = SWITCH if self._pulse else self.intent
            self._pulse = 0
            probs, jaw = self.decoder.step(intent)
            on_frame(dict(t=now, probs=probs.tolist(), jaw=bool(jaw), intent=int(intent)))
            next_t += self.cfg.dt
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))


class UdpEngine(Engine):
    name = "udp"

    def __init__(self, host="127.0.0.1", port=9871):
        self.host, self.port, self.transport = host, port, None

    async def run(self, on_frame, on_event):
        loop = asyncio.get_running_loop()

        class Proto(asyncio.DatagramProtocol):
            def datagram_received(_, data, addr):
                try:
                    msg = json.loads(data.decode())
                except ValueError:
                    return
                if not isinstance(msg, dict):
                    return
                if msg.get("type") == "frame" and isinstance(msg.get("probs"), (list, tuple)) and len(msg["probs"]) >= 3:
                    on_frame(dict(t=time.monotonic(), probs=normalise(msg["probs"]), jaw=bool(msg.get("jaw")), intent=None))
                elif msg.get("type") in ("cue", "trained", "log"):
                    on_event(msg)

        self.transport, _ = await loop.create_datagram_endpoint(Proto, local_addr=(self.host, self.port))
        await asyncio.Event().wait()

    def close(self):
        if self.transport:
            self.transport.close()
