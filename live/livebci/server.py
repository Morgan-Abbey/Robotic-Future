"""The bridge: decoder frames in, interpreter, arm out, browser UI over a WebSocket.

The browser is only a display and a set of controls. The interpreter and the arm output run here, so closing or
freezing the tab cannot leave the arm running: with no browser heard from for heartbeat_timeout_s the arm is
disabled, and with no decoder frames for frame_timeout_s nothing is driven.
"""
import asyncio
import hmac
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import websockets
from websockets.asyncio.server import serve

from .arm import ArmError, ArmPort, open_port
from .engine import MockEngine, UdpEngine
from .plant import Plant, make_controller
from .runner import Runner

WEB = Path(__file__).resolve().parents[2] / "sim" / "web" / "bci_demo_standalone.html"
INTERPRETERS = ("direct", "evidence", "proportional", "shared")
BUDGETS = ("0.05", "0.25", "1.0")


class Bridge:
    def __init__(self, cfg, allow_send=False, config_dir=".", html_path=WEB, arm_opener=open_port, engine=None, clock=time.monotonic):
        self.cfg, self.clock, self.html_path = cfg, clock, Path(html_path)
        self.token = secrets.token_urlsafe(16)
        self.engine = engine or (MockEngine(cfg["dt"]) if cfg["engine"] == "mock" else UdpEngine(cfg["host"], cfg["udp_port"]))
        self.arm = ArmPort(cfg["arm"], allow_send, arm_opener, clock)
        self.settings = dict(interp=cfg["interpreter"], budget=str(cfg["budget"]), profile=int(cfg["profile_accuracy"]))
        self._build_controller()
        self.runner = Runner(cfg["scripts"], self._script_line, self._script_exit, config_dir)
        self.clients, self.last_frame, self.last_seen, self.stale = set(), None, clock(), False
        self.notice = None
        self.last_state = None

    # ------------------------------------------------------------------ pipeline
    def _build_controller(self):
        s = self.settings
        sim_cfg, self.ctrl = make_controller(s["interp"], self.cfg["dt"], s["profile"], s["budget"], self.cfg["targets"])
        old = getattr(self, "plant", None)
        self.plant = Plant(sim_cfg)
        if old is not None:
            self.plant.q, self.plant.j, self.plant.t = old.q, old.j, old.t

    def broadcast(self, msg):
        if self.clients:
            websockets.broadcast(self.clients, json.dumps(msg))

    def on_frame(self, f):
        self.last_frame = self.clock()
        if self.stale:
            self.stale = False
        probs = np.asarray(f["probs"], float)
        action = self.ctrl.step(probs, bool(f["jaw"]), self.plant.obs())
        move = self.plant.apply(action)
        try:
            sent = self.arm.update(self.plant.q)
        except ArmError as e:
            sent, self.notice = [0, 0, 0], str(e)
        auto = getattr(self.ctrl, "auto", None)
        self.last_state = dict(type="state", t=time.time(), probs=probs.tolist(), jaw=bool(f["jaw"]), intent=f.get("intent"), move=move,
                               q=self.plant.q.tolist(), j=int(self.plant.j), auto=None if auto is None else int(auto),
                               arm=dict(enabled=self.arm.enabled, sent=sent, pending=self.arm.pending(self.plant.q)))
        self.broadcast(self.last_state)

    def on_event(self, e):
        self.broadcast(dict(e, ts=time.time()))

    def _script_line(self, name, line):
        self.broadcast(dict(type="log", name=name, line=line, ts=time.time()))

    def _script_exit(self, name, code):
        self.broadcast(dict(type="script_exit", name=name, code=code))
        self.broadcast(self.status())

    def status(self):
        return dict(type="status", engine=self.engine.name, settings=self.settings, arm=self.arm.status(), scripts=self.runner.running(),
                    stale=self.stale, notice=self.notice, available=dict(interpreters=INTERPRETERS, budgets=BUDGETS),
                    scripts_defined=sorted(self.cfg["scripts"]), dt=self.cfg["dt"])

    # ------------------------------------------------------------------ browser messages
    async def handle(self, msg):
        kind, self.notice = msg.get("type"), None
        if kind == "ping":
            return
        if kind == "settings":
            s = dict(self.settings)
            if msg.get("interp") in INTERPRETERS:
                s["interp"] = msg["interp"]
            if str(msg.get("budget")) in BUDGETS:
                s["budget"] = str(msg["budget"])
            if isinstance(msg.get("profile"), (int, float)):
                s["profile"] = int(msg["profile"])
            self.settings = s
            self._build_controller()
        elif kind == "intent":
            self.engine.set_intent(int(msg.get("value", 0)))
            return
        elif kind == "arm":
            self._arm_action(msg)
        elif kind == "script":
            await self._script_action(msg)
        else:
            self.notice = "unknown message type %r" % (kind,)
        self.broadcast(self.status())

    def _arm_action(self, msg):
        act, arm = msg.get("action"), self.arm
        try:
            if act == "enable":
                arm.enable()
            elif act == "disable":
                arm.disable()
            elif act == "estop":
                arm.estop()
            elif act == "reset_estop":
                arm.reset_estop()
            elif act == "zero":
                arm.zero()
                self.plant.zero()
            elif act == "jog":
                j = int(msg.get("joint", 0))
                step = arm.jog(j, int(msg.get("steps", 0)))
                spu = self.cfg["arm"]["steps_per_unit"][j] or 1
                self.plant.q[j] += (step[j] / spu) * (-1 if self.cfg["arm"]["invert"][j] else 1)
            else:
                self.notice = "unknown arm action %r" % (act,)
        except ArmError as e:
            self.notice = str(e)

    async def _script_action(self, msg):
        name, act = msg.get("name"), msg.get("action", "start")
        if act == "stop":
            await self.runner.stop(name)
        elif name == "record" and self.engine.name == "mock":
            self.engine.start_demo_session()
            self.broadcast(dict(type="log", name="record", line="mock engine: running a short demo training session", ts=time.time()))
        elif name == "live" and self.engine.name == "mock":
            self.broadcast(dict(type="log", name="live", line="mock engine: simulated decoder frames are already streaming", ts=time.time()))
        else:
            try:
                await self.runner.start(name)
            except (KeyError, OSError) as e:
                self.notice = "could not start %s: %s" % (name, e)

    async def handler(self, ws):
        try:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 3))
        except Exception:
            await ws.close(4001, "no hello")
            return
        if hello.get("type") != "hello" or not hmac.compare_digest(str(hello.get("token", "")), self.token):
            await ws.close(4003, "bad token")
            return
        self.clients.add(ws)
        self.last_seen = self.clock()
        try:
            await ws.send(json.dumps(dict(type="hello_ok", engine=self.engine.name, version=1)))
            await ws.send(json.dumps(self.status()))
            if self.last_state:
                await ws.send(json.dumps(self.last_state))
            async for raw in ws:
                self.last_seen = self.clock()
                try:
                    msg = json.loads(raw)
                    if isinstance(msg, dict):
                        await self.handle(msg)
                except (ValueError, TypeError, KeyError) as e:
                    await ws.send(json.dumps(dict(type="error", message=str(e))))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    # ------------------------------------------------------------------ housekeeping
    async def watchdog(self):
        while True:
            await asyncio.sleep(0.25)
            now = self.clock()
            stale = self.last_frame is not None and now - self.last_frame > self.cfg["frame_timeout_s"]
            if stale and not self.stale:
                self.stale = True
                self.arm._stop() if self.arm.enabled else None
                self.broadcast(self.status())
            if self.arm.enabled and now - self.last_seen > self.cfg["heartbeat_timeout_s"]:
                self.arm.disable()
                self.notice = "browser not heard from: arm output disabled"
                self.broadcast(self.status())

    def page(self):
        inject = "<script>window.BCI_LIVE = %s;</script>\n" % json.dumps(dict(ws="ws://%s:%d/" % (self.cfg["host"], self.cfg["ws_port"]), token=self.token))
        html = self.html_path.read_text()
        marker = '<script src="https://cdnjs'
        return (html.replace(marker, inject + marker, 1) if marker in html else html + inject).encode()

    def start_http(self):
        bridge = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body, ctype, code = bridge.page(), "text/html; charset=utf-8", 200
                elif self.path == "/healthz":
                    body, ctype, code = b"ok", "text/plain", 200
                else:
                    body, ctype, code = b"not found", "text/plain", 404
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        httpd = ThreadingHTTPServer((self.cfg["host"], self.cfg["http_port"]), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd

    async def run(self, ready=None):
        httpd = self.start_http()
        origins = ["http://%s:%d" % (h, self.cfg["http_port"]) for h in (self.cfg["host"], "localhost", "127.0.0.1")] + [None]
        tasks = []
        try:
            async with serve(self.handler, self.cfg["host"], self.cfg["ws_port"], origins=origins):
                tasks = [asyncio.create_task(self.engine.run(self.on_frame, self.on_event)), asyncio.create_task(self.watchdog())]
                if ready:
                    ready.set()
                await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            self.engine.close()
            await self.runner.stop_all()
            self.arm.close()
            httpd.shutdown()
