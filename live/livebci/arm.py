"""Arm output with the safety rules in one place.

Nothing is written unless all of these hold: sending was allowed on the command line, a port is set, every joint's
steps_per_unit is calibrated (> 0), the operator pressed Enable, and no e-stop is latched. The software only
dead-reckons the arm's pose from the steps it sent (steppers are open loop), so "zero" means "wherever the arm
is right now" and must be set by hand after homing.
"""
import time


class ArmError(Exception):
    pass


class FileSink:
    """Stands in for a serial port: appends each line to a file. Useful for a dry run with no arm."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def write(self, data):
        self.f.write(data.decode() if isinstance(data, bytes) else data)

    def close(self):
        self.f.close()


def open_port(port, baud):
    if str(port).startswith("file:"):
        return FileSink(str(port)[5:])
    import serial                                        # pyserial, imported late so dry runs don't need it
    return serial.Serial(port, baud, timeout=0, write_timeout=1)


class ArmPort:
    def __init__(self, cfg, allow_send=False, opener=open_port, clock=time.monotonic):
        self.c, self.allow_send, self.opener, self.clock = cfg, allow_send, opener, clock
        self.enabled, self.estopped = False, False
        self.sent = [0, 0, 0]                            # steps sent since the last zero
        self.dry = [0, 0, 0]                             # steps that would have been sent
        self.sink, self.last_send, self.n_sent, self.n_dry, self.last_error = None, -1e9, 0, 0, None

    # ------------------------------------------------------------------ gating
    def blockers(self):
        out = []
        if not self.allow_send:
            out.append("start the bridge with --send")
        if not self.c.get("send"):
            out.append('set "send": true in the config')
        if not self.c.get("port"):
            out.append('set the serial "port" in the config')
        if any(s <= 0 for s in self.c["steps_per_unit"]):
            out.append('calibrate "steps_per_unit" for all three joints')
        return out

    def status(self):
        return dict(enabled=self.enabled, estopped=self.estopped, blockers=self.blockers(), sent=list(self.sent),
                    would_send=list(self.dry), n_sent=self.n_sent, n_dry=self.n_dry, port=self.c.get("port"),
                    protocol_verified=bool(self.c.get("protocol_verified")), error=self.last_error)

    def enable(self):
        if self.estopped:
            raise ArmError("e-stop is latched; press Reset e-stop first")
        if self.blockers():
            raise ArmError("cannot enable: " + "; ".join(self.blockers()))
        if self.sink is None:
            self.sink = self.opener(self.c["port"], self.c["baud"])
        self.enabled = True
        self.last_error = None

    def disable(self):
        was = self.enabled
        self.enabled = False
        if was:
            self._stop()

    def estop(self):
        self.enabled, self.estopped = False, True
        self._stop()

    def reset_estop(self):
        self.estopped = False

    def close(self):
        self.disable()
        if self.sink is not None:
            try:
                self.sink.close()
            finally:
                self.sink = None

    def zero(self):
        """Declare the arm's present pose as zero. Only the operator can know it is homed."""
        self.sent, self.dry = [0, 0, 0], [0, 0, 0]

    def _stop(self):
        line = self.c.get("stop_line")
        if line and self.sink is not None:
            self._write(line)

    # ------------------------------------------------------------------ motion
    def _steps_for(self, q):
        lim = self.c["limit_units"]
        out = []
        for j in range(3):
            u = max(-lim, min(lim, float(q[j]))) * (-1 if self.c["invert"][j] else 1)
            out.append(int(round(u * self.c["steps_per_unit"][j])))
        return out

    def update(self, q):
        """Move toward pose q (simulator units). Returns what was sent (or would have been) this call."""
        target = self._steps_for(q) if all(s > 0 for s in self.c["steps_per_unit"]) else list(self.dry)
        base = self.sent if self.enabled else self.dry
        delta = [t - b for t, b in zip(target, base)]
        if not any(delta):
            return [0, 0, 0]
        now = self.clock()
        if self.enabled and now - self.last_send < 1.0 / self.c["max_rate_hz"] - 1e-9:
            return [0, 0, 0]                              # rate limit: keep the remainder for the next call
        cap = self.c["max_steps_per_message"]
        step = [max(-cap, min(cap, d)) for d in delta]
        if self.enabled:
            self._send(step)
        else:
            self.dry = [b + s for b, s in zip(base, step)]
            self.n_dry += 1
        return step

    def jog(self, joint, steps):
        """Small manual move of one joint, for checking direction and scale. Returns the steps actually sent."""
        if not self.enabled:
            raise ArmError("enable the arm before jogging")
        cap = self.c["max_jog_steps"]
        steps = max(-cap, min(cap, int(steps)))
        step = [0, 0, 0]
        step[int(joint)] = steps
        self._send(step)
        return step

    def pending(self, q):
        if not all(s > 0 for s in self.c["steps_per_unit"]):
            return [0, 0, 0]
        target = self._steps_for(q)
        base = self.sent if self.enabled else self.dry
        return [t - b for t, b in zip(target, base)]

    def _send(self, step):
        a, b, c = step
        self._write(self.c["format"].format(a=a, b=b, c=c))
        self.sent = [s + d for s, d in zip(self.sent, step)]
        self.last_send, self.n_sent = self.clock(), self.n_sent + 1

    def _write(self, text):
        try:
            self.sink.write(text.encode())
        except Exception as e:                           # a failed write must never leave the arm "enabled"
            self.last_error = "write failed: %s" % e
            self.enabled = False
            raise ArmError(self.last_error)
