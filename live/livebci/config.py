import copy
import json
from pathlib import Path

DEFAULTS = {
    "host": "127.0.0.1",
    "http_port": 8765,
    "ws_port": 8766,
    "udp_port": 9871,
    "engine": "mock",                    # "mock" (simulated decoder) or "udp" (your pipeline sends frames)
    "interpreter": "proportional",       # direct | evidence | proportional | shared
    "budget": "0.25",                    # strictness: "0.05" strict, "0.25" balanced, "1.0" loose
    "profile_accuracy": 65,              # which tuned settings to use (45, 55, 65, 75 or 85)
    "dt": 0.25,                          # seconds per decoder frame; match your pipeline's window hop
    "frame_timeout_s": 1.5,              # no frames for this long: stop driving the arm
    "heartbeat_timeout_s": 5.0,          # no browser connected for this long: disable the arm
    "arm": {
        "port": None,                    # e.g. "COM15". "file:path.log" writes the lines to a file instead
        "baud": 9600,
        "send": False,                   # must be true, and --send given on the command line, before anything is written
        "steps_per_unit": [0, 0, 0],     # motor steps per simulator unit (-1..1) for each joint; 0 = not calibrated
        "invert": [False, False, False],
        "limit_units": 1.0,              # software travel limit either side of the zero pose
        "max_steps_per_message": 40,
        "max_rate_hz": 4.0,
        "max_jog_steps": 20,
        "format": "{a} {b} {c}\n",       # UNVERIFIED: three integers per line, as the Dec 2024 WinForms app is said to send
        "stop_line": None,               # sent on e-stop/disable if your firmware has a stop command
        "protocol_verified": False,
    },
    "targets": [[0.6, -0.4, 0.3], [-0.5, 0.5, 0.6], [0.3, 0.7, -0.5], [-0.6, -0.5, -0.3]],   # candidates for shared autonomy
    "scripts": {                         # commands the Training panel can launch; edit to match your pipeline's README
        "record": {"cwd": ".", "cmd": ["python", "record_session.py", "--board", "cyton", "--layout", "6eeg", "--out", "data/sessions/live.npz"]},
        "train": {"cwd": ".", "cmd": ["python", "train.py", "data/sessions/live.npz", "--out", "data/models/live.pkl"]},
        "live": {"cwd": ".", "cmd": ["python", "run_bci.py", "--model", "data/models/live.pkl"]},
    },
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = merge(cfg, json.loads(Path(path).read_text()))
    return merge(cfg, overrides)
