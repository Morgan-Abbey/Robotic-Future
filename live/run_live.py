"""Start the local bridge and open the UI.

    python run_live.py                              # simulated decoder, arm output off
    python run_live.py --engine udp                 # your pipeline sends frames (see HANDOFF.md)
    python run_live.py --engine udp --send          # allow the arm to be enabled (also needs the config's arm settings)
"""
import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
from livebci import config                                                   # noqa: E402
from livebci.server import Bridge                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file overriding live/config defaults (see config.example.json)")
    ap.add_argument("--engine", choices=["mock", "udp"])
    ap.add_argument("--send", action="store_true", help="allow the arm to be enabled; without it nothing is ever written to the port")
    ap.add_argument("--arm-port", help='serial port, e.g. COM15; "file:arm.log" writes the lines to a file for a dry run')
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    over = {}
    if a.engine:
        over["engine"] = a.engine
    if a.arm_port:
        over["arm"] = {"port": a.arm_port}
    cfg = config.load(a.config, over)
    bridge = Bridge(cfg, allow_send=a.send, config_dir=str(Path(a.config).resolve().parent) if a.config else ".")
    url = "http://%s:%d/" % (cfg["host"], cfg["http_port"])
    print("Bridge on %s   engine: %s   arm output: %s" % (url, cfg["engine"], "allowed (needs Enable in the UI)" if a.send else "OFF"))
    if not a.no_browser:
        webbrowser.open(url)
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
