"""Copy this file next to your pipeline's scripts. One line per event, no dependencies, never raises.

    from bci_emit import emit
    emit({"type": "frame", "probs": [p_rest, p_left, p_right], "jaw": bool(jaw_event)})          # once per decoder window
    emit({"type": "cue", "class": "left", "phase": "start", "trial": 3, "n": 40, "dur": 4.0})    # record_session.py, each cue
    emit({"type": "cue", "class": "left", "phase": "end", "trial": 3, "n": 40})
    emit({"type": "trained", "cv_acc": 0.71})                                                      # train.py, when done
"""
import json
import socket

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ADDRESS = ("127.0.0.1", 9871)


def emit(msg):
    try:
        _sock.sendto(json.dumps(msg).encode(), ADDRESS)
    except OSError:
        pass
