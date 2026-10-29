"""Build the interactive demo from template.html plus the numbers the Python simulator produced.

    python build.py              # writes bci_demo.html (artifact-style fragment) and bci_demo_standalone.html

SIGNAL maps decoder accuracy (%) to the signal strength of the generative decoder; TUNED holds the interpreter
settings found by run_sweep.py for each accuracy, interpreter and false-motion budget.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from armsim.decoder import accuracy_to_signal                    # noqa: E402

RESULTS = HERE.parents[1] / "reports" / "arm_sim_results.json"


def data_block():
    signal = {str(a): round(accuracy_to_signal(round(a / 100, 4)), 5) for a in range(35, 100)}
    tuned = json.loads(RESULTS.read_text())["tuned"]
    table = {str(round(float(acc) * 100)): {name: by_b for name, by_b in d.items() if name != "direct"} for acc, d in tuned.items()}
    return "const SIGNAL = %s;\nconst TUNED = %s;" % (json.dumps(signal), json.dumps(table))


def main():
    page = (HERE / "template.html").read_text().replace("/*__DATA__*/", data_block())
    (HERE / "bci_demo.html").write_text(page)
    standalone = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                  '</head><body>\n%s\n</body></html>\n' % page)
    (HERE / "bci_demo_standalone.html").write_text(standalone)
    print("wrote", HERE / "bci_demo.html", "and bci_demo_standalone.html", len(page), "bytes")


if __name__ == "__main__":
    main()
