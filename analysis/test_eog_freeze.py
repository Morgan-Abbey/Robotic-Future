import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eog_freeze import METHODS, FrozenEyeCleaner
from eog_freeze_synthetic import detection, score
from synthetic import LAYOUTS, person, session

FS = 250
L3, L6 = "3 EEG (C3 Cz C4)", "6 EEG (+CP3 CP4 FCz)"


def make(layout, seed=3, scale=1.0):
    rng = np.random.default_rng(seed)
    who = person(rng, layout)
    return (session(rng, who, seconds=150, rates=(0.25, 0.15, 0.15), prop_scale=scale),
            session(rng, who, seconds=200, rates=(0.2, 0.05, 0.05), prop_scale=scale))


def test_fit_recovers_the_eye_sources():
    cal, _ = make(L6)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    S = cl.W.T @ (cal["eog"] - cl.eog_mean)
    true = np.vstack([cal["v"], cal["h"]])
    coef = np.linalg.lstsq(S.T, true.T, rcond=None)[0]       # fitted sources span the true vertical/horizontal plane
    assert np.mean((true.T - S.T @ coef) ** 2) / np.mean(true ** 2) < 0.1


def test_propagation_estimate_is_close_to_the_truth():
    cal, _ = make(L6, scale=4.0)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    S = cl.W.T @ (cal["eog"] - cl.eog_mean)
    # EEG reconstructed from the fitted propagation and the EOG-derived sources leaves only brain-sized residue
    resid = (cal["eeg"] - cl.eeg_mean) - cl.B @ S
    assert np.sqrt(np.mean((resid - (cal["brain"] - cal["brain"].mean(1, keepdims=True))) ** 2)) < 0.3 * cal["brain"].std()


@pytest.mark.parametrize("layout", [L3, L6])
def test_eog_regression_with_eog_at_run_time_removes_the_eye_signal(layout):
    cal, test = make(layout)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    raw = score(test["eeg"], test)
    out = score(cl.transform(test["eeg"], "eog_regression", eog=test["eog"]), test)
    assert out["err_eye"] < 0.6 * raw["err_eye"] and out["corr_v"] < 0.5 * raw["corr_v"]


def test_projecting_out_two_eye_directions_from_three_channels_destroys_the_brain_signal():
    cal, test = make(L3)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    out = score(cl.transform(test["eeg"], "proj_orth"), test)
    assert out["err_quiet"] > 0.6          # two of three dimensions removed: about sqrt(2/3) of the signal
    assert np.linalg.matrix_rank(cl.P) == 1


def test_gate_protects_quiet_data_and_the_slow_filter_protects_mu_beta():
    cal, test = make(L6)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    lp, gated = score(cl.transform(test["eeg"], "gls_lp"), test), score(cl.transform(test["eeg"], "gls_lp_gate"), test)
    broad = score(cl.transform(test["eeg"], "gls"), test)
    assert gated["err_quiet"] < 0.5 * lp["err_quiet"]
    assert lp["err_mubeta"] < 0.5 * broad["err_mubeta"]


def test_frozen_cleaner_works_when_the_eye_signal_is_strong_on_the_scalp():
    cal, test = make(L6, scale=6.0)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    raw = score(test["eeg"], test)
    out = score(cl.transform(test["eeg"], "gls_gate"), test)
    assert out["err_eye"] < 0.5 * raw["err_eye"] and out["corr_v"] < 0.5 * raw["corr_v"]
    assert out["err_quiet"] < 0.3
    det = detection(cl, test)
    assert det["blink"] > 0.9 and det["false_per_min"] < 5


def test_gate_leaves_eye_free_data_untouched():
    cal, test = make(L6)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    quiet = test["brain"]
    gated, always = cl.transform(quiet, "gls_lp_gate"), cl.transform(quiet, "gls_lp")
    assert np.mean((gated - quiet) ** 2) < 0.5 * np.mean((always - quiet) ** 2)
    assert cl.detect(quiet).mean() < 0.05


def test_transform_and_detect_are_causal():
    cal, test = make(L6)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    x, cut = test["eeg"].copy(), 20 * FS
    x2 = x.copy()
    x2[:, cut:] += 500.0                                       # change the future
    for m in ("gls_lp_gate", "gls_gate", "gls_lp", "gls"):
        assert np.allclose(cl.transform(x2, m)[:, :cut], cl.transform(x, m)[:, :cut]), m
    assert (cl.detect(x2)[:cut] == cl.detect(x)[:cut]).all()


def test_input_validation():
    cal, test = make(L3)
    cl = FrozenEyeCleaner(FS).fit(cal["eeg"], cal["eog"])
    with pytest.raises(ValueError):
        cl.transform(test["eeg"], "eog_regression")
    with pytest.raises(ValueError):
        cl.transform(test["eeg"], "magic")
    assert set(METHODS) == {"eog_regression", "proj_orth", "gls", "gls_lp", "gls_gate", "gls_lp_gate"}
    assert set(LAYOUTS) == {L3, L6}
