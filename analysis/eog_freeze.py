"""Eye-artifact removal that needs EOG electrodes only during calibration.

    cleaner = FrozenEyeCleaner(fs).fit(eeg_calibration, eog_calibration)   # EOG recorded here only
    clean = cleaner.transform(eeg)                                         # EEG alone from here on

The calibration learns how eye activity shows up on each EEG channel (propagation factors). At run time the
eye amplitude is estimated from the EEG itself by generalised least squares, treating brain activity as the
noise, so the estimate is as little contaminated by brain signal as the channel layout allows. The estimate
can be restricted to slow frequencies (where eye activity lives, below the mu/beta band the decoder uses)
and switched on only while it is clearly above its resting level.

Arrays are channels x samples. The cleaner is linear plus a causal low-pass and a causal gate, so it can be
run on a stream by carrying `sosfilt` state (not done here, `transform` handles whole arrays).
"""
import numpy as np
from scipy.signal import butter, sosfilt

METHODS = ("eog_regression", "proj_orth", "gls", "gls_lp", "gls_gate", "gls_lp_gate")


class FrozenEyeCleaner:
    def __init__(self, fs, n_sources=2, lowpass_hz=7.0, gate_k=3.0, hold_s=0.3, ridge=1e-3, shrink=1e-2):
        self.fs, self.r, self.gate_k, self.ridge, self.shrink = fs, n_sources, gate_k, ridge, shrink
        self.hold = max(1, int(round(hold_s * fs)))
        self.sos = butter(4, lowpass_hz, btype="low", fs=fs, output="sos")

    # ------------------------------------------------------------------ calibration
    def fit(self, eeg, eog):
        eeg, eog = np.asarray(eeg, float), np.asarray(eog, float)
        self.eog_mean = eog.mean(1, keepdims=True)
        eig, vec = np.linalg.eigh(np.cov(eog))
        self.W = vec[:, ::-1][:, :self.r]                               # EOG -> eye sources (PCA)
        S = self.W.T @ (eog - self.eog_mean)
        self.eeg_mean = eeg.mean(1, keepdims=True)
        X = eeg - self.eeg_mean
        reg = self.ridge * np.trace(S @ S.T) / self.r * np.eye(self.r)
        self.B = (X @ S.T) @ np.linalg.inv(S @ S.T + reg)             # propagation: sources -> EEG
        R = X - self.B @ S                                              # what is left: brain + noise
        Sigma = np.cov(R)
        Sigma += self.shrink * np.trace(Sigma) / len(Sigma) * np.eye(len(Sigma))
        Si = np.linalg.inv(Sigma)
        self.G = np.linalg.solve(self.B.T @ Si @ self.B, self.B.T @ Si)  # EEG -> eye amplitude (GLS)
        self.P = np.eye(len(X)) - self.B @ np.linalg.pinv(self.B)       # orthogonal projection out of span(B)
        # resting level of the slow eye estimate, from the calibration frames where the EOG is quiet
        Ssl = sosfilt(self.sos, S, axis=1)
        act = np.linalg.norm(Ssl / (1.4826 * np.median(np.abs(Ssl - np.median(Ssl, 1, keepdims=True)), 1, keepdims=True) + 1e-12), axis=0)
        quiet = act < 2.0
        self.rest_sd = sosfilt(self.sos, self.G @ X, axis=1)[:, quiet].std(1) + 1e-12
        return self

    # ------------------------------------------------------------------ run time
    def _gate(self, s_lp):
        a = np.linalg.norm(s_lp / self.rest_sd[:, None], axis=0) / np.sqrt(self.r)
        on = (a > self.gate_k).astype(float)
        return (np.convolve(on, np.ones(self.hold), "full")[:on.size] > 0).astype(float)   # causal hold

    def detect(self, eeg):
        """Causal on/off flag per sample: is the EEG-only eye estimate clearly above its resting level?"""
        s_lp = sosfilt(self.sos, self.G @ (np.asarray(eeg, float) - self.eeg_mean), axis=1)
        return self._gate(s_lp) > 0

    def transform(self, eeg, method="gls_lp_gate", eog=None):
        X = np.asarray(eeg, float) - self.eeg_mean
        if method == "eog_regression":
            if eog is None:
                raise ValueError("eog_regression needs the EOG at run time")
            return X - self.B @ (self.W.T @ (np.asarray(eog, float) - self.eog_mean)) + self.eeg_mean
        if method == "proj_orth":
            return self.P @ X + self.eeg_mean
        s = self.G @ X
        if method == "gls":
            return X - self.B @ s + self.eeg_mean
        s_lp = sosfilt(self.sos, s, axis=1)
        if method == "gls_gate":                       # broadband subtraction, but only while an eye event is flagged
            return X - self.B @ (s * self._gate(s_lp)) + self.eeg_mean
        if method == "gls_lp":
            return X - self.B @ s_lp + self.eeg_mean
        if method == "gls_lp_gate":
            return X - self.B @ (s_lp * self._gate(s_lp)) + self.eeg_mean
        raise ValueError(method)
