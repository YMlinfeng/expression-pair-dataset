"""Audio-driven temporal alignment for prescribed-speech corpora.

Two actors reading the same sentence do not read it at the same speed, so their
video frames do not correspond one-to-one and any frame-synchronous comparison
is comparing different phonemes. Aligning on audio rather than on the face is
the point: the alignment must be derived from a signal that is *independent of
the thing being measured*, otherwise the aligner can manufacture the expression
agreement the verifier is supposed to test.

Log-mel + DTW with a Sakoe-Chiba band, implemented on numpy/scipy so the demo
carries no extra dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                   fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c == l:
            c = min(l + 1, n_fft // 2)
        if r == c:
            r = min(c + 1, n_fft // 2)
        fb[m, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


def log_mel(path: str, n_mels: int = 40, win_ms: float = 25.0,
            hop_ms: float = 10.0) -> tuple[np.ndarray, float]:
    """-> (T, n_mels) log-mel and the hop in seconds."""
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    x /= max(np.abs(x).max(), 1e-6)
    n_fft = int(2 ** np.ceil(np.log2(sr * win_ms / 1000.0)))
    hop = max(1, int(sr * hop_ms / 1000.0))
    _, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                   padded=False, boundary=None)
    power = np.abs(Z).astype(np.float32) ** 2
    mel = mel_filterbank(sr, n_fft, n_mels) @ power
    lm = np.log(mel + 1e-8).T.astype(np.float32)
    lm = (lm - lm.mean(axis=0)) / (lm.std(axis=0) + 1e-6)
    return lm, hop / sr


def dtw_path(a: np.ndarray, b: np.ndarray, band_frac: float = 0.25
             ) -> tuple[np.ndarray, float]:
    """DTW on cosine distance with a Sakoe-Chiba band. -> (path (N,2), mean cost)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    C = 1.0 - an @ bn.T
    n, m = C.shape
    band = max(8, int(round(band_frac * max(n, m))))

    INF = np.float32(1e18)
    D = np.full((n + 1, m + 1), INF, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, int((i - 1) * m / n) - band + 1)
        j_hi = min(m, int((i - 1) * m / n) + band + 1)
        prev, cur = D[i - 1], D[i]
        Ci = C[i - 1]
        for j in range(j_lo, j_hi + 1):
            best = prev[j - 1]
            if prev[j] < best:
                best = prev[j]
            if cur[j - 1] < best:
                best = cur[j - 1]
            if best < INF:
                cur[j] = Ci[j - 1] + best

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        opts = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        k = int(np.argmin(opts))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    p = np.asarray(path, dtype=int)
    cost = float(D[n, m] / max(len(p), 1)) if D[n, m] < INF else float("inf")
    return p, cost


def align_videos(wav_a: str, wav_b: str, fps_a: float, fps_b: float,
                 n_a: int, n_b: int) -> tuple[np.ndarray, float]:
    """Audio DTW -> (N,2) video-frame index pairs, plus the mean DTW cost.

    The cost is retained and thresholded downstream: a bad alignment produces
    frame pairs that are simply not corresponding moments, and no expression
    metric can recover from that.
    """
    A, hop_a = log_mel(wav_a)
    B, hop_b = log_mel(wav_b)
    path, cost = dtw_path(A, B)
    if path.size == 0:
        return np.zeros((0, 2), dtype=int), float("inf")

    # audio-frame -> video-frame, taking the median target for each ref frame
    t_a = path[:, 0] * hop_a
    t_b = path[:, 1] * hop_b
    va = np.clip((t_a * fps_a).astype(int), 0, n_a - 1)
    vb = np.clip((t_b * fps_b).astype(int), 0, n_b - 1)
    out = []
    for f in range(n_a):
        sel = vb[va == f]
        if sel.size:
            out.append((f, int(np.median(sel))))
    return np.asarray(out, dtype=int), cost
