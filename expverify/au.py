"""M3 -- Action Unit channel from OpenFace 3.0's multitask model.

Its value in the ensemble is independence, not accuracy. M1 and M2 both descend
from the same MediaPipe landmark regression -- the blendshape head literally
consumes landmark coordinates -- so they can be wrong together, and a
conjunctive gate over two correlated metrics buys far less than it appears to.
M3 is a different architecture (EfficientNet-B0 + a graph AU head) trained on
different data, reading pixels rather than landmark geometry, so its errors are
uncorrelated with theirs. `redundancy()` checks that claim rather than assuming
it.

Scope correction worth stating plainly: the released OpenFace 3.0 multitask head
predicts **8** AU channels as logits, not the 12-dimensional 0-5 FACS intensity
vector often assumed. The channels are therefore treated as an unnamed
8-dimensional AU-activation descriptor -- still neutral-subtracted, still
usable in the gate, but not something to hand a FACS coder as intensities.

Licence: OpenFace 3.0 weights are research-only, so this metric belongs in the
offline verifier and must not be redistributed inside a dataset.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MTL_URL = "https://huggingface.co/nutPace/openface_weights/resolve/main/MTL_backbone.pth"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MTL_PATH = MODEL_DIR / "MTL_backbone.pth"
N_AU = 8


def ensure_mtl(path: Path = MTL_PATH) -> Path | None:
    if path.exists() and path.stat().st_size > 10_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MTL_URL, path)
    except Exception as e:  # noqa: BLE001
        print(f"[au] OpenFace weights unavailable ({e})")
        return None
    return path


class AUExtractor:
    def __init__(self, model_path: Path | None = None, device: str | None = None):
        import torch
        from openface.multitask_model import MultitaskPredictor

        path = model_path or ensure_mtl()
        if path is None:
            raise FileNotFoundError("OpenFace MTL weights unavailable")
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.model = MultitaskPredictor(str(path), device=device)

    def run(self, frames_bgr: list[np.ndarray], ok: np.ndarray,
            batch: int = 16) -> np.ndarray:
        """(T, 8) AU activations; rows for invalid frames are left at zero."""
        T = len(frames_bgr)
        out = np.zeros((T, N_AU), dtype=np.float32)
        idx = [int(t) for t in np.flatnonzero(ok)]
        for i in range(0, len(idx), batch):
            chunk = idx[i:i + batch]
            tensors = [self.model.preprocess(frames_bgr[t]) for t in chunk]
            x = self.torch.cat(tensors, dim=0)
            with self.torch.no_grad():
                _, _, au = self.model.model(x)
            out[chunk] = au.detach().float().cpu().numpy()
        return out


def neutral_au(au: np.ndarray, ok: np.ndarray, keep_frac: float = 0.4) -> np.ndarray:
    v = au[ok]
    if v.shape[0] == 0:
        return np.zeros(au.shape[1], dtype=np.float32)
    m1 = np.median(v, axis=0)
    if v.shape[0] < 8:
        return m1.astype(np.float32)
    d = np.linalg.norm(v - m1, axis=1)
    k = max(4, int(round(v.shape[0] * keep_frac)))
    return np.median(v[np.argsort(d)[:k]], axis=0).astype(np.float32)


def au_distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray | None = None) -> float:
    d = np.abs(a - b)
    if scale is not None:
        d = d / np.maximum(scale, 1e-4)
    return float(d.mean())


def redundancy(m_a: np.ndarray, m_b: np.ndarray) -> float:
    """Pearson correlation between two metrics' per-frame distances.

    A near-1 correlation means the second metric adds no independent evidence,
    and a conjunctive gate over the pair is really a single gate applied twice.
    """
    if m_a.size < 3 or m_b.size < 3:
        return float("nan")
    a = m_a - m_a.mean()
    b = m_b - m_b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 1e-9 else float("nan")
