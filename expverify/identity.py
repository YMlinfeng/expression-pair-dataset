"""Identity embedding, used only to prove the two people are *different*.

Two separate checks depend on this:

  1. ref and target must not be the same person (the pair is worthless otherwise);
  2. for generated pairs, the output must keep its own source identity and must
     NOT drift towards the driver. LivePortrait's absolute-expression mode copies
     the driver's expression keypoints verbatim, and its documented cost is that
     the driver's face shape leaks through. That failure is invisible to any
     expression metric, so it needs its own detector.

Licence note: buffalo_l / ArcFace weights are InsightFace's and are licensed for
non-commercial research use. They are used here only inside the offline
verifier, never shipped as part of a dataset. If the weights are absent the
module falls back to a licence-clean geometric proxy.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
ARCFACE_PATH = MODEL_DIR / "w600k_r50.onnx"

# Canonical ArcFace 112x112 template: left eye, right eye, nose, left mouth,
# right mouth, with "left" meaning smaller x in image space.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

LEFT_IRIS, RIGHT_IRIS = 468, 473
NOSE_TIP = 1
MOUTH_A, MOUTH_B = 61, 291


def ensure_arcface(path: Path = ARCFACE_PATH) -> Path | None:
    if path.exists() and path.stat().st_size > 10_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = path.parent / "buffalo_l.zip"
    try:
        if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
            urllib.request.urlretrieve(BUFFALO_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            member = next(n for n in z.namelist() if n.endswith("w600k_r50.onnx"))
            with z.open(member) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as e:  # noqa: BLE001
        print(f"[identity] ArcFace unavailable ({e}); using geometric fallback")
        return None
    finally:
        if zip_path.exists():
            zip_path.unlink()
    return path


def keypoints5(lm: np.ndarray) -> np.ndarray:
    """478 landmarks (pixel space) -> ArcFace 5-point set, ordered by image x."""
    eyes = np.array([lm[LEFT_IRIS, :2], lm[RIGHT_IRIS, :2]], dtype=np.float32)
    mouth = np.array([lm[MOUTH_A, :2], lm[MOUTH_B, :2]], dtype=np.float32)
    eyes = eyes[np.argsort(eyes[:, 0])]
    mouth = mouth[np.argsort(mouth[:, 0])]
    return np.stack([eyes[0], eyes[1], lm[NOSE_TIP, :2], mouth[0], mouth[1]]).astype(np.float32)


def similarity_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact least-squares 2D similarity (Umeyama) as a 2x3 affine matrix.

    Deliberately not `cv2.estimateAffinePartial2D`: its robust estimators fit on
    random subsets, and with only five points a degenerate subset is likely. A
    slightly misaligned chip does not throw -- it quietly pulls every embedding
    toward a mean face, which inflates cosine similarity between different
    people and silently disables the identity gate.
    """
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(d.T @ s / len(src))
    W = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[1, 1] = -1.0
    R = U @ W @ Vt
    var = float((s ** 2).sum() / len(src))
    c = float(np.trace(np.diag(D) @ W) / max(var, 1e-12))
    return np.hstack([c * R, (mu_d - c * (R @ mu_s)).reshape(2, 1)]).astype(np.float32)


def align112(frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    M = similarity_2d(kps.astype(np.float64), ARCFACE_TEMPLATE.astype(np.float64))
    return cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0.0)


class ArcFace:
    def __init__(self, model_path: Path | None = None):
        import onnxruntime as ort
        path = model_path or ensure_arcface()
        if path is None or not Path(path).exists():
            raise FileNotFoundError("ArcFace weights unavailable")
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def embed_aligned(self, chips_bgr: list[np.ndarray]) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(chips_bgr, 1.0 / 127.5, (112, 112),
                                      (127.5, 127.5, 127.5), swapRB=True)
        out = self.sess.run(None, {self.input_name: blob})[0]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    def embed_video(self, frames_bgr: list[np.ndarray], lms: np.ndarray,
                    ok: np.ndarray, max_frames: int = 12) -> np.ndarray | None:
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return None
        idx = idx[np.linspace(0, idx.size - 1, min(max_frames, idx.size)).astype(int)]
        chips = []
        for t in idx:
            try:
                chips.append(align112(frames_bgr[t], keypoints5(lms[t])))
            except Exception:  # noqa: BLE001
                continue
        if not chips:
            return None
        emb = self.embed_aligned(chips).mean(axis=0)
        return emb / (np.linalg.norm(emb) + 1e-9)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def identity_separation(emb: dict[str, np.ndarray | None],
                        person_of: dict[str, str]) -> dict[str, float]:
    """Same-person vs different-person cosine distributions over a corpus.

    A gate is only worth having if it separates; this measures whether it does,
    on the actual footage, instead of trusting a threshold copied from a paper.
    Also reports the equal-error threshold, which is what the gate should use if
    the published one turns out to be wrong for this resolution.
    """
    keys = [k for k, v in emb.items() if v is not None]
    same, diff = [], []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            c = cosine(emb[a], emb[b])
            (same if person_of[a] == person_of[b] else diff).append(c)
    out: dict[str, float] = {"n_same": len(same), "n_diff": len(diff)}
    if same:
        out |= {"same_median": float(np.median(same)), "same_q05": float(np.quantile(same, 0.05))}
    if diff:
        out |= {"diff_median": float(np.median(diff)), "diff_q95": float(np.quantile(diff, 0.95)),
                "diff_max": float(np.max(diff))}
    if same and diff:
        s, d = np.asarray(same), np.asarray(diff)
        grid = np.unique(np.concatenate([s, d]))
        far = np.array([(d > t).mean() for t in grid])       # different people accepted as same
        frr = np.array([(s <= t).mean() for t in grid])      # same person rejected
        i = int(np.argmin(np.abs(far - frr)))
        out |= {"eer_threshold": float(grid[i]), "eer": float((far[i] + frr[i]) / 2)}
    return out


@dataclass
class GeometricIdentity:
    """Licence-clean fallback: neutral face geometry as an identity proxy.

    Much weaker than ArcFace and only meaningful for the "are these clearly
    different people" direction, which is the direction we need.
    """

    shape: np.ndarray

    @staticmethod
    def from_neutral(canon_neutral: np.ndarray) -> "GeometricIdentity":
        return GeometricIdentity(shape=canon_neutral.reshape(-1).astype(np.float32))

    def distance(self, other: "GeometricIdentity") -> float:
        return float(np.linalg.norm(self.shape - other.shape) / np.sqrt(self.shape.size))
