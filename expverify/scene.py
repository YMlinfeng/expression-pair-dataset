"""Non-expression dissimilarity: head pose and background.

"Same expression, everything else different" is only a real specification if the
"everything else" half is checked too. A pair that passes every expression test
but shares a background and a head pose is a near-duplicate, not a pair.

Background is measured on the outer border ring of the frame, which in portrait
video is almost always background, plus a face-hull exclusion. This is
deliberately model-free: adding a segmentation network buys marginal accuracy
for a dependency that would then need its own licence review.
"""

from __future__ import annotations

import cv2
import numpy as np


def pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """(T,3) and (T,3) degrees -> (T,3) absolute per-axis difference."""
    d = np.abs(pose_a - pose_b)
    return np.minimum(d, 360.0 - d)


def border_ring_mask(h: int, w: int, margin: float = 0.15) -> np.ndarray:
    m = np.ones((h, w), dtype=np.uint8)
    mh, mw = int(round(h * margin)), int(round(w * margin))
    m[mh:h - mh, mw:w - mw] = 0
    return m


def face_hull_mask(h: int, w: int, lm: np.ndarray, dilate: float = 0.35) -> np.ndarray:
    pts = lm[:, :2].astype(np.int32)
    hull = cv2.convexHull(pts)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    k = max(3, int(round(dilate * np.sqrt(cv2.contourArea(hull.astype(np.float32)) + 1))))
    return cv2.dilate(m, np.ones((k, k), np.uint8))


def _hist(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def _dhash(frame_bgr: np.ndarray, mask: np.ndarray, size: int = 16) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[mask == 0] = float(g[mask > 0].mean()) if (mask > 0).any() else 0.0
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def background_similarity(frame_a: np.ndarray, lm_a: np.ndarray,
                          frame_b: np.ndarray, lm_b: np.ndarray) -> dict[str, float]:
    """Higher = more similar backgrounds. `hist` in [-1,1], `hash` in [0,1]."""
    ha, wa = frame_a.shape[:2]
    hb, wb = frame_b.shape[:2]
    ma = border_ring_mask(ha, wa) & (1 - face_hull_mask(ha, wa, lm_a))
    mb = border_ring_mask(hb, wb) & (1 - face_hull_mask(hb, wb, lm_b))
    if ma.sum() < 100 or mb.sum() < 100:
        return {"hist": 1.0, "hash": 1.0}
    hist = float(cv2.compareHist(_hist(frame_a, ma), _hist(frame_b, mb), cv2.HISTCMP_CORREL))
    da, db = _dhash(frame_a, ma), _dhash(frame_b, mb)
    return {"hist": hist, "hash": float((da == db).mean())}


def sample_indices(ok: np.ndarray, n: int = 5) -> np.ndarray:
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return idx
    return idx[np.linspace(0, idx.size - 1, min(n, idx.size)).astype(int)]
