"""Expression descriptors M1 (blendshape) and M2 (landmark deformation field).

Both are built so that identity cancels out:

M1 selects a curated blendshape subset and is always used neutral-subtracted.
M2 expresses every landmark in an anatomically-anchored frame derived from the
face's own rigid points, then subtracts that person's neutral shape. What
survives is muscle displacement in units of the subject's own interocular
distance, which is comparable across faces of different size and shape; raw
landmark geometry is not, because it is dominated by face shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .landmarks import EYE_OUTER_L, EYE_OUTER_R, REGIONS, RIGID_IDS

# MediaPipe's blendshape head regresses from 146 2-D landmark coordinates, not
# pixels, and several of its channels are known-broken. Excluding them is not
# cosmetic: a dead channel contributes pure noise to an L1 distance, and an
# over-firing one manufactures false differences.
BS_DEAD = {
    "noseSneerLeft", "noseSneerRight",      # never fires
    "mouthFrownLeft", "mouthFrownRight",    # never fires
    "jawForward",                           # never fires
    "cheekSquintLeft", "cheekSquintRight",  # never fires
    "cheekPuff",                            # never fires
}
BS_UNRELIABLE = {
    "eyeWideLeft", "eyeWideRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthPucker",
}
# eyeSquint* fires on every blink, so it is masked rather than dropped: it is
# informative when the eye is actually open.
BS_SQUINT = ("eyeSquintLeft", "eyeSquintRight")
BS_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")
BLINK_MASK_THRESHOLD = 0.3

GAZE_CHANNELS = (
    ("eyeLookOutLeft", "eyeLookInLeft", "eyeLookUpLeft", "eyeLookDownLeft"),
    ("eyeLookOutRight", "eyeLookInRight", "eyeLookUpRight", "eyeLookDownRight"),
)

EXPRESSION_POINTS = sorted({i for pts in REGIONS.values() for i in pts})


@dataclass
class ChannelPlan:
    """Name-addressed blendshape layout. Never index MediaPipe blendshapes
    positionally: `_neutral` occupies slot 0 and `tongueOut` is absent, so every
    ARKit-named channel is shifted relative to canonical ARKit-52 order."""

    feat_names: list[str]
    sym_pairs: list[tuple[str, int, int]]
    singles: list[tuple[str, int]]
    gaze_idx: list[tuple[int, int, int, int]]
    blink_idx: tuple[int, int] | None
    squint_idx: tuple[int, int] | None
    region_of: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.feat_names)


def _region_for(name: str) -> str:
    n = name.lower()
    if n.startswith("brow"):
        return "brow"
    if n.startswith("eye"):
        return "eyelid"
    if n.startswith("nose") or n.startswith("cheek"):
        return "nose"
    if n.startswith("jaw"):
        return "jaw"
    return "lip"


def build_channel_plan(bs_names: list[str]) -> ChannelPlan:
    idx = {n: i for i, n in enumerate(bs_names)}
    usable = [
        n for n in bs_names
        if n != "_neutral" and n not in BS_DEAD and n not in BS_UNRELIABLE
        and not n.startswith("eyeLook")
    ]

    sym_pairs: list[tuple[str, int, int]] = []
    singles: list[tuple[str, int]] = []
    seen: set[str] = set()
    for n in usable:
        if n in seen:
            continue
        if n.endswith("Left"):
            base, other = n[:-4], n[:-4] + "Right"
            if other in idx:
                sym_pairs.append((base, idx[n], idx[other]))
                seen.update({n, other})
                continue
        if n.endswith("Right"):
            base, other = n[:-5], n[:-5] + "Left"
            if other in idx:
                sym_pairs.append((base, idx[other], idx[n]))
                seen.update({n, other})
                continue
        singles.append((n, idx[n]))
        seen.add(n)

    feat_names: list[str] = []
    region_of: list[str] = []
    for base, _, _ in sym_pairs:
        # The mean carries the expression; |L-R| carries asymmetry, which is a
        # genuine fine-grained cue that a symmetric summary would erase.
        feat_names += [f"{base}.mean", f"{base}.asym"]
        region_of += [_region_for(base)] * 2
    for name, _ in singles:
        feat_names.append(name)
        region_of.append(_region_for(name))

    gaze_idx = [tuple(idx[c] for c in grp) for grp in GAZE_CHANNELS if all(c in idx for c in grp)]
    blink = tuple(idx[c] for c in BS_BLINK) if all(c in idx for c in BS_BLINK) else None
    squint = tuple(idx[c] for c in BS_SQUINT) if all(c in idx for c in BS_SQUINT) else None
    return ChannelPlan(feat_names, sym_pairs, singles, gaze_idx, blink, squint, region_of)


def bs_features(bs: np.ndarray, plan: ChannelPlan) -> np.ndarray:
    """(T, 52) raw blendshape scores -> (T, D) curated features."""
    bs = np.atleast_2d(np.asarray(bs, dtype=np.float32)).copy()
    if plan.squint_idx and plan.blink_idx:
        for s, b in zip(plan.squint_idx, plan.blink_idx):
            bs[bs[:, b] > BLINK_MASK_THRESHOLD, s] = 0.0

    cols = []
    for _, li, ri in plan.sym_pairs:
        cols.append(0.5 * (bs[:, li] + bs[:, ri]))
        cols.append(np.abs(bs[:, li] - bs[:, ri]))
    for _, i in plan.singles:
        cols.append(bs[:, i])
    return np.stack(cols, axis=1).astype(np.float32)


def gaze_features(bs: np.ndarray, plan: ChannelPlan) -> np.ndarray:
    """(T, 52) -> (T, 2*n_eyes): signed horizontal / vertical gaze per eye."""
    bs = np.atleast_2d(np.asarray(bs, dtype=np.float32))
    cols = []
    for out_i, in_i, up_i, down_i in plan.gaze_idx:
        cols.append(bs[:, out_i] - bs[:, in_i])
        cols.append(bs[:, up_i] - bs[:, down_i])
    if not cols:
        return np.zeros((bs.shape[0], 0), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)


def face_frame(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Three-point anatomical frame. Kept only as a bootstrap for fitting the
    shared reference; too noisy to use directly, because the up-axis depends on
    the forehead apex, a texture-free point whose localisation error rotates the
    entire face and shows up as large phantom displacement at the chin.
    """
    o = p[RIGID_IDS].mean(axis=0)
    x = p[EYE_OUTER_R] - p[EYE_OUTER_L]
    s = float(np.linalg.norm(x))
    if s < 1e-6:
        raise ValueError("degenerate interocular distance")
    x = x / s
    up = p[10] - o
    y = up - np.dot(up, x) * x
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        raise ValueError("degenerate vertical axis")
    y = y / ny
    z = np.cross(x, y)
    return o, np.stack([x, y, z], axis=0).astype(np.float32), s


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity transform with dst ~= c * src @ R.T + t."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    var_s = (sc ** 2).sum() / n
    c = float(np.trace(np.diag(D) @ W) / max(var_s, 1e-12))
    t = mu_d - c * (R @ mu_s)
    return R.astype(np.float32), c, t.astype(np.float32)


_REFERENCE: np.ndarray | None = None
REFERENCE_PATH = None  # set by set_reference_path()


def fit_rigid_reference(samples: np.ndarray, iters: int = 4) -> np.ndarray:
    """Generalised Procrustes over rigid-point sets -> shared reference.

    Averaging the fit over all 13 rigid points instead of constructing axes from
    2-3 of them is what removes the frame-estimation noise; the reference is
    rescaled so interocular distance is exactly 1, keeping every downstream
    number interpretable in interocular units.
    """
    ref = samples[0].copy()
    ref -= ref.mean(axis=0)
    ref /= np.sqrt((ref ** 2).sum() / ref.shape[0])
    for _ in range(iters):
        acc = np.zeros_like(ref)
        for s in samples:
            R, c, t = umeyama(s, ref)
            acc += c * (s @ R.T) + t
        ref = acc / len(samples)
        ref -= ref.mean(axis=0)
        ref /= np.sqrt((ref ** 2).sum() / ref.shape[0])
    li, ri = RIGID_IDS.index(EYE_OUTER_L), RIGID_IDS.index(EYE_OUTER_R)
    ref = ref / float(np.linalg.norm(ref[ri] - ref[li]))
    return ref.astype(np.float32)


def rigid_samples(landmarks: np.ndarray, ok: np.ndarray, per_clip: int = 6) -> np.ndarray:
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.zeros((0, len(RIGID_IDS), 3), dtype=np.float32)
    pick = idx[np.linspace(0, idx.size - 1, min(per_clip, idx.size)).astype(int)]
    return landmarks[pick][:, RIGID_IDS, :].astype(np.float32)


def set_reference(ref: np.ndarray | None) -> None:
    global _REFERENCE
    _REFERENCE = None if ref is None else np.asarray(ref, dtype=np.float32)


def get_reference() -> np.ndarray | None:
    return _REFERENCE


def canonicalize(p: np.ndarray) -> np.ndarray:
    """(478,3) pixel-space landmarks -> (478,3) in interocular units.

    With a shared reference this is a 13-point least-squares alignment; without
    one it falls back to the noisier 3-point frame.
    """
    if _REFERENCE is None:
        o, R, s = face_frame(p)
        return ((p - o) @ R.T / s).astype(np.float32)
    R, c, t = umeyama(p[RIGID_IDS], _REFERENCE)
    return (c * (p @ R.T) + t).astype(np.float32)


def canonicalize_seq(lms: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = lms.shape[0]
    out = np.zeros_like(lms)
    good = np.zeros(T, dtype=bool)
    for t in range(T):
        if not ok[t]:
            continue
        try:
            out[t] = canonicalize(lms[t])
            good[t] = True
        except ValueError:
            pass
    return out, good


def region_distance(d_ref: np.ndarray, d_tgt: np.ndarray) -> dict[str, float]:
    """Per-region RMS disagreement between two deformation fields."""
    diff = d_ref - d_tgt
    out: dict[str, float] = {}
    for name, ids in REGIONS.items():
        v = diff[ids]
        out[name] = float(np.sqrt((v ** 2).sum(axis=1).mean()))
    v = diff[EXPRESSION_POINTS]
    out["all"] = float(np.sqrt((v ** 2).sum(axis=1).mean()))
    return out


def expressiveness(d: np.ndarray) -> float:
    """How far this frame sits from the subject's own neutral, in interocular units."""
    v = d[EXPRESSION_POINTS]
    return float(np.sqrt((v ** 2).sum(axis=1).mean()))
