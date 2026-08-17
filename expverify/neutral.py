"""Per-person neutral estimation.

This is the single highest-impact calibration step in the whole verifier and it
is not optional. Without it, both descriptors are dominated by permanent facial
structure -- resting brow height, lid shape, wrinkles, facial hair -- which AU
and blendshape estimators systematically read as active muscle movement. The
cross-identity comparison then measures face shape instead of expression.

Estimation follows the median-over-sequence idea (Baltrusaitis et al., FG 2015)
with a second pass: the plain median is biased when a strong expression occupies
a large fraction of the clip, so we re-estimate over the frames closest to the
first-pass median.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .descriptors import (ChannelPlan, bs_features, canonicalize_seq,
                          expressiveness, gaze_features)
from .landmarks import VideoFeats


@dataclass
class Neutral:
    person: str
    bs_feat: np.ndarray    # (D,)
    gaze: np.ndarray       # (G,)
    canon: np.ndarray      # (478, 3)
    pose: np.ndarray       # (3,)
    n_frames: int
    source: str            # "explicit" | "median"
    au: np.ndarray | None = None   # (8,) set only when M3 is enabled


def _two_pass_median(x: np.ndarray, keep_frac: float = 0.4) -> np.ndarray:
    """Median, then median over the frames nearest that median."""
    if x.shape[0] == 0:
        raise ValueError("no valid frames for neutral estimation")
    m1 = np.median(x, axis=0)
    if x.shape[0] < 8:
        return m1
    d = np.linalg.norm((x - m1).reshape(x.shape[0], -1), axis=1)
    k = max(4, int(round(x.shape[0] * keep_frac)))
    keep = np.argsort(d)[:k]
    return np.median(x[keep], axis=0)


def estimate_neutral(clips: list[VideoFeats], plan: ChannelPlan, person: str,
                     source: str = "median") -> Neutral:
    bs_all, gz_all, canon_all, pose_all = [], [], [], []
    for f in clips:
        canon, good = canonicalize_seq(f.landmarks, f.ok)
        if not good.any():
            continue
        bs_all.append(bs_features(f.blendshapes[good], plan))
        gz_all.append(gaze_features(f.blendshapes[good], plan))
        canon_all.append(canon[good])
        pose_all.append(f.pose[good])
    if not canon_all:
        raise ValueError(f"no valid frames for person {person}")

    bs = np.concatenate(bs_all, axis=0)
    gz = np.concatenate(gz_all, axis=0)
    canon = np.concatenate(canon_all, axis=0)
    pose = np.concatenate(pose_all, axis=0)

    return Neutral(
        person=person,
        bs_feat=_two_pass_median(bs),
        gaze=_two_pass_median(gz) if gz.shape[1] else np.zeros(0, dtype=np.float32),
        canon=_two_pass_median(canon),
        pose=np.median(pose, axis=0),
        n_frames=int(canon.shape[0]),
        source=source,
    )


@dataclass
class SeqDescriptor:
    """Neutral-subtracted, per-frame descriptors for one clip."""

    name: str
    person: str
    ok: np.ndarray          # (T,) bool
    bs: np.ndarray          # (T, D)  neutral-subtracted blendshape features
    gaze: np.ndarray        # (T, G)  neutral-subtracted gaze
    deform: np.ndarray      # (T, 478, 3) canonical deformation from own neutral
    energy: np.ndarray      # (T,) expressiveness in interocular units
    pose: np.ndarray        # (T, 3) degrees
    au: np.ndarray | None = None   # (T, 8) neutral-subtracted AU activations (M3)

    def __len__(self) -> int:
        return int(self.ok.shape[0])


def describe(f: VideoFeats, plan: ChannelPlan, neutral: Neutral,
             au: np.ndarray | None = None) -> SeqDescriptor:
    canon, good = canonicalize_seq(f.landmarks, f.ok)
    ok = f.ok & good
    bs = bs_features(f.blendshapes, plan) - neutral.bs_feat
    gz = gaze_features(f.blendshapes, plan)
    if gz.shape[1] and neutral.gaze.size:
        gz = gz - neutral.gaze
    deform = canon - neutral.canon
    deform[~ok] = 0.0
    energy = np.array([expressiveness(deform[t]) if ok[t] else 0.0
                       for t in range(len(ok))], dtype=np.float32)
    au_rel = None
    if au is not None and au.shape[0] == len(ok):
        from .au import neutral_au
        # A person-level neutral is preferred: a single clip may never show this
        # person at rest, and then its own median is not a neutral at all.
        base = neutral.au if neutral.au is not None else neutral_au(au, ok)
        au_rel = (au - base).astype(np.float32)
    return SeqDescriptor(name=f.name, person=neutral.person, ok=ok, bs=bs,
                         gaze=gz, deform=deform, energy=energy, pose=f.pose, au=au_rel)


def save_neutral(n: Neutral, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, person=n.person, bs_feat=n.bs_feat, gaze=n.gaze,
                        canon=n.canon, pose=n.pose, n_frames=n.n_frames, source=n.source,
                        au=n.au if n.au is not None else np.zeros(0, dtype=np.float32))


def load_neutral(path: str | Path) -> Neutral:
    d = np.load(path, allow_pickle=True)
    au = d["au"] if "au" in d.files else np.zeros(0)
    return Neutral(person=str(d["person"]), bs_feat=d["bs_feat"], gaze=d["gaze"],
                   canon=d["canon"], pose=d["pose"], n_frames=int(d["n_frames"]),
                   source=str(d["source"]), au=au if au.size else None)
