"""Strict pair verification.

The headline test is *rank-1 temporal identifiability*, and it exists because
absolute thresholds cannot express "fine-grained enough". A threshold says
"these two faces are within epsilon". It cannot say "this match is finer than
the video's own frame-to-frame expression change", which is the actual
requirement. The rank-1 test says exactly that, and it needs no hand-tuned
constant:

    for reference frame t, the target frame minimising expression distance must
    be t itself (+/-1), and the aligned distance must be smaller than the
    distance from ref frame t to ref frame t+/-k of the *same* video.

The second clause is the strict one. The hardest possible negatives for a pair
of synchronised clips are the temporally adjacent frames -- they share identity,
pose, lighting and background, and differ only by a few milliseconds of muscle
movement. Requiring the cross-identity match to beat them is what separates
"the same expression" from "a similar expression".

Frames in a static segment cannot support this test (a held expression really is
equal to its own neighbours), so they are reported as *untestable* rather than
silently passed or failed.

All gates combine conjunctively. Averaging metrics would let one confident
metric mask another's rejection, which is precisely the failure mode a strict
verifier exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .descriptors import EXPRESSION_POINTS
from .landmarks import REGIONS
from .neutral import SeqDescriptor


@dataclass
class DatasetSpec:
    """A dataset contract: what must match, and what must NOT."""

    name: str

    # --- expression agreement (absolute; calibrate.py fits these) ---
    max_bs: float = 0.060           # weighted mean |delta| over M1 features
    max_deform: float = 0.045       # RMS displacement disagreement, interocular units
    max_gaze: float = 0.400         # mean |delta| over signed gaze channels
    max_region: float = 1.000       # worst region's disagreement / its own motion
    max_au: float | None = None     # M3; enforced only when AU features exist

    # --- rank-1 temporal identifiability ---
    rank1_window: int = 12          # +/- frames searched around the aligned index
    rank1_offset_k: int = 3         # nearest offset counted as a hard negative
    rank1_tolerance: int = 1        # argmin may land within +/- this many frames
    max_ratio: float = 0.80         # cross-aligned distance / hardest-negative distance
    min_self_contrast: float = 0.006  # below this the frame is untestable, not failed

    # --- expressiveness: near-neutral frames match trivially and teach nothing ---
    min_energy: float = 0.018

    # --- non-expression dissimilarity ("everything else must differ") ---
    max_identity_cos: float | None = 0.25
    min_pose_delta_deg: float | None = 8.0
    max_bg_hist: float | None = 0.60

    # --- generation-specific leakage guards ---
    min_source_identity_cos: float | None = None
    max_driver_identity_cos: float | None = None

    # --- pair-level acceptance ---
    min_pass_frames: int = 8
    min_pass_rate: float = 0.40
    min_testable_frames: int = 5


# Cross-identity pairs with genuinely different pose, background and person.
REFERENCE_SPEC = DatasetSpec(name="reference")

# Relaxed contract: expression must still match exactly, but pose/background are
# unconstrained. This is the right spec for studio corpora such as CREMA-D,
# where every actor is filmed frontally against the same backdrop -- demanding
# background dissimilarity there would reject everything for the wrong reason.
EDITING_SPEC = DatasetSpec(
    name="editing",
    max_identity_cos=0.25,
    min_pose_delta_deg=None,
    max_bg_hist=None,
)

SPECS = {"reference": REFERENCE_SPEC, "editing": EDITING_SPEC}


@dataclass
class Weights:
    """Per-feature inverse scales from the cross-identity null distribution."""

    bs: np.ndarray | None = None

    def bs_weights(self, dim: int) -> np.ndarray:
        if self.bs is None or self.bs.shape[0] != dim:
            return np.ones(dim, dtype=np.float32) / dim
        w = 1.0 / np.maximum(self.bs, 1e-4)
        return (w / w.sum()).astype(np.float32)


def bs_distance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float(np.abs(a - b) @ w)


def deform_distance(da: np.ndarray, db: np.ndarray) -> float:
    v = (da - db)[EXPRESSION_POINTS]
    return float(np.sqrt((v ** 2).sum(axis=1).mean()))


def gaze_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.abs(a - b).mean())


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt((v ** 2).sum(axis=1).mean()))


def region_worst(da: np.ndarray, db: np.ndarray, regions: dict[str, list[int]],
                 min_motion: float = 0.010) -> tuple[str, float]:
    """Worst region, scored *relative to how much that region actually moved*.

    An unnormalised worst-region RMS is nearly collinear with the global one
    (r = 0.99 measured on Demo 1 pairs): the lips move most, so they dominate
    both, and the second gate re-states the first instead of adding evidence.
    Dividing by the region's own motion asks a different question -- "is any one
    part of the face doing proportionally the wrong thing?" -- which is what
    catches a matching mouth over a mismatched brow. The floor keeps a barely
    moving region from producing a huge ratio out of noise.
    """
    diff = da - db
    worst_name, worst = "", 0.0
    for name, ids in regions.items():
        motion = max(_rms(da[ids]), _rms(db[ids]), min_motion)
        r = _rms(diff[ids]) / motion
        if r > worst:
            worst_name, worst = name, r
    return worst_name, worst


@dataclass
class PairResult:
    ref: str
    tgt: str
    ref_person: str
    tgt_person: str
    spec: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, list] = field(default_factory=dict)

    def to_json(self, include_frames: bool = True) -> str:
        d = asdict(self)
        if not include_frames:
            d.pop("frames")
        return json.dumps(d, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def verify_pair(
    ref: SeqDescriptor,
    tgt: SeqDescriptor,
    spec: DatasetSpec,
    align: np.ndarray | None = None,
    weights: Weights | None = None,
    identity_cos: float | None = None,
    pose_delta: np.ndarray | None = None,
    bg_hist: float | None = None,
    source_identity_cos: float | None = None,
    driver_identity_cos: float | None = None,
    extra: dict[str, Any] | None = None,
) -> PairResult:
    """Verify one candidate pair.

    `align` is an (N, 2) array of (ref_index, tgt_index). Track A pairs are
    frame-synchronous by construction so the default identity map applies;
    Track B pairs arrive with a DTW alignment.
    """
    weights = weights or Weights()
    w = weights.bs_weights(ref.bs.shape[1])

    if align is None:
        n = min(len(ref), len(tgt))
        align = np.stack([np.arange(n), np.arange(n)], axis=1)
    align = np.asarray(align, dtype=int)

    W, K, tol = spec.rank1_window, spec.rank1_offset_k, spec.rank1_tolerance
    T_t = len(tgt)

    rows: list[dict[str, Any]] = []
    for i in range(align.shape[0]):
        tr, tt = int(align[i, 0]), int(align[i, 1])
        if not (ref.ok[tr] and tgt.ok[tt]):
            continue

        d_bs = bs_distance(ref.bs[tr], tgt.bs[tt], w)
        d_df = deform_distance(ref.deform[tr], tgt.deform[tt])
        d_gz = gaze_distance(ref.gaze[tr], tgt.gaze[tt])
        worst_region, d_rg = region_worst(ref.deform[tr], tgt.deform[tt], REGIONS)

        # rank-1 search over target frames near the aligned index
        lo, hi = max(0, tt - W), min(T_t, tt + W + 1)
        cand = [s for s in range(lo, hi) if tgt.ok[s]]
        if len(cand) >= 3:
            dists = np.array([deform_distance(ref.deform[tr], tgt.deform[s]) for s in cand])
            best = int(cand[int(np.argmin(dists))])
            far = [j for j, s in enumerate(cand) if abs(s - tt) >= K]
            d_far = float(dists[far].min()) if far else np.inf
        else:
            best, d_far = tt, np.inf

        # self-contrast: how much this video changes over +/-K frames
        self_cand = [s for s in (tr - K, tr + K) if 0 <= s < len(ref) and ref.ok[s]]
        d_self = (min(deform_distance(ref.deform[tr], ref.deform[s]) for s in self_cand)
                  if self_cand else 0.0)

        testable = d_self >= spec.min_self_contrast
        ratio = d_df / d_self if d_self > 1e-9 else np.inf
        rank1_ok = abs(best - tt) <= tol
        # The strict clause: beat the video's own temporal neighbour.
        beats_self = (ratio <= spec.max_ratio) if testable else True
        beats_far = (d_df <= spec.max_ratio * d_far) if np.isfinite(d_far) else True

        energy = float(min(ref.energy[tr], tgt.energy[tt]))
        expressive = energy >= spec.min_energy

        d_au = None
        if ref.au is not None and tgt.au is not None:
            from .au import au_distance
            d_au = au_distance(ref.au[tr], tgt.au[tt])

        gates = dict(
            g_bs=bool(d_bs <= spec.max_bs),
            g_deform=bool(d_df <= spec.max_deform),
            g_gaze=bool(d_gz <= spec.max_gaze),
            g_region=bool(d_rg <= spec.max_region),
            g_rank1=bool(rank1_ok),
            g_ratio=bool(beats_self and beats_far),
            g_energy=bool(expressive),
        )
        if d_au is not None and spec.max_au is not None:
            gates["g_au"] = bool(d_au <= spec.max_au)
        rows.append(dict(
            t_ref=tr, t_tgt=tt, d_bs=d_bs, d_deform=d_df, d_gaze=d_gz, d_au=d_au,
            d_region=d_rg, worst_region=worst_region, d_self=d_self,
            d_far=None if not np.isfinite(d_far) else d_far,
            ratio=None if not np.isfinite(ratio) else ratio,
            argmin_offset=best - tt, testable=testable, energy=energy,
            **gates, accepted=bool(all(gates.values())),
        ))

    reasons: list[str] = []
    n_eval = len(rows)
    if n_eval == 0:
        return PairResult(ref.name, tgt.name, ref.person, tgt.person, spec.name,
                          False, ["no jointly valid frames"], {}, {})

    acc = np.array([r["accepted"] for r in rows], dtype=bool)
    testable = np.array([r["testable"] for r in rows], dtype=bool)
    n_pass, n_test = int(acc.sum()), int(testable.sum())
    n_pass_testable = int((acc & testable).sum())
    pass_rate = n_pass / n_eval

    if n_test < spec.min_testable_frames:
        reasons.append(f"only {n_test} testable frames (< {spec.min_testable_frames}); "
                       "clip too static to prove fine-grained agreement")
    if n_pass_testable < spec.min_pass_frames:
        reasons.append(f"{n_pass_testable} accepted testable frames "
                       f"(< {spec.min_pass_frames})")
    if pass_rate < spec.min_pass_rate:
        reasons.append(f"frame pass rate {pass_rate:.2f} < {spec.min_pass_rate}")

    # --- non-expression dissimilarity ---
    if spec.max_identity_cos is not None and identity_cos is not None:
        if identity_cos > spec.max_identity_cos:
            reasons.append(f"identity cosine {identity_cos:.3f} > {spec.max_identity_cos} "
                           "(same or too-similar person)")
    if spec.min_pose_delta_deg is not None:
        if pose_delta is None:
            reasons.append("pose delta unavailable but required by spec")
        else:
            pd = float(np.median(pose_delta[:, :2].max(axis=1)))
            if pd < spec.min_pose_delta_deg:
                reasons.append(f"median pose delta {pd:.1f}deg < {spec.min_pose_delta_deg}deg "
                               "(head motion not different enough)")
    if spec.max_bg_hist is not None:
        if bg_hist is None:
            reasons.append("background similarity unavailable but required by spec")
        elif bg_hist > spec.max_bg_hist:
            reasons.append(f"background similarity {bg_hist:.2f} > {spec.max_bg_hist}")

    # --- generation leakage guards ---
    if spec.min_source_identity_cos is not None and source_identity_cos is not None:
        if source_identity_cos < spec.min_source_identity_cos:
            reasons.append(f"output drifted from its source identity "
                           f"({source_identity_cos:.3f} < {spec.min_source_identity_cos})")
    if spec.max_driver_identity_cos is not None and driver_identity_cos is not None:
        if driver_identity_cos > spec.max_driver_identity_cos:
            reasons.append(f"driver identity leaked into output "
                           f"({driver_identity_cos:.3f} > {spec.max_driver_identity_cos})")

    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    gate_rates = {k: float(np.mean([r[k] for r in rows]))
                  for k in rows[0] if k.startswith("g_")}
    summary = dict(
        gate_pass_rates=gate_rates,
        n_frames=n_eval, n_pass=n_pass, n_testable=n_test,
        n_pass_testable=n_pass_testable, pass_rate=pass_rate,
        med_d_bs=med("d_bs"), med_d_deform=med("d_deform"), med_d_gaze=med("d_gaze"),
        med_d_region=med("d_region"), med_energy=med("energy"),
        med_d_au=med("d_au") if rows[0].get("d_au") is not None else None,
        med_ratio=float(np.median([r["ratio"] for r in rows if r["ratio"] is not None]))
        if any(r["ratio"] is not None for r in rows) else None,
        rank1_rate=float(np.mean([r["g_rank1"] for r in rows])),
        identity_cos=identity_cos, bg_hist=bg_hist,
        pose_delta_med=float(np.median(pose_delta[:, :2].max(axis=1)))
        if pose_delta is not None and len(pose_delta) else None,
        source_identity_cos=source_identity_cos,
        driver_identity_cos=driver_identity_cos,
    )
    if extra:
        summary.update(extra)

    keys = ("t_ref", "t_tgt", "d_bs", "d_deform", "d_gaze", "d_au", "d_region", "d_self",
            "ratio", "argmin_offset", "testable", "energy", "accepted",
            "g_bs", "g_deform", "g_gaze", "g_region", "g_rank1", "g_ratio",
            "g_energy", "g_au")
    frames = {k: [r[k] for r in rows] for k in keys if k in rows[0]}

    return PairResult(ref.name, tgt.name, ref.person, tgt.person, spec.name,
                      accepted=not reasons, reasons=reasons, summary=summary, frames=frames)
